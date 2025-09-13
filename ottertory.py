import os
# Disable torch.compile early, print diagnostics for CUDA/Triton, and add try/except fallback to CPU for model loading
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import flet as ft
import torch
import sounddevice as sd
import numpy as np
import threading
from dataclasses import dataclass
from moshi.models import loaders, LMGen, MimiModel, LMModel
import sentencepiece
import queue
import time
import textwrap
import re
import requests
import pyperclip
import keyboard
import tkinter as tk
from tkinter import ttk
from models import build_model, generate_speech, list_available_voices


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    device: torch.device
    frame_size: int

    def __init__(self, mimi, text_tokenizer, lm, batch_size, device):
        self.mimi = mimi
        self.lm = lm
        self.text_tokenizer = text_tokenizer
        self.batch_size = batch_size
        self.lm_gen = LMGen(lm, temp=0, temp_text=0, use_sampling=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        # Do not start persistent streaming here; will manage per-run streaming context
        self._stream_ctx = None

    def start_streaming(self):
        # Enter a streaming context for LMGen for the current run
        if self._stream_ctx is not None:
            # already streaming
            return
        try:
            self._stream_ctx = self.lm_gen.streaming(self.batch_size)
            # enter the context
            self._stream_ctx.__enter__()
        except Exception as e:
            print("Failed to enter LMGen streaming context:", e)
            self._stream_ctx = None

    def stop_streaming(self):
        # Exit the streaming context if present
        if self._stream_ctx is None:
            return
        try:
            # call __exit__ with no exception
            self._stream_ctx.__exit__(None, None, None)
        except Exception as e:
            print("Failed to exit LMGen streaming context:", e)
        finally:
            self._stream_ctx = None

    def reset_streaming(self, batch_size=None):
        # Reinitialize generator state to start a fresh streaming session
        if batch_size is None:
            batch_size = self.batch_size
        try:
            # Stop any existing streaming context first
            try:
                self.stop_streaming()
            except Exception:
                pass
            # Recreate the LMGen instance to reset internal state; do not auto-enter
            self.lm_gen = LMGen(self.lm, temp=0, temp_text=0, use_sampling=False)
        except Exception as e:
            print("Failed to reset streaming state:", e)

    def process_chunk(self, chunk):
        if len(chunk) != self.frame_size:
            raise ValueError(f"Chunk must be exactly {self.frame_size} samples long")

        # Shape: [1, 1, frame_size]
        pcm = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)

        codes = self.mimi.encode(pcm)
        tokens = None
        # Assume caller has entered streaming context via start_streaming()
        try:
            # Check if we're in a streaming context
            if self._stream_ctx is None:
                print("Warning: Not in streaming context, starting streaming...")
                self.start_streaming()
            tokens = self.lm_gen.step(codes)
        except Exception as e:
            print("LMGen.step error:", e)
            tokens = None

        if tokens is None:
            return ""

        tok = tokens[0, 0].cpu().item()
        if tok in [0, 3]:  # Skip special tokens
            return ""
        txt = self.text_tokenizer.id_to_piece(tok).replace("▁", " ")
        return txt


# Load Kyutai model once (may take some time)
# Use a device string for the loaders (they expect 'cuda' or 'cpu')
# We'll detect CUDA/triton and fall back to CPU if loading on CUDA fails.
import importlib

cuda_available = torch.cuda.is_available()
device_str = "cuda" if cuda_available else "cpu"
# torch device for tensors
device = torch.device(device_str)
print(f"CUDA available: {cuda_available}, using device: {device_str}")
# Check Triton availability (may be missing on Windows)
triton_ok = importlib.util.find_spec("triton") is not None
print(f"triton available: {triton_ok}")

ck = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
try:
    # Try loading on requested device
    mimi = ck.get_mimi(device=device_str)
    tokenizer = ck.get_text_tokenizer()
    lm = ck.get_moshi(device=device_str)
    state = InferenceState(mimi, tokenizer, lm, batch_size=1, device=device)
except Exception as e:
    print(f"Model load failed on {device_str}:", e)
    if device_str == "cuda":
        print("Falling back to CPU and retrying model load...")
        try:
            device_str = "cpu"
            device = torch.device(device_str)
            mimi = ck.get_mimi(device=device_str)
            tokenizer = ck.get_text_tokenizer()
            lm = ck.get_moshi(device=device_str)
            state = InferenceState(mimi, tokenizer, lm, batch_size=1, device=device)
        except Exception as e2:
            print("Model load failed on CPU too:", e2)
            raise
    else:
        raise

# Shared state
recording_flag = threading.Event()
pause_flag = threading.Event()
audio_queue = queue.Queue()
current_run_id = 0
last_audio_level = 0.0
last_audio_time = 0
audio_level_lock = threading.Lock()
audio_source_is_microphone = True  # True = microphone, False = system audio

# TTS control event to allow stopping playback
tts_stop_event = threading.Event()

# Import ClosedCaptionOverlay from the new module
from caption_overlay import ClosedCaptionOverlay

# Function to toggle between microphone and system audio
def toggle_audio_source():
    """Toggle between microphone and system audio capture"""
    global audio_source_is_microphone
    audio_source_is_microphone = not audio_source_is_microphone
    source_name = "Microphone" if audio_source_is_microphone else "System Audio"
    print(f"Audio source switched to: {source_name}")
    return source_name

# Global overlay instance
overlay = ClosedCaptionOverlay()


def audio_stream_thread(run_id: int):
    global audio_source_is_microphone
    samplerate = int(state.mimi.sample_rate)
    frames_per_chunk = int(state.frame_size)

    def callback(indata, frames, time_info, status):
        # Only enqueue while recording and not paused and still same run
        if not recording_flag.is_set() or pause_flag.is_set() or run_id != current_run_id:
            return
        try:
            audio_queue.put(indata.copy())
            # compute RMS level for UI
            rms = float(np.sqrt(np.mean(indata.astype('float32') ** 2)))
            with audio_level_lock:
                global last_audio_level, last_audio_time
                last_audio_level = rms
                last_audio_time = time.time()
        except Exception:
            pass

    try:
        # Determine which device to use
        device_id = None
        if not audio_source_is_microphone:
            # Find the default output device for system audio capture
            try:
                # Get default output device info
                default_output = sd.query_devices(kind='output')
                
                # Try to find a corresponding input device (like "Stereo Mix" or similar)
                # This is platform-specific and may not work on all systems
                devices = sd.query_devices()
                
                # Look for devices that might capture system audio
                system_audio_keywords = ['vb-audio', 'cable output', 'vb-cable', 'loopback']
                system_audio_device = None
                vb_audio_devices = []
                
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        device_name_lower = device['name'].lower()
                        if 'vb-audio' in device_name_lower or 'cable' in device_name_lower:
                            vb_audio_devices.append((i, device))
                        elif any(keyword in device_name_lower for keyword in system_audio_keywords):
                            system_audio_device = i
                
                # Prefer VB-Audio devices if found
                if vb_audio_devices:
                    # Sort to prefer devices with both 'vb-audio' and 'cable' in the name
                    vb_audio_devices.sort(key=lambda x: ('vb-audio' in x[1]['name'].lower(), 'cable' in x[1]['name'].lower()), reverse=True)
                    device_id = vb_audio_devices[0][0]
                    print(f"Using VB-Audio device: {vb_audio_devices[0][1]['name']}")
                elif system_audio_device is not None:
                    device_id = system_audio_device
                    print(f"Using system audio device: {devices[system_audio_device]['name']}")
                else:
                    print("Warning: No suitable system audio capture device found, falling back to default input")
                    device_id = None
                    
            except Exception as e:
                print(f"Error setting up system audio capture: {e}")
                device_id = None
        else:
            # Use default microphone
            device_id = None
            print("Using default microphone")

        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=samplerate,
            blocksize=frames_per_chunk,
            dtype="float32",
            callback=callback,
        ):
            # Keep the stream open while recording_flag is set and run_id matches
            while recording_flag.is_set() and run_id == current_run_id:
                time.sleep(0.1)
    except Exception as e:
        error_msg = f"Audio input stream error: {e}"
        if not audio_source_is_microphone:
            error_msg += "\nNote: System audio capture may not be available on your system. Try enabling 'Stereo Mix' in Windows sound settings or using audio software that supports loopback."
        print(error_msg)


def transcriber_loop(result_field: ft.TextField, page: ft.Page, run_id: int, text_to_command_mode=False, command_field=None, auto_process_func=None, show_overlay=False):
    buffer = ""
    accumulated = np.empty((0,), dtype=np.float32)
    frame_size = state.frame_size
    last_update_time = 0
    min_update_interval = 0.1  # More frequent updates for better responsiveness
    last_audio_time = time.time()
    silence_threshold = 0.5  # Seconds of silence before forcing an update
    
    # Initialize overlay if showing AND it's not already visible
    if show_overlay and not overlay.is_visible:
        overlay.show()
    # If not showing overlay, make sure it's hidden
    elif not show_overlay and overlay.is_visible:
        overlay.hide()

    def update_ui(current_text):
        nonlocal last_update_time, buffer
        current_time = time.time()
        
        # Always update the buffer with the latest text
        buffer = current_text
        
        # Update the UI if enough time has passed or if we detect the end of a sentence
        should_update = (
            (current_time - last_update_time >= min_update_interval) or
            ('.' in current_text or '?' in current_text or '!' in current_text)
        )
        
        if should_update and run_id == current_run_id:
            if text_to_command_mode and command_field is not None:
                command_field.value = buffer.strip()
                page.update()
                if show_overlay:
                    overlay.update_text(buffer.strip(), is_live_transcription=False)
            else:
                result_field.value = textwrap.fill(buffer.strip(), width=80)
                page.update()
                if show_overlay:
                    overlay.update_text(buffer.strip(), is_live_transcription=True)
            
            last_update_time = current_time
            return True
        return False

    try:
        # Keep transcribing while recording or while there is still audio in the queue
        while (recording_flag.is_set() or not audio_queue.empty() or len(accumulated) >= frame_size) and run_id == current_run_id:
            current_time = time.time()
            
            # Process any pending audio chunks
            try:
                # Try to get more audio if we don't have enough for a full frame
                if len(accumulated) < frame_size and recording_flag.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)  # Shorter timeout for more responsive updates
                        audio_samples = chunk[:, 0]  # 1D array
                        accumulated = np.concatenate((accumulated, audio_samples))
                        last_audio_time = current_time
                    except queue.Empty:
                        # If we're not recording anymore, process what we have
                        if not recording_flag.is_set() and len(accumulated) > 0:
                            # Pad with zeros if needed to process the remaining audio
                            if len(accumulated) < frame_size:
                                padding = np.zeros(frame_size - len(accumulated), dtype=np.float32)
                                accumulated = np.concatenate((accumulated, padding))
            except Exception as e:
                print(f"Error getting audio chunk: {e}")
                continue

            # Process complete frames
            text_updated = False
            while len(accumulated) >= frame_size and run_id == current_run_id:
                to_process = accumulated[:frame_size]
                accumulated = accumulated[frame_size:]
                
                try:
                    text = state.process_chunk(to_process)
                    if text:
                        buffer += text
                        text_updated = True
                        last_audio_time = current_time
                except Exception as e:
                    print(f"Processing error: {e}")
                    continue
            
            # Check if we should update the UI
            if text_updated or (current_time - last_audio_time < silence_threshold and buffer):
                update_ui(buffer)
            
            # Small sleep to prevent busy waiting
            time.sleep(0.01)
            
        # Final update with any remaining text
        if buffer and run_id == current_run_id:
            update_ui(buffer)
            
    except Exception as e:
        print(f"Error in transcriber loop: {e}")
    finally:
        # Ensure the final text is displayed
        if buffer and run_id == current_run_id:
            update_ui(buffer)


# Simple lazy-loaded Kokoro model handle
kokoro_model = None

def load_kokoro_model(device_str="cpu"):
    """Lazy load kokoro TTS model (build_model from models.py)."""
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    try:
        kokoro_model = build_model("kokoro-v1_0.pth", device=device_str)
    except Exception as e:
        print("Failed to load kokoro model:", e)
        kokoro_model = None
    return kokoro_model


def get_llm_models():
    """Return list of available LLM models from both Ollama and LM Studio.
    
    Returns:
        list: List of model names with 'lmstudio::' prefix for LM Studio models
    """
    models = []
    
    # Get Ollama models
    ollama_models = []
    try:
        resp = requests.get('http://localhost:11434/api/tags', timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if 'models' in data:
                ollama_models = [m['name'] for m in data['models']]
    except Exception as e:
        print(f"Warning: Could not fetch Ollama models: {e}")
    
    # Get LM Studio models
    lmstudio_models = []
    try:
        resp = requests.get('http://localhost:1234/v1/models', timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                # Deduplicate models by ID, keeping the first occurrence
                seen = set()
                for model in data['data']:
                    if 'id' in model and model['id'] not in seen:
                        seen.add(model['id'])
                        lmstudio_models.append(f"lmstudio::{model['id']}")
    except Exception as e:
        print(f"Warning: Could not fetch LM Studio models: {e}")
    
    # Combine models with LM Studio models at the bottom
    models = ollama_models + lmstudio_models
    
    # If no models found, provide a default
    if not models and not ollama_models and not lmstudio_models:
        models = ['gemma3:4b-it-qat']  # Default fallback
    
    return models


def save_last_used_ollama_model(model_name):
    """Save the last used model name to a file."""
    if not model_name:
        return
    cfg = os.path.join(os.path.expanduser("~"), ".antistentorian_last_llm_model.txt")
    try:
        with open(cfg, "w", encoding="utf-8") as f:
            f.write(model_name)
    except Exception as e:
        print(f"Warning: Could not save last used model: {e}")


def load_last_used_ollama_model():
    """Load the last used model name from file."""
    cfg = os.path.join(os.path.expanduser("~"), ".antistentorian_last_llm_model.txt")
    try:
        if os.path.exists(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        print(f"Warning: Could not load last used model: {e}")
    return None


def call_llm_custom_command(selected, text, command):
    """Call LM Studio (prefixed) or Ollama to run a simple instruction on text."""
    if not selected:
        raise RuntimeError("No LLM model selected")
    if selected.startswith('lmstudio::'):
        model = selected[len('lmstudio::'):]
        url = 'http://localhost:1234/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Respond in plain text."},
                {"role": "user", "content": f"Process the following text as per the instruction below.\n\nText:\n{text}\n\nInstruction:\n{command}"}
            ]
        }
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data['choices'][0]['message']['content']
    else:
        # Ollama chat wrapper
        try:
            from ollama import chat
            response = chat(
                model=selected,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond in plain text."},
                    {"role": "user", "content": f"Process the following text as per the instruction below.\n\nText:\n{text}\n\nInstruction:\n{command}"}
                ]
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")


def clear_ollama_memory(selected):
    """Clear the LLM's conversation memory by sending a reset command"""
    if not selected:
        print("No LLM model selected")
        return
    if selected.startswith('lmstudio::'):
        # For LM Studio, we can't directly clear memory, but we can send a reset message
        model = selected[len('lmstudio::'):]
        url = 'http://localhost:1234/v1/chat/completions'
        headers = {'Content-Type': 'application/json'}
        try:
            # Send a system message to reset the conversation
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": "Conversation history cleared."}]
            }
            requests.post(url, json=payload, headers=headers, timeout=5)
            print("LM Studio memory cleared")
        except Exception:
            pass  # Ignore errors when clearing memory
    else:
        # For Ollama, we can't directly clear memory, but we can send a reset message
        try:
            from ollama import chat
            chat(model=selected, messages=[{"role": "system", "content": "Conversation history cleared."}])
            print("Ollama memory cleared")
        except Exception:
            pass  # Ignore errors when clearing memory


def _play_tts_if_enabled(text, voice_name, device_str, speed=1.0):
    """Helper function to play TTS if auto-TTS is enabled"""
    global auto_tts_active
    if auto_tts_active and text.strip():
        try:
            print("Auto-TTS: Playing response...")
            play_tts_ui(voice_name, text, device_str, speed)
        except Exception as e:
            print(f"Error in auto-TTS playback: {e}")

def summarize_text_ui(selected_model, result_field, page):
    """Fetch transcript from result_field and summarize using selected_model."""
    transcript = result_field.value or ""
    if not transcript.strip():
        print("No text to summarize")
        return
    try:
        summary = call_llm_custom_command(selected_model, transcript, "Summarize the following text.")
        result_field.value = summary
        page.update()
        # Trigger TTS if enabled
        _play_tts_if_enabled(summary, voice_dropdown.value, device_str, speed_slider.value)
    except Exception as e:
        print("Summarize failed:", e)


def run_bullet_points_ui(selected_model, result_field, page):
    txt = result_field.value or ""
    if not txt.strip():
        print("No text for bullet points")
        return
    try:
        out = call_llm_custom_command(selected_model, txt, "Extract the main topics from the following text and list them. Dont begin your response with anything but the summary. make sure each sentence ends with a period.")
        result_field.value = out
        page.update()
        # Trigger TTS if enabled
        _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value)
    except Exception as e:
        print("Bullet points failed:", e)


def run_proofread_ui(selected_model, result_field, page):
    txt = result_field.value or ""
    if not txt.strip():
        print("No text for proofreading")
        return
    try:
        out = call_llm_custom_command(selected_model, txt, "Proofread the following text and correct any punctuation errors, including fixing any excessive periods.")
        result_field.value = out
        page.update()
        # Trigger TTS if enabled
        _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value)
    except Exception as e:
        print("Proofread failed:", e)


def play_tts_ui(voice_name, text, device_str, speed=1.0):
    """Generate audio with kokoro and play it. Improved: sentence-split, chunk, retry and fallback."""
    # Clear stop flag at start
    tts_stop_event.clear()
    if not text or not text.strip():
        print("No text to play")
        return
    # Normalize whitespace and remove forced line breaks from wrapped live text
    normalized_text = re.sub(r"\s+", " ", text).strip()
    model = load_kokoro_model(device_str)
    if model is None:
        print("Kokoro model not available")
        return

    # Split into sentences while keeping punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', normalized_text) if s.strip()]
    if not sentences:
        sentences = [normalized_text]

    # Group sentences into chunks (max chars per chunk)
    max_chars = 400
    chunks = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    import numpy as _np
    audio_pieces = []
    silence = _np.zeros(int(24000 * 0.06), dtype=_np.float32)  # 60ms silence between pieces

    for idx, chunk in enumerate(chunks):
        if tts_stop_event.is_set():
            print("TTS stopped by user")
            return
        if not chunk.endswith(('.', '!', '?')):
            chunk += '.'
        print(f"TTS: generating chunk {idx+1}/{len(chunks)} (chars={len(chunk)})")

        # Try generation with retries
        audio_tensor = None
        for attempt in range(3):
            if tts_stop_event.is_set():
                print("TTS stopped during generation")
                return
            try:
                audio_tensor, _ = generate_speech(model, chunk, voice=voice_name, speed=speed)
            except Exception as e:
                print(f"generate_speech exception: {e}")
                audio_tensor = None
            if audio_tensor is not None:
                break
            time.sleep(0.25)

        # Fallback: if whole chunk failed, try per-sentence generation
        if audio_tensor is None:
            print(f"Chunk generation failed, attempting per-sentence fallback for chunk {idx+1}")
            sub_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if s.strip()]
            for ss in sub_sentences:
                if tts_stop_event.is_set():
                    print("TTS stopped during fallback generation")
                    return
                if not ss.endswith(('.', '!', '?')):
                    ss += '.'
                for attempt in range(2):
                    try:
                        at, _ = generate_speech(model, ss, voice=voice_name, speed=speed)
                    except Exception as e:
                        print(f"generate_speech exception (fallback): {e}")
                        at = None
                    if at is not None:
                        audio_pieces.append(at.numpy().astype(_np.float32))
                        audio_pieces.append(silence)
                        break
                    time.sleep(0.15)
            # continue to next chunk after fallback
            continue

        # Append generated piece plus small silence to avoid seam issues
        piece = audio_tensor.numpy().astype(_np.float32)
        audio_pieces.append(piece)
        audio_pieces.append(silence)

    if not audio_pieces:
        print("No audio generated")
        return

    try:
        concatenated = _np.concatenate(audio_pieces)
    except Exception as e:
        print("Error concatenating audio pieces:", e)
        return

    # Normalize modestly to avoid clipping
    peak = max(1e-9, float(_np.max(_np.abs(concatenated))))
    if peak > 1.0:
        concatenated = concatenated / peak

    try:
        import sounddevice as _sd
        _sd.play(concatenated, 24000)
        # During playback poll for stop event
        start_time = time.time()
        duration = len(concatenated) / 24000.0
        while (time.time() - start_time) < duration:
            if tts_stop_event.is_set():
                _sd.stop()
                print("Playback stopped by user")
                break
            time.sleep(0.05)
    except Exception as e:
        print("Error playing TTS:", e)


# Replace download_tts_ui with improved version
def download_tts_ui(voice_name, text, device_str, speed=1.0):
    """Generate full TTS audio and save as WAV. Improved chunking, retries and stable write."""
    if not text or not text.strip():
        print("No text to save")
        return
    # Normalize whitespace/newlines to avoid wrapped-line artifacts from live text
    normalized_text = re.sub(r"\s+", " ", text).strip()
    model = load_kokoro_model(device_str)
    if model is None:
        print("Kokoro model not available")
        return

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', normalized_text) if s.strip()]
    if not sentences:
        sentences = [normalized_text]

    max_chars = 400
    chunks = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)

    import numpy as _np, soundfile as _sf, os, time
    audio_pieces = []
    silence = _np.zeros(int(24000 * 0.06), dtype=_np.float32)

    for idx, chunk in enumerate(chunks):
        if not chunk.endswith(('.', '!', '?')):
            chunk += '.'
        print(f"TTS download: generating chunk {idx+1}/{len(chunks)}")

        audio_tensor = None
        for attempt in range(3):
            try:
                audio_tensor, _ = generate_speech(model, chunk, voice=voice_name, speed=speed)
            except Exception as e:
                print(f"generate_speech exception: {e}")
                audio_tensor = None
            if audio_tensor is not None:
                break
            time.sleep(0.25)

        if audio_tensor is None:
            # Fallback to sentence-level
            print(f"Chunk generation failed for download; falling back to sentences for chunk {idx+1}")
            sub_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if s.strip()]
            for ss in sub_sentences:
                if not ss.endswith(('.', '!', '?')):
                    ss += '.'
                for attempt in range(2):
                    try:
                        at, _ = generate_speech(model, ss, voice=voice_name, speed=speed)
                    except Exception as e:
                        print(f"generate_speech exception (fallback): {e}")
                        at = None
                    if at is not None:
                        audio_pieces.append(at.numpy().astype(_np.float32))
                        audio_pieces.append(silence)
                        break
                    time.sleep(0.15)
            continue

        piece = audio_tensor.numpy().astype(_np.float32)
        audio_pieces.append(piece)
        audio_pieces.append(silence)

    if not audio_pieces:
        print("No audio generated for download")
        return

    try:
        concatenated = _np.concatenate(audio_pieces)
    except Exception as e:
        print("Error concatenating audio pieces:", e)
        return

    out_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(out_dir, f"tts_{ts}.wav")

    try:
        _sf.write(wav_path, concatenated, 24000)
        print(f"Saved TTS WAV: {wav_path}")
    except Exception as e:
        print("Error writing WAV file:", e)

def stop_tts_ui():
    """Signal TTS playback to stop."""
    tts_stop_event.set()
    try:
        import sounddevice as _sd
        _sd.stop()
    except Exception:
        pass
    print("TTS stop signaled")


def copy_text_ui(result_field: ft.TextField):
    try:
        txt = result_field.value or ""
        if txt.strip():
            pyperclip.copy(txt)
            print("Copied text to clipboard")
        else:
            print("No text to copy")
    except Exception as e:
        print("Copy failed:", e)


def main(page: ft.Page):
    page.title = " Ottertory "
    page.scroll = ft.ScrollMode.AUTO
    result_text = ft.TextField(
        label="Live Transcription", multiline=True, min_lines=8, expand=True
    )
    
    # Global variable to track text destination mode
    text_to_command_mode = False
    overlay_enabled = False

    # Audio source toggle button
    audio_source_toggle = ft.ElevatedButton(
        "Source: Mic",
        tooltip="Toggle between microphone and system audio capture"
    )
    
    def _toggle_audio_source(e):
        """Handle audio source toggle button click"""
        source_name = toggle_audio_source()
        if audio_source_is_microphone:
            audio_source_toggle.text = "Source: Mic"
            audio_source_toggle.bgcolor = None
        else:
            audio_source_toggle.text = "Source: System"
            audio_source_toggle.bgcolor = ft.Colors.ORANGE_400
        page.update()
    
    audio_source_toggle.on_click = _toggle_audio_source
    
    # Microphone status UI
    mic_status = ft.Text("Mic: Idle")
    level_bar = ft.ProgressBar(value=0, width=200)

    # >>> Insert Ollama and TTS controls here
    ollama_models = get_llm_models()
    ollama_opts = [ft.dropdown.Option(m) for m in ollama_models] if ollama_models else []
    last_ollama_model = load_last_used_ollama_model()
    if last_ollama_model and last_ollama_model in ollama_models:
        default_ollama = last_ollama_model
    else:
        default_ollama = ollama_models[0] if ollama_models else None
    def _on_model_changed(e):
        """Save the selected model when changed"""
        if e.control.value:
            save_last_used_ollama_model(e.control.value)
            print(f"Saved model selection: {e.control.value}")
    
    ollama_dropdown = ft.Dropdown(
        label="LLM Model", 
        options=ollama_opts, 
        value=default_ollama,
        on_change=_on_model_changed
    )
    summarize_button = ft.ElevatedButton("Summarize", on_click=lambda e: threading.Thread(target=summarize_text_ui, args=(ollama_dropdown.value, result_text, page), daemon=True).start())
    bullets_button = ft.ElevatedButton("Bullet Points", on_click=lambda e: threading.Thread(target=run_bullet_points_ui, args=(ollama_dropdown.value, result_text, page), daemon=True).start())
    proof_button = ft.ElevatedButton("Proofread", on_click=lambda e: threading.Thread(target=run_proofread_ui, args=(ollama_dropdown.value, result_text, page), daemon=True).start())
    refresh_button = ft.ElevatedButton("⟳", on_click=lambda e: (ollama_dropdown.options.clear(), [ollama_dropdown.options.append(ft.dropdown.Option(m)) for m in get_llm_models()], page.update()))

    # Add Ollama command input and Run/Clear buttons
    command_field = ft.TextField(label="Ollama Command", expand=True)
    
    # Text destination toggle button
    text_dest_toggle = ft.ElevatedButton("Text → Live", on_click=None)  # Will be set below
    
    # Initialize overlay state properly
    overlay_enabled = False  # Start with overlay disabled
    overlay_toggle = ft.ElevatedButton(
        "Overlay: OFF",
        tooltip="Toggle closed captioning overlay"
    )
    
    def _toggle_overlay(e):
        """Toggle closed captioning overlay"""
        nonlocal overlay_enabled
        overlay_enabled = not overlay_enabled
        if overlay_enabled:
            overlay_toggle.text = "Overlay: ON"
            overlay_toggle.bgcolor = ft.Colors.GREEN_400
            # Show overlay immediately when enabled
            overlay.show()
            print("Overlay enabled")
        else:
            overlay_toggle.text = "Overlay: OFF"
            overlay_toggle.bgcolor = None
            # Hide overlay immediately when disabled
            overlay.hide()
            overlay.clear_text_only()
            print("Overlay disabled")
        page.update()
    # Auto-TTS toggle button and state
    auto_tts_toggle = ft.ElevatedButton("Auto-TTS: OFF", on_click=None)
    global auto_tts_active, last_processed_text
    auto_tts_active = False
    last_processed_text = ""
    
    def _run_llm_command_thread():
        try:
            cmd = command_field.value or ""
            if not cmd.strip():
                print("No command to run")
                return
            
            # Special command to clear LLM memory (accepts both /clear and just /)
            if cmd.lower() in ("/clear", "/"):
                clear_ollama_memory(ollama_dropdown.value)
                result_text.value = ""
                command_field.value = ""
                page.update()
                print("Output and LLM memory cleared")
                return
            
            out = call_llm_custom_command(ollama_dropdown.value, result_text.value or "", cmd)
            result_text.value = out
            command_field.value = ""  # Clear the command field after running
            page.update()
            # Trigger TTS if enabled
            _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value)
        except Exception as err:
            print("LLM command error:", err)
    
    def _on_command_enter(e):
        """Handle Ollama command submission"""
        cmd = command_field.value or ""
        if not cmd.strip():
            return
            
        def _run_command():
            try:
                # Use the command field value as the instruction
                out = call_llm_custom_command(ollama_dropdown.value, result_text.value or "", cmd)
                result_text.value = out
                page.update()
                command_field.value = ""  # Clear the command field after successful execution
                page.update()
                # Trigger TTS if enabled
                _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value)
            except Exception as ex:
                print("Error running command:", ex)
        
        # Run in a separate thread to avoid blocking the UI
        threading.Thread(target=_run_command, daemon=True).start()
    
    def _toggle_text_destination(e):
        """Toggle between text going to live transcription or command field"""
        nonlocal text_to_command_mode
        text_to_command_mode = not text_to_command_mode
        if text_to_command_mode:
            text_dest_toggle.text = "Text → Command"
            text_dest_toggle.color = ft.Colors.WHITE
            text_dest_toggle.bgcolor = ft.Colors.ORANGE
        else:
            text_dest_toggle.text = "Text → Live"
            text_dest_toggle.color = None
            text_dest_toggle.bgcolor = None
        page.update()
        print(f"Text destination: {'Command (Auto-Ollama)' if text_to_command_mode else 'Live Transcription'}")
    
    # Overlay is now automatic, no toggle needed
    
    def _toggle_auto_tts(e):
        global auto_tts_active
        auto_tts_active = not auto_tts_active
        if auto_tts_active:
            auto_tts_toggle.text = "Auto-TTS: ON"
            auto_tts_toggle.bgcolor = ft.Colors.BLUE_400
            # If there's text when enabling auto-tts, read it
            if result_text.value.strip():
                threading.Thread(
                    target=play_tts_ui, 
                    args=(voice_dropdown.value, result_text.value, device_str, speed_slider.value),
                    daemon=True
                ).start()
        else:
            auto_tts_toggle.text = "Auto-TTS: OFF"
            auto_tts_toggle.bgcolor = None
        page.update()
        print(f"Auto-TTS: {'ON' if auto_tts_active else 'OFF'}")
        
    def _process_final_text(text):
        global last_processed_text
        # Always use the full text from the result field
        full_text = result_text.value
        if not full_text.strip():
            return
            
        # Only process if text is different from last processed
        if full_text != last_processed_text:
            last_processed_text = full_text
            
            def _play_tts(text_to_speak):
                try:
                    print(f"Playing TTS: {text_to_speak[:50]}...")
                    play_tts_ui(voice_dropdown.value, text_to_speak, device_str, speed_slider.value)
                except Exception as e:
                    print(f"Error in TTS playback: {e}")
            
            # Use the full transcription text
            if auto_tts_active and full_text.strip():
                print(f"Starting TTS for full transcription...")
                t = threading.Thread(
                    target=_play_tts,
                    args=(full_text,),
                    daemon=True
                )
                t.start()
    
    def _auto_process_with_ollama(text):
        """Automatically process text with Ollama when in command mode"""
        if not text.strip():
            return
        try:
            # Use the transcribed text as a direct command to the LLM
            # If there's existing text in the result field, use it as context
            context_text = result_text.value or ""
            if context_text.strip():
                # Use existing text as context and transcribed text as command
                out = call_llm_custom_command(ollama_dropdown.value, context_text, text)
            else:
                # No context, just execute the transcribed text as a command
                out = call_llm_custom_command(ollama_dropdown.value, "", text)
            result_text.value = out
            page.update()
        except Exception as err:
            print("Auto-Ollama processing error:", err)
    
    # Bind Enter key to command field
    command_field.on_submit = _on_command_enter
    text_dest_toggle.on_click = _toggle_text_destination
    overlay_toggle.on_click = _toggle_overlay
    auto_tts_toggle.on_click = _toggle_auto_tts
    
    run_command_button = ft.ElevatedButton("Run", on_click=lambda e: threading.Thread(target=_run_llm_command_thread, daemon=True).start())
    clear_command_button = ft.ElevatedButton("Clear", on_click=lambda e: (command_field.update(value=""), page.update()))

    # TTS controls
    voices = list_available_voices()
    if not voices:
        voices = ["af_bella"]
    # TTS action buttons (now with icons)
    play_tts_button = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        tooltip="Play TTS",
        on_click=lambda e: threading.Thread(
            target=play_tts_ui, 
            args=(voice_dropdown.value, result_text.value, device_str, speed_slider.value), 
            daemon=True
        ).start()
    )
    stop_tts_button = ft.IconButton(
        icon=ft.Icons.STOP,
        tooltip="Stop TTS",
        on_click=lambda e: stop_tts_ui()
    )
    # Voice and other TTS controls
    voice_dropdown = ft.Dropdown(
        label="Voice", 
        options=[ft.dropdown.Option(v) for v in voices], 
        value=voices[0],
        expand=1
    )
    speed_slider = ft.Slider(
        min=1, 
        max=1.5, 
        divisions=20, 
        value=1, 
        label="Speed",
        expand=2
    )
    download_tts_button = ft.ElevatedButton(
        "Download Audio", 
        on_click=lambda e: threading.Thread(
            target=download_tts_ui, 
            args=(voice_dropdown.value, result_text.value, device_str, speed_slider.value), 
            daemon=True
        ).start()
    )
    copy_button = ft.ElevatedButton("Copy Text", on_click=lambda e: copy_text_ui(result_text))

    start_button = ft.ElevatedButton("Start", disabled=False)
    stop_button = ft.ElevatedButton("Stop", disabled=True)
    # Flag to indicate that when recording stops we should run the Ollama command
    run_llm_on_stop = False

    def start_stream_for_ollama(e):
        nonlocal run_llm_on_stop
        run_llm_on_stop = True
        start_stream(e)

    # Button to record then run Ollama command automatically
    record_llm_button = ft.ElevatedButton("Record → Run", on_click=lambda e: start_stream_for_ollama(e))

    # Add new controls into the page layout
    # Insert the new rows into the Column later when building the UI

    def start_stream(e):
        nonlocal run_llm_on_stop, text_to_command_mode, overlay_enabled
        global current_run_id
        # increment run id to invalidate any previous threads
        current_run_id += 1
        my_run = current_run_id

        start_button.disabled = True
        stop_button.disabled = False
        page.update()

        # Reset model streaming state so we start fresh BEFORE recording begins
        try:
            state.reset_streaming(batch_size=1)
            # explicitly enter streaming context for this run
            state.start_streaming()
        except Exception as e:
            print("Warning: failed to reset/start model streaming state:", e)

        pause_flag.clear()
        recording_flag.set()

        # reset audio level
        with audio_level_lock:
            global last_audio_level, last_audio_time
            last_audio_level = 0.0
            last_audio_time = 0.0

        # clear visible text so we don't flash old content
        if text_to_command_mode:
            command_field.value = ""
        else:
            result_text.value = ""
        
        # Only show overlay when starting recording if overlay is enabled
        if overlay_enabled:
            overlay.show()
            overlay.clear_text_only()
        # If overlay is disabled, make sure it stays hidden
        elif not overlay_enabled:
            overlay.hide()
        
        page.update()

        threading.Thread(target=audio_stream_thread, args=(my_run,), daemon=True).start()
        threading.Thread(
            target=transcriber_loop, args=(result_text, page, my_run, text_to_command_mode, command_field, _auto_process_with_ollama, overlay_enabled), daemon=True
        ).start()

    def stop_stream(e):
        nonlocal run_llm_on_stop, text_to_command_mode, overlay_enabled
        global current_run_id
        # Stop recording and allow the transcriber to drain the queue
        pause_flag.set()
        recording_flag.clear()

        # Increment run id to invalidate active threads
        current_run_id += 1

        # Clear any queued audio so next run starts fresh
        try:
            while not audio_queue.empty():
                audio_queue.get_nowait()
        except Exception:
            pass

        # Exit streaming context so LMGen is not left open
        try:
            state.stop_streaming()
        except Exception as e:
            print("Warning: failed to stop model streaming state on stop:", e)

        # Only hide overlay when recording stops if overlay is disabled
        # If overlay is enabled, keep it visible but clear text
        if overlay_enabled:
            overlay.clear_text_only()  # Keep overlay visible but clear text
        else:
            overlay.hide()  # Hide overlay if disabled
        
        start_button.disabled = False
        stop_button.disabled = True
        
        # Trigger Auto-TTS if enabled
        if auto_tts_active and result_text.value.strip():
            print("Auto-TTS: Processing final text after recording")
            _process_final_text(result_text.value)
            
        page.update()

        # If recording was started specifically to run the LLM command, do so now in background
        if run_llm_on_stop:
            def _run_after_record():
                try:
                    cmd = command_field.value or ""
                    if not cmd.strip():
                        print("No Ollama command provided; skipping")
                    else:
                        out = call_llm_custom_command(ollama_dropdown.value, result_text.value or "", cmd)
                        result_text.value = out
                        page.update()
                except Exception as ex:
                    print("Error running Ollama command after record:", ex)
            threading.Thread(target=_run_after_record, daemon=True).start()
            run_llm_on_stop = False
        
        # If in command mode, auto-process the transcribed text
        if text_to_command_mode and command_field.value.strip():
            def _auto_process_after_record():
                try:
                    transcribed_text = command_field.value.strip()
                    if transcribed_text:
                        # Use transcribed text as a direct command to the LLM
                        # If there's existing text in the result field, use it as context
                        context_text = result_text.value or ""
                        if context_text.strip():
                            # Use existing text as context and transcribed text as command
                            out = call_llm_custom_command(ollama_dropdown.value, context_text, transcribed_text)
                        else:
                            # No context, just execute the transcribed text as a command
                            out = call_llm_custom_command(ollama_dropdown.value, "", transcribed_text)
                        result_text.value = out
                        command_field.value = ""  # Clear command field after processing
                        page.update()
                        print(f"Executed voice command: '{transcribed_text}'")
                except Exception as ex:
                    print("Error auto-processing transcribed text:", ex)
            threading.Thread(target=_auto_process_after_record, daemon=True).start()

    start_button.on_click = start_stream
    stop_button.on_click = stop_stream
    record_llm_button.on_click = start_stream_for_ollama
    
    # Add Alt+Space hotkey functionality
    def _alt_space_hotkey():
        """Handle Alt+Space hotkey to toggle recording"""
        if start_button.disabled:
            # Currently recording, stop it
            stop_stream(None)
        else:
            # Not recording, start it
            start_stream(None)
    
    # Register Alt+Z hotkey
    try:
        keyboard.add_hotkey('alt+z', _alt_space_hotkey, suppress=False)
        print("Alt+Z hotkey registered for recording toggle")
    except Exception as e:
        print(f"Failed to register Alt+Space hotkey: {e}")
    
    # Create overlay window immediately
    try:
        print("Creating overlay window...")
        overlay.create_overlay()
        if overlay.root is not None:
            print("Overlay window created successfully")
        else:
            print("Failed to create overlay window")
    except Exception as e:
        print(f"Error creating overlay: {e}")
        import traceback
        traceback.print_exc()

    # UI refresher shows mic status and level
    def ui_refresher():
        while True:
            with audio_level_lock:
                lvl = last_audio_level
                t = last_audio_time
            now = time.time()
            status = "No input" if (now - t) > 1.0 else "Receiving"
            mic_status.value = f"Mic: {status}"
            # map RMS to progress bar (tweak scale as needed)
            level_bar.value = min(1.0, lvl / 0.1)
            try:
                page.update()
                # Update tkinter overlay if it exists
                if overlay.root is not None:
                    overlay.root.update()
            except Exception:
                pass
            time.sleep(0.2)

    threading.Thread(target=ui_refresher, daemon=True).start()

    page.add(
        ft.Column(
            [
                ft.Text("Ottertory", size=18),
                # TTS Controls at the top (with icon buttons first)
                ft.Row([
                    play_tts_button,
                    stop_tts_button,
                    voice_dropdown,
                    speed_slider,
                    download_tts_button,
                    copy_button,
                    auto_tts_toggle
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                # Recording controls
                ft.Row([
                    start_button, 
                    stop_button, 
                    audio_source_toggle, 
                    overlay_toggle, 
                    mic_status, 
                    level_bar
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                # LLM model selection and actions
                ft.Row([ollama_dropdown, summarize_button, bullets_button, proof_button, refresh_button]),
                # Command input area (above transcription)
                ft.Row([text_dest_toggle, command_field, run_command_button, clear_command_button]),
                # Transcription area
                result_text,
            ]
        )
    )

    # Save last used model on exit and cleanup hotkey
    def on_exit(e):
        save_last_used_ollama_model(ollama_dropdown.value)
        try:
            keyboard.unhook_all()
            print("Hotkeys unregistered")
        except Exception as ex:
            print(f"Error unregistering hotkeys: {ex}")
        
        # Stop any active recording
        if recording_flag.is_set():
            recording_flag.clear()
            print("Stopped active recording")
        
        # Cleanup overlay
        try:
            overlay.destroy()
            print("Overlay destroyed")
        except Exception as ex:
            print(f"Error destroying overlay: {ex}")
            
        # Reset audio source to default (microphone) for next launch
        global audio_source_is_microphone
        audio_source_is_microphone = True

    page.on_close = on_exit


if __name__ == "__main__":
    ft.app(target=main)
