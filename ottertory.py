import os
# Disable torch.compile early, print diagnostics for CUDA/Triton, and add try/except fallback to CPU for model loading
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Language code mapping for multilingual TTS - must match chatterboxexample.py
LANGUAGE_CODES = {
    "Arabic": "ar",
    "Chinese": "zh",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Turkish": "tr"
}

from dotenv import load_dotenv, set_key, dotenv_values
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
from openai import OpenAI
import json
import tempfile
import shutil
from pathlib import Path


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

# Language dropdown (global)
language_dropdown = None

# TTS state
use_openai_tts = False  # Default to Kokoro TTS

# Auto-translate state
auto_translate_active = False
translation_queue = queue.Queue()
last_translated_segment = ""
translate_worker_thread = None
sentence_endings = ('.', '!', '?', '。', '？', '！')  # Include various language punctuation


def audio_stream_thread(run_id: int):
    global audio_source_is_microphone, state, mimi, tokenizer, lm
    
    # Only load STT models (not TTS)
    try:
        load_models(load_tts=False)  # Don't load TTS model here
    except Exception as e:
        print(f"Failed to load STT models in audio thread: {e}")
        return
        
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
            samplerate=int(state.mimi.sample_rate),
            blocksize=int(state.frame_size),
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
    global state, last_audio_time, audio_queue, current_run_id, auto_translate_active, translation_queue, overlay, mimi, tokenizer, lm
    
    current_run_id = run_id
    print(f"Transcriber loop started (run_id={run_id})")
    
    # Ensure models are loaded
    try:
        load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
    
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

        # For tracking sentences for translation
    sentence_buffer = ""
    
    def update_ui(current_text):
        nonlocal last_update_time, buffer, sentence_buffer
        current_time = time.time()
        
        # Always update the buffer with the latest text
        buffer = current_text
        
        # Check for complete sentences for translation
        if auto_translate_active and current_text:
            # Extract new text since last sentence
            new_text = current_text[len(sentence_buffer):].strip()
            
            if new_text:
                # Check if we have a complete sentence or segment
                has_sentence_ending = any(ending in new_text for ending in sentence_endings)
                segment_length = len(new_text)
                
                # Translate if we have a sentence ending or reached ~70 chars
                if has_sentence_ending or segment_length >= 70:
                    # Find the last sentence or take the whole segment
                    if has_sentence_ending:
                        # Find the last sentence ending
                        last_ending_pos = -1
                        for ending in sentence_endings:
                            pos = new_text.rfind(ending)
                            if pos > last_ending_pos:
                                last_ending_pos = pos
                        
                        if last_ending_pos > -1:
                            segment_to_translate = new_text[:last_ending_pos + 1].strip()
                            sentence_buffer = current_text[:len(sentence_buffer) + last_ending_pos + 1]
                        else:
                            segment_to_translate = new_text.strip()
                            sentence_buffer = current_text
                    else:
                        # Take the last ~70 chars as a segment
                        segment_to_translate = new_text[-70:].strip()
                        sentence_buffer = current_text
                    
                    # Queue for translation
                    if segment_to_translate:
                        translation_queue.put(segment_to_translate)
        
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
                # Only update the result field if auto-translate is not active
                if not auto_translate_active:
                    result_field.value = textwrap.fill(buffer.strip(), width=80)
                    page.update()
                
                # Always update the overlay if shown, but mark as translation when auto-translate is active
                if show_overlay:
                    if auto_translate_active:
                        # Don't update the overlay with live text when translating
                        pass
                    else:
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

def load_models(load_tts=False):
    """Load or reload all models if they're not already loaded.
    
    Args:
        load_tts: If True, also loads the TTS model. Otherwise, only loads STT models.
    """
    global state, mimi, tokenizer, lm, kokoro_model
    
    # Only reload STT models if not already loaded
    if state is None or mimi is None or lm is None or tokenizer is None:
        print("Loading STT models...")
        try:
            ck = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
            mimi = ck.get_mimi(device=device_str)
            tokenizer = ck.get_text_tokenizer()
            lm = ck.get_moshi(device=device_str)
            state = InferenceState(mimi, tokenizer, lm, batch_size=1, device=device)
            print("STT models loaded successfully")
        except Exception as e:
            print(f"Error loading STT models: {e}")
            state = mimi = tokenizer = lm = None
            raise
    
    # Only load TTS model if explicitly requested and not already loaded
    if load_tts and kokoro_model is None:
        load_kokoro_model(device_str)
    
    return state

def load_kokoro_model(device_str="cpu"):
    """Lazy load kokoro TTS model (build_model from models.py)."""
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    try:
        print("Loading Kokoro TTS model...")
        kokoro_model = build_model("kokoro-v1_0.pth", device=device_str)
        print("Kokoro TTS model loaded successfully")
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


def translate_segment(text, target_language, ollama_model):
    """Translate a segment of text using Ollama"""
    if not text or not text.strip():
        return ""
    
    try:
        # Create a concise translation prompt
        prompt = f"Translate to {target_language}: {text}"
        
        # Call the LLM for translation
        if ollama_model.startswith('lmstudio::'):
            model = ollama_model[len('lmstudio::'):]
            url = 'http://localhost:1234/v1/chat/completions'
            headers = {'Content-Type': 'application/json'}
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": f"You are a translator. Translate text to {target_language}. Only output the translation, nothing else."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "max_tokens": 100
            }
            r = requests.post(url, json=payload, headers=headers, timeout=5)
            r.raise_for_status()
            data = r.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            # Ollama
            from ollama import chat
            response = chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": f"You are a translator. Translate text to {target_language}. Only output the translation, nothing else."},
                    {"role": "user", "content": text}
                ],
                options={"temperature": 0.3}
            )
            return response['message']['content'].strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return ""


def translation_worker(ollama_model, target_language, text_field, page, overlay_enabled):
    """Background worker to process translation queue"""
    global auto_translate_active, last_translated_segment
    
    # Store translations in a list to maintain context
    translations = []
    
    while auto_translate_active:
        try:
            # Get text from queue with timeout
            text = translation_queue.get(timeout=0.5)
            
            if not text or not auto_translate_active:
                continue
                
            # Translate the segment
            translated = translate_segment(text, target_language, ollama_model)
            
            if translated and auto_translate_active:
                # Add to our translations list
                translations.append(translated)
                
                # Keep only the last few translations to avoid memory issues
                if len(translations) > 5:  # Keep last 5 translations for context
                    translations = translations[-5:]
                
                # Update overlay with the latest translation if enabled
                if overlay_enabled:
                    overlay.update_text(translated, is_live_transcription=False)
                
                # Update the text field with translations on the same line
                if text_field:
                    # Get the latest translation
                    latest_translation = translations[-1] if translations else ""
                    
                    # Get existing text without the last translation if it exists
                    existing_text = text_field.value or ""
                    
                    # Split by double newlines to separate translations
                    existing_translations = [t.strip() for t in existing_text.split('\n\n') if t.strip()]
                    
                    # Keep only the last 4 translations to avoid clutter
                    if len(existing_translations) >= 4:
                        existing_translations = existing_translations[-4:]
                    
                    # Add the latest translation if it's new
                    if latest_translation and latest_translation not in existing_text:
                        existing_translations.append(latest_translation)
                    
                    # Update the text field with all translations on one line
                    if existing_translations:
                        text_field.value = ' '.join(existing_translations)
                        page.update()
                
                last_translated_segment = translated
                print(f"Translated: '{text[:30]}...' -> '{translated[:30]}...'")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Translation worker error: {e}")
            # Try to update the UI with error message
            try:
                if text_field:
                    text_field.value = f"Translation error: {str(e)}"
                    page.update()
            except Exception as update_error:
                print(f"Failed to update UI: {update_error}")


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


def _play_tts_if_enabled(text, voice_name, device_str, speed=1.0, page=None, target_language=None):
    """Helper function to play TTS if auto-TTS is enabled
    
    Args:
        text: Text to speak
        voice_name: Name or path of the voice to use
        device_str: Device string for TTS
        speed: Playback speed (0.25 - 4.0)
        page: Reference to the Flet page for UI updates
        target_language: Target language for TTS (e.g., 'French', 'Spanish')
    """
    global auto_tts_active
    if auto_tts_active and text.strip():
        try:
            print("Auto-TTS: Playing response...")
            # Get language code if target_language is provided
            language_code = None
            if target_language:
                language_code = LANGUAGE_CODES.get(target_language, None)
                print(f"Auto-TTS: Using language {target_language} (code: {language_code})")
            
            play_tts_ui(voice_name, text, device_str, speed, page=page, language_code=language_code)
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
        _play_tts_if_enabled(summary, voice_dropdown.value, device_str, speed_slider.value, page=page)
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
        _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value, page=page)
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
        _play_tts_if_enabled(out, voice_dropdown.value, device_str, speed_slider.value, page=page)
    except Exception as e:
        print("Proofread failed:", e)


def play_tts_ui(voice_name, text, device_str, speed=1.0, page=None):
    """Generate audio with either Kokoro or OpenAI TTS and play it.
    
    Args:
        voice_name: Name of the voice (e.g., 'af_alloy') or path to the voice file
        text: Text to be spoken
        device_str: Device to use for TTS
        speed: Playback speed (0.25 - 4.0)
        page: Optional page reference for accessing UI state
    """
    global use_openai_tts, kokoro_model
    
    if not text.strip():
        print("No text to speak")
        return
    
    # For OpenAI TTS, we need to process language settings if ID tag is enabled
    if use_openai_tts:
        # Default to no language code (will use voice's default language)
        language_code = None
        multilingual_attr = False
        
        # Check if ID tag (multilingual mode) is enabled
        if hasattr(page, 'multilingual_mode'):
            multilingual_attr = page.multilingual_mode
        elif hasattr(page, 'page') and hasattr(page.page, 'multilingual_mode'):
            multilingual_attr = page.page.multilingual_mode
        
        print(f"ID Tag (Multilingual Mode): {'Enabled' if multilingual_attr else 'Disabled'}")
        
        # Only process language settings if ID tag is enabled
        if multilingual_attr:
            # Debug: Print page attributes
            print("\n--- DEBUG: Page Attributes ---")
            print(f"Page attributes: {dir(page)}")
            if hasattr(page, 'page'):
                print(f"Page.page attributes: {dir(page.page)}")
            
            # Get the selected language from the global dropdown
            global language_dropdown
            
            # Debug info
            print("\n--- DEBUG: Language Dropdown State ---")
            print(f"Language dropdown exists: {language_dropdown is not None}")
            if language_dropdown is not None:
                print(f"Dropdown value: {getattr(language_dropdown, 'value', 'NO VALUE')}")
                print(f"Dropdown options: {getattr(language_dropdown, 'options', 'NO OPTIONS')}")
            
            if not language_dropdown or not hasattr(language_dropdown, 'value'):
                raise ValueError("Language dropdown not properly initialized")
                
            selected_language = language_dropdown.value
            if not selected_language:
                raise ValueError("No language selected in dropdown")
            print(f"Selected language: {selected_language}")
            language_name = None
            
            # Find matching language name
            for name, code in LANGUAGE_CODES.items():
                if name.lower() == selected_language.lower():
                    language_name = name
                    break
            
            # If no exact match, try partial match
            if not language_name:
                for name in LANGUAGE_CODES:
                    if selected_language.lower() in name.lower():
                        language_name = name
                        break
            
            language_name = language_name or selected_language
            language_code = LANGUAGE_CODES.get(language_name, 'en')
            
            print("\n--- OpenAI TTS Language Settings (ID Tag Enabled) ---")
            print(f"Selected Language: {selected_language}")
            print(f"Matched Language: {language_name}")
            print(f"Language Code: {language_code}")
        else:
            print("\n--- OpenAI TTS Language Settings (ID Tag Disabled) ---")
            print("Using voice's default language")
        
        # Resolve the voice path for OpenAI TTS
        if not os.path.isabs(voice_name) and not voice_name.lower().endswith(('.wav', '.mp3', '.ogg')):
            # This is a voice name, not a path - get the path from the environment
            voice_path = os.getenv('tts_voice_path')
            if voice_path and not os.path.isabs(voice_path):
                # Make path absolute relative to the script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                voice_path = os.path.join(script_dir, voice_path)
            
            # Use the configured voice path if it exists
            if voice_path and os.path.exists(voice_path):
                voice_to_use = voice_path
            else:
                # Try to use the default voice file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                default_voice = os.path.join(script_dir, 'voices', 'tempclone.wav')
                if os.path.exists(default_voice):
                    voice_to_use = default_voice
                    print(f"Using default voice file: {default_voice}")
                else:
                    raise FileNotFoundError(
                        f"OpenAI TTS requires a valid voice file. "
                        f"Default voice file not found at: {default_voice}"
                    )
        else:
            # This is already a path or a special identifier
            voice_to_use = voice_name
            
        print(f"\n--- OpenAI TTS Request ---")
        print(f"Voice: {os.path.basename(voice_to_use) if os.path.exists(voice_to_use) else voice_to_use}")
        print(f"Voice Path: {voice_to_use}")
        print(f"Language Code: {language_code}")
        print(f"Speed: {speed}")
        print(f"Text Length: {len(text)} characters")
        print("----------------------")
        
        _play_openai_tts(voice_to_use, text, speed, language_code=language_code, 
                        multilingual=multilingual_attr if multilingual_attr is not None else False)
    else:
        # For Kokoro TTS, we don't need language settings
        print("\n--- Kokoro TTS Request ---")
        print(f"Voice: {voice_name}")
        print(f"Speed: {speed}")
        print(f"Text Length: {len(text)} characters")
        print("----------------------")
        
        # Only load Kokoro TTS model when actually needed for playback
        try:
            # Only load Kokoro model, not STT models
            if kokoro_model is None:
                print("Loading Kokoro TTS model...")
                load_kokoro_model(device_str)
            _play_kokoro_tts(voice_name, text, device_str, speed)
        except Exception as e:
            print(f"Failed to load Kokoro TTS model: {e}")
            # Re-raise the error to prevent falling back to OpenAI TTS
            print(f"Kokoro TTS failed to load: {e}")
            raise

def _play_openai_tts(voice_name, text, speed=1.0, language_code=None, multilingual=False):
    """Generate and stream audio using OpenAI TTS API with parallel processing.
    
    Args:
        voice_name: Path to the voice file
        text: Text to be spoken
        speed: Playback speed (0.25 - 4.0)
        language_code: 2-letter language code (e.g., 'en', 'fr', 'ru')
        multilingual: Whether to use multilingual mode
    """
    import os  # Import os at the function level to ensure it's available
    
    tts_stop_event.clear()
    if not text or not text.strip():
        print("No text to play")
        return
        
    # Resolve the voice path
    voice_path = voice_name  # Start with the provided path
    
    # If path is not absolute, try to resolve it relative to the script directory
    if voice_path and not os.path.isabs(voice_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # First try the path as is (could be relative to script dir)
        temp_path = os.path.join(script_dir, voice_path)
        if os.path.exists(temp_path):
            voice_path = temp_path
        else:
            # If not found, try looking in the voices directory
            voices_dir = os.path.join(script_dir, 'voices')
            temp_path = os.path.join(voices_dir, os.path.basename(voice_path))
            if os.path.exists(temp_path):
                voice_path = temp_path
    
    # Debug output
    print(f"Looking for voice file at: {voice_path}")
    if not voice_path or not os.path.exists(voice_path):
        print(f"Warning: Voice file not found at {voice_path}")
        # Try one last time with just the filename in the voices directory
        if voice_path:
            last_try = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voices', os.path.basename(voice_path))
            if os.path.exists(last_try):
                voice_path = last_try
                print(f"Found voice file at: {voice_path}")
            else:
                print("Falling back to Kokoro TTS")
                return _play_kokoro_tts(voice_name, text, 'cpu', speed)
    
    # At this point, voice_path should be a valid absolute path
    voice_id = voice_path
    
    print(f"Using OpenAI TTS with voice: {voice_id}")
    
    import queue
    import threading
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    import os
    import time
    from openai import OpenAI
    
    # Initialize queue and control variables
    audio_queue = queue.Queue(maxsize=3)  # Limit queue size to prevent memory issues
    generation_complete = threading.Event()
    stop_event = threading.Event()
    
    # Prepare the request parameters
    params = {
        'model': os.getenv('tts_model', 'chatterbox'),  # Default to chatterbox model
        'voice': voice_id,
        'input': text,
        'speed': min(max(0.5, speed), 1.5),  # Ensure speed is within valid range
    }
    
    # Add language parameters for multilingual support
    if language_code:
        params['extra_body'] = {
            'params': {
                'language_id': language_code,
                'model_name': 'multilingual',
                'use_gpu': True,
                'multilingual': True
            }
        }
    
    print(f"Sending TTS request with params: {params}")
    
    def audio_worker():
        """Worker thread that plays audio chunks from the queue."""
        try:
            while not stop_event.is_set() and not (generation_complete.is_set() and audio_queue.empty()):
                try:
                    # Get next audio chunk with timeout to check stop_event
                    try:
                        temp_path = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                        
                    if temp_path is None:  # Sentinel value to stop
                        break
                        
                    if tts_stop_event.is_set() or stop_event.is_set():
                        sd.stop()
                        break
                    
                    try:
                        # Load and play the audio chunk
                        data, samplerate = sf.read(temp_path)
                        sd.play(data, samplerate)
                        sd.wait()
                    finally:
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                            
                except Exception as e:
                    print(f"Error in audio worker: {e}")
                    break
        finally:
            # Ensure we clean up properly
            sd.stop()
    
    def generate_worker():
        """Worker thread that generates audio chunks and adds them to the queue."""
        try:
            # Split text into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not sentences:
                sentences = [text]
            
            # Initialize client once at the start
            client = OpenAI(
                base_url='http://127.0.0.1:7778/v1',
                api_key='dummy_key'
            )
            
            for sentence in sentences:
                if tts_stop_event.is_set() or stop_event.is_set():
                    break
                    
                if not sentence.strip():
                    continue
                    
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                    
                print(f"TTS: generating sentence (chars={len(sentence)}): {sentence[:50]}...")
                
                # Try generation with retries
                for attempt in range(3):
                    if tts_stop_event.is_set() or stop_event.is_set():
                        break
                        
                    try:
                        # Create a temporary file for this sentence
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        # Prepare the request parameters for this sentence
                        request_params = {
                            'model': 'chatterbox',
                            'voice': voice_path,
                            'input': sentence,
                            'speed': min(max(0.5, float(speed)), 1.5)
                        }
                        
                        # Add multilingual parameters if needed
                        if multilingual and language_code:
                            request_params['extra_body'] = {
                                'params': {
                                    'language_id': language_code,
                                    'model_name': 'multilingual',
                                    'use_gpu': True,
                                    'multilingual': True
                                }
                            }
                            
                            # Log the complete request parameters for debugging
                            print("\n--- TTS Request Parameters ---")
                            print(f"Multilingual mode enabled")
                            print(f"Language code: {language_code}")
                            print(f"Using model: {request_params['model']} with multilingual support")
                            
                            import json
                            print("Complete request parameters:")
                            print(json.dumps(request_params, indent=2, ensure_ascii=False))
                        
                        # Make the API call for this sentence
                        with client.audio.speech.with_streaming_response.create(**request_params) as response:
                            # Save the response to a temporary file
                            response.stream_to_file(temp_path)
                            
                            # Add the file path to the queue for playback
                            audio_queue.put(temp_path)
                            break  # Success, exit retry loop
                            
                    except Exception as e:
                        print(f"Error generating TTS (attempt {attempt + 1}/3): {e}")
                        if attempt == 2:  # If this was the last attempt
                            print(f"Failed to generate audio for: {sentence}")
                        time.sleep(0.5)  # Wait before retry
                    
        except Exception as e:
            print(f"Error in generate_worker: {e}")
            
        finally:
            # Signal that generation is complete
            generation_complete.set()
    
    # Start audio playback and generation threads
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    generator_thread = threading.Thread(target=generate_worker, daemon=True)
    
    audio_thread.start()
    generator_thread.start()
    
    try:
        # Wait for generation to complete and audio to finish playing
        while (generator_thread.is_alive() or not audio_queue.empty()) and not tts_stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("TTS interrupted by user")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish with timeout
        generator_thread.join(timeout=1.0)
        audio_thread.join(timeout=1.0)
        
        # Clean up any remaining temporary files in the queue
        while not audio_queue.empty():
            try:
                temp_path = audio_queue.get_nowait()
                try:
                    os.unlink(temp_path)
                except:
                    pass
            except queue.Empty:
                break

def _play_kokoro_tts(voice_name, text, device_str, speed=1.0):
    """Generate and stream audio with Kokoro TTS, playing chunks as they're generated."""
    tts_stop_event.clear()
    if not text or not text.strip():
        print("No text to play")
        return
        
    print(f"Using Kokoro TTS with voice: {voice_name}")
    
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

    import numpy as _np
    import sounddevice as _sd
    import queue
    import threading
    
    # Initialize audio queue and control variables
    audio_queue = queue.Queue(maxsize=3)  # Limit queue size to prevent memory issues
    generation_complete = threading.Event()
    stop_event = threading.Event()
    sample_rate = 24000
    
    def audio_worker():
        """Worker thread that plays audio chunks from the queue."""
        try:
            while not stop_event.is_set() and not (generation_complete.is_set() and audio_queue.empty()):
                try:
                    # Get next audio chunk with timeout to check stop_event
                    try:
                        audio_data = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                        
                    if audio_data is None:  # Sentinel value to stop
                        break
                        
                    if tts_stop_event.is_set():
                        _sd.stop()
                        break
                        
                    # Normalize the chunk to avoid clipping
                    peak = max(1e-9, float(_np.max(_np.abs(audio_data))))
                    if peak > 1.0:
                        audio_data = audio_data / peak
                    
                    # Play the chunk
                    _sd.play(audio_data, sample_rate)
                    _sd.wait()
                    
                except Exception as e:
                    print(f"Error in audio worker: {e}")
                    break
        finally:
            # Ensure we clean up properly
            _sd.stop()
    
    def generate_worker():
        """Worker thread that generates audio chunks and adds them to the queue."""
        try:
            current_chunk = ""
            max_chunk_chars = 100  # Smaller chunks for more responsive playback
            
            for sentence in sentences:
                if tts_stop_event.is_set() or stop_event.is_set():
                    break
                    
                # Add sentence to current chunk
                if not current_chunk:
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
                    
                # If chunk is big enough or this is the last sentence, process it
                if len(current_chunk) >= max_chunk_chars or sentence is sentences[-1]:
                    if not current_chunk.endswith(('.', '!', '?')):
                        current_chunk += '.'
                        
                    print(f"TTS: generating chunk (chars={len(current_chunk)})")
                    
                    # Try generation with retries
                    audio_tensor = None
                    for attempt in range(3):
                        if tts_stop_event.is_set() or stop_event.is_set():
                            break
                            
                        try:
                            audio_tensor, _ = generate_speech(model, current_chunk, voice=voice_name, speed=speed)
                            if audio_tensor is not None:
                                # Add to playback queue (block if queue is full)
                                audio_queue.put(audio_tensor.numpy().astype(_np.float32), timeout=1.0)
                                break
                        except queue.Full:
                            print("Audio queue full, waiting...")
                            time.sleep(0.1)
                            continue
                        except Exception as e:
                            print(f"generate_speech exception: {e}")
                            if attempt == 2:  # Last attempt
                                print(f"Failed to generate audio for: {current_chunk}")
                        time.sleep(0.1)
                        
                    # Reset chunk for next iteration
                    current_chunk = ""
        finally:
            # Signal that generation is complete
            generation_complete.set()

    # Start audio playback and generation threads
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    generator_thread = threading.Thread(target=generate_worker, daemon=True)
    
    audio_thread.start()
    generator_thread.start()
    
    try:
        # Wait for generation to complete and audio to finish playing
        while (generator_thread.is_alive() or not audio_queue.empty()) and not tts_stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("TTS interrupted by user")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish with timeout
        generator_thread.join(timeout=1.0)
        audio_thread.join(timeout=1.0)
        
        # Clean up any remaining temporary files in the queue
        while not audio_queue.empty():
            try:
                temp_path = audio_queue.get_nowait()
                try:
                    os.unlink(temp_path)
                except:
                    pass
            except queue.Empty:
                break

def _download_kokoro_tts(voice_name, text, device_str, speed=1.0):
    """Generate and save audio using Kokoro TTS."""
    # Normalize whitespace and remove forced line breaks from wrapped live text
    normalized_text = re.sub(r"\s+", " ", text).strip()
    model = load_kokoro_model(device_str)
    if model is None:
        print("Kokoro model not available")
        return None

    # Split into sentences while keeping punctuation
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', normalized_text) if s.strip()]
    if not sentences:
        sentences = [normalized_text]

    import numpy as _np
    audio_pieces = []
    silence = _np.zeros(int(24000 * 0.06), dtype=_np.float32)

    for idx, chunk in enumerate(sentences):
        if not chunk.endswith(('.', '!', '?')):
            chunk += '.'
        print(f"TTS download: generating chunk {idx+1}/{len(sentences)}")

        audio_tensor = None
        for attempt in range(3):
            try:
                audio_tensor, _ = generate_speech(model, chunk, voice=voice_name, speed=speed)
                if audio_tensor is not None:
                    break
            except Exception as e:
                print(f"generate_speech exception: {e}")
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
                        if at is not None:
                            audio_pieces.append(at.numpy().astype(_np.float32))
                            audio_pieces.append(silence)
                            break
                    except Exception as e:
                        print(f"generate_speech exception (fallback): {e}")
                    time.sleep(0.15)
            continue

        piece = audio_tensor.numpy().astype(_np.float32)
        audio_pieces.append(piece)
        audio_pieces.append(silence)

    if not audio_pieces:
        print("No audio generated for download")
        return None

    try:
        return _np.concatenate(audio_pieces), 24000
    except Exception as e:
        print("Error concatenating audio pieces:", e)
        return None

def _download_openai_tts(voice_name, text, speed=1.0):
    """Generate and save audio using OpenAI TTS."""
    import os
    import tempfile
    import numpy as np
    from scipy.io import wavfile
    from openai import OpenAI
    
    # Load environment variables
    load_dotenv()
    
    # Get voice path from environment and resolve it relative to the script location
    voice_path = os.getenv('tts_voice_path')
    if not voice_path:
        print("Warning: No voice path configured in environment")
        return None
        
    # Convert relative path to absolute path based on script location
    if not os.path.isabs(voice_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        voice_path = os.path.join(script_dir, voice_path)
        print(f"Resolved voice path to: {voice_path}")
        
    if not os.path.exists(voice_path):
        print(f"Error: Voice file not found at {voice_path}")
        # Try to find the voice file in the voices directory
        voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voices')
        alt_path = os.path.join(voices_dir, 'tempclone.wav')
        if os.path.exists(alt_path):
            print(f"Found voice file at alternative location: {alt_path}")
            voice_path = alt_path
        else:
            print("No valid voice file found. Please set a voice file first.")
            return None
    
    # Convert full path to the format expected by the API
    try:
        path_parts = os.path.normpath(voice_path).split(os.sep)
        if len(path_parts) >= 3:
            voice_id = f"voices/{path_parts[-3]}/{path_parts[-2]}/{path_parts[-1]}"
        else:
            voice_id = f"voices/{os.path.basename(voice_path)}"
    except Exception as e:
        print(f"Error parsing voice path: {e}")
        voice_id = os.getenv('speaker', 'default')
    
    print(f"Using OpenAI TTS with voice: {voice_id}")
    
    try:
        client = OpenAI(
            base_url=os.getenv('openai_base_url'),
            api_key=os.getenv('openai_api_key')
        )
        
        # Prepare the request parameters
        params = {
            'model': os.getenv('tts_model', 'tts-1'),
            'voice': voice_id,
            'input': text,
            'speed': speed,
        }
        
        # Always include language parameters as they're required by chatterbox
        if language_code:
            params['extra_body'] = {
                'params': {
                    'language_id': language_code,
                    'model_name': 'multilingual',  # Required by chatterbox
                    'use_gpu': True,  # Enable GPU acceleration
                    'multilingual': True  # Explicitly enable multilingual
                }
            }
        
        print(f"Sending TTS request with params: {params}")
        
        # Ensure speed is within valid range
        params['speed'] = min(max(0.5, speed), 1.5)
        
        # Generate the speech
        response = client.audio.speech.create(**params)
        
        # Save to a temporary file and read it back to get the audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Save the audio data
        response.stream_to_file(temp_path)
        
        # Read the WAV file to get sample rate and audio data
        sample_rate, audio_data = wavfile.read(temp_path)
        
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / (2**15 if audio_data.dtype == np.int16 else 2**31)
            
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error in OpenAI TTS download: {e}")
        return None

def download_tts_ui(voice_name, text, device_str, speed=1.0):
    """Generate full TTS audio and save as WAV. Uses the currently selected TTS engine."""
    global use_openai_tts
    
    if not text or not text.strip():
        print("No text to save")
        return
    
    print(f"Using {'OpenAI' if use_openai_tts else 'Kokoro'} TTS with voice: {voice_name} for download")
    
    # Use the appropriate TTS engine
    if use_openai_tts:
        result = _download_openai_tts(voice_name, text, speed)
    else:
        result = _download_kokoro_tts(voice_name, text, device_str, speed)
    
    if result is None:
        print("Failed to generate audio for download")
        return
    
    audio_data, sample_rate = result
    
    # Save to WAV file
    import os
    import time
    from scipy.io import wavfile
    
    out_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    wav_path = os.path.join(out_dir, f"tts_{ts}.wav")
    
    try:
        wavfile.write(wav_path, sample_rate, audio_data.astype(np.float32))
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


def update_voice_path(file_path: str):
    """Update the TTS voice path by using Flet's file picker to save the file"""
    try:
        print(f"\n--- Starting voice file update ---")
        print(f"Source file: {os.path.abspath(file_path)}")
        
        # Define the voices directory relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(script_dir, 'voices')
        print(f"Voices directory: {voices_dir}")
        os.makedirs(voices_dir, exist_ok=True)
        
        # Define the target path (tempclone.wav in voices directory)
        target_path = os.path.join(voices_dir, 'tempclone.wav')
        print(f"Target path: {target_path}")
        
        # Use Flet's file picker to save the file
        try:
            # First, let's make sure we can write to the directory
            test_file = os.path.join(voices_dir, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            # Now try to copy the file using a different approach
            # Read in binary mode and write in binary mode
            with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    dst.flush()  # Force write to disk
                    os.fsync(dst.fileno())  # Ensure it's written to disk
            
            # Verify the copy
            if os.path.exists(target_path):
                source_size = os.path.getsize(file_path)
                target_size = os.path.getsize(target_path)
                print(f"Successfully copied to {target_path}")
                print(f"Source size: {source_size} bytes, Target size: {target_size} bytes")
                
                if source_size != target_size:
                    raise RuntimeError(f"File size mismatch: source={source_size}, target={target_size}")
            else:
                raise RuntimeError("Failed to verify copied file")
                
        except Exception as e:
            print(f"Error during file copy: {e}")
            # Try one more time with a direct copy
            try:
                import shutil
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.copy2(file_path, target_path)
                print(f"Used shutil.copy2 as fallback")
                
                # Verify again
                if not os.path.exists(target_path):
                    raise RuntimeError("shutil.copy2 failed - file not created")
                    
            except Exception as e2:
                print(f"Fallback copy also failed: {e2}")
                # Try one last approach using Windows API if on Windows
                if os.name == 'nt':
                    try:
                        import ctypes
                        from ctypes import wintypes
                        
                        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                        
                        # Define Windows API types
                        BOOL = ctypes.c_bool
                        LPCWSTR = wintypes.LPCWSTR
                        
                        # Set up the CopyFileW function
                        CopyFileW = kernel32.CopyFileW
                        CopyFileW.argtypes = (LPCWSTR, LPCWSTR, BOOL)
                        CopyFileW.restype = BOOL
                        
                        # Convert paths to Windows wide strings
                        src_w = os.path.abspath(file_path).replace('/', '\\')
                        dst_w = target_path.replace('/', '\\')
                        
                        # Call the Windows API
                        if not CopyFileW(src_w, dst_w, False):  # False = overwrite existing
                            error = ctypes.get_last_error()
                            raise ctypes.WinError(error)
                            
                        print("Used Windows API copy successfully")
                    except Exception as e3:
                        print(f"Windows API copy also failed: {e3}")
                        raise RuntimeError(f"All copy methods failed: {e} | {e2} | {e3}")
                else:
                    raise RuntimeError(f"All copy methods failed: {e} | {e2}")
        
        # Get the original filename for display
        original_filename = os.path.basename(file_path)
        
        # Store the original filename in a separate file for UI display
        with open(os.path.join(voices_dir, 'current_voice.txt'), 'w') as f:
            f.write(original_filename)
        
        # Update the environment variable to use relative path
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        normalized_path = 'voices/tempclone.wav'  # Always use this relative path
        
        # Update the .env file
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Find and update the tts_voice_path line
        updated = False
        for i, line in enumerate(lines):
            if line.startswith('tts_voice_path'):
                lines[i] = f"tts_voice_path='{normalized_path}'\n"
                updated = True
                break
        
        if not updated:
            lines.append(f"tts_voice_path='{normalized_path}'\n")
        
        # Write the updated .env file
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        # Reload environment variables
        load_dotenv(override=True)
        print(f"Updated TTS voice to: {original_filename}")
        return True, original_filename
    except Exception as e:
        print(f"Error updating voice path: {e}")
        return False, str(e)

def main(page: ft.Page):
    page.title = " Ottertory "
    page.scroll = ft.ScrollMode.AUTO
    
    # Make page available to other functions
    global page_ref
    page_ref = page
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

    # Voice file drop target
    def handle_drop(e: ft.DragTargetEvent):
        if e.data and 'files' in e.data:
            files = e.data['files']
            if files:
                file_path = files[0]['path']
                if file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    success, result = update_voice_path(file_path)
                    if success:
                        voice_drop_text.value = f"Voice: {result}"  # result is already the original filename
                        voice_drop_text.color = ft.Colors.GREEN
                    else:
                        voice_drop_text.value = f"Error: {result}"
                        voice_drop_text.color = ft.Colors.RED
                    page.update()
                    return
        voice_drop_text.value = "CLONE"
        voice_drop_text.color = None
        page.update()

    voice_drop = ft.Container(
        content=ft.Column([
            ft.Icon(ft.Icons.UPLOAD_FILE, size=40, color=ft.Colors.BLUE_400),
            ft.Text("CLONE", size=12, weight=ft.FontWeight.BOLD)
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        border=ft.border.all(2, ft.Colors.BLUE_200),
        border_radius=10,
        padding=15,
        on_hover=lambda e: (
            setattr(voice_drop.border, 'color', ft.Colors.BLUE_400 if e.data == "true" else ft.Colors.BLUE_200),
            voice_drop.update()
        ),
        on_click=lambda e: (
            page.launch_files(
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=['wav', 'mp3', 'ogg', 'flac'],
                allow_multiple=False
            )
        ),
        data=None,
    )

    # Function to get the current voice display name
    def get_voice_display_name():
        try:
            voice_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voices', 'current_voice.txt')
            if os.path.exists(voice_file):
                with open(voice_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return "Not set"

    voice_drop_text = ft.Text(
        f"Voice: {get_voice_display_name()}",
        size=12,
        weight=ft.FontWeight.BOLD,
        text_align=ft.TextAlign.CENTER
    )

    # Set up drag target
    def handle_drag_enter(e):
        voice_drop.border.color = ft.Colors.GREEN_400
        voice_drop.update()

    def handle_drag_leave(e):
        voice_drop.border.color = ft.Colors.BLUE_200
        voice_drop.update()

    voice_drop.on_drop = handle_drop
    voice_drop.on_will_accept = lambda e: (
        setattr(voice_drop.border, 'color', ft.Colors.GREEN_400 if e.data == "true" else ft.Colors.BLUE_200),
        voice_drop.update()
    )
    voice_drop.on_leave = handle_drag_leave
    voice_drop.on_enter = handle_drag_enter

    # Handle file picker result
    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            if file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                success, result = update_voice_path(file_path)
                if success:
                    voice_drop_text.value = f"Voice: {result}"  # result is already the original filename
                    voice_drop_text.color = ft.Colors.GREEN
                else:
                    voice_drop_text.value = f"Error: {result}"
                    voice_drop_text.color = ft.Colors.RED
                page.update()

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)

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
    
    # Initialize the overlay (already defined as a global variable)
    overlay = globals().get('overlay')
    
    # Keyboard event handler
    def on_keyboard(e):
        if e.key == '/':
            # Focus the command field and scroll to it
            command_field.focus()
            command_field.update()
            page.update()
    
    # Add keyboard event listener
    page.on_keyboard_event = on_keyboard
    
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
                    args=(voice_dropdown.value, result_text.value, device_str, speed_slider.value, page),
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
                    play_tts_ui(voice_dropdown.value, text_to_speak, device_str, speed_slider.value, page=page)
                except Exception as e:
                    print(f"Error in TTS playback: {e}")
            
            # Use the full transcription text
            if auto_tts_active and full_text.strip():
                # Check if we're in ID (multilingual) mode
                if hasattr(page, 'multilingual_mode') and page.multilingual_mode and language_dropdown.value:
                    print("Auto-translating text before TTS...")
                    try:
                        from ollama import chat
                        target_language = language_dropdown.value
                        print(f"Translating to {target_language}...")
                        response = chat(
                            model=ollama_dropdown.value,
                            messages=[
                                {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Only output the translation, nothing else."},
                                {"role": "user", "content": full_text}
                            ],
                            options={"temperature": 0.3}
                        )
                        translated_text = response['message']['content'].strip()
                        print(f"Translated text: {translated_text[:50]}...")
                        
                        # Update the result field with the full translated text
                        result_text.value = f"{full_text}\n\n--- TRANSLATION ---\n{translated_text}"
                        page.update()
                        
                        # Use translated text for TTS
                        t = threading.Thread(
                            target=_play_tts,
                            args=(translated_text,),
                            daemon=True
                        )
                        t.start()
                        return  # Exit early since we've handled the TTS with translation
                    except Exception as e:
                        print(f"Error in auto-translation: {e}")
                        # Fall back to non-translated TTS
                        print("Falling back to non-translated TTS")
                
                # Default TTS without translation
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
        
    # TTS Engine Toggle
    def toggle_tts_engine(e):
        global use_openai_tts
        use_openai_tts = not use_openai_tts
        tts_engine_btn.text = f"TTS: {'OpenAI' if use_openai_tts else 'Kokoro'}"
        tts_engine_btn.bgcolor = ft.Colors.BLUE_400 if use_openai_tts else None
        page.update()
        print(f"TTS engine set to: {'OpenAI' if use_openai_tts else 'Kokoro'}")
    
    tts_engine_btn = ft.ElevatedButton(
        "TTS: Kokoro",
        tooltip="Toggle between Kokoro and OpenAI TTS",
        on_click=toggle_tts_engine,
        width=150
    )
    
    # Play TTS button
    play_tts_button = ft.IconButton(
        icon=ft.Icons.PLAY_ARROW,
        tooltip="Play TTS",
        on_click=lambda e: threading.Thread(
            target=play_tts_ui, 
            args=(voice_dropdown.value, result_text.value, device_str, speed_slider.value, page), 
            daemon=True
        ).start()
    )
    
    # Stop TTS button
    stop_tts_button = ft.IconButton(
        icon=ft.Icons.STOP,
        tooltip="Stop TTS",
        on_click=lambda e: stop_tts_ui()
    )
    
    # TTS Controls Row
    tts_controls = ft.Row(
        [
            tts_engine_btn,
            play_tts_button,
            stop_tts_button
        ]
    )
    # Voice file drop target
    def handle_drop(e: ft.DragTargetEvent):
        if e.data and 'files' in e.data:
            files = e.data['files']
            if files:
                file_path = files[0]['path']
                if file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    success, result = update_voice_path(file_path)
                    if success:
                        voice_drop_text.value = f"Voice: {result}"  # result is already the original filename
                        voice_drop_text.color = ft.Colors.GREEN
                    else:
                        voice_drop_text.value = f"Error: {result}"
                        voice_drop_text.color = ft.Colors.RED
                    page.update()
                    return
        voice_drop_text.value = "CLONE"
        voice_drop_text.color = None
        page.update()

    voice_drop = ft.ElevatedButton(
        "CLONE",
        icon=ft.Icons.UPLOAD_FILE,
        tooltip="Select voice file to clone",
        style=ft.ButtonStyle(
            padding=10,
            shape=ft.RoundedRectangleBorder(radius=5),
            overlay_color=ft.Colors.TRANSPARENT,
        ),
        on_hover=lambda e: (
            voice_drop.update()
        ),
        data=None,
    )

    # Function to get the current voice display name
    def get_voice_display_name():
        try:
            voice_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voices', 'current_voice.txt')
            if os.path.exists(voice_file):
                with open(voice_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        return "Not set"

    voice_drop_text = ft.Text(
        f"Voice: {get_voice_display_name()}",
        size=10,
        weight=ft.FontWeight.BOLD,
        text_align=ft.TextAlign.CENTER
    )

    # Set up drag target
    def handle_drag_enter(e):
        voice_drop.border.color = ft.Colors.GREEN_400
        voice_drop.update()

    def handle_drag_leave(e):
        voice_drop.border.color = ft.Colors.BLUE_200
        voice_drop.update()

    voice_drop.on_drop = handle_drop
    voice_drop.on_will_accept = lambda e: (
        setattr(voice_drop.border, 'color', ft.Colors.GREEN_400 if e.data == "true" else ft.Colors.BLUE_200),
        voice_drop.update()
    )
    voice_drop.on_leave = handle_drag_leave
    voice_drop.on_enter = handle_drag_enter

    # Handle file picker result
    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            if file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                success, result = update_voice_path(file_path)
                if success:
                    voice_drop_text.value = f"Voice: {result}"  # result is already the original filename
                    voice_drop_text.color = ft.Colors.GREEN
                else:
                    voice_drop_text.value = f"Error: {result}"
                    voice_drop_text.color = ft.Colors.RED
                page.update()

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)
    
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
    
    # Add click handler to voice drop container to open file picker
    voice_drop.on_click = lambda e: file_picker.pick_files(
        file_type=ft.FilePickerFileType.CUSTOM,
        allowed_extensions=['wav', 'mp3', 'ogg', 'flac'],
        allow_multiple=False
    )

    # Translation controls
    translate_button = ft.ElevatedButton(
        "Translate",
        tooltip="Translate the text using the selected language",
        on_click=lambda e: threading.Thread(target=translate_text, args=(result_text, language_dropdown.value), daemon=True).start(),
        height=40
    )
    
    # Initialize the global language dropdown
    global language_dropdown
    language_dropdown = ft.Dropdown(
        label="Language",
        hint_text="Select a language for translation",
        options=[
            ft.dropdown.Option(""),  # Empty default option
            ft.dropdown.Option("Finnish"),
            ft.dropdown.Option("Norwegian"),
            ft.dropdown.Option("Swedish"),
            ft.dropdown.Option("Danish"),
            ft.dropdown.Option("English"),
            ft.dropdown.Option("Dutch"),
            ft.dropdown.Option("German"),
            ft.dropdown.Option("Polish"),
            ft.dropdown.Option("Russian"),
            ft.dropdown.Option("French"),
            ft.dropdown.Option("Spanish"),
            ft.dropdown.Option("Portuguese"),
            ft.dropdown.Option("Italian"),
            ft.dropdown.Option("Greek"),
            ft.dropdown.Option("Turkish"),
            ft.dropdown.Option("Arabic"),
            ft.dropdown.Option("Hebrew"),
            ft.dropdown.Option("Hindi"),
            ft.dropdown.Option("Malay"),
            ft.dropdown.Option("Chinese"),
            ft.dropdown.Option("Korean"),
            ft.dropdown.Option("Japanese"),
            ft.dropdown.Option("Swahili")
        ],
        value="French",
        width=150
    )
    
    def translate_text(text_field, target_language):
        """Translate the text in the result field to the target language using Ollama.
        If recording is active, toggles auto-translate mode. Otherwise does a one-time translation."""
        global auto_translate_active, translate_worker_thread
        
        # If recording, toggle auto-translate mode
        if recording_flag.is_set():
            auto_translate_active = not auto_translate_active
            
            if auto_translate_active:
                # Start translation worker if not already running
                if translate_worker_thread is None or not translate_worker_thread.is_alive():
                    translate_worker_thread = threading.Thread(
                        target=translation_worker,
                        args=(ollama_dropdown.value, target_language, text_field, page, overlay_enabled),
                        daemon=True
                    )
                    translate_worker_thread.start()
                
                # Clear any previous translation from the overlay (if enabled) and result field
                if overlay_enabled:
                    overlay.update_text("Translating...", is_live_transcription=False)
                text_field.value = ""  # Clear the result field
                page.update()
                print(f"Auto-translate ENABLED for {target_language}")
            else:
                # Clear the overlay (if enabled) when turning off auto-translate
                if overlay_enabled:
                    overlay.update_text("", is_live_transcription=False)
                # Restore the original transcription if available
                if buffer.strip():
                    text_field.value = textwrap.fill(buffer.strip(), width=80)
                    page.update()
                print("Auto-translate DISABLED")
            
            # Update button appearance
            translate_button.text = f"{'Stop ' if auto_translate_active else ''}Translate"
            translate_button.bgcolor = ft.Colors.GREEN_400 if auto_translate_active else None
            page.update()
            return
        
        # If not recording, do a one-time translation
        if not text_field.value.strip():
            print("No text to translate")
            return
            
        try:
            # Save original text and show loading state
            original_text = text_field.value
            text_field.disabled = True
            text_field.value = f"Translating to {target_language}..."
            page.update()
            
            # Call the LLM for translation
            if ollama_dropdown.value.startswith('lmstudio::'):
                model = ollama_dropdown.value[len('lmstudio::'):]
                url = 'http://localhost:1234/v1/chat/completions'
                headers = {'Content-Type': 'application/json'}
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are a translator. Translate text to {target_language}. EXTREMELY IMPORTANT you only output the translation, nothing else!!!! If you cant translate output - instead."},
                        {"role": "user", "content": original_text}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                translated = data['choices'][0]['message']['content'].strip()
            else:
                # Ollama
                from ollama import chat
                try:
                    response = chat(
                        model=ollama_dropdown.value,
                        messages=[
                            {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Only output the translation, nothing else."},
                            {"role": "user", "content": original_text}
                        ],
                        options={"temperature": 0.3}
                    )
                    translated = response['message']['content'].strip()
                    if not translated:
                        raise ValueError("Empty translation received")
                except Exception as e:
                    print(f"Error in Ollama translation: {e}")
                    # Fall back to Kokoro TTS if available
                    translated = f"[Translation Error: {str(e)}]"
            
            # Update the text field with the translation
            text_field.value = translated
            text_field.disabled = False
            page.update()
            
            # Auto-play TTS if enabled
            if auto_tts_active:
                _play_tts_if_enabled(
                    translated, 
                    voice_dropdown.value, 
                    device_str, 
                    speed_slider.value,
                    page=page,
                    target_language=target_language  # Pass the target language
                )
            
        except Exception as e:
            print(f"Translation error: {e}")
            try:
                text_field.value = f"{original_text}\n\n[Translation Error: {str(e)}]"
                text_field.disabled = False
                page.update()
            except Exception as update_error:
                print(f"Error updating UI: {update_error}")
            
        return

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

        # Clean up translation worker if active
        global auto_translate_active, translate_worker_thread
        if auto_translate_active:
            auto_translate_active = False
            print("Auto-translate disabled due to recording stop")
            
            # Reset translate button state
            translate_button.text = "Translate"
            translate_button.bgcolor = None
            page.update()
        
        # Clear translation queue
        with translation_queue.mutex:
            translation_queue.queue.clear()
        
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

    # Function to unload the current model
    def unload_model(e):
        global state, kokoro_model, mimi, tokenizer, lm
        if state is not None or mimi is not None or lm is not None:
            print("Unloading models to free up memory...")
            # Clean up state
            if state is not None:
                if hasattr(state, 'lm_gen') and state.lm_gen is not None:
                    if hasattr(state.lm_gen, 'lm'):
                        state.lm_gen.lm = None
                    state.lm_gen = None
                state = None
            
            # Clean up individual model components
            if mimi is not None:
                if hasattr(mimi, 'model'):
                    mimi.model = None
                mimi = None
                
            if lm is not None:
                if hasattr(lm, 'model'):
                    lm.model = None
                lm = None
                
            if tokenizer is not None:
                tokenizer = None
                
            if kokoro_model is not None:
                # Clean up Kokoro model resources
                if hasattr(kokoro_model, 'model'):
                    kokoro_model.model = None
                if hasattr(kokoro_model, 'vocoder'):
                    kokoro_model.vocoder = None
                kokoro_model = None
                
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print("Models unloaded from memory")
            
            # Show a notification
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Models unloaded from memory. Will reload when needed."),
                action="OK",
                action_color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_400,
            )
            page.snack_bar.open = True
            page.update()
        else:
            print("No models are currently loaded")
            page.snack_bar = ft.SnackBar(
                content=ft.Text("No models are currently loaded"),
                action="OK",
                action_color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE_400,
            )
            page.snack_bar.open = True
            page.update()

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

    # Start the UI refresher in a separate thread
    threading.Thread(target=ui_refresher, daemon=True).start()
    
    # Main UI Layout
    page.add(
        ft.Column(
            [
                # Title and Controls
                ft.Row([
                    ft.Text("Ottertory", size=18, weight=ft.FontWeight.BOLD, expand=True),
                    # Add Unload button with tooltip
                    ft.ElevatedButton(
                        "Unload Model",
                        icon=ft.Icons.UNARCHIVE,
                        on_click=unload_model,
                        tooltip="Unload model to free up memory (will reload on next record)",
                        width=150
                    )
                ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # Recording controls - Row 1
                ft.Row([
                    start_button, 
                    stop_button,
                    audio_source_toggle,
                    overlay_toggle,
                    mic_status,
                    level_bar
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # TTS Controls - Row 2
                ft.Row([
                    tts_engine_btn,
                    play_tts_button,
                    stop_tts_button,
                    voice_dropdown,
                    ft.Column([
                        voice_drop,
                        voice_drop_text
                    ], width=120, spacing=0, tight=True, alignment=ft.MainAxisAlignment.CENTER),
                    speed_slider,
                    auto_tts_toggle,
                    download_tts_button,
                    copy_button,
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # Translation and LLM Controls - Row 3
                ft.Row([
                    translate_button,
                    language_dropdown,
                    # ID Toggle (Multilingual Mode)
                    ft.Row([
                        ft.Text("ID:", size=12, color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
                        ft.Switch(
                            value=False,
                            active_color=ft.Colors.BLUE_200,
                            inactive_thumb_color=ft.Colors.WHITE,
                            inactive_track_color=ft.Colors.GREY_600,
                            scale=0.8,
                            on_change=lambda e: setattr(page, 'multilingual_mode', e.control.value)
                        )
                    ], spacing=5, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    # LLM controls
                    ft.Container(width=10),  # Spacer
                    ollama_dropdown,
                    refresh_button,
                    summarize_button,
                    bullets_button,
                    proof_button,
                ], spacing=5, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # Command input area - Row 4
                ft.Row([
                    text_dest_toggle,
                    command_field,
                    run_command_button,
                    clear_command_button
                ], spacing=5, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                # Transcription area - Takes remaining space
                ft.Container(
                    content=result_text,
                    expand=True,
                    margin=ft.margin.only(top=10)
                ),
            ],
            spacing=10,
            expand=True,
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
        
        # Clean up translation resources
        global auto_translate_active, translate_worker_thread, translation_queue
        if auto_translate_active:
            auto_translate_active = False
            print("Stopped auto-translation")
        
        # Clear translation queue
        with translation_queue.mutex:
            translation_queue.queue.clear()
        
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
