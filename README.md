# Ottertory  

Ottertory provides a **GUI** for handling transcription outputs with a **pop-out live caption box** (adjustable opacity).  
It can output transcriptions in real time to your **default microphone input** (default recording device) and to **VB-Cable Output** (a virtual audio output device). 

Outputs can be copied, have Kokoro TTS read it out loud, Ollama/LMstudio can summarize or any custom command (eg. translate). Recording usually goes in the live transcription box but you can change it to type in the Ollama Custom Command box. 

The shortcut to toggle record is **Alt+Z**. 

> For best accuracy, I still recommend my other app [Antistentorian](https://github.com/pointave/Antistentorian), which uses **Whisper** or **Parakeet**.  
> However, the **Kyutai Moshi** model is very responsive and accurate (trained only on French and English).  

---

## Installation  

1. Create and activate a new environment:  
   ```bash
   conda create -n ottertory python=3.12 -y
   conda activate ottertory

2. Clone the repository and install requirements:

   ```bash
   git clone https://github.com/pointave/Ottertory
   cd Ottertory
   pip install -r requirements.txt
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

---

## (Optional) System Audio Capture with VB-Cable

If you want to record system audio in addition to microphone input:

1. **Install [VB-Cable](https://vb-audio.com/Cable/)** and restart your computer.
2. Open **Sound Settings** → set **Cable Input** as your *default playback device*.
3. Go to the **Recording tab** → right-click **Cable Output** → **Properties**:

   * Check **Listen to this device**
   * Set playback to your normal output device (e.g., headphones, TV, or speakers)
4. Keep your microphone as the **default recording device**.

This routes system audio into Ottertory while preserving your microphone input. Changing default audio device will change your volume knob though...

TIP :  Win+Ctrl+V   will pop up Windows quick audio device selector 

---

## Kokoro
This will download all the models, but you may need Espeak NG in your Environment Variables.
PHONEMIZER_ESPEAK_LIBRARY  C:\Program Files\eSpeak NG\libespeak-ng.dll
PHONEMIZER_ESPEAK_PATH   C:\Program Files\eSpeak NG\
There's probably an export comamnd but you can add it manually.
GLHF


---

## Status

* [x] GUI with live captions (adjustable opacity)
* [x] Real-time transcription to microphone + VB-Cable Output

---

## 📜 License

[MIT](LICENSE)

