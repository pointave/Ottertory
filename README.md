# Ottertory  

Captions on any screen transcribing many popular languages, get realtime translation, multilingual text-to-speech capable. This flet app allows users to connect local large language models to query, summarize, generate tutoring session, by typing or speaking commands to apply on the Live transcription box. 

Using OpenAi TTS endpoint or kokoro, you can control how much memory the program requires. With the transcription model Kyutai and a small light LLM like Granite4-micro with 4k context you can slide just under 8gb, allowing realtime translations. If you add the recommended TTS-Webui openai-tts server and use the Chatterbox model, you can have multilingual realtime with under 16gb. 

It can output transcriptions in real time from your **microphone** (default recording device) or from **VB-Cable Output** (virtual audio output device).

Outputs can be copied, saved, or have the audio downloaded.

The shortcut to toggle record is **Alt+Z** and Play/Stop is **Alt+Space**. Highly recommend having "system volume sounds" at like 10% in Windows volume mixer. Depending on program you'll switch between holding z first and tapping alt or the reverse. 


> For best accuracy, I still recommend my other app [Antistentorian](https://github.com/pointave/Antistentorian), which uses **Whisper** or **Parakeet**.  
> 
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

## (Optional) System Audio Capture with VB-Cable (useful for live captions)

If you want to record system audio in addition to microphone input:

1. **Install [VB-Cable](https://vb-audio.com/Cable/)** and restart your computer.
2. Open **Volume Mixer** → choose programs you want to record/listen output to *Cable Input*.
3. Go to the **Recording tab** → right-click **Cable Output** → **Properties**:

   * Check **Listen to this device**
   * Set playback to your normal output device (e.g., headphones, TV, or speakers)
4. Keep your microphone as the **default recording device**.

---
## Openai-TTS Endpoint

I use https://github.com/rsxdalv/TTS-WebUI, specifically there is an option on the tools page called "OpenAI TTS API" and I have the auto enable checked. You are going to want to install the chatterbox extension as well. I made a python file to specifically only open the endpoint and it boots in like 5 seconds.

***YOU NEED TO CHANGE .env.example to .env*** 

Chatterbox uses like 6gb of vram but is EXTREMELY fast now, like a 10x improvement than a month or two ago. To use the multilingual model you'll have to select the ID checkbox. This pulls the language id tag and the other chatterbox model. I dont know if I should just stick with only multimodel and default to English language id tag instead of using the original model as the English TTS so if you want to test that and report back that'd be helpful, because its kind of a drag swapping the models, but I havent checked how the API endpoint behaves with other models like xttsv2 or w/e yet. 

If you enable auto-tts and have the ID checked it will show both the spoken language and the translation. I recommend using a small model like a Gemma3:2b/Qwen3:1.6/LLama3.2:2b to have everything on at once and as fast as possible.   


There are bugs like not being able to download any tts output for chatterbox, crashing when pushing stop and quickly pushing play or record, ...

---
## Kokoro

This will download all the models, but you may need Espeak NG in your Environment Variables.  ( MOST LIKELY Unnecessary)

PHONEMIZER_ESPEAK_LIBRARY  C:\Program Files\eSpeak NG\libespeak-ng.dll

PHONEMIZER_ESPEAK_PATH   C:\Program Files\eSpeak NG\

There's probably an export comamnd but you can add it manually.
GLHF


---
## Captions
The new settings menu is accessible by right clicking on the captions. I recommend reccording someone speaking when wanting to deep dive into it because it gets hidden at the same time the speaker stops talking. In the settings you can change

* Host and Port
* Model Name
* Voice
* Parameters like speed

I tested with a few APIs like vibevoice and a few chatterbox endpoints, and TTS-Webui was the best. To get those working you'll have to change everytime you restart ottertory. You'll have to try combos of model name and voice. Like for example devnen/Chatterbox-TTS-Server I changed its yaml to set ottertory as the reference audio and set voice in settings to tempclone.wav so I could change inside the speaker in the GUI. Turbo model didnt real seem like it saved that much vram but is pretty cool with the [Laugh] [Clears throat] expressions. Vibevoice was still too slow, there is probably a better way of implementing it though like as you are adding live transcriptions.


## Status

* [x] GUI with live captions (adjustable opacity)
* [x] Real-time transcription to microphone + VB-Cable Output
* [X] OpenAI Speech API connection

---

### CREDITS 

Kokoro
Chatterbox
Kyutai

@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}

@techreport{kyutai2025streaming,
      title={Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling}, 
      author={Neil Zeghidour and Eugene Kharitonov and Manu Orsini and Václav Volhejn and Gabriel de Marmiesse and Edouard Grave and Patrick Pérez and Laurent Mazaré and Alexandre Défossez},
      year={2025},
      eprint={2509.08753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.08753}, 
}

---

## 📜 License

[MIT](LICENSE)

