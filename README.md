# Healthcare-Translation-Web-App-with-Generative-AI


A real-time multilingual **voice translation app** designed for **patients and healthcare providers**. This prototype leverages **Generative AI**, **Web Speech API**, and **text-to-speech synthesis** to break language barriers in clinical environments.

---

##  Project Overview

This project is a 48-hour prototype challenge to build a healthcare translation app that can:

- Convert **spoken medical input** into a **text transcript**
- Translate that transcript into a **target language** using **Generative AI (Groq)**
- Provide **text-to-speech playback** of the translation
- Run in **mobile-first, browser-based environments** (Streamlit)

---

##  Core Features

| Feature               | Description |
|----------------------|-------------|
| 🎙 Voice Input        | Captures user speech using the browser's microphone via Web Speech API |
|  Transcript Display | Shows original and translated transcripts side-by-side in real-time |
|  AI Translation     | Uses Groq LLM (e.g., Mixtral or LLaMA3) for accurate multilingual translation |
|  Audio Playback     | Reads translated text aloud in the target language |
|  Mobile-First Design| Fully responsive and accessible on phones, tablets, and desktops |

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (Python)
- **Speech Recognition:** Web Speech API (JavaScript) or Whisper Speech
- **Translation Engine:** Groq API with LLMs (e.g., Mixtral, LLaMA3)
- **Text-to-Speech:** Web Speech API (`SpeechSynthesis`) or whisper speech
- **Custom JS Communication:** `streamlit.components.v1`

---

