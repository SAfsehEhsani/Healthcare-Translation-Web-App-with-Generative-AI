

import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import io
from gtts import gTTS
from gtts import lang as gtts_lang
import base64
from streamlit_mic_recorder import mic_recorder

# Import faster_whisper libraries
from faster_whisper import WhisperModel
import tempfile
import os # Re-import os just to be explicit if needed for os.remove


# --- Streamlit UI Configuration (MUST BE FIRST) ---
# This needs to be the very first Streamlit command called.
st.set_page_config(
    page_title="Healthcare Translation Web App with Generative AI",
    layout="wide",
    initial_sidebar_state="auto"
)


# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if Groq API key is loaded
if not GROQ_API_KEY:
    st.error("Groq API key not found. Please set GROQ_API_KEY in your .env file or secrets.toml.")
    st.stop()

# --- Initialize clients and Whisper Model using st.cache_resource ---
@st.cache_resource
def init_groq_client(api_key):
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        # Note: Using st.error inside cached resource is generally discouraged,
        # but kept here for immediate feedback if key is missing.
        # A cleaner approach might be to check the key *before* calling the cached function.
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop() # stopping execution is safe here

@st.cache_resource
def init_whisper_model(model_size="base"):
    """Initializes the  model."""
    try:
        # model_size can be "tiny", "base", "small", "medium", "large-v3"
        # or quantized versions like "base.en", "small.en" etc.
        # "base" is a good balance for prototype on CPU. "base.en" is English only.
        # device="cuda" for GPU, device="cpu" for CPU
        # compute_type="int8" or "float16" can affect performance/accuracy
        # Using a placeholder while loading
        loading_message = st.empty()
        loading_message.info(f"Loading Whisper model: {model_size} (This may take a moment)...")

        model = WhisperModel(model_size, device="cpu", compute_type="int8") # Using CPU and int8 for broader compatibility

        loading_message.success(f"Whisper model {model_size} loaded.")
        loading_message.empty()
        # loading_message.empty() # Optional: Clear the message after loading
        return model
    except Exception as e:
        # Note: See comment in init_groq_client about st.error in cached functions.
        st.error(f"Failed to load Whisper model: {e}")
        st.stop() # stopping execution is safe here


# Initialize clients and Whisper Model
# These calls will trigger the cached functions on the first run
groq_client = init_groq_client(GROQ_API_KEY)
# The whisper model loading message will appear above the rest of the UI
whisper_model = init_whisper_model("base") # Choose your desired model size

# Supported languages (Mapping full name to gTTS code and Whisper code)
LANGUAGES = {
    "English": {"gtts": "en", "whisper_stt": "en"},
    "Spanish": {"gtts": "es", "whisper_stt": "es"},
    "French": {"gtts": "fr", "whisper_stt": "fr"},
    "German": {"gtts": "de", "whisper_stt": "de"},
    "Chinese (Simplified)": {"gtts": "zh-CN", "whisper_stt": "zh"},
    "Portuguese": {"gtts": "pt", "whisper_stt": "pt"},
    "Russian": {"gtts": "ru", "whisper_stt": "ru"},
    "Arabic": {"gtts": "ar", "whisper_stt": "ar"},
    "Japanese": {"gtts": "ja", "whisper_stt": "ja"},
    "Korean": {"gtts": "ko", "whisper_stt": "ko"},
    "Italian": {"gtts": "it", "whisper_stt": "it"},
    "Dutch": {"gtts": "nl", "whisper_stt": "nl"},
    "Swedish": {"gtts": "sv", "whisper_stt": "sv"},
}


# --- Helper Function for Text-to-Speech ---
def text_to_audio(text, lang_code):
    """Converts text to speech and returns audio as bytes."""
    if not text:
        return None
    try:
        supported_languages = gtts_lang.tts_langs()
        if lang_code not in supported_languages:
             base_lang = lang_code.split('-')[0]
             if base_lang in supported_languages:
                 lang_code = base_lang
             else:
                 st.warning(f"Audio playback might not be fully supported for language code '{lang_code}' via gTTS.")

        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating audio for language code '{lang_code}': {e}")
        return None

# --- Faster Whisper Speech-to-Text Function ---
def transcribe_audio_whisper(audio_bytes, language_code):
    """Transcribes audio bytes using faster-whisper (local model)."""
    if not audio_bytes:
        return "Error: No audio data received."

    try:
        # Save bytes to a temporary file for faster-whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_filepath = tmp_file.name

        st.info(f"Transcribing audio using Whisper ({language_code})...")

        segments, info = whisper_model.transcribe( # Use the cached model
            tmp_filepath,
            language=language_code,
        )

        # Join segments into a single transcript string
        transcript_parts = [segment.text for segment in segments]
        transcript = "".join(transcript_parts).strip()

        # Clean up the temporary file
        os.remove(tmp_filepath)

        if transcript:
             return transcript
        else:
             return "Could not transcribe audio. Please try speaking again or adjust model settings."

    except Exception as e:
        # Ensure temporary file is cleaned up even if error occurs
        if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
             os.remove(tmp_filepath)
        st.error(f"Error during Whisper transcription: {e}")
        return f"Transcription failed: {e}"


# --- Generative AI (Groq) Translation Function ---
def translate_text(text, source_lang_name, target_lang_name):
    """Translates text using the Groq API."""
    if not text or text.strip() == "":
        return ""

    prompt = f"""Translate the following healthcare-related text from {source_lang_name} to {target_lang_name}.
    Focus on accurate and natural-sounding translation suitable for communication between patients and healthcare providers.
    Translate only the provided text and return only the translated text. Do not include any conversational phrases, explanations, or the original text in your response.

    Text to translate:
    "{text}"
    """

    try:
        chat_completion = groq_client.chat.completions.create( # Use cached client
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and accurate translator assistant specialized in healthcare contexts. You translate text as requested by the user.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="llama3-8b-8192", # Or other suitable Groq model
            temperature=0.1,
            max_tokens=1000,
        )
        translated_text = chat_completion.choices[0].message.content.strip()

        if translated_text.startswith('"') and translated_text.endswith('"'):
             translated_text = translated_text[1:-1]
        unwanted_prefixes = [
            f"Translated text in {target_lang_name}:", f"{target_lang_name}:",
            "Translation:", "Here is the translation:", "The translation is:",
            "Result:"
        ]
        for prefix in unwanted_prefixes:
            if translated_text.lower().startswith(prefix.lower()):
                translated_text = translated_text[len(prefix):].strip()
                break

        return translated_text

    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return "Translation failed."


# --- Initialize session state ---
# Initialize these *after* st.set_page_config
if 'current_transcript' not in st.session_state:
    st.session_state.current_transcript = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'last_recorded_audio_bytes' not in st.session_state:
    st.session_state.last_recorded_audio_bytes = None
if 'trigger_audio_translation' not in st.session_state:
    st.session_state.trigger_audio_translation = False


# --- Main App UI Content (After configuration and initialization) ---

st.title("Healthcare Translation Web App with Generative AI")
st.markdown("""
Welcome to the Healthcare Translator Generative AI Application.
Real-time voice-to-voice translation for patients and healthcare providers.

Speech-to-Text: Speak in your native language using the mic

AI Translation: Translate medical speech with generative AI (Groq)

Text-to-Speech: Listen to translated output in the target language


Select languages and click Translate (auto-translates after voice input).
""")

# Language Selection
col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox(
        "Select Input Language (for Voice)",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index("English"),
        key="source_lang"
    )
with col2:
    target_lang_name = st.selectbox(
        "Select Output Language",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index("Spanish"),
        key="target_lang"
    )

st.subheader("Input: Speak or Type")

# Microphone Recorder - Audio Capture
audio_info = mic_recorder(key='recorder')

# --- Logic to handle audio input and trigger STT ---
if audio_info and audio_info['bytes']:
    if 'last_recorded_audio_bytes' not in st.session_state or audio_info['bytes'] != st.session_state.last_recorded_audio_bytes:
        st.session_state.last_recorded_audio_bytes = audio_info['bytes']
        current_source_lang_info = LANGUAGES.get(st.session_state.source_lang)

        if current_source_lang_info and current_source_lang_info.get("whisper_stt"):
            whisper_stt_lang_code = current_source_lang_info["whisper_stt"]

            # --- Perform Actual Speech-to-Text using Faster Whisper ---
            actual_transcript = transcribe_audio_whisper(audio_info['bytes'], whisper_stt_lang_code)

            st.session_state.current_transcript = actual_transcript
            st.session_state.translated_text = ""
            # Only trigger translation if transcription was successful
            st.session_state.trigger_audio_translation = not actual_transcript.startswith("Error:") and not actual_transcript.startswith("Transcription failed:")

            st.rerun()
        else:
            st.warning(f"Whisper Speech-to-Text language code not defined for '{st.session_state.source_lang}'. Cannot transcribe.")
            st.session_state.current_transcript = f"Audio recorded, but transcription failed. Language '{st.session_state.source_lang}' may not be supported or configured correctly for Whisper STT."
            st.session_state.translated_text = ""
            st.session_state.trigger_audio_translation = False
            st.rerun()


# Text Area for Manual Input and Displaying Transcribed Text
input_text_area_value = st.text_area(
    "Original Transcript",
    value=st.session_state.current_transcript,
    height=150,
    key="transcript_input_area"
)

# Sync the text area's current value back to session state whenever it changes
st.session_state.current_transcript = input_text_area_value


# --- Translate Button (for manual text or re-translation) ---
if st.button("Translate", key="translate_button"):
     if st.session_state.current_transcript and st.session_state.current_transcript.strip() != "" and not st.session_state.current_transcript.startswith("Error:") and not st.session_state.current_transcript.startswith("Transcription failed:"):
         current_source_lang_name = st.session_state.source_lang
         current_target_lang_name = st.session_state.target_lang
         with st.spinner(f"Translating from {current_source_lang_name} to {current_target_lang_name}..."):
             translated_text = translate_text(st.session_state.current_transcript, current_source_lang_name, current_target_lang_name)
             st.session_state.translated_text = translated_text
     else:
         st.warning("Please enter text or use voice input first, or check for transcription errors.")


# --- Auto-trigger translation after audio transcription ---
if st.session_state.trigger_audio_translation:
    st.session_state.trigger_audio_translation = False

    if st.session_state.current_transcript and st.session_state.current_transcript.strip() != "" and not st.session_state.current_transcript.startswith("Error:") and not st.session_state.current_transcript.startswith("Transcription failed:"):
         current_source_lang_name = st.session_state.source_lang
         current_target_lang_name = st.session_state.target_lang

         with st.spinner(f"Translating transcribed text from {current_source_lang_name} to {current_target_lang_name}..."):
             translated_text = translate_text(st.session_state.current_transcript, current_source_lang_name, current_target_lang_name)
             st.session_state.translated_text = translated_text
    elif st.session_state.current_transcript.startswith("Error:") or st.session_state.current_transcript.startswith("Transcription failed:"):
        st.error(f"Transcription resulted in an error: {st.session_state.current_transcript}")
        st.session_state.translated_text = ""
    else:
         st.warning("No transcript available to translate.")
         st.session_state.translated_text = ""


# --- Display Translation Output ---
if st.session_state.translated_text:
    st.subheader("Transcript and Translation")

    st.text_area("Original Text", st.session_state.current_transcript, height=150, disabled=True, key="display_original")
    st.text_area("Translated Text", st.session_state.translated_text, height=150, disabled=True, key="display_translated")

    # Audio Playback
    if st.session_state.translated_text != "Translation failed.":
        target_lang_info = LANGUAGES.get(st.session_state.target_lang)
        if target_lang_info and target_lang_info.get("gtts"):
            target_gtts_lang_code = target_lang_info["gtts"]
            audio_bytes = text_to_audio(st.session_state.translated_text, target_gtts_lang_code)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
        else:
            st.warning(f"Audio playback language code not defined or supported by gTTS for '{st.session_state.target_lang}'.")
    elif st.session_state.translated_text == "Translation failed.":
        st.error("Could not generate audio due to translation failure.")

# Optional: Add a clear button
if st.button("Clear", key="clear_button"):
    st.session_state.current_transcript = ""
    st.session_state.translated_text = ""
    st.session_state.last_recorded_audio_bytes = None
    st.session_state.trigger_audio_translation = False
    st.session_state.transcript_input_area = ""
    st.rerun()


# --- Notes ---
st.markdown("---")
st.markdown("""
**Notes & Considerations Built By Syed Afseh Ehsani:**

#*   **Processing Power:** Transcription runs on the server CPU/GPU. Choose a smaller model ("base" or "small") for faster results on typical machines. "large-v3" is more accurate but much slower and requires more RAM/VRAM.
*   **Model Download:** The model files are downloaded the first time runs. This can take several minutes depending on the model size and internet speed.
*   **Real-time STT:** This implementation processes the *entire* recording after stopping. Real-time transcription is more complex and not supported directly by the synchronous Speech API.
*   **Accuracy:** Accuracy depends heavily on the chosen Whisper model size, audio quality, background noise, accents, and specific terminology. For medical terms, it might require trying larger models or exploring fine-tuned versions if available.
*   **Language Support:** Ensure the language code selected in the app maps correctly to a language supported by the specific Whisper model version you load.
*   **Dependencies:** Installing might pull in large dependencies like PyTorch/TensorFlow (depending on compute_type).
*   **Error Handling:** Added checks for transcription errors and adjusted translation trigger.
""")