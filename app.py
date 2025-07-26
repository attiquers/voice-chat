# app.py

import streamlit as st
import tempfile
import soundfile as sf

from fastwhisper_stt import VoskSTT
from kokoro_tts import KokoroTTS
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Initialize components
st.set_page_config(page_title="Voice Chat", layout="centered")
st.title("🗣️ Streamlit Voice Chat App")

whisper = VoskSTT(model_path="models/vosk")
tts = KokoroTTS()
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Capture user voice
audio_input = st.audio_input("🎤 Speak something")

if audio_input:
    st.audio(audio_input)

    # Save recorded audio to temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_input.getbuffer())
        tmpfile.flush()

        # Transcribe with Vosk
        transcription = whisper.transcribe(tmpfile.name)
        st.success(f"📝 Transcription: {transcription}")

        # Generate LLM response
        with st.spinner("Thinking..."):
            response = llm([HumanMessage(content=transcription)])
            text_reply = response.content

        st.markdown(f"💬 Gemini: {text_reply}")

        # Synthesize response using Kokoro
        audio_data = tts.speak(text_reply)
        st.audio(audio_data, format="audio/wav")
