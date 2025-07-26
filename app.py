import os
import streamlit as st
import soundfile as sf
import tempfile

from fastwhisper_stt import VoskSTT
from kokoro_tts import KokoroTTS

# ✅ Correct import from `langchain_google_genai`
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Page configuration
st.set_page_config(page_title="🎙️ Local Voice Chat", layout="centered")
st.title("🧠 Voice Chat — Gemini, Vosk & Kokoro")

# Load models
stt = VoskSTT(model_path="models/vosk")
tts = KokoroTTS()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Record user message
audio_file = st.audio_input("🎤 Record your voice")

if audio_file:
    st.audio(audio_file)  # play back recorded audio

    # Save WAV locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getbuffer())
        tmp.flush()
        wav_path = tmp.name

    # Transcribe speech
    transcription = stt.transcribe(wav_path)
    st.success(f"You said: {transcription}")

    # Get Gemini LLM reply
    response = llm.invoke([HumanMessage(content=transcription)])
    gemini_reply = response.content
    st.markdown(f"💬 Gemini: {gemini_reply}")

    # Convert reply to speech
    audio_data = tts.speak(gemini_reply)
    st.audio(audio_data, format="audio/wav")
