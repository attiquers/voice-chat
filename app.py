# app.py

import streamlit as st
import tempfile
import soundfile as sf

from fastwhisper_stt import VoskSTT
from kokoro_tts import KokoroTTS

from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage


# Page config
st.set_page_config(page_title="🎙️ Local Voice Chat", layout="centered")
st.title("🗣️ Local Voice Chat with Gemini + Vosk + Kokoro")

# Initialize models
stt = VoskSTT(model_path="models/vosk")
tts = KokoroTTS()
llm = ChatGoogleGenerativeAI(model="gemini-pro")


# Audio input
audio_file = st.audio_input("🎤 Record your voice")

if audio_file:
    st.audio(audio_file)

    # Save to temp .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getbuffer())
        tmp.flush()
        wav_path = tmp.name

    # Transcribe with STT
    st.markdown("🧠 **Transcribing...**")
    transcription = stt.transcribe(wav_path)
    st.success(f"📝 You said: `{transcription}`")

    # LLM response
    with st.spinner("🤖 Gemini is thinking..."):
        gemini_reply = llm([HumanMessage(content=transcription)]).content
    st.markdown(f"💬 Gemini says: {gemini_reply}")

    # TTS response
    st.markdown("🗣️ **Speaking reply...**")
    audio_output = tts.speak(gemini_reply)
    st.audio(audio_output, format="audio/wav")
