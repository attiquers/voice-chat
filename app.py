# app.py

import os
import streamlit as st
import soundfile as sf
import tempfile
import numpy as np

from kokoro_tts import KokoroTTS
from fastwhisper_stt import FasterWhisperSTT

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------------------------------------------------------
# Init models (Kokoro TTS + FasterWhisper STT + Gemini LLM)
# ------------------------------------------------------------------------------

@st.cache_resource
def load_tts():
    return KokoroTTS(
        model_path="models/kokoro-v1.0.int8.onnx",
        voices_path="models/voices-v1.0.bin"
    )

@st.cache_resource
def load_stt():
    return FasterWhisperSTT(model_size="base")

@st.cache_resource
def get_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return (
        RunnablePassthrough.assign(
            chat_history=lambda x: [
                HumanMessage(content=m["content"]) if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in x.get("chat_history", [])
            ]
        )
        | prompt
        | llm
        | StrOutputParser()
    )

kokoro = load_tts()
whisper = load_stt()
chain = get_chain()

# ------------------------------------------------------------------------------
# UI setup
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Voice Chat", layout="wide")
st.title("🎤 Voice Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------------------
# Capture Audio Input
# ------------------------------------------------------------------------------

audio_file = st.audio_input("🎙️ Record your message")
user_input = None

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    audio_data, _ = sf.read(tmp_path)
    if audio_data.size > 0:
        try:
            user_input = whisper.transcribe(audio_data)
            st.success(f"You said: {user_input}")
        except Exception as e:
            st.error(f"STT failed: {e}")
    else:
        st.warning("No audio captured.")

# ------------------------------------------------------------------------------
# Optional Text Input
# ------------------------------------------------------------------------------

text_input = st.chat_input("💬 Type your message instead")
if text_input:
    user_input = text_input

# ------------------------------------------------------------------------------
# Process Input → LLM → Response + TTS
# ------------------------------------------------------------------------------

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({
            "input": user_input,
            "chat_history": st.session_state.chat_history[:-1]
        }):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        try:
            audio = kokoro.synthesize(full_response.strip()).astype(np.float32)
            if audio.size > 0:
                st.audio(audio, sample_rate=24000)
        except Exception as e:
            st.error(f"TTS failed: {e}")

# ------------------------------------------------------------------------------
# Clear Chat
# ------------------------------------------------------------------------------

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
