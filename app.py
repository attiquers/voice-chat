import streamlit as st
import os
import logging
import numpy as np
import tempfile
import soundfile as sf

from typing import Iterator

# --- Custom modules ---
from kokoro_tts import KokoroTTS
from fastwhisper_stt import FasterWhisperSTT

# --- LangChain & Gemini ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
TTS_SAMPLE_RATE = 24000

# --- Initialize Models ---
@st.cache_resource
def initialize_kokoro_tts_model():
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        return KokoroTTS(
            model_path=os.path.join(model_dir, "kokoro-v1.0.int8.onnx"),
            voices_path=os.path.join(model_dir, "voices-v1.0.bin")
        )
    except Exception as e:
        logger.error(f"TTS init error: {e}", exc_info=True)
        st.error("Kokoro TTS init failed")
        return None

@st.cache_resource
def initialize_gemini_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY")
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=api_key
        )
    except Exception as e:
        logger.error("Gemini init error", exc_info=True)
        st.error("Gemini LLM init failed")
        return None

@st.cache_resource
def initialize_whisper_stt_model():
    try:
        return FasterWhisperSTT(model_size="base")
    except Exception as e:
        logger.error("STT init error", exc_info=True)
        st.error("Whisper STT init failed")
        return None

kokoro_tts = initialize_kokoro_tts_model()
llm = initialize_gemini_llm()
whisper_stt = initialize_whisper_stt_model()

# --- LangChain Prompt ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and concisely."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

main_chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in x.get("chat_history", [])
        ]
    )
    | prompt_template
    | llm
    | StrOutputParser()
) if llm else None

# --- Helper functions ---
def synthesize_text(text: str) -> np.ndarray:
    if kokoro_tts and text.strip():
        try:
            audio = kokoro_tts.synthesize(text.strip())
            return audio.astype(np.float32) if audio is not None else np.array([])
        except Exception as e:
            logger.error("TTS error", exc_info=True)
    return np.array([])

# --- Streamlit UI ---
st.set_page_config(page_title="LLM Voice Assistant", layout="wide")
st.title("🧠🎤 Voice + Text Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_input_text" not in st.session_state:
    st.session_state.voice_input_text = ""

col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    text_input = st.chat_input("Type your message...")

with col2:
    st.header("🎙️ Voice Input")
    st.write("Use the audio recorder below.")

    audio_file = st.audio_input("Record your voice")
    transcribed_text_placeholder = st.empty()

    if audio_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name

            audio_data, sample_rate = sf.read(tmp_path)
            st.audio(tmp_path)

            if audio_data.size > 0 and whisper_stt:
                transcribed = whisper_stt.transcribe(audio_data)
                if transcribed:
                    st.session_state.voice_input_text = transcribed
                    transcribed_text_placeholder.success(f"Transcribed: \"{transcribed}\"")
                else:
                    transcribed_text_placeholder.warning("No speech detected.")
            else:
                transcribed_text_placeholder.warning("Invalid audio or STT unavailable")
        except Exception as e:
            logger.error("Audio input error", exc_info=True)
            transcribed_text_placeholder.error(f"Error: {e}")

    if st.session_state.voice_input_text:
        transcribed_text_placeholder.info(f"Last voice input: {st.session_state.voice_input_text}")

# --- Process Input ---
llm_input = text_input or st.session_state.voice_input_text
if llm_input:
    st.session_state.chat_history.append({"role": "user", "content": llm_input})
    st.session_state.voice_input_text = ""

    with col1:
        with st.chat_message("user"):
            st.markdown(llm_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            try:
                if main_chain:
                    for chunk in main_chain.stream({
                        "input": llm_input,
                        "chat_history": st.session_state.chat_history[:-1]
                    }):
                        response += chunk
                        placeholder.markdown(response + "▌")

                    placeholder.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                    audio = synthesize_text(response)
                    if audio.size > 0:
                        st.audio(audio, sample_rate=TTS_SAMPLE_RATE)
                else:
                    placeholder.markdown("LLM not initialized")
            except Exception as e:
                logger.error("Chat error", exc_info=True)
                placeholder.markdown(f"Error: {e}")
