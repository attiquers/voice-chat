import streamlit as st
import os
import logging
import numpy as np

# --- TTS Model ---
from kokoro_tts import KokoroTTS

# --- LangChain imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LLM (Google Gemini) ---
from langchain_google_genai import ChatGoogleGenerativeAI

from typing import Iterator

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 24000
INITIAL_SILENCE_DURATION = 0.1
INITIAL_SILENCE_SAMPLES = int(SAMPLE_RATE * INITIAL_SILENCE_DURATION)
INITIAL_SILENCE = np.zeros(INITIAL_SILENCE_SAMPLES, dtype=np.float32)

# --- Initialize TTS ---
@st.cache_resource
def initialize_kokoro_tts():
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(model_dir, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(model_dir, "voices-v1.0.bin")
        return KokoroTTS(model_path, voices_path)
    except Exception as e:
        logger.error("Kokoro TTS init failed", exc_info=True)
        return None

kokoro_tts = initialize_kokoro_tts()

# --- Initialize Gemini LLM ---
@st.cache_resource
def initialize_gemini_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not set")
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            streaming=True,
            google_api_key=api_key
        )
    except Exception as e:
        logger.error("Gemini init failed", exc_info=True)
        return None

llm_stream_instance = initialize_gemini_llm()

# --- LangChain Prompt & Chain ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer with each point or sentence on a separate line."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

def _synthesize_text_chunk(text_chunk: str) -> np.ndarray:
    if not text_chunk.strip() or not kokoro_tts:
        return np.array([])
    try:
        return kokoro_tts.synthesize(text_chunk.strip()) or np.array([])
    except Exception as e:
        logger.error(f"TTS synthesis failed: {text_chunk}", exc_info=True)
        return np.array([])

def _process_text_for_audio(full_text: str) -> Iterator[np.ndarray]:
    sentences = []
    buffer = ""
    for line in full_text.split('\n'):
        for char in line:
            buffer += char
            if char in ".!?":
                sentences.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            sentences.append(buffer.strip())
            buffer = ""
    for sentence in sentences:
        audio = _synthesize_text_chunk(sentence)
        if audio.size > 0:
            yield sentence, audio

main_chain = None
if llm_stream_instance:
    main_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in x["chat_history"]]
        )
        | prompt_template
        | llm_stream_instance
        | StrOutputParser()
    )

# --- Streamlit UI ---
st.set_page_config(page_title="LLM to TTS Chat", layout="wide")
st.title("💬🔊 Streaming Chat with Kokoro TTS")

if kokoro_tts:
    st.success("Kokoro TTS ready")
else:
    st.warning("Kokoro TTS not initialized")

if llm_stream_instance:
    st.success("Gemini LLM ready")
else:
    st.error("Gemini LLM not available")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chat_input = {
        "input": user_input,
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]
    }

    full_response = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for chunk in main_chain.stream(chat_input):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            # Play each sentence independently
            for idx, (sentence, audio) in enumerate(_process_text_for_audio(full_response)):
                with st.container():
                    st.markdown(f"🔈 Sentence {idx + 1}: {sentence}")
                    st.audio(audio, sample_rate=SAMPLE_RATE, format="audio/wav", autoplay=True)

        except Exception as e:
            logger.error("LLM or audio error", exc_info=True)
            placeholder.markdown(f"Error: {str(e)}")
