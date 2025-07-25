import streamlit as st
import os
import logging
import numpy as np
from typing import Iterator, List, Dict, Any

# --- TTS Model ---
# Assuming KokoroTTS class is in a separate file named kokoro_tts.py
# If not, you would need to include its definition here or ensure it's importable.
from kokoro_tts import KokoroTTS

# --- LangChain imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- LLM (Google Gemini) ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 24000
# Initial silence is not used in the current `_process_text_for_audio` logic,
# but keeping it for completeness if future audio processing needs it.
INITIAL_SILENCE_DURATION = 0.1
INITIAL_SILENCE_SAMPLES = int(SAMPLE_RATE * INITIAL_SILENCE_DURATION)
INITIAL_SILENCE = np.zeros(INITIAL_SILENCE_SAMPLES, dtype=np.float32)

# --- Initialize TTS ---
@st.cache_resource
def initialize_kokoro_tts():
    """Initializes and caches the KokoroTTS model."""
    try:
        # Construct model and voices paths relative to the current script's directory
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(model_dir, "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(model_dir, "voices-v1.0.bin")

        logger.info(f"Initializing Kokoro TTS...")
        logger.info(f"Checking for model file: {model_path} - Exists: {os.path.exists(model_path)}")
        logger.info(f"Checking for voices file: {voices_path} - Exists: {os.path.exists(voices_path)}")

        tts_instance = KokoroTTS(model_path, voices_path)
        logger.info("Kokoro TTS initialized successfully.")
        return tts_instance
    except FileNotFoundError as fnfe:
        logger.error(f"Required TTS model or voices file not found: {fnfe}", exc_info=True)
        st.error(f"Error initializing TTS: {fnfe}. Please ensure 'models' directory and its contents are present.")
        return None
    except Exception as e:
        logger.error("Kokoro TTS initialization failed", exc_info=True)
        st.error(f"Error initializing TTS: {e}")
        return None

kokoro_tts = initialize_kokoro_tts()

# --- Initialize Gemini LLM ---
@st.cache_resource
def initialize_gemini_llm():
    """Initializes and caches the Google Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not set. Please set it to use Gemini LLM.")
        logger.error("GOOGLE_API_KEY is not set. Gemini LLM cannot be initialized.")
        return None
    try:
        # Removed 'streaming=True' from constructor as it's a warning and handled by chain.stream()
        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Ensure this model is available and supported for your API key
            temperature=0.7,
            google_api_key=api_key
        )
        logger.info("Gemini LLM initialized successfully.")
        return llm_instance
    except Exception as e:
        logger.error("Gemini LLM initialization failed", exc_info=True)
        st.error(f"Error initializing Gemini LLM: {e}")
        return None

llm_stream_instance = initialize_gemini_llm()

# --- TTS Synthesis Function (with ValueError fix) ---
def _synthesize_text_chunk(text_chunk: str) -> np.ndarray:
    """
    Synthesizes a single text chunk into audio.
    Handles empty text and potential errors during synthesis.
    """
    if not text_chunk.strip() or not kokoro_tts:
        return np.array([]) # Return empty array if no text or TTS not initialized
    try:
        # Attempt to synthesize the text
        audio = kokoro_tts.synthesize(text_chunk.strip())

        # Check if the synthesized audio array is valid and not empty
        # The ValueError fix is applied here: check .size instead of truthiness
        if audio is None or audio.size == 0:
            logger.warning(f"Synthesized audio for chunk '{text_chunk[:50]}...' was empty or None.")
            return np.array([])
        
        return audio.astype(np.float32) # Ensure consistent float32 dtype for audio
    except Exception as e:
        logger.error(f"TTS synthesis failed for chunk: '{text_chunk[:50]}...'", exc_info=True)
        return np.array([])

def _process_text_for_audio(full_text: str) -> Iterator[tuple[str, np.ndarray]]:
    """
    Processes the full LLM response, splitting it into sentences
    and yielding each sentence along with its synthesized audio.
    """
    sentences = []
    buffer = ""
    # Split by newline first, then process each line for sentence endings
    for line in full_text.split('\n'):
        line = line.strip() # Trim whitespace from lines
        if not line: # Skip empty lines
            continue
        for char in line:
            buffer += char
            # Consider common sentence-ending punctuation
            if char in ".!?":
                if buffer.strip(): # Add sentence if not just punctuation
                    sentences.append(buffer.strip())
                buffer = ""
        # If there's remaining text in the buffer after processing line, add it as a sentence
        if buffer.strip():
            sentences.append(buffer.strip())
            buffer = ""
    
    # If after all processing, there's still a buffer (e.g., text without punctuation), add it
    if buffer.strip():
        sentences.append(buffer.strip())

    for idx, sentence in enumerate(sentences):
        logger.info(f"Processing sentence {idx + 1}/{len(sentences)} for audio: '{sentence[:70]}...'")
        audio = _synthesize_text_chunk(sentence)
        if audio.size > 0:
            yield sentence, audio
        else:
            logger.warning(f"Skipping empty audio for sentence: '{sentence[:70]}...'")

# --- LangChain Prompt & Chain ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer with each point or sentence on a separate line."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

main_chain = None
if llm_stream_instance:
    # Define the LangChain processing chain
    main_chain = (
        # Prepare chat history for the prompt template
        RunnablePassthrough.assign(
            chat_history=lambda x: [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) 
                for msg in x.get("chat_history", []) # Use .get with default for safety
            ]
        )
        | prompt_template # Apply the chat prompt template
        | llm_stream_instance # Pass to the Gemini LLM for generation
        | StrOutputParser() # Parse the LLM's output into a string
    )
else:
    st.warning("LLM not initialized, chat functionality will be limited.")

# --- Streamlit UI ---
st.set_page_config(page_title="LLM to TTS Chat", layout="wide")
st.title("💬🔊 Streaming Chat with Kokoro TTS")

# Display initialization status
if kokoro_tts:
    st.success("Kokoro TTS ready")
else:
    st.warning("Kokoro TTS not initialized. Audio output will not work.")
if llm_stream_instance:
    st.success("Gemini LLM ready")
else:
    st.error("Gemini LLM not available. Chat responses will not be generated.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Clear chat button in sidebar
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun() # Rerun to clear chat display immediately

# User input for chat
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare input for the LangChain
    chain_input = {
        "input": user_input,
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1] # Exclude the current user input for history
        ]
    }

    full_response = ""
    with st.chat_message("assistant"):
        placeholder = st.empty() # Placeholder for streaming LLM response
        try:
            if main_chain:
                # Stream response from LLM
                for chunk in main_chain.stream(chain_input):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌") # Show typing indicator
                
                # Final display of the complete response
                placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                # Play each sentence independently
                # This ensures audio starts playing as soon as a sentence is complete
                for idx, (sentence_text, audio_data) in enumerate(_process_text_for_audio(full_response)):
                    # Optionally display the sentence being played, or just play audio
                    # with st.container():
                    #     st.markdown(f"🔈 Sentence {idx + 1}: {sentence_text}")
                    st.audio(audio_data, sample_rate=SAMPLE_RATE, format="audio/wav", autoplay=True)
                logger.info("Stream completed. All audio played.")
            else:
                error_message = "LLM is not initialized. Cannot generate response."
                placeholder.markdown(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})

        except Exception as e:
            logger.error("LLM or audio processing error during streaming", exc_info=True)
            error_message = f"An error occurred: {str(e)}"
            placeholder.markdown(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

