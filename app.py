# app.py
import streamlit as st
import os
import logging
import numpy as np
from typing import Iterator, List, Dict, Any
import queue
import threading
import time

# For Audio Processing and Speech-to-Text
import av # PyAV, used by streamlit-webrtc
from scipy.signal import resample # Using scipy for resampling

# Streamlit WebRTC for audio input
# Removed ClientSettings import
from streamlit_webrtc import WebRtcMode, webrtc_streamer, AudioProcessorBase

# --- Import custom modules ---
from kokoro_tts import KokoroTTS
from fastwhisper_stt import FasterWhisperSTT

# --- LangChain imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LLM (Google Gemini) ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
TTS_SAMPLE_RATE = 24000 # Sample rate for Kokoro TTS output
WHISPER_SAMPLE_RATE = 16000 # Sample rate expected by Whisper
WEBRTC_DEFAULT_SAMPLE_RATE = 48000 # Default sample rate from streamlit-webrtc

# --- Global Queue for Audio Processing ---
audio_queue = queue.Queue()

# --- Audio Processor for WebRTC ---
class AudioFrameProcessor(AudioProcessorBase):
    """
    A WebRTC audio processor that buffers audio frames when recording is active
    and puts the concatenated, resampled audio into a global queue upon stopping.
    """
    def __init__(self) -> None:
        self.audio_buffer = []
        self.is_recording = False
        self.input_sample_rate = WEBRTC_DEFAULT_SAMPLE_RATE 
        logger.info("AudioFrameProcessor initialized.")

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Receives an audio frame from the WebRTC stream and buffers it if recording."""
        if self.is_recording:
            # Removed 'format="s16"' as it causes TypeError in some av versions
            audio_array = frame.to_ndarray(layout="mono").flatten().astype(np.float32) / 32768.0
            self.audio_buffer.append(audio_array)
        return frame

    def start_recording(self):
        """Starts the audio recording."""
        self.is_recording = True
        self.audio_buffer.clear()
        logger.info("Recording started by AudioFrameProcessor.")

    def stop_recording(self):
        """Stops the audio recording and processes the buffered audio."""
        self.is_recording = False
        logger.info("Recording stopped by AudioFrameProcessor.")
        if self.audio_buffer:
            concatenated_audio = np.concatenate(self.audio_buffer)
            try:
                # Resample audio using scipy.signal.resample
                num_output_samples = int(len(concatenated_audio) * (WHISPER_SAMPLE_RATE / self.input_sample_rate))
                resampled_audio = resample(concatenated_audio, num_output_samples)
                
                audio_queue.put(resampled_audio.astype(np.float32)) # Ensure float32 type
                logger.info(f"Recorded audio (original samples: {len(concatenated_audio)}) resampled to {len(resampled_audio)} samples and put into queue.")
            except Exception as e:
                logger.error(f"Error during audio resampling: {e}", exc_info=True)
                audio_queue.put(np.array([])) # Put empty array on error
        else:
            audio_queue.put(np.array([])) # Put empty array if nothing was recorded
        self.audio_buffer.clear() # Clear buffer after processing

# --- Initialize Models ---
@st.cache_resource
def initialize_kokoro_tts_model():
    """Initializes and caches the KokoroTTS model."""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        tts_instance = KokoroTTS(
            model_path=os.path.join(model_dir, "kokoro-v1.0.int8.onnx"),
            voices_path=os.path.join(model_dir, "voices-v1.0.bin")
        )
        logger.info("Kokoro TTS initialized successfully.")
        return tts_instance
    except Exception as e:
        logger.error(f"Error initializing Kokoro TTS: {e}", exc_info=True)
        st.error(f"Error initializing TTS: {e}. Please ensure 'models' directory and its contents are present.")
        return None

@st.cache_resource
def initialize_gemini_llm():
    """Initializes and caches the Google Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY environment variable not set. Please set it to use Gemini LLM.")
        logger.error("GOOGLE_API_KEY is not set. Gemini LLM cannot be initialized.")
        return None
    try:
        llm_instance = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=api_key
        )
        logger.info("Gemini LLM initialized successfully.")
        return llm_instance
    except Exception as e:
        logger.error(f"Error initializing Gemini LLM: {e}", exc_info=True)
        st.error(f"Error initializing Gemini LLM: {e}")
        return None

@st.cache_resource
def initialize_whisper_stt_model():
    """Initializes and caches the FasterWhisperSTT model."""
    try:
        stt_instance = FasterWhisperSTT(model_size="base") # Use "base" model for balance
        logger.info("Faster-Whisper STT initialized successfully.")
        return stt_instance
    except Exception as e:
        logger.error(f"Error initializing Faster-Whisper STT: {e}", exc_info=True)
        st.error(f"Error initializing Speech-to-Text model: {e}")
        return None

# Load all models at startup
kokoro_tts = initialize_kokoro_tts_model()
llm_stream_instance = initialize_gemini_llm()
whisper_stt = initialize_whisper_stt_model()

# --- TTS Synthesis Function ---
def _synthesize_text_chunk(text_chunk: str) -> np.ndarray:
    """Synthesizes a single text chunk into audio using KokoroTTS."""
    if not text_chunk.strip() or not kokoro_tts:
        return np.array([])
    try:
        audio = kokoro_tts.synthesize(text_chunk.strip())
        if audio is None or audio.size == 0:
            logger.warning(f"Synthesized audio for chunk '{text_chunk[:50]}...' was empty or None.")
            return np.array([])
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"TTS synthesis failed for chunk: '{text_chunk[:50]}...'", exc_info=True)
        return np.array([])

def _process_text_for_audio(full_text: str) -> Iterator[tuple[str, np.ndarray]]:
    """Processes LLM response into sentences and yields each with its synthesized audio."""
    sentences = []
    buffer = ""
    for line in full_text.split('\n'):
        line = line.strip()
        if not line: continue
        for char in line:
            buffer += char
            if char in ".!?":
                if buffer.strip(): sentences.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            sentences.append(buffer.strip())
            buffer = ""
    
    if buffer.strip(): sentences.append(buffer.strip())

    for idx, sentence in enumerate(sentences):
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
    main_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) 
                for msg in x.get("chat_history", [])
            ]
        )
        | prompt_template
        | llm_stream_instance
        | StrOutputParser()
    )
else:
    st.warning("LLM not initialized, chat functionality will be limited.")

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="LLM to TTS Voice Chat", layout="wide")
st.title("💬🔊 Voice & Text Chat Assistant")

# Display initialization status
if kokoro_tts: st.success("Kokoro TTS ready")
else: st.warning("Kokoro TTS not initialized. Audio output will not work.")
if llm_stream_instance: st.success("Gemini LLM ready")
else: st.error("Gemini LLM not available. Chat responses will not be generated.")
if whisper_stt: st.success("Speech-to-Text (Whisper) model ready")
else: st.error("Speech-to-Text (Whisper) model not available. Voice input will not work.")

# Initialize chat history and voice input states
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "recording_active" not in st.session_state: st.session_state.recording_active = False
if "processing_audio" not in st.session_state: st.session_state.processing_audio = False
if "voice_input_text" not in st.session_state: st.session_state.voice_input_text = ""

# Create two vertical columns for layout
col1, col2 = st.columns([2, 1]) # Chat takes 2/3 width, Voice Input takes 1/3

with col1: # Chat Conversation Section
    st.header("Chat Conversation")
    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Clear chat button
    if st.button("🗑️ Clear Chat", key="clear_chat_button"):
        st.session_state.chat_history = []
        st.session_state.voice_input_text = ""
        st.rerun()

    # Text input for chat (still available as an alternative to voice)
    text_user_input = st.chat_input("Type your message...")

with col2: # Voice Input Section
    st.header("Voice Input")
    st.write("Click 'Start Recording' to begin, 'Stop Recording' to end and transcribe.")

    # WebRTC streamer for audio input
    webrtc_ctx = webrtc_streamer(
        key="speech_to_text_stream",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioFrameProcessor,
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
        # Removed client_settings argument entirely to avoid ClientSettings import issue
        # This will use default client settings, which usually include common STUN servers.
    )

    # Control buttons for recording
    if webrtc_ctx.state.playing:
        if st.session_state.recording_active:
            if st.button("🔴 Stop Recording", key="stop_record_button"):
                if webrtc_ctx.audio_processor:
                    webrtc_ctx.audio_processor.stop_recording()
                st.session_state.recording_active = False
                st.session_state.processing_audio = True
                st.rerun()
        else:
            st.info("Microphone is active, ready to record.")
            if st.button("🎤 Start Recording", key="start_record_button"):
                if webrtc_ctx.audio_processor:
                    webrtc_ctx.audio_processor.start_recording()
                st.session_state.recording_active = True
                st.session_state.processing_audio = False
                st.rerun()
    else:
        st.warning("Waiting for microphone access...")

    # Placeholder for transcription status/output
    transcribed_text_placeholder = st.empty()

    # Process recorded audio when the 'processing_audio' flag is set
    if st.session_state.processing_audio:
        transcribed_text_placeholder.info("Processing audio for transcription...")
        try:
            recorded_audio = audio_queue.get(timeout=20) # 20-second timeout
            
            if recorded_audio.size > 0 and whisper_stt:
                transcribed_text = whisper_stt.transcribe(recorded_audio)
                if transcribed_text:
                    transcribed_text_placeholder.success(f"Transcribed: \"{transcribed_text}\"")
                    st.session_state.voice_input_text = transcribed_text
                else:
                    transcribed_text_placeholder.warning("No speech detected or transcription was empty.")
                    st.session_state.voice_input_text = ""
                st.session_state.processing_audio = False
                st.rerun()
            else:
                transcribed_text_placeholder.warning("No audio recorded or Speech-to-Text model not loaded.")
                st.session_state.processing_audio = False
                st.session_state.voice_input_text = ""
        except queue.Empty:
            transcribed_text_placeholder.error("Audio processing timed out. No audio received from recorder.")
            st.session_state.processing_audio = False
            st.session_state.voice_input_text = ""
        except Exception as e:
            transcribed_text_placeholder.error(f"Error during transcription: {e}")
            logger.error("Transcription error", exc_info=True)
            st.session_state.processing_audio = False
            st.session_state.voice_input_text = ""

    # Display last transcribed text if available and not currently processing
    if st.session_state.voice_input_text and not st.session_state.processing_audio:
        transcribed_text_placeholder.write(f"Last voice input: {st.session_state.voice_input_text}")


# --- Main Chat Processing Logic ---
# Determine the actual input for the LLM (text box or voice input)
current_llm_input = None
if text_user_input:
    current_llm_input = text_user_input
elif st.session_state.voice_input_text:
    current_llm_input = st.session_state.voice_input_text
    st.session_state.voice_input_text = "" # Clear voice input after use

if current_llm_input:
    st.session_state.chat_history.append({"role": "user", "content": current_llm_input})
    with col1:
        with st.chat_message("user"):
            st.markdown(current_llm_input)

    chain_input = {
        "input": current_llm_input,
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ]
    }

    full_response = ""
    with col1:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                if main_chain:
                    for chunk in main_chain.stream(chain_input):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    
                    placeholder.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

                    for idx, (sentence_text, audio_data) in enumerate(_process_text_for_audio(full_response)):
                        st.audio(audio_data, sample_rate=TTS_SAMPLE_RATE, format="audio/wav", autoplay=True)
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