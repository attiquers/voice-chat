import os
import json
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
from scipy.io import wavfile

from vosk import Model, KaldiRecognizer
from kokoro_tts import KokoroTTS
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# === ENV ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in .env"

# === STT ===
vosk_model = Model("models/vosk")

def transcribe(audio_path):
    with sf.SoundFile(audio_path) as audio_file:
        audio_data = audio_file.read(dtype="int16")
        samplerate = audio_file.samplerate

    recognizer = KaldiRecognizer(vosk_model, samplerate)
    audio_bytes = audio_data.tobytes()

    if recognizer.AcceptWaveform(audio_bytes):
        result = json.loads(recognizer.Result())
        return result.get("text", "")
    else:
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")

# === TTS ===
tts = KokoroTTS(model_path="models/kokoro-v1.0.int8.onnx")

def synthesize(text, speaker="am_amber"):
    audio_bytes = tts.tts(text, speaker=speaker)
    return audio_bytes

# === LLM via OpenRouter ===
llm: BaseChatModel = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=OPENROUTER_API_KEY,
    model="mistralai/devistral-small-2505",
)

chain = RunnableLambda(lambda x: llm.invoke([HumanMessage(content=x["text"])]))

# === Streamlit UI ===
st.title("🎤 Voice Chat with OpenRouter + Kokoro + Vosk")

audio_bytes = st.audio_input("Record your question")

if audio_bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        input_path = tmp.name

    # Convert to 16-bit mono, 16kHz if needed
    with sf.SoundFile(input_path) as f:
        data = f.read()
        samplerate = f.samplerate

    if f.channels > 1:
        data = np.mean(data, axis=1)

    wavfile.write(input_path, 16000, (data * 32767).astype(np.int16))

    transcription = transcribe(input_path)
    st.markdown(f"**You said:** `{transcription}`")

    if transcription:
        response = chain.invoke({"text": transcription})
        reply = response.content
        st.markdown(f"**Assistant:** {reply}")

        # Synthesize and play
        audio_output = synthesize(reply)
        st.audio(audio_output, format="audio/wav")
