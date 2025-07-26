import os
import wave
import streamlit as st
import tempfile
import soundfile as sf
from dotenv import load_dotenv

from vosk import Model, KaldiRecognizer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from kokoro_tts import KokoroTTS

# Load environment variables from .env
load_dotenv()

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    model="mistralai/devstral-small-2505:free",
)

# Initialize Kokoro TTS
tts = KokoroTTS(model_path="models/kokoro-v1.0.int8.onnx", config_path="models/voices-v1.0.bin")

# Initialize Vosk STT
vosk_model = Model("models/vosk")


def convert_audio_to_16bit_mono(input_path, output_path):
    """Convert audio to 16-bit mono WAV for Vosk."""
    data, samplerate = sf.read(input_path)
    if data.ndim > 1:
        data = data[:, 0]  # convert to mono
    sf.write(output_path, data, samplerate, subtype='PCM_16')


def transcribe_vosk(audio_path: str) -> str:
    """Transcribe audio using Vosk STT."""
    recognizer = KaldiRecognizer(vosk_model, 16000)

    with wave.open(audio_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

    result = recognizer.FinalResult()
    import json
    return json.loads(result).get("text", "")


def synthesize_tts(text: str, output_path: str):
    """Generate audio using Kokoro TTS."""
    wav_data = tts.synthesize(text)
    with open(output_path, "wb") as f:
        f.write(wav_data)


# Streamlit UI
st.set_page_config(page_title="🎙️ Voice Chat with LLM", layout="centered")
st.title("🎙️ Voice Chat (Vosk + Kokoro + OpenRouter)")

audio_data = st.audio_input("Speak something...", type="wav")

if audio_data:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
        temp_input.write(audio_data.getvalue())
        temp_input.flush()

        converted_path = temp_input.name.replace(".wav", "_converted.wav")
        convert_audio_to_16bit_mono(temp_input.name, converted_path)

        transcription = transcribe_vosk(converted_path)
        st.markdown(f"**You said:** {transcription}")

        if transcription:
            response = llm.invoke([HumanMessage(content=transcription)])
            reply = response.content
            st.markdown(f"**LLM says:** {reply}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                synthesize_tts(reply, temp_output.name)
                st.audio(temp_output.name, format="audio/wav")
