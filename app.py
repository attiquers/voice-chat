import os
import tempfile
import streamlit as st
from vosk import Model, KaldiRecognizer
import wave
import subprocess
from kokoro_tts import KokoroTTS
from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Kokoro TTS
tts = KokoroTTS(model_path="models/kokoro-v1.0.int8.onnx")

# Initialize Vosk STT
vosk_model = Model("models/vosk")

# Initialize OpenRouter (LangChain with mistral)
llm: BaseChatModel = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/devstral-small-2505:free"
)

# WAV converter to Vosk-compatible format
def convert_to_vosk_compatible(src_path, dst_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", src_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        dst_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Transcribe WAV using Vosk
def transcribe(wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)
    result = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = rec.Result()
            result += res
    final_res = rec.FinalResult()
    result += final_res
    import json
    try:
        return json.loads(final_res)["text"]
    except:
        return ""

# Streamlit UI
st.set_page_config(page_title="🎙 Voice Chat", layout="centered")
st.title("🎙 Voice Chat with Kokoro + Vosk + OpenRouter")

# Record voice
audio = st.audio_input("🎤 Record your question (Speak clearly)")
if audio is not None:
    st.audio(audio)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as inp:
        inp.write(audio.getvalue())
        inp.flush()
        src = inp.name

    dst = src.replace(".wav", "_mono.wav")
    convert_to_vosk_compatible(src, dst)

    transcription = transcribe(dst)
    st.markdown(f"**You said:** `{transcription}`")

    if transcription:
        with st.spinner("🧠 Thinking..."):
            response = llm.invoke([HumanMessage(content=transcription)])
        reply = response.content.strip()
        st.markdown(f"**LLM says:** {reply}")

        audio_bytes = tts.synthesize(reply, voice="am_amber")
        st.audio(audio_bytes, format="audio/wav")
