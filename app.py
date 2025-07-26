import os
import wave
import json
import tempfile
import streamlit as st
import soundfile as sf
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from kokoro_tts import KokoroTTS

# Load API key
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Init components
st.set_page_config(page_title="Voice Chat (Local STT + TTS + OpenRouter LLM)", layout="centered")
st.title("🎙️ Local Voice Chat — Vosk + Kokoro + OpenRouter")

stt_model = Model("models/vosk")
tts = KokoroTTS(model_path="models/kokoro-v1.0.int8.onnx")
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model="mistralai/devstral-small-2505:free",
    temperature=0.7
)

def convert_to_vosk_compatible(src, dst):
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data[:, 0]
    sf.write(dst, data, sr, subtype="PCM_16")

def transcribe(path: str) -> str:
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(stt_model, wf.getframerate())
    text = ""
    while True:
        chunk = wf.readframes(4000)
        if not chunk:
            break
        if rec.AcceptWaveform(chunk):
            text += json.loads(rec.Result()).get("text", "") + " "
    text += json.loads(rec.FinalResult()).get("text", "")
    return text.strip()

# UI: audio input
audio = st.audio_input("🎤 Record your question (Speak clearly)", type="wav")
if audio is not None:
    st.audio(audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as inp:
        inp.write(audio.getvalue())
        inp.flush()
        src = inp.name
    dst = src.replace(".wav", "_mono.wav")
    convert_to_vosk_compatible(src, dst)

    transcription = transcribe(dst)
    st.markdown(f"**You said:** {transcription}")

    # LLM response
    if transcription:
        with st.spinner("🧠 Thinking..."):
            resp = llm.invoke([HumanMessage(content=transcription)])
        reply = resp.content.strip()
        st.markdown(f"**LLM says:** {reply}")

        # Synthesize with chosen voice preset
        audio_bytes = tts.synthesize(reply, voice="am_amber")  # use voice like "am_amber"
        st.audio(audio_bytes, format="audio/wav")
