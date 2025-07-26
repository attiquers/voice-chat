# kokoro_tts.py
import numpy as np
from typing import Iterator
import logging
import os

logger = logging.getLogger(__name__)

# --- GPU Detection and Conditional Import Setup ---
HAS_GPU_TTS = False # Using a different variable name to avoid conflict if both files are imported
try:
    import torch
    if torch.cuda.is_available():
        # Only try to import onnxruntime-gpu if CUDA is available
        try:
            import onnxruntime as rt
            if "CUDAExecutionProvider" in rt.get_available_providers():
                HAS_GPU_TTS = True
                logger.info("GPU (CUDA) detected and ONNX Runtime CUDA provider available. KokoroTTS will attempt to use it.")
            else:
                logger.info("CUDA GPU detected, but ONNX Runtime CUDA provider not available. KokoroTTS will use CPU.")
        except ImportError:
            logger.info("CUDA GPU detected, but onnxruntime-gpu not installed. KokoroTTS will use CPU.")
    else:
        logger.info("No CUDA GPU detected. KokoroTTS will use CPU.")
except ImportError:
    logger.info("PyTorch not found. KokoroTTS will use CPU.")


# Ensure kokoro_onnx is available
try:
    from kokoro_onnx import Kokoro
except ImportError:
    logger.error("The 'kokoro_onnx' library is not installed. Please install it as per the instructions in the original Gradio app setup.")
    class MockKokoro:
        def __init__(self, *args, **kwargs):
            logger.warning("Using MockKokoro as kokoro_onnx is not found.")
        def create(self, *args, **kwargs):
            logger.warning("MockKokoro.create called. No audio will be generated.")
            return np.array([]), 24000
    Kokoro = MockKokoro


class KokoroTTS:
    def __init__(self, model_path: str = "models/kokoro-v1.0.int8.onnx",
                 voices_path: str = "models/voices-v1.0.bin"):
        """Initialize Kokoro ONNX TTS model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"Voices file not found: {voices_path}")

        try:
            # We no longer pass 'providers' directly to Kokoro as it doesn't accept it.
            # Instead, we rely on kokoro_onnx automatically picking up onnxruntime-gpu
            # if it's installed and available in the environment.
            if HAS_GPU_TTS:
                 logger.info("KokoroTTS is expecting onnxruntime-gpu to be auto-detected by kokoro_onnx.")
            else:
                 logger.info("KokoroTTS will use CPU for ONNX Runtime.")

            # THIS IS THE CRUCIAL LINE: NO 'providers=providers' HERE
            self.kokoro = Kokoro(model_path, voices_path) 
            logger.info("Kokoro TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise

    def synthesize(self, text: str, voice_id: str = "af_heart", speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text"""
        try:
            logger.info(f"Synthesizing text: {text[:50]}...")
            samples, sample_rate = self.kokoro.create(text, voice=voice_id, speed=speed)
            logger.info(f"Generated audio shape: {samples.shape}, sample_rate: {sample_rate}")
            return samples.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([])

    def get_available_voices(self) -> list:
        return ["af_heart", "af_sky", "am_mystic"]

    def stream_synthesize(self, text: str, voice_id: str = "af_heart", chunk_size: int = 1024) -> Iterator[np.ndarray]:
        """Stream synthesize speech from text (example, actual streaming for Kokoro might be different)"""
        audio = self.synthesize(text, voice_id)
        if audio.size > 0:
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]