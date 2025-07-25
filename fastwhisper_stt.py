# fastwhisper_stt.py
import logging
import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class FasterWhisperSTT:
    def __init__(self, model_size: str = "base", device: str = None, compute_type: str = None):
        """
        Initializes the Faster-Whisper Speech-to-Text model.
        Args:
            model_size (str): The size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
            device (str): Device to use for inference ("cpu" or "cuda"). Defaults to "cuda" if available, else "cpu".
            compute_type (str): Compute type ("int8", "float16", "float32"). Defaults to "float16" for CUDA, "int8" for CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8" # int8 is usually faster on CPU

        logger.info(f"Loading Whisper model '{model_size}' on device: {device}, compute type: {compute_type}")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("Faster-Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}", exc_info=True)
            raise

    def transcribe(self, audio_array: np.ndarray) -> str:
        """
        Transcribes an audio NumPy array to text.
        Args:
            audio_array (np.ndarray): A 1D NumPy array of audio samples (float32, 16kHz).
        Returns:
            str: The transcribed text. Returns an empty string if no audio or transcription fails.
        """
        if audio_array.size == 0:
            logger.warning("Attempted to transcribe an empty audio array.")
            return ""
        
        try:
            segments, info = self.model.transcribe(audio_array, beam_size=5) 
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            logger.info(f"Transcribed audio to: '{transcribed_text[:70]}...'")
            return transcribed_text
        except Exception as e:
            logger.error(f"Error during transcription: {e}", exc_info=True)
            return ""