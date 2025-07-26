# fastwhisper_stt.py
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger(__name__)

# --- GPU Detection and Conditional Import Setup ---
HAS_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        HAS_GPU = True
        logger.info("GPU (CUDA) detected. Faster-Whisper will attempt to use it.")
    else:
        logger.info("No CUDA GPU detected. Faster-Whisper will use CPU.")
except ImportError:
    logger.info("PyTorch not found or no CUDA GPU detected. Faster-Whisper will use CPU.")


class FastWhisperSTT:
    def __init__(self):
        if HAS_GPU:
            device = "cuda"
            compute_type = "float16" # Generally recommended for GPU for speed and memory efficiency
        else:
            device = "cpu"
            compute_type = "int8" # Optimized for CPU

        logger.info(f"Initializing Faster-Whisper model from 'models/faster-whisper-tiny.en' with device: {device}, compute_type: {compute_type}")
        self.model = WhisperModel(
            model_size_or_path="models/faster-whisper-tiny.en",
            device=device,
            compute_type=compute_type,
            local_files_only=True
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments)