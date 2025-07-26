# 🎙️ Streamlit Voice Assistant

This is a voice-enabled assistant built with Streamlit, Faster-Whisper for Speech-to-Text (STT), Kokoro-ONNX for Text-to-Speech (TTS), and an LLM (Gemma via OpenRouter) for conversational AI. The project is designed to automatically optimize for GPU usage if a compatible NVIDIA GPU is detected, while also providing a seamless experience for CPU-only users.

## Features

* **🎙️ Voice Input:** Record your message directly in the browser using Streamlit's audio input.
* **📝 Speech-to-Text (STT):** Powered by `faster-whisper` for efficient and accurate transcription of your voice messages.
* **💬 Conversational AI:** Integrates with a Large Language Model (Gemma via OpenRouter) for natural and engaging responses.
* **🗣️ Text-to-Speech (TTS):** Uses `kokoro-onnx` to synthesize AI responses into natural-sounding speech.
* **🚀 Automatic Optimization:** Detects NVIDIA GPUs (CUDA) and configures STT and TTS models for accelerated performance. Falls back gracefully to CPU if no GPU is found or configured.
* **📜 Chat History:** Keeps track of your conversation with the AI.

## Demo

(You can add a GIF or screenshot of the app running here if you wish)

## Setup and Installation

### Prerequisites

* Python 3.9+
* **For GPU Acceleration (NVIDIA CUDA):**
    * An NVIDIA GPU with CUDA compute capability.
    * NVIDIA CUDA Toolkit (matching your PyTorch CUDA version, e.g., CUDA 12.1 for `cu121`).
    * cuDNN library.
    * **Crucially, ensure these are installed on your system *before* installing the Python packages.** Refer to NVIDIA's official documentation for detailed installation guides for your operating system.

### Installation Steps

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/streamlitVoiceChat.git](https://github.com/your-username/streamlitVoiceChat.git)
    cd streamlitVoiceChat
    ```

2.  **Choose your installation path (CPU-only or GPU-enabled):**

    #### A. For CPU-Only Users (No NVIDIA GPU)

    This is the simplest installation and will use your CPU for all computations.

    ```bash
    pip install -r requirements-cpu.txt
    ```

    #### B. For GPU-Enabled Users (With NVIDIA CUDA GPU)

    This path will install CUDA-accelerated versions of the libraries for maximum performance.

    ```bash
    # IMPORTANT: Ensure NVIDIA CUDA Toolkit and cuDNN are installed on your system first!

    # 1. Install PyTorch with CUDA support.
    #    Replace 'cu121' with the CUDA version corresponding to your system's CUDA Toolkit.
    #    You can find the correct command for your specific setup here: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # 2. Install the rest of the GPU-optimized Python packages
    pip install -r requirements-gpu.txt
    ```

3.  **Download Models:**
    The `models/` directory should contain:
    * `faster-whisper-tiny.en` (or other Faster-Whisper model sizes)
    * `kokoro-v1.0.int8.onnx`
    * `voices-v1.0.bin`

    If these are not present, you'll need to download them. For `faster-whisper` models, they are typically downloaded automatically on first run if not found, but you can also manually place them. For Kokoro, you might need to obtain them from the original Kokoro project source.

4.  **Set up OpenRouter API Key:**
    The LLM (`google/gemma`) is accessed via OpenRouter. You need to provide your OpenRouter API key. Create a `.streamlit/secrets.toml` file in your project root with the following content:

    ```toml
    # .streamlit/secrets.toml
    OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
    ```
    Replace `"YOUR_OPENROUTER_API_KEY"` with your actual API key from [OpenRouter.ai](https://openrouter.ai/).

## Running the Application

Once all dependencies are installed and the API key is set, run the Streamlit application:

```bash
streamlit run app.py