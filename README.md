# TTS — Local Text-to-Speech Service

Self-hosted text-to-speech stack using **Kokoro ONNX**, with a Python server, Docker support, and voice model downloads (including Google Drive / Hugging Face helpers).

## Features
- Local ONNX TTS inference (`kokoro-onnx`)
- Python API / generation scripts
- Docker deployment
- PDF / text input helpers (`pypdf`, loudness normalization)

## Quick start
```bash
pip install -r requirements.txt
python download_model.py
python server.py
```

Or use the included `Dockerfile`.
