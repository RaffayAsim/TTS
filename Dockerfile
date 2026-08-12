FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY index.html .
COPY kokoro-v1.0.onnx .
COPY voices-v1.0.bin .

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860
ENV PORT=7860

# Run the FastAPI server
CMD ["python", "server.py"]
