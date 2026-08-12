import os
import sys

# Crucial: Add espeak-ng path to Windows environment variable so phonemizer can find it
ESPEAK_PATH = r"C:\Program Files\eSpeak NG"
if os.path.exists(ESPEAK_PATH):
    # Add to beginning of PATH
    os.environ["PATH"] = ESPEAK_PATH + ";" + os.environ.get("PATH", "")
    print("Configured espeak-ng path successfully.")
else:
    print("Warning: eSpeak NG not found at default path. Phonemization may fail.")

try:
    from kokoro_onnx import Kokoro
    import soundfile as sf
    print("Successfully imported kokoro-onnx and soundfile.")
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def main():
    onnx_file = "kokoro-v1.0.onnx"
    voices_file = "voices-v1.0.bin"
    
    if not os.path.exists(onnx_file) or not os.path.exists(voices_file):
        print("Error: Model weights not found in current directory. Please run download_model.py first.")
        sys.exit(1)
        
    print("Loading model (this might take a moment)...")
    kokoro = Kokoro(onnx_file, voices_file)
    
    text = "Hello! Kokoro text to speech is running successfully in your local Python environment."
    voice = "af_bella"
    output_file = "output.wav"
    
    print(f"Generating audio for text: '{text}' using voice: '{voice}'...")
    try:
        samples, sample_rate = kokoro.create(
            text, 
            voice=voice, 
            speed=1.0, 
            lang="en-us"
        )
        
        print(f"Saving generated audio to {output_file}...")
        sf.write(output_file, samples, sample_rate)
        print("Audio saved successfully to output.wav!")
        
    except Exception as e:
        print(f"Failed to generate audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
