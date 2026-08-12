import os
import urllib.request

MODELS = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
}

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    
    # Custom opener to handle redirects if any
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    # Progress callback
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent % 10 == 0:
            print(f"Progress: {percent}% ({count * block_size}/{total_size} bytes)", end="\r")
            
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\nFinished downloading {dest}.")

if __name__ == "__main__":
    for name, url in MODELS.items():
        if not os.path.exists(name):
            download_file(url, name)
        else:
            print(f"{name} already exists. Skipping download.")
