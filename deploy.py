import os
import sys
from huggingface_hub import HfApi

def deploy():
    print("=== Hugging Face Spaces Deployer ===")
    
    # Read token
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face WRITE Token: ").strip()
        if not token:
            print("Error: Hugging Face Token is required.")
            sys.exit(1)
            
    api = HfApi(token=token)
    
    try:
        user = api.whoami()
        username = user['name']
        print(f"Authenticated as: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)
        
    repo_name = os.environ.get("HF_REPO_NAME")
    if not repo_name:
        repo_name = input("Enter Space name (default: narrator-studio): ").strip() or "narrator-studio"
    repo_id = f"{username}/{repo_name}"
    
    print(f"Creating Space repository: {repo_id}...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=True, # Private by default to protect credentials and drive folder config
            exist_ok=True
        )
        print("Space repository verified/created.")
    except Exception as e:
        print(f"Error creating space repo: {e}")
        sys.exit(1)
        
    files_to_upload = [
        "server.py",
        "index.html",
        "requirements.txt",
        "Dockerfile",
        "kokoro-v1.0.onnx",
        "voices-v1.0.bin"
    ]
    
    # Filter files that exist
    files_to_upload = [f for f in files_to_upload if os.path.exists(f)]
    
    print("\nUploading project files to Hugging Face Space (this includes model files, please wait)...")
    for f in files_to_upload:
        print(f"Uploading {f}...")
        try:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=repo_id,
                repo_type="space"
            )
            print(f"SUCCESS: Uploaded {f} successfully.")
        except Exception as e:
            print(f"ERROR: Failed to upload {f}: {e}")
            
    print(f"\n=== Deployment Completed successfully! ===")
    print(f"Your space is building at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy()
