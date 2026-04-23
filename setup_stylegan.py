import os
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    print("--- GAP Path 1 Setup ---")
    
    # 1. Clone NVIDIA StyleGAN2-ADA PyTorch repository
    if not os.path.exists("stylegan2-ada-pytorch"):
        run_command("git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git")
    else:
        print("StyleGAN2 repo already exists.")

    # 2. Add repo to PYTHONPATH so pickle can find 'dnnlib'
    # We will do this in the app.py directly as well
    sys.path.append(os.path.abspath("stylegan2-ada-pytorch"))
    
    # 3. Install dependencies required for StyleGAN2
    print("Installing StyleGAN2 dependencies (ninja, click, requests, tqdm, pyspng)...")
    run_command("pip install ninja click requests tqdm pyspng")
    
    # 4. Download weights if not already present
    print("Running weight download script...")
    if os.path.exists("download_weights.py"):
        run_command("python download_weights.py")
    
    print("\n--- Setup Complete! ---")
    print("You can now run 'python app.py' to use the High-Quality Age Projection.")

if __name__ == "__main__":
    main()
