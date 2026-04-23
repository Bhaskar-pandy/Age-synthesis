import os
import urllib.request
import tqdm

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    
    response = urllib.request.urlopen(url)
    total_size = int(response.getheader('Content-Length').strip())
    block_size = 1024 * 8
    
    with open(filename, 'wb') as f:
        # Use tqdm.tqdm to be explicit and avoid 'module' vs 'class' ambiguity
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

def main():
    # Directory to store weights
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    # Pre-trained weights for StyleGAN2-ADA (NVIDIA)
    # Note: These are large (~360MB)
    stylegan_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    stylegan_path = os.path.join(weights_dir, "stylegan2_ffhq.pkl")
    
    # Age direction vector (InterFaceGAN style)
    # Using the verified working URL from the official InterFaceGAN repo
    # Age direction vector (InterFaceGAN style)
    age_vector_url = "https://github.com/genforce/interfacegan/raw/master/boundaries/stylegan_ffhq_age_boundary.npy"
    age_vector_path = os.path.join(weights_dir, "age_boundary.npy")

    # Gender orientation vector (to fix swaps)
    gender_vector_url = "https://github.com/genforce/interfacegan/raw/master/boundaries/stylegan_ffhq_gender_boundary.npy"
    gender_vector_path = os.path.join(weights_dir, "gender_boundary.npy")
    
    if not os.path.exists(stylegan_path):
        download_file(stylegan_url, stylegan_path)
    else:
        print(f"{stylegan_path} already exists.")
        
    if not os.path.exists(age_vector_path):
        download_file(age_vector_url, age_vector_path)
    else:
        print(f"{age_vector_path} already exists.")

    if not os.path.exists(gender_vector_path):
        download_file(gender_vector_url, gender_vector_path)
    else:
        print(f"{gender_vector_path} already exists.")

    print("\nAll weights downloaded successfully! You can now run Path 1.")

if __name__ == "__main__":
    # Ensure tqdm is installed for progress bar
    try:
        import tqdm
    except ImportError:
        print("Installing tqdm for download progress bars...")
        os.system("pip install tqdm")
    
    main()
