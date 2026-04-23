import torch
import numpy as np
import os
import sys
import warnings
from PIL import Image

# 1. Setup paths
sys.path.append(os.path.abspath("stylegan2-ada-pytorch"))
from gap_engine import StyleGAN2AgingEngine

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Config
WEIGHTS = "weights/stylegan2_ffhq.pkl"
AGE_V = "weights/age_boundary.npy"
GENDER_V = "weights/gender_boundary.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_model():
    print(f"--- Diagnostic Check: StyleGAN2 Model ---")
    if not os.path.exists(WEIGHTS):
        print("FAIL: Model weights missing.")
        return

    try:
        engine = StyleGAN2AgingEngine(WEIGHTS, AGE_V, GENDER_V, device=DEVICE)
        engine.load_model()
        print("SUCCESS: Model loaded into memory.")
        
        # Test 1: Generate 3 random faces (Identity check)
        print("Testing Generator (3 seeds)...")
        results = []
        for seed in [100, 200, 300]:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, engine.G.z_dim)).to(DEVICE)
            with torch.no_grad():
                w = engine.G.mapping(z, None)
                img = engine.generate_at_age(w, 0, 0) # Neutral age/gender
                results.append(img)
                print(f"  - Seed {seed} generated successfully.")

        # Save a grid to verify visually
        final_img = Image.fromarray(np.concatenate(results, axis=1))
        final_img.save("test_generator_output.png")
        print("SUCCESS: Diagnostic image saved to 'test_generator_output.png'.")
        
        # Test 2: Verify Latent Math (Age Vector check)
        print("Testing Latent Math (Age shift)...")
        z = torch.from_numpy(np.random.RandomState(100).randn(1, engine.G.z_dim)).to(DEVICE)
        with torch.no_grad():
            w = engine.G.mapping(z, None)
            # Generate young and old
            img_young = engine.generate_at_age(w, -5.0)
            img_old = engine.generate_at_age(w, 5.0)
            
            # Check if they are different
            if np.array_equal(img_young, img_old):
                 print("FAIL: Latent age vector has NO effect.")
            else:
                 print("SUCCESS: Latent age vector shifts the image.")
                 Image.fromarray(np.concatenate([img_young, results[0], img_old], axis=1)).save("test_age_shift.png")

    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model()
