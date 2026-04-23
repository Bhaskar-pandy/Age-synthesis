import torch
import numpy as np
import pickle
import sys
import os

# We will need the StyleGAN2 repository code to load the .pkl files.
# Instead of cloning, we'll try to use a more direct method if possible, 
# or advise the user to run a setup script.

class StyleGAN2AgingEngine:
    def __init__(self, stylegan_path, age_vector_path, gender_vector_path, device="cuda"):
        self.device = torch.device(device)
        self.stylegan_path = stylegan_path
        self.age_vector_path = age_vector_path
        self.gender_vector_path = gender_vector_path
        self.G = None
        self.age_direction = None
        self.gender_direction = None
        
    def load_model(self):
        print(f"Loading StyleGAN2 from {self.stylegan_path}...")
        
        # StyleGAN2-ADA weights are stored in pickle files with complex object structures.
        # To load them, we typically need the original repository in the PYTHONPATH.
        # We will add a check for the user.
        
        if not os.path.exists(self.stylegan_path):
            raise FileNotFoundError("StyleGAN2 weights not found. Please run download_weights.py first.")
            
        with open(self.stylegan_path, 'rb') as f:
            # Note: This requires the 'dnnlib' and 'torch_utils' from the StyleGAN2 repo
            # We will handle the setup in a separate step or provided as a utility.
            self.G = pickle.load(f)['G_ema'].to(self.device)
            
        print("Loading Age Boundary vector...")
        self.age_direction = np.load(self.age_vector_path)
        self.age_direction = torch.from_numpy(self.age_direction).to(self.device).float()

        print("Loading Gender Boundary vector...")
        if os.path.exists(self.gender_vector_path):
            self.gender_direction = np.load(self.gender_vector_path)
            self.gender_direction = torch.from_numpy(self.gender_direction).to(self.device).float()

    def generate_at_age(self, latent_w, age_coeff, gender_coeff=0.0):
        """
        latent_w: The W space latent (shape: 1, 18, 512)
        age_coeff: Higher = Older
        gender_coeff: Shifting between Male/Female
        """
        w_aged = latent_w.clone()
        
        # Apply Age manipulation
        w_aged += age_coeff * self.age_direction.view(1, 1, 512)
        
        # Apply Gender correction
        if self.gender_direction is not None:
             w_aged += gender_coeff * self.gender_direction.view(1, 1, 512)
        
        # Generate the image
        with torch.no_grad():
            img = self.G.synthesis(w_aged, noise_mode='const')
            
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img.cpu().numpy()[0]

    def project_image(self, target_img_pil, num_steps=100):
        """
        Inverts a real image into StyleGAN2 latent space (W+).
        This takes ~15-30 seconds on a GTX 1650.
        """
        import torch.nn.functional as F
        
        # Preprocess image
        target_img = np.array(target_img_pil.convert('RGB'))
        target_img = torch.from_numpy(target_img).permute(2, 0, 1).unsqueeze(0).to(self.device).to(torch.float32)
        target_img = (target_img / 127.5 - 1.0)
        target_img = sys.modules['torch.nn.functional'].interpolate(target_img, size=(self.G.img_resolution, self.G.img_resolution), mode='area')

        # Clear memory before heavy lifting
        torch.cuda.empty_cache()

        # Find initial W (average W)
        w_avg = self.G.mapping.w_avg
        w_pivot = w_avg.clone().detach().unsqueeze(0).repeat(1, self.G.mapping.num_ws, 1)
        w_opt = w_pivot.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([w_opt], lr=0.1)
        
        print(f"Starting Inversion (Identity Lock)...")
        try:
            for step in range(num_steps):
                optimizer.zero_grad()
                synth_img = self.G.synthesis(w_opt, noise_mode='const')
                loss = sys.modules['torch.nn.functional'].mse_loss(synth_img, target_img)
                loss.backward()
                optimizer.step()
                if step % 2 == 0:
                    print(f"  Step {step}/{num_steps}, Loss: {loss.item():.4f}")
            print("Inversion Complete.")
        except Exception as e:
            print(f"Critical Error during inversion: {e}")
            import traceback
            traceback.print_exc()
            return w_pivot.detach() # Return average face as fallback
        finally:
            torch.cuda.empty_cache()
                
        return w_opt.detach()

# Instructions for the USER:
# To run this, we need the StyleGAN2-ADA repo.
# I will create a setup_path1.py script to handle this automatically.
