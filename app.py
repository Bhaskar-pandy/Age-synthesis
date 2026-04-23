import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import warnings

# Suppress the MSVC/Compiler warnings from StyleGAN2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# This forces StyleGAN2 to use the slower but stable PyTorch fallback without screaming
warnings.filterwarnings("ignore", category=UserWarning, module="torch_utils.ops")

# Add StyleGAN2 repo to path
sys.path.append(os.path.abspath("stylegan2-ada-pytorch"))

# Silence the "Setting up PyTorch plugin ... Failed!" messages
try:
    import torch_utils.custom_ops as custom_ops
    custom_ops.verbosity = 'none'
except ImportError:
    pass

# --- LOCAL ENGINE (COMMENTED OUT) ---
# try:
#     from gap_engine import StyleGAN2AgingEngine
# except ImportError:
#     print("Warning: gap_engine not found. Using fallback.")
#
# STYLEGAN_WEIGHTS = "weights/stylegan2_ffhq.pkl"
# AGE_VECTOR = "weights/age_boundary.npy"
# GENDER_VECTOR = "weights/gender_boundary.npy"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# engine = None
# try:
#     if os.path.exists(STYLEGAN_WEIGHTS):
#         engine = StyleGAN2AgingEngine(STYLEGAN_WEIGHTS, AGE_VECTOR, GENDER_VECTOR, device=DEVICE)
#         engine.load_model()
#     else:
#         print("Model weights not found. Please run 'python setup_stylegan.py' first.")
# except Exception as e:
#     print(f"Error loading StyleGAN2 engine: {e}")

# --- API ENGINE (CLOUD) ---
from gradio_client import Client, handle_file
HF_API_URL = "Robys01/Face-Aging"
api_client = None

def get_client():
    global api_client
    if api_client is None:
        print(f"Connecting to Free API: {HF_API_URL}...")
        api_client = Client(HF_API_URL)
    return api_client

# State to hold the projected latent vector
current_latent_w = gr.State(None)

def analyze_image(image):
    # API handles inversion and aging in one call, so we just pass the image back to UI
    if image is None:
        return None, None
    return image, image

"""
# PRESERVED LOCAL LOGIC
def analyze_image_local(image):
    try:
        if image is None or engine is None:
            return None, None
        pil_img = Image.fromarray(image)
        w_opt = engine.project_image(pil_img, num_steps=10)
        reconstructed_img = engine.generate_at_age(w_opt, 0, 0)
        return w_opt, reconstructed_img
    except Exception as e:
        print(f"UI Error in analyze_image: {e}")
        return None, None
"""

def age_projection(img_path, current_age, target_age):
    if img_path is None:
        return None
    
    try:
        client = get_client()
        # API expects: (image_path, source_age, target_age)
        # Using POSITIONAL arguments to avoid keyword mismatch
        result = client.predict(
            handle_file(img_path), # image_path
            float(current_age),    # source_age
            float(target_age),     # target_age
            api_name="/predict"
        )
        # result is typically a string path or a dict with 'path'
        if isinstance(result, dict) and 'path' in result:
            return result['path']
        return result
    except Exception as e:
        print(f"API Error: {e}")
        return None

"""
# PRESERVED LOCAL LOGIC
def age_projection_local(latent_w, target_age):
    if engine is None:
        return None
    if latent_w is None:
        seed = 100
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, engine.G.z_dim)).to(DEVICE)
        with torch.no_grad():
            latent_w = engine.G.mapping(z, None)
    age_coeff = (target_age - 50.0) / 10.0
    projected_img = engine.generate_at_age(latent_w, age_coeff)
    return projected_img
"""

# UI Style Configuration
with open("style.css", "r") as f:
    custom_css = f.read()

# Gradio UI Setup
with gr.Blocks(title="GAP - Neural Human Synthesis") as app:
    # Header Section
    with gr.Column(elem_id="header-container"):
        gr.HTML("""
            <div class='status-badge'>Neural Core Process Active</div>
            <h1 id='header-logo'>GAP AI</h1>
            <h2 id='header-subtitle'>PROPRIETARY HUMAN AGE SYNTHESIS</h2>
        """)
    
    with gr.Row():
        with gr.Column(elem_classes="glass-card"):
            input_img = gr.Image(label="Source Portrait", type="filepath", elem_classes="input-image")
            gr.HTML("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>")
            with gr.Column():
                curr_age_bar = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Input Age Reference", interactive=True)
                target_age_bar = gr.Slider(minimum=10, maximum=95, step=1, value=60, label="Synthesis Target", interactive=True)
            
            run_btn = gr.Button("⚡ Run Neural Synthesis", variant="primary", elem_id="run-btn")
            
        with gr.Column(elem_classes="glass-card"):
            output_img = gr.Image(label="Synthesis Output", elem_classes="output-image")
            gr.HTML("""
                <hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 20px 0;'>
                <p class='status-text'><strong>Neural State</strong>: Identity architecture locked. Select the chronological epoch and click run to begin remapping.</p>
            """)
            
    # Trigger on button click
    run_btn.click(
        fn=age_projection, 
        inputs=[input_img, curr_age_bar, target_age_bar], 
        outputs=output_img
    )
    
    # Custom HTML Footer
    gr.HTML("""
    <div id='custom-footer'>
        <p>GAP AI ENGINE v2.5 | Neural Synthesis Core | Proprietary Technology</p>
    </div>
    """)

if __name__ == "__main__":
    # Enable Queuing to prevent timeouts during long analysis
    app.queue()
    # In Gradio 6+, css and theme should be passed to launch()
    app.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        theme=gr.themes.Default(),
        css=custom_css,
        max_threads=20
    )
