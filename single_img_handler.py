import runpod
import torch
import io
import os
import sys
import argparse
import tempfile
import numpy as np
import requests
from PIL import Image, ImageOps

# --- Dynamic Path Setup ---
current_dir = os.getcwd()
repo_root = os.path.join(current_dir, "sam-3d-objects")
notebook_path = os.path.join(repo_root, "notebook")

if os.path.exists(repo_root):
    sys.path.append(repo_root)
    sys.path.append(notebook_path)
else:
    # Fallback for production Docker paths
    sys.path.append("/app/sam-3d-objects")
    sys.path.append("/app/sam-3d-objects/notebook")

try:
    from inference import Inference
except ImportError:
    print(f"WARNING: Could not import Inference. Sys.path is: {sys.path}")

# --- Global Initialization ---
inference_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_model():
    global inference_pipeline
    if inference_pipeline is not None:
        return
        
    print("Initializing SAM 3D Pipeline...")

    # --- DYNAMIC VOLUME CHECK ---
    if os.path.exists("/runpod-volume"):
        base_storage = "/runpod-volume"
    else:
        base_storage = "/workspace"

    # Search for pipeline.yaml
    possible_paths = [
        os.path.join(base_storage, "models", "sam-3d-objects", "checkpoints", "pipeline.yaml"),
        os.path.join(base_storage, "sam-3d-objects", "checkpoints", "pipeline.yaml"),
        os.path.join(base_storage, "checkpoints", "pipeline.yaml")
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
            
    if config_path is None:
        config_path = os.environ.get("SAM3D_CONFIG_PATH", "sam-3d-objects/checkpoints/pipeline.yaml")
        if not os.path.exists(config_path):
             # Fallback for local testing if not in standard structure
             if os.path.exists("sam-3d-objects/checkpoints/pipeline.yaml"):
                 config_path = "sam-3d-objects/checkpoints/pipeline.yaml"
             else:
                 raise FileNotFoundError(f"Could not find pipeline.yaml in {base_storage} or local dirs.")

    inference_pipeline = Inference(config_path, compile=False)
    print("SAM 3D Pipeline loaded successfully.")

# --- Shared Logic ---

def preprocess_inputs(image_pil, mask_pil):
    """
    Centralized logic for rotation, resizing, and numpy conversion.
    """
    # 1. Apply EXIF Rotation
    image_pil = ImageOps.exif_transpose(image_pil)
    mask_pil = ImageOps.exif_transpose(mask_pil)

    # 2. Smart Rotation Check
    # FIX: First check if they ALREADY match (Square Image Fix). 
    # If they match exactly, skip the rotation logic entirely.
    if image_pil.size != mask_pil.size:
        
        # Only check for swapped dimensions if they don't already match
        if image_pil.size == (mask_pil.size[1], mask_pil.size[0]):
            print(f"Auto-rotating mask to match image orientation.")
            mask_pil = mask_pil.transpose(Image.ROTATE_90)
            
            # If +90 didn't fix it (it's still not matching), try flip 180
            if image_pil.size != mask_pil.size:
                 mask_pil = mask_pil.transpose(Image.ROTATE_180)

    # 3. Final Safety Resize
    if image_pil.size != mask_pil.size:
        print(f"Resizing mask from {mask_pil.size} to {image_pil.size}")
        mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)

    # 4. Convert to NumPy Arrays
    image = np.array(image_pil)  # (H, W, 3)
    mask_np = np.array(mask_pil) # (H, W)
    
    # 5. Binarize and Ensure 2D
    mask = (mask_np > 128).astype(np.uint8)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
        
    return image, mask

def process_glb_output(output):
    """Helper to process the GLB mesh (color fix) and return bytes."""
    if "glb" in output:
        mesh_obj = output["glb"]
        
        # Color Fix: Ensure vertex colors are opaque
        if hasattr(mesh_obj.visual, 'vertex_colors') and len(mesh_obj.visual.vertex_colors) > 0:
                if mesh_obj.visual.vertex_colors.shape[1] == 4:
                    mesh_obj.visual.vertex_colors[:, 3] = 255

        # Export to Bytes
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            mesh_obj.export(tmp.name)
            tmp.name_to_read = tmp.name
            tmp.close()
            
            with open(tmp.name_to_read, "rb") as f:
                glb_bytes = f.read()
            
            os.unlink(tmp.name_to_read)
        return glb_bytes
    return None

def download_image_from_url(url):
    headers = {"User-Agent": "RunPod-Worker/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to download image from {url}: {str(e)}")

def upload_to_url(data_bytes, url):
    headers = {
        'x-ms-blob-type': 'BlockBlob',
        'Content-Type': 'model/gltf-binary'
    }
    try:
        print(f"Uploading {len(data_bytes)} bytes to Azure...")
        response = requests.put(url, data=data_bytes, headers=headers, timeout=60)
        response.raise_for_status()
        print("Upload successful.")
    except Exception as e:
        raise RuntimeError(f"Failed to upload result to output_location: {str(e)}")

# --- Main Handler (Production) ---

def handler(job):
    job_input = job.get("input", {})
    
    image_url = job_input.get("image_url")
    mask_url = job_input.get("mask_url")
    output_location = job_input.get("output_location")
    seed = job_input.get("seed", 42)

    if not image_url or not mask_url or not output_location:
        return {"status": "failed", "error": "Missing image_url, mask_url, or output_location"}

    try:
        init_model()
        
        # Download
        print(f"Downloading inputs...")
        image_pil = download_image_from_url(image_url)
        mask_pil = download_image_from_url(mask_url).convert("L")

        # Preprocess (Rotation/Resize)
        image_np, mask_np = preprocess_inputs(image_pil, mask_pil)

        # Inference
        output = inference_pipeline(image_np, mask_np, seed=seed)
        
        # Post-Process & Upload
        glb_bytes = process_glb_output(output)
        
        if glb_bytes:
            upload_to_url(glb_bytes, output_location)
            return {"status": "success", "error": None}
        else:
            return {"status": "failed", "error": "Model failed to generate GLB output."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": handler})