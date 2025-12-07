import runpod
import torch
import base64
import io
import os
import sys
import argparse
import tempfile
import numpy as np
from PIL import Image, ImageOps

# --- Dynamic Path Setup ---
# This ensures Python finds the 'inference' module and the 'sam3d_objects' package
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
    # 1. Determine the storage base path
    if os.path.exists("/runpod-volume"):
        print("DEBUG: Network Volume found. Using /runpod-volume as base.")
        base_storage = "/runpod-volume"
    else:
        print("DEBUG: No volume found. Using /workspace as base.")
        base_storage = "/workspace"

    # 2. Search for the config file in probable locations relative to the base
    # We look for 'pipeline.yaml' because that is the entry point for the model
    possible_paths = [
        # Path structure seen in your logs: BASE/models/sam-3d-objects/checkpoints
        os.path.join(base_storage, "models", "sam-3d-objects", "checkpoints", "pipeline.yaml"),
        # Alternative structure: BASE/sam-3d-objects/checkpoints
        os.path.join(base_storage, "sam-3d-objects", "checkpoints", "pipeline.yaml"),
        # Direct structure: BASE/checkpoints
        os.path.join(base_storage, "checkpoints", "pipeline.yaml")
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            print(f"DEBUG: Found config at: {config_path}")
            break
            
    if config_path is None:
        # Final fallback to environment variable or local relative path
        config_path = os.environ.get("SAM3D_CONFIG_PATH", "sam-3d-objects/checkpoints/pipeline.yaml")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Could not find pipeline.yaml in {base_storage} subdirectories.")

    inference_pipeline = Inference(config_path, compile=False)
    print("SAM 3D Pipeline loaded successfully.")

def decode_base64_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string))).convert("RGB")

# --- Handler ---

def handler(job):
    job_input = job.get("input", {})
    
    if "image" not in job_input or "mask" not in job_input:
        return {"error": "Input must contain 'image' and 'mask' base64 strings."}
        
    try:
        init_model()
        
        # 1. Decode inputs to PIL
        image_pil = decode_base64_image(job_input["image"])
        mask_pil = decode_base64_image(job_input["mask"]).convert("L")

        # 2. Apply EXIF Rotation
        image_pil = ImageOps.exif_transpose(image_pil)
        mask_pil = ImageOps.exif_transpose(mask_pil)

        # 3. Smart Rotation Check
        # If dimensions are swapped (e.g. Landscape vs Portrait), rotate mask to match
        if image_pil.size == (mask_pil.size[1], mask_pil.size[0]):
            print(f"Auto-rotating mask to match image orientation.")
            mask_pil = mask_pil.transpose(Image.ROTATE_90)
            if image_pil.size != mask_pil.size:
                 mask_pil = mask_pil.transpose(Image.ROTATE_180)

        # 4. Final Safety Resize
        # Forces the mask to match the image pixel-perfectly
        if image_pil.size != mask_pil.size:
            print(f"Resizing mask from {mask_pil.size} to {image_pil.size}")
            mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)

        # 5. Convert to NumPy Arrays
        image = np.array(image_pil)  # (H, W, 3)
        mask_np = np.array(mask_pil) # (H, W)
        
        # 6. Binarize and Ensure 2D
        mask = (mask_np > 128).astype(np.uint8)
        
        # If the mask accidentally has 3 dimensions (H, W, 1), flatten it to (H, W)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        seed = job_input.get("seed", 42)
        
        # Run Inference
        output = inference_pipeline(image, mask, seed=seed)
        
        # Export GLB to buffer
        if "glb" in output:
            mesh_obj = output["glb"]
            
            # Optional: Ensure vertex colors are not transparent
            if hasattr(mesh_obj.visual, 'vertex_colors') and len(mesh_obj.visual.vertex_colors) > 0:
                 if mesh_obj.visual.vertex_colors.shape[1] == 4:
                     mesh_obj.visual.vertex_colors[:, 3] = 255

            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                mesh_obj.export(tmp.name)
                tmp.name_to_read = tmp.name
                tmp.close()
                
                with open(tmp.name_to_read, "rb") as f:
                    glb_bytes = f.read()
                
                os.unlink(tmp.name_to_read)
                
            b64_glb = base64.b64encode(glb_bytes).decode("utf-8")
            return {"glb_file": b64_glb}
        else:
            return {"error": "Model failed to generate GLB output."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# --- Local Testing Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SAM 3D locally")
    parser.add_argument("--image", required=True, help="Input RGB image path")
    parser.add_argument("--mask", required=True, help="Input Mask image path")
    parser.add_argument("--output", default="output.glb", help="Output GLB path")
    
    args = parser.parse_args()
    
    print("--- Running SAM 3D Local Test ---")
    
    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open(args.mask, "rb") as f:
        mask_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    test_job = {
        "input": {
            "image": img_b64,
            "mask": mask_b64,
            "seed": 42
        }
    }
    
    init_model()
    result = handler(test_job)
    
    if "glb_file" in result:
        glb_data = base64.b64decode(result["glb_file"])
        with open(args.output, "wb") as f:
            f.write(glb_data)
        print(f"Success! 3D model saved to {args.output}")
    else:
        print(f"Failed: {result}")

else:
    runpod.serverless.start({"handler": handler})