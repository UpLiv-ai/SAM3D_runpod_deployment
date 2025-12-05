import runpod
import torch
import base64
import io
import os
import sys
import argparse
import tempfile
import numpy as np
from PIL import Image

# Setup path to import from the repo
sys.path.append(os.path.join(os.getcwd(), "notebook"))
try:
    from inference import Inference
except ImportError:
    # Fallback if running inside the repo folder directly
    sys.path.append("/app/sam-3d-objects/notebook")
    from inference import Inference

# --- Global Initialization ---
inference_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.environ.get("SAM3D_CHECKPOINT_DIR", "/runpod-volume/sam3d/checkpoints/hf")

def init_model():
    global inference_pipeline
    if inference_pipeline is not None:
        return
        
    print("Initializing SAM 3D Pipeline...")
    config_path = os.path.join(CHECKPOINT_DIR, "pipeline.yaml")
    
    # Verify config exists
    if not os.path.exists(config_path):
        # Fallback for local testing if not using volumes
        if os.path.exists("checkpoints/hf/pipeline.yaml"):
            config_path = "checkpoints/hf/pipeline.yaml"
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")

    inference_pipeline = Inference(config_path, compile=False)
    print("SAM 3D Pipeline loaded.")

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
        
        image = decode_base64_image(job_input["image"])
        # Mask needs to be L mode (grayscale)
        mask = decode_base64_image(job_input["mask"]).convert("L")
        seed = job_input.get("seed", 42)
        
        # Run Inference
        output = inference_pipeline(image, mask, seed=seed)
        
        # Export GLB to buffer
        if "glb" in output:
            mesh_obj = output["glb"]
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                mesh_obj.export(tmp.name)
                tmp.close()
                
                with open(tmp.name, "rb") as f:
                    glb_bytes = f.read()
                
                os.unlink(tmp.name)
                
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
    
    # Load and encode inputs
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
    
    # Initialize logic manually
    # Note: Ensure you are in the repo root or have paths set up correctly for this to run
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