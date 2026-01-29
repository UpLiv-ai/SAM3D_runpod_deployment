import runpod
import torch
import io
import os
import sys
import tempfile
import numpy as np
import requests
from PIL import Image, ImageOps

# --- Dynamic Path Setup ---
current_dir = os.getcwd()

# 1. Setup MV-SAM3D Path
mv_sam3d_root = os.path.join(current_dir, "MV-SAM3D")
notebook_path = os.path.join(mv_sam3d_root, "notebook")

if os.path.exists(mv_sam3d_root):
    sys.path.append(mv_sam3d_root)
    sys.path.append(notebook_path)

# 2. Setup Depth Anything 3 Path
da3_root = os.path.join(current_dir, "Depth-Anything-3")
if os.path.exists(da3_root):
    sys.path.append(da3_root)
    if os.path.exists(os.path.join(da3_root, "src")):
        sys.path.append(os.path.join(da3_root, "src"))

# --- Import Models ---
try:
    from inference import Inference  # MV-SAM3D
except ImportError:
    print(f"WARNING: Could not import Inference. Sys.path: {sys.path}")

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("WARNING: Could not import DepthAnything3.")

# --- Global Initialization ---
sam3d_pipeline = None
da3_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_models():
    global sam3d_pipeline, da3_model
    
    # --- 1. Init Depth Anything 3 ---
    if da3_model is None:
        print("Initializing Depth Anything 3...")
        
        possible_da3_paths = [
            "/runpod-volume/models/depth-anything-3",
            "/workspace/models/depth-anything-3"
        ]
        
        weights_path = "depth-anything/DA3NESTED-GIANT-LARGE" # Fallback
        
        for path in possible_da3_paths:
            if os.path.exists(path) and os.path.isdir(path):
                print(f"Found local DA3 weights at: {path}")
                weights_path = path
                break
        
        try:
            da3_model = DepthAnything3.from_pretrained(weights_path).to(device)
            print("Depth Anything 3 loaded successfully.")
        except Exception as e:
            print(f"Error loading DA3 from {weights_path}: {e}")
            raise e

    # --- 2. Init MV-SAM3D ---
    if sam3d_pipeline is None:
        print("Initializing MV-SAM3D Pipeline...")
        
        if os.path.exists("/runpod-volume"):
            base_storage = "/runpod-volume"
        else:
            base_storage = "/workspace"

        possible_paths = [
            os.path.join(base_storage, "models", "MV-SAM3D", "checkpoints", "pipeline.yaml"),
            os.path.join(base_storage, "MV-SAM3D", "checkpoints", "pipeline.yaml"),
            os.path.join(mv_sam3d_root, "checkpoints", "pipeline.yaml")
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
             config_path = os.path.join(mv_sam3d_root, "checkpoints", "pipeline.yaml")

        if os.path.exists(config_path):
            sam3d_pipeline = Inference(config_path, compile=False)
            print(f"MV-SAM3D Pipeline loaded with config: {config_path}")
        else:
             raise FileNotFoundError(f"Could not find pipeline.yaml. Checked: {possible_paths}")

def download_image(url):
    """Downloads a single image from a URL."""
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to download image from {url}: {e}")

def preprocess_single_pair(image_pil, mask_pil):
    """Aligns a single mask to its corresponding image."""
    image_pil = ImageOps.exif_transpose(image_pil)
    mask_pil = ImageOps.exif_transpose(mask_pil)

    # Rotation check
    if image_pil.size != mask_pil.size:
        if image_pil.size == (mask_pil.size[1], mask_pil.size[0]):
            mask_pil = mask_pil.transpose(Image.ROTATE_90)
            if image_pil.size != mask_pil.size:
                 mask_pil = mask_pil.transpose(Image.ROTATE_180)

    # Resize check
    if image_pil.size != mask_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, resample=Image.NEAREST)

    image_np = np.array(image_pil)
    mask_np = np.array(mask_pil)
    
    # Binarize Mask
    mask = (mask_np > 128).astype(np.uint8)
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
        
    return image_np, mask

def process_batch_inputs(image_urls, mask_urls):
    """Downloads and processes lists of images and masks."""
    if len(image_urls) != len(mask_urls):
        raise ValueError(f"Mismatch: {len(image_urls)} images vs {len(mask_urls)} masks provided.")

    images_np_list = []
    masks_np_list = []

    print(f"Processing {len(image_urls)} views...")
    
    for img_url, msk_url in zip(image_urls, mask_urls):
        img_pil = download_image(img_url)
        msk_pil = download_image(msk_url).convert("L") # Ensure mask is grayscale
        
        img_np, msk_np = preprocess_single_pair(img_pil, msk_pil)
        
        images_np_list.append(img_np)
        masks_np_list.append(msk_np)

    return images_np_list, masks_np_list

def process_glb_output(output):
    """Extracts GLB from pipeline output."""
    mesh_obj = output.get("glb")
    
    if mesh_obj:
        # Vertex Color Fix
        if hasattr(mesh_obj.visual, 'vertex_colors') and len(mesh_obj.visual.vertex_colors) > 0:
                if mesh_obj.visual.vertex_colors.shape[1] == 4:
                    mesh_obj.visual.vertex_colors[:, 3] = 255

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            mesh_obj.export(tmp.name)
            tmp_name = tmp.name
            
        with open(tmp_name, "rb") as f:
            glb_bytes = f.read()
        os.unlink(tmp_name)
        return glb_bytes
    return None

def upload_to_url(data, url):
    if url.startswith("http"):
        headers = {'x-ms-blob-type': 'BlockBlob', 'Content-Type': 'model/gltf-binary'}
        requests.put(url, data=data, headers=headers).raise_for_status()
    else:
        # Local file save
        print(f"Saving output locally to: {url}")
        with open(url, "wb") as f:
            f.write(data)

# --- Main Handler ---

def handler(job):
    job_input = job.get("input", {})
    
    # 1. Parse Inputs (Handle both list and string for backward compatibility if needed)
    image_input = job_input.get("image_urls") or job_input.get("image_url")
    mask_input = job_input.get("mask_urls") or job_input.get("mask_url")
    output_location = job_input.get("output_location")
    seed = job_input.get("seed", 42)

    # Normalize to lists
    if isinstance(image_input, str): image_input = [image_input]
    if isinstance(mask_input, str): mask_input = [mask_input]

    if not image_input or not mask_input or not output_location:
        return {"status": "failed", "error": "Missing image_urls, mask_urls, or output_location"}

    try:
        init_models()
        
        # 2. Batch Preprocessing
        print("Downloading and preprocessing input batch...")
        # Returns List[np.ndarray]
        images_list, masks_list = process_batch_inputs(image_input, mask_input)

        # 3. Multi-View Depth & Pose Estimation (DA3)
        print(f"Running Depth Anything 3 on {len(images_list)} views...")
        
        # DA3's inference typically accepts a list of numpy images for multi-view contexts
        da3_output = da3_model.inference(image=images_list, export_format="mini_npz")
        
        # Extract Extrinsics/Intrinsics (Expecting arrays corresponding to N views)
        if isinstance(da3_output, dict):
            extrinsics = da3_output.get("extrinsics")
            intrinsics = da3_output.get("intrinsics")
        else:
            extrinsics = getattr(da3_output, "extrinsics", None)
            intrinsics = getattr(da3_output, "intrinsics", None)

        if extrinsics is None:
            raise RuntimeError("Depth Anything 3 failed to generate camera poses.")

        print(f"Poses obtained. Shape: {extrinsics.shape}")

        # 4. MV-SAM3D Generation
        print("Running MV-SAM3D with multi-view context...")
        
        # We pass the full lists. MV-SAM3D pipeline handles List[np.ndarray]
        output = sam3d_pipeline(
            images_list, 
            masks_list, 
            seed=seed,
            camera_poses=extrinsics, 
            intrinsics=intrinsics
        )
        
        # 5. Save/Upload
        glb_bytes = process_glb_output(output)
        
        if glb_bytes:
            upload_to_url(glb_bytes, output_location)
            return {"status": "success"}
        else:
            return {"status": "failed", "error": "No GLB generated"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

# --- Local Testing Block ---
if __name__ == "__main__":
    print("--- STARTING MULTI-VIEW LOCAL TEST ---")
    
    # Define dummy Multi-View inputs
    # Using the same image twice just to simulate a 2-view input list
    # In reality, you would provide different views of the same object
    dummy_img = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    
    test_job = {
        "input": {
            "image_urls": [dummy_img, dummy_img], 
            "mask_urls": [dummy_img, dummy_img], # Using image as mask for test
            "output_location": "test_mv_output.glb",
            "seed": 42
        }
    }
    
    result = handler(test_job)
    
    print(f"\nTest Result: {result}")
    if result["status"] == "success":
        print(f"Output saved to: {test_job['input']['output_location']}")
    else:
        print("Test failed.")

else:
    runpod.serverless.start({"handler": handler})