import runpod
import os
import subprocess
import shutil
import requests
import glob
import sys
import zipfile
import io
from pathlib import Path
from PIL import Image, ImageOps

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MV_SAM3D_DIR = os.path.join(BASE_DIR, "MV-SAM3D")
DA3_SCRIPT_PATH = os.path.join(MV_SAM3D_DIR, "scripts", "run_da3.py") 
INFERENCE_SCRIPT_PATH = os.path.join(MV_SAM3D_DIR, "run_inference_weighted.py")

if not os.path.exists(DA3_SCRIPT_PATH):
    DA3_SCRIPT_PATH = os.path.join(MV_SAM3D_DIR, "run_da3.py")

def get_subprocess_env():
    env = os.environ.copy()
    if "CONDA_PREFIX" not in env:
        env["CONDA_PREFIX"] = os.path.dirname(os.path.dirname(sys.executable))
    return env

def find_local_da3_weights():
    if os.path.exists("/runpod-volume"):
        base = "/runpod-volume"
    else:
        base = "/workspace"
    possible_paths = [
        os.path.join(base, "models", "depth-anything-3"),
        os.path.join(base, "depth-anything-3"),
        "/workspace/models/depth-anything-3"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found local DA3 weights at: {path}")
            return path
    print("No local DA3 weights found. Script will attempt download.")
    return None

def process_and_save_image(source, save_path, is_mask=False):
    try:
        if source.startswith("http"):
            resp = requests.get(source, stream=True, timeout=30)
            resp.raise_for_status()
            image_data = io.BytesIO(resp.content)
            img = Image.open(image_data)
        else:
            img = Image.open(source)

        img = ImageOps.exif_transpose(img)

        if is_mask:
            if img.mode != "RGBA":
                print(f"Warning: Mask {source} is {img.mode}, converting to RGBA.")
                img = img.convert("RGBA")
        else:
            img = img.convert("RGB") 

        img.save(save_path, format="PNG")
        
    except Exception as e:
        raise RuntimeError(f"Failed to process {source}: {e}")

def package_debug_zip(work_dir, mv_sam3d_vis_dir, output_zip_path):
    print(f"Packaging debug info into {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, work_dir)
                zipf.write(abs_path, rel_path)
        
        if os.path.exists(mv_sam3d_vis_dir):
             for root, dirs, files in os.walk(mv_sam3d_vis_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.join("mv_sam3d_results", os.path.relpath(abs_path, mv_sam3d_vis_dir))
                    zipf.write(abs_path, rel_path)
    print("Debug ZIP created successfully.")

def handler(job):
    job_input = job.get("input", {})
    
    image_urls = job_input.get("image_urls")
    mask_urls = job_input.get("mask_urls")
    output_location = job_input.get("output_location", "debug_output.zip")
    mask_folder_name = job_input.get("mask_prompt", "object")

    if not image_urls or not mask_urls:
        return {"status": "failed", "error": "Both image_urls and mask_urls are required."}
    if len(image_urls) != len(mask_urls):
        return {"status": "failed", "error": "Mismatch between image and mask counts."}

    work_dir = os.path.abspath("temp_work_dir")
    inputs_root = os.path.join(work_dir, "data", "inputs")
    input_images_dir = os.path.join(inputs_root, "images")
    input_masks_dir = os.path.join(inputs_root, mask_folder_name)
    da3_output_dir = os.path.join(work_dir, "da3_output")
    
    if os.path.exists(work_dir): shutil.rmtree(work_dir)
    os.makedirs(input_images_dir, exist_ok=True)
    os.makedirs(input_masks_dir, exist_ok=True)
    os.makedirs(da3_output_dir, exist_ok=True)

    custom_env = get_subprocess_env()

    try:
        print(f"Processing {len(image_urls)} pairs...")
        image_names_no_ext = []
        
        for i, (img_url, mask_url) in enumerate(zip(image_urls, mask_urls)):
            filename = f"{i}.png" 
            img_save_path = os.path.join(input_images_dir, filename)
            mask_save_path = os.path.join(input_masks_dir, filename)
            
            process_and_save_image(img_url, img_save_path, is_mask=False)
            process_and_save_image(mask_url, mask_save_path, is_mask=True)
            
            image_names_no_ext.append(str(i))

        print("--- Running Depth Anything 3 ---")
        da3_weights_path = find_local_da3_weights()
        da3_cmd = [
            "python", DA3_SCRIPT_PATH,
            "--image_dir", input_images_dir,
            "--output_dir", da3_output_dir,
        ]
        if da3_weights_path:
            da3_cmd.extend(["--model_path", da3_weights_path])
        
        subprocess.run(da3_cmd, cwd=MV_SAM3D_DIR, env=custom_env, check=True)
        
        da3_npz = os.path.join(da3_output_dir, "da3_output.npz")
        if not os.path.exists(da3_npz):
            raise RuntimeError("DA3 failed to generate da3_output.npz")

        vis_output_dir = os.path.join(MV_SAM3D_DIR, "visualization")
        if os.path.exists(vis_output_dir):
            print(f"--- Cleaning up stale data in {vis_output_dir} ---")
            shutil.rmtree(vis_output_dir)

        print(f"--- Running MV-SAM3D Inference (Using Entropy Weighting) ---")
        img_names_arg = ",".join(image_names_no_ext)
        inference_cmd = [
            "python", INFERENCE_SCRIPT_PATH,
            "--input_path", inputs_root,
            "--mask_prompt", mask_folder_name,
            "--image_names", img_names_arg,
            "--no_stage1_weighting",
            # "--stage1_entropy_alpha", "80.0", 
            # "--stage1_entropy_layer", "9",
            # "--da3_output", da3_npz,
            # "--stage2_weight_source", "mixed",       # Combine Entropy + Visibility
            # "--stage2_entropy_alpha", "60.0",        # Sharpen selection 
            # "--stage2_visibility_alpha", "60.0",     # Sharpen occlusion
            # "--stage2_visibility_weight_ratio", "0.6", 
            # "--self_occlusion_tolerance", "6.0"      
        ]

        subprocess.run(inference_cmd, cwd=MV_SAM3D_DIR, env=custom_env, check=True)

        print("--- Packaging Debug Data ---")
        zip_path = os.path.abspath("final_debug.zip")
        package_debug_zip(work_dir, vis_output_dir, zip_path)

        with open(zip_path, "rb") as f:
            zip_bytes = f.read()

        if output_location.startswith("http"):
            headers = {'Content-Type': 'application/zip'}
            requests.put(output_location, data=zip_bytes, headers=headers).raise_for_status()
        else:
            if not output_location.endswith(".zip"): output_location += ".zip"
            with open(output_location, "wb") as f_out:
                f_out.write(zip_bytes)
                
        if os.path.exists(zip_path): os.remove(zip_path)
        return {"status": "success", "message": f"Debug ZIP generated at {output_location}"}

    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error": f"Script execution failed: {e}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    print("--- STARTING LOCAL DEBUG TEST ---")
    test_images = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png"]
    test_masks = ["1_mask.png", "2_mask.png", "3_mask.png", "4_mask.png", "5_mask.png", "6_mask.png", "7_mask.png", "8_mask.png"]
    
    valid_pairs = []
    for img, mask in zip(test_images, test_masks):
        if os.path.exists(img) and os.path.exists(mask):
            valid_pairs.append((img, mask))

    if valid_pairs:
        valid_imgs, valid_msks = zip(*valid_pairs)
        test_job = {
            "input": {
                "image_urls": list(valid_imgs),
                "mask_urls": list(valid_msks),
                "output_location": "local_debug_bear_hq.zip",
                "mask_prompt": "object"
            }
        }
        print(f"Processing {len(valid_imgs)} pairs...")
        result = handler(test_job)
        print(f"\nFinal Result: {result}")
    else:
        print("No valid image/mask pairs found!")
else:
    runpod.serverless.start({"handler": handler})