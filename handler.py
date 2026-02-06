import runpod
import os
import subprocess
import shutil
import requests
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

def load_image_from_source(source):
    try:
        if source.startswith("http"):
            resp = requests.get(source, stream=True, timeout=30)
            resp.raise_for_status()
            image_data = io.BytesIO(resp.content)
            img = Image.open(image_data)
        else:
            img = Image.open(source)
        img = ImageOps.exif_transpose(img)
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {source}: {e}")

def process_and_save_image(source, save_path):
    try:
        img = load_image_from_source(source)
        img = img.convert("RGB")
        img.save(save_path, format="PNG")
    except Exception as e:
        raise RuntimeError(f"Failed to save image {source}: {e}")

def save_rgba_mask_from_inputs(image_source, mask_source, save_path):
    try:
        img = load_image_from_source(image_source)
        img = img.convert("RGB")
        mask = load_image_from_source(mask_source)
        mask = mask.convert("L")
        if mask.size != img.size:
            print(f"Resizing mask {mask_source} to match image dimensions {img.size}")
            mask = mask.resize(img.size, Image.NEAREST)
        img.putalpha(mask)
        img.save(save_path, format="PNG")
    except Exception as e:
        raise RuntimeError(f"Failed to create RGBA mask from {image_source} and {mask_source}: {e}")

def upload_single_file(file_path, upload_url):
    """Uploads a single file (result.glb) to the output location."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File to upload not found: {file_path}")

    print(f"Uploading single file: {file_path} to {upload_url[:50]}...")
    
    with open(file_path, "rb") as f_data:
        data_bytes = f_data.read()

    if upload_url.startswith("http"):
        headers = {
            'Content-Type': 'model/gltf-binary', 
            'x-ms-blob-type': 'BlockBlob'
        }
        resp = requests.put(upload_url, data=data_bytes, headers=headers)
        resp.raise_for_status()
    else:
        # Local testing fallback
        with open(upload_url, "wb") as f_out:
            f_out.write(data_bytes)
    return os.path.basename(file_path)

def package_zip_folder(folder_to_zip, upload_url):
    """Zips a folder and uploads it."""
    zip_path = os.path.abspath("output_package.zip")
    print(f"Zipping folder {folder_to_zip}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_to_zip):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_to_zip)
                zipf.write(abs_path, rel_path)

    print(f"Uploading ZIP to {upload_url[:50]}...")
    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    if upload_url.startswith("http"):
        headers = {'Content-Type': 'application/zip', 'x-ms-blob-type': 'BlockBlob'}
        requests.put(upload_url, data=zip_bytes, headers=headers).raise_for_status()
    else:
        # Local testing fallback
        if not upload_url.endswith(".zip"): upload_url += ".zip"
        with open(upload_url, "wb") as f_out:
            f_out.write(zip_bytes)
    
    if os.path.exists(zip_path): os.remove(zip_path)
    return "output_package.zip"

def handler(job):
    job_input = job.get("input", {})
    
    image_urls = job_input.get("image_urls")
    mask_urls = job_input.get("mask_urls")
    output_location = job_input.get("output_location", "output")
    mask_folder_name = job_input.get("mask_prompt", "object")
    save_all = job_input.get("save_all", False) # Default to False (GLB only)

    if not image_urls or not mask_urls:
        return {"status": "failed", "error": "Both image_urls and mask_urls are required."}
    if len(image_urls) != len(mask_urls):
        return {"status": "failed", "error": "Mismatch between image and mask counts."}

    # Setup directories
    work_dir = os.path.abspath("temp_work_dir")
    inputs_root = os.path.join(work_dir, "data", "inputs")
    input_images_dir = os.path.join(inputs_root, "images")
    input_masks_dir = os.path.join(inputs_root, mask_folder_name)
    da3_output_dir = os.path.join(work_dir, "da3_output")
    
    # OUTPUT DIRECTORY: This is where we tell inference to put files
    inference_output_dir = os.path.join(work_dir, "final_inference_output")

    if os.path.exists(work_dir): shutil.rmtree(work_dir)
    os.makedirs(input_images_dir, exist_ok=True)
    os.makedirs(input_masks_dir, exist_ok=True)
    os.makedirs(da3_output_dir, exist_ok=True)
    os.makedirs(inference_output_dir, exist_ok=True)

    custom_env = get_subprocess_env()

    try:
        print(f"Processing {len(image_urls)} pairs...")
        image_names_no_ext = []
        
        for i, (img_url, mask_url) in enumerate(zip(image_urls, mask_urls)):
            filename = f"{i}.png" 
            img_save_path = os.path.join(input_images_dir, filename)
            mask_save_path = os.path.join(input_masks_dir, filename)
            
            process_and_save_image(img_url, img_save_path)
            save_rgba_mask_from_inputs(img_url, mask_url, mask_save_path)
            
            image_names_no_ext.append(str(i))

        print("--- Running Depth Anything 3 ---")
        da3_weights_path = find_local_da3_weights()
        da3_cmd = [
            "python", DA3_SCRIPT_PATH,
            "--image_dir", input_images_dir,
            "--output_dir", da3_output_dir,
            "--process_res", "756"
        ]
        if da3_weights_path:
            da3_cmd.extend(["--model_path", da3_weights_path])
        
        subprocess.run(da3_cmd, cwd=MV_SAM3D_DIR, env=custom_env, check=True)
        
        da3_npz = os.path.join(da3_output_dir, "da3_output.npz")
        if not os.path.exists(da3_npz):
            raise RuntimeError("DA3 failed to generate da3_output.npz")

        print(f"--- Running MV-SAM3D Inference ---")
        img_names_arg = ",".join(image_names_no_ext)
        
        inference_cmd = [
            "python", INFERENCE_SCRIPT_PATH,
            "--input_path", inputs_root,
            "--mask_prompt", mask_folder_name,
            "--image_names", img_names_arg,
            "--da3_output", da3_npz,
            
            # DIRECT the output to our known folder
            "--handoff_dir", inference_output_dir, 
            
            # Weighting Params
            "--no_stage1_weighting",
            "--stage2_weight_source", "mixed",       
            # "--stage2_entropy_alpha", "60.0",        
            # "--stage2_visibility_alpha", "60.0",
            "--stage2_visibility_weight_ratio", "0.4",
            # "--self_occlusion_tolerance", "6.0"      
        ]
        
        # If user wants EVERYTHING, pass the flag to the script
        if save_all:
            inference_cmd.append("--save_all")

        print("DEBUG COMMAND:", " ".join(inference_cmd))
        subprocess.run(inference_cmd, cwd=MV_SAM3D_DIR, env=custom_env, check=True)

        # --- UPLOAD LOGIC ---
        if save_all:
            print("--- Save All requested: Zipping entire work directory ---")
            # If save_all is True, we zip the entire work_dir (inputs + outputs) for debugging
            uploaded_name = package_zip_folder(work_dir, output_location)
        else:
            print("--- Standard Mode: Uploading Result GLB only ---")
            # If save_all is False, we just grab result.glb from our known location
            expected_glb = os.path.join(inference_output_dir, "result.glb")
            uploaded_name = upload_single_file(expected_glb, output_location)

        return {
            "status": "success", 
            "message": f"Successfully uploaded {uploaded_name} to output location.",
            "mode": "save_all" if save_all else "single_glb"
        }

    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error": f"Script execution failed: {e}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}
    finally:
        # CLEANUP: This runs regardless of success or failure
        if os.path.exists(work_dir):
            print(f"Cleaning up {work_dir}...")
            shutil.rmtree(work_dir)

runpod.serverless.start({"handler": handler})
# if __name__ == "__main__":
#     print("--- STARTING LOCAL DEBUG TEST ---")
#     test_images = ["1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png"] 
#     test_masks = ["1_mask.png", "2_mask.png", "3_mask.png", "4_mask.png", "5_mask.png", "6_mask.png", "7_mask.png", "8_mask.png"] 
    
#     valid_pairs = []
#     for img, mask in zip(test_images, test_masks):
#         if os.path.exists(img) and os.path.exists(mask):
#             valid_pairs.append((img, mask))

#     if valid_pairs:
#         valid_imgs, valid_msks = zip(*valid_pairs)
        
#         # Test Case 1: Standard (GLB Only)
#         print("\n>>> TEST 1: save_all = False")
#         test_job_std = {
#             "input": {
#                 "image_urls": list(valid_imgs),
#                 "mask_urls": list(valid_msks),
#                 "output_location": "local_result.glb",
#                 "save_all": False
#             }
#         }
#         handler(test_job_std)

#         # # Test Case 2: Save All (Zip)
#         # print("\n>>> TEST 2: save_all = True")
#         # test_job_all = {
#         #     "input": {
#         #         "image_urls": list(valid_imgs),
#         #         "mask_urls": list(valid_msks),
#         #         "output_location": "local_debug_bundle", # will append .zip
#         #         "save_all": True
#         #     }
#         # }
#         # handler(test_job_all)
#     else:
#         print("No local test files found.")
# else:
#     runpod.serverless.start({"handler": handler})