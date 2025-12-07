import requests
import base64
import json
import os
import time

# --- CONFIGURATION ---
API_KEY = "REPLACE_WITH_KEY" # <--- Paste your key here
ENDPOINT_ID = "vf7lpguz4eszh3"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

# Input Files
IMAGE_PATH = "test_image_1.jpg"
MASK_PATH = "test_image_1_mask.png"
OUTPUT_FILENAME = "output_async.glb"

def encode_image(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 1. Prepare Payload
    print(f"--- Encoding inputs for Endpoint: {ENDPOINT_ID} ---")
    try:
        payload = {
            "input": {
                "image": encode_image(IMAGE_PATH),
                "mask": encode_image(MASK_PATH),
                "seed": 42
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Submit Job (Async)
    # We change '/runsync' to '/run'
    submit_url = f"{BASE_URL}/run"
    print("Submitting job...")
    response = requests.post(submit_url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Submission Error: {response.text}")
        return

    job_data = response.json()
    job_id = job_data['id']
    print(f"Job submitted successfully. ID: {job_id}")
    print("Waiting for completion (Polling every 5 seconds)...")

    # 3. Poll for Results
    status_url = f"{BASE_URL}/status/{job_id}"
    
    while True:
        status_res = requests.get(status_url, headers=headers)
        status_data = status_res.json()
        
        status = status_data.get('status')
        
        if status == 'COMPLETED':
            print("\nJob Completed!")
            output = status_data.get('output', {})
            
            # Check for internal handler errors
            if isinstance(output, dict) and "error" in output:
                print(f"Model Error: {output['error']}")
            
            # Save Result
            elif isinstance(output, dict) and "glb_file" in output:
                print("Decoding GLB file...")
                glb_bytes = base64.b64decode(output["glb_file"])
                with open(OUTPUT_FILENAME, "wb") as f:
                    f.write(glb_bytes)
                print(f"SUCCESS: Saved 3D model to {os.path.abspath(OUTPUT_FILENAME)}")
            else:
                print(f"Unexpected output format: {output.keys() if isinstance(output, dict) else output}")
            break
            
        elif status == 'FAILED':
            print("\nJob Failed.")
            print(f"Error Info: {status_data}")
            break
            
        elif status in ['IN_QUEUE', 'IN_PROGRESS']:
            # Simple loading animation
            print(".", end="", flush=True)
            time.sleep(5)
            
        else:
            print(f"\nUnknown status: {status}")
            break

if __name__ == "__main__":
    main()