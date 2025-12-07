import requests
import base64
import json
import os

# --- CONFIGURATION ---
API_KEY = "REPLACE_WITH_KEY"  # Replace with your actual key
ENDPOINT_ID = "vf7lpguz4eszh3"
URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# File paths to send
IMAGE_PATH = "test_image_1.jpg"
MASK_PATH = "test_image_1_mask.png"
OUTPUT_FILENAME = "output_from_api.glb"

def encode_image(file_path):
    """Reads an image file and encodes it to a Base64 string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
        
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def main():
    print(f"--- Sending request to Endpoint: {ENDPOINT_ID} ---")

    # 1. Encode Inputs
    try:
        print("Encoding images...")
        b64_image = encode_image(IMAGE_PATH)
        b64_mask = encode_image(MASK_PATH)
    except Exception as e:
        print(f"Error preparing inputs: {e}")
        return

    # 2. Construct Payload
    # Matches the structure your handler.py expects: job['input']['image']
    payload = {
        "input": {
            "image": b64_image,
            "mask": b64_mask,
            "seed": 42
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # 3. Send Request
    # We use 'runsync' which waits for the response (good for testing).
    # For production apps, you usually use 'run' (async) and poll for status.
    print("Sending POST request (this may take 10-60 seconds)...")
    try:
        response = requests.post(URL, json=payload, headers=headers, timeout=300)
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model might be taking too long for 'runsync'.")
        return

    # 4. Handle Response
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(response.text)
        return

    result = response.json()
    
    # Check for RunPod-level errors
    if 'error' in result:
        print(f"RunPod Error: {result['error']}")
        return

    # Check for Handler-level status
    status = result.get('status')
    if status == 'COMPLETED':
        output = result.get('output', {})
        
        # Check if our handler sent back an error inside the successful execution
        if "error" in output:
            print(f"Model Handler Error: {output['error']}")
        
        elif "glb_file" in output:
            # 5. Decode and Save GLB
            print("Success! Decoding GLB file...")
            glb_bytes = base64.b64decode(output["glb_file"])
            
            with open(OUTPUT_FILENAME, "wb") as f:
                f.write(glb_bytes)
            
            print(f"Saved 3D model to: {os.path.abspath(OUTPUT_FILENAME)}")
        else:
            print(f"Unexpected output format: {output.keys()}")
            
    else:
        print(f"Job Status: {status}")
        print(f"Full Response: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()