# Use RunPod's PyTorch 2.4 image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. System Dependencies for 3D Rendering
RUN apt-get update && apt-get install -y \
    git wget unzip libgl1-mesa-glx libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
RUN pip install --no-cache-dir \
    runpod \
    scipy \
    trimesh \
    imageio[ffmpeg] \
    transformers \
    accelerate

# 3. Install PyTorch3D (Critical Step)
# Since we are on a specific CUDA version (12.4), we try to install prebuilt if available, 
# or compile. Compiling takes time but is safest on the Devel image.
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 4. Install SAM 3D
WORKDIR /app
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git
WORKDIR /app/sam-3d-objects
RUN pip install -r requirements.txt

# 5. Setup Handler
COPY handler.py /app/handler.py

# 6. Config
ENV SAM3D_CHECKPOINT_DIR="/runpod-volume/sam3d/checkpoints/hf"

WORKDIR /app
CMD [ "python", "-u", "handler.py" ]