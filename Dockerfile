# Use the specific RunPod base image you successfully tested with
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    git wget unzip libgl1-mesa-glx libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# --- Python Environment Setup ---

# 1. Install Basic Tools & RunPod SDK
RUN pip install --no-cache-dir runpod scipy trimesh imageio[ffmpeg] transformers accelerate

# 2. Install PyTorch3D (Slow compile, do early)
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# 3. Fix 'blinker' conflict
RUN pip install --no-cache-dir blinker --ignore-installed

# 4. Install Kaolin
RUN pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# 5. Install Custom Rendering Engines
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
RUN pip install --no-cache-dir ninja jaxtyping rich && \
    pip install --no-cache-dir gsplat

# 6. Install Specific Utils
RUN pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git

# 7. Install Remaining Heavy Dependencies
RUN pip install --no-cache-dir \
    seaborn omegaconf hydra-core einops timm \
    gradio rembg loguru open3d opencv-python \
    scikit-image lightning jsonlines auto-gptq bitsandbytes

# --- BAKING THE WEIGHTS ---
# We COPY from the 'models' folder in your git repo (which we will fill in the next step)
# to /app/models inside the image.
COPY models/ /app/models/

# Set Environment Variables so handler.py knows where to look
# (Note: These point to where they live INSIDE the container, not the builder pod)
ENV LOCAL_DA3_PATH="/app/models/depth-anything-3"
ENV LOCAL_SAM3D_PATH="/app/models/sam3d"

# --- COMPILATION WARMUP ---
# Copy the warmup script
COPY warmup.py .
# Run it during the build to bake the compiled kernels into the image
RUN python warmup.py

# --- Project Setup ---
# This copies everything else (handler.py, code folders)
COPY . /app

# Final Requirements Check
# We use '|| true' or careful ordering to prevent simple failures from stopping the build
RUN pip install -r sam-3d-objects/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install -r Depth-Anything-3/requirements.txt

# Overwrite the CMD to run your handler
CMD [ "python", "-u", "handler.py" ]