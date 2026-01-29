# Use the specific RunPod base image you successfully tested with 
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# --- System Dependencies ---
# Added 'ninja-build' which is critical for compiling gsplat/kaolin
RUN apt-get update && apt-get install -y \
    git wget unzip libgl1-mesa-glx libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# --- Python Environment Setup ---

# 1. Install Basic Tools & RunPod SDK
RUN pip install --no-cache-dir runpod scipy trimesh imageio[ffmpeg] transformers accelerate

# 2. Install PyTorch3D
# We do this early because it takes 10-15 minutes to compile.
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
RUN pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"

# 3. Fix 'blinker' conflict (System vs Pip)
RUN pip install --no-cache-dir blinker --ignore-installed

# 4. Install Kaolin (Pre-compiled Wheel)
# Using the specific version/index you verified (0.18.0 for Torch 2.5.1) [cite: 3]
# This will trigger a PyTorch upgrade to 2.5.1, which is expected.
RUN pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# 5. Install Custom Rendering Engines (NVDiffrast & GSplat)
RUN pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
RUN pip install --no-cache-dir ninja jaxtyping rich && \
    pip install --no-cache-dir gsplat

# 6. Install Specific Utils (Fixes 'utils3d' version mismatch)
RUN pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git

# 7. Install Remaining Heavy Dependencies
RUN pip install --no-cache-dir \
    seaborn omegaconf hydra-core einops timm \
    gradio rembg loguru open3d opencv-python \
    scikit-image lightning jsonlines auto-gptq bitsandbytes

# --- Project Setup ---

# 8. Copy your project files
# Expects: handler.py, MV-SAM3D/, and Depth-Anything-3/ in the build context
COPY . /app

# 9. Install Repo Requirements
# We use the extra-index-url to ensure CUDA versions are found if needed [cite: 6]
RUN if [ -f MV-SAM3D/requirements.txt ]; then \
    pip install -r MV-SAM3D/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121; \
    fi

RUN if [ -f Depth-Anything-3/requirements.txt ]; then \
    pip install -r Depth-Anything-3/requirements.txt; \
    fi

# --- Deployment ---

# Overwrite the CMD to run your handler
CMD [ "python", "-u", "handler.py" ]