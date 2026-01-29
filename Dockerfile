# --- 1. System Dependencies ---
apt-get update && apt-get install -y \
    git wget unzip libgl1-mesa-glx libglib2.0-0 build-essential ninja-build

# --- 2. Basic Python Tools ---
pip install --upgrade pip
pip install runpod scipy trimesh imageio[ffmpeg] transformers accelerate

# --- 3. PyTorch3D (Compiling from source - Takes ~10 mins) ---
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# --- 4. Kaolin & Custom Renderers ---
# Fix blinker conflict
pip install blinker --ignore-installed

# Install Kaolin (using pre-built wheel for speed)
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html

# Install NVDiffrast, GSplat, and utilities
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install ninja jaxtyping rich
pip install gsplat
pip install git+https://github.com/EasternJournalist/utils3d.git

# --- 5. Heavy Python Dependencies ---
pip install seaborn omegaconf hydra-core einops timm \
    gradio rembg loguru open3d opencv-python \
    scikit-image lightning jsonlines auto-gptq bitsandbytes

# --- 6. Repository Requirements ---
# Assuming you are in the folder containing 'MV-SAM3D' and 'Depth-Anything-3'
# Adjust paths if your folders are named differently

if [ -d "MV-SAM3D" ]; then
    echo "Installing MV-SAM3D requirements..."
    pip install -r MV-SAM3D/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
fi

if [ -d "Depth-Anything-3" ]; then
    echo "Installing Depth-Anything-3 requirements..."
    pip install -r Depth-Anything-3/requirements.txt
fi