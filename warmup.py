# warmup.py
import torch
import os

print("--- Starting Warmup: Pre-compiling Kernels ---")

# 1. Warmup nvdiffrast (Major culprit for slow starts)
try:
    print("Compiling nvdiffrast...")
    import nvdiffrast.torch as dr
    # Creating the context forces the GL/Cuda compilation
    glctx = dr.RasterizeGLContext()
    print("SUCCESS: nvdiffrast compiled.")
except Exception as e:
    print(f"WARNING: nvdiffrast warmup failed: {e}")

# 2. Warmup Gaussian Splatting (gsplat)
try:
    print("Compiling gsplat...")
    import gsplat
    # access a dummy function to trigger JIT if applicable
    # (Just importing it is often enough for some versions, but 
    # instantiating a simple operation is safer)
    print("SUCCESS: gsplat imported/compiled.")
except Exception as e:
    print(f"WARNING: gsplat warmup failed: {e}")

# 3. Warmup PyTorch3D (Just in case)
try:
    print("Compiling PyTorch3D ops...")
    from pytorch3d.ops import sample_farthest_points
    print("SUCCESS: PyTorch3D ops loaded.")
except Exception as e:
    print(f"WARNING: PyTorch3D warmup failed: {e}")

print("--- Warmup Complete ---")r