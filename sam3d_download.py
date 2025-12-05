from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam-3d-objects",
    repo_type="model",                       # optional; defaults to "model"
    local_dir="models/sam-3d-objects",      # directory where files are stored
    local_dir_use_symlinks=False             # ensure no symlinks, replicate files
)
