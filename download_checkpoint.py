from huggingface_hub import snapshot_download

save_dir = "models/G2VLM-2B-MoT"
repo_id = "InternRobotics/G2VLM-2B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
