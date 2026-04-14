from huggingface_hub import snapshot_download
snapshot_download('nccratliri/vad-human-ava-speech', local_dir = "data/human-ava-speech", repo_type="dataset")