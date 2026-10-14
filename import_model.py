import os
from huggingface_hub import hf_hub_download

def download_model(model_path):
    repo_id: str = "Zacharia01/zachGPT-30M"
    if not os.path.exists(model_path):
        print(f"No model found at '{model_path}'.")
        print(f"Download from : ({repo_id})")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=os.path.basename(model_path),
            local_dir=os.path.dirname(model_path) or ".",
        )
        return downloaded_path
    print('Model already downloaded')
    return model_path

if __name__ == "__main__":
    model_path = "zachGPT_pretrained.pt"
    download_model(model_path)


