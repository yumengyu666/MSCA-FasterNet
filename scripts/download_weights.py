"""Download FasterNet-T0 pretrained weights from GitHub."""
import os
import urllib.request
import json
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def download_file(url, save_path):
    """Download a file with progress display."""
    print(f"Downloading: {url}")
    print(f"Saving to: {save_path}")
    
    def reporthook(count, block_size, total_size):
        percent = count * block_size / total_size * 100
        downloaded = count * block_size / 1024 / 1024
        total = total_size / 1024 / 1024
        sys.stdout.write(f"\rProgress: {downloaded:.1f}/{total:.1f} MB ({percent:.1f}%)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, save_path, reporthook)
    size = os.path.getsize(save_path)
    print(f"\nDone! File size: {size/1024/1024:.1f} MB")
    return size

def main():
    # Step 1: Get release assets from GitHub API
    api_url = "https://api.github.com/repos/JierunChen/FasterNet/releases/tags/v1.0"
    headers = {"User-Agent": "Python"}
    
    print("Querying GitHub API for FasterNet releases...")
    req = urllib.request.Request(api_url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"GitHub API failed: {e}")
        print("Trying direct URL patterns...")
        # Try common URL patterns
        urls = [
            "https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_t0.pth",
            "https://github.com/JierunChen/FasterNet/releases/download/v1.0/FasterNet_T0.pth",
        ]
        for url in urls:
            try:
                save_path = os.path.join(CHECKPOINT_DIR, "fasternet_t0.pth")
                download_file(url, save_path)
                if os.path.getsize(save_path) > 1_000_000:  # > 1MB means success
                    print("Download successful!")
                    return
                else:
                    os.remove(save_path)
                    print("File too small, likely 404 page. Trying next...")
            except Exception as e2:
                print(f"Failed: {e2}")
        print("All direct URLs failed.")
        return
    
    # Step 2: Find fasternet_t0 weight file
    assets = data.get("assets", [])
    print(f"Found {len(assets)} assets in release v1.0:")
    
    target = None
    for asset in assets:
        name = asset["name"]
        size_mb = asset["size"] / 1024 / 1024
        print(f"  - {name} ({size_mb:.1f} MB)")
        if "t0" in name.lower() and name.endswith(".pth"):
            target = asset
    
    if target is None:
        print("ERROR: fasternet_t0.pth not found in release assets!")
        print("Available .pth files:")
        for a in assets:
            if a["name"].endswith(".pth"):
                print(f"  {a['name']}")
        return
    
    # Step 3: Download
    download_url = target["browser_download_url"]
    save_path = os.path.join(CHECKPOINT_DIR, target["name"])
    
    if os.path.exists(save_path) and os.path.getsize(save_path) == target["size"]:
        print(f"File already exists and size matches: {save_path}")
        return
    
    download_file(download_url, save_path)
    
    # Step 4: Verify
    print("\nVerifying weight file...")
    import torch
    state_dict = torch.load(save_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        print(f"Keys: {len(state_dict)} parameters")
        first_key = list(state_dict.keys())[0]
        print(f"First key: {first_key}, shape: {state_dict[first_key].shape}")
        print("Verification PASSED!")
    else:
        print(f"Unexpected format: {type(state_dict)}")

if __name__ == "__main__":
    main()
