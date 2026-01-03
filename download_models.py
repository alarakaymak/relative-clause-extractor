#!/usr/bin/env python3
"""
Download SuPar parser models required for relative clause extraction.

This script downloads the necessary parser models from the SuPar repository
and places them in the models/ directory.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination with progress."""
    print(f"Downloading {os.path.basename(destination)}...")
    print(f"URL: {url}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n✓ Successfully downloaded to {destination}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        return False

def main():
    """Download required SuPar models."""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs from SuPar
    # Note: These are the official SuPar model URLs
    models = {
        "ptb.biaffine.dep.lstm.char.zip": "https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dep.lstm.char.zip",
        "ptb.crf.con.lstm.char.zip": "https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.con.lstm.char.zip"
    }
    
    print("="*70)
    print("SuPar Model Downloader")
    print("="*70)
    print(f"\nModels will be downloaded to: {models_dir.absolute()}")
    print(f"Total size: ~665 MB (this may take several minutes)\n")
    
    success_count = 0
    for filename, url in models.items():
        destination = models_dir / filename
        
        # Check if already exists
        if destination.exists():
            print(f"\n{filename} already exists. Skipping...")
            success_count += 1
            continue
        
        if download_file(url, destination):
            success_count += 1
        else:
            print(f"\nFailed to download {filename}")
            print("You can manually download from:")
            print(f"  {url}")
    
    print("\n" + "="*70)
    if success_count == len(models):
        print("✓ All models downloaded successfully!")
        print("\nNote: The models are zipped. The extractor will automatically")
        print("unzip them when first used. You can also manually unzip if needed.")
    else:
        print(f"⚠ Only {success_count}/{len(models)} models downloaded.")
        print("Please download the remaining models manually.")
    print("="*70)

if __name__ == "__main__":
    main()

