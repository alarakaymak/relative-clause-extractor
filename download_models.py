#!/usr/bin/env python3
"""
Download SuPar parser models required for relative clause extraction.

This script downloads the necessary parser models and places them in the models/ directory.
"""

import os
import urllib.request
import zipfile
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

def unzip_model(zip_path, models_dir):
    """Unzip a model file if needed."""
    model_name = zip_path.stem  # filename without .zip
    extracted_path = models_dir / model_name
    
    if extracted_path.exists():
        print(f"  Model already extracted: {model_name}")
        return True
    
    print(f"  Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        print(f"  ✓ Extracted to {extracted_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error extracting: {e}")
        return False

def main():
    """Download required SuPar models."""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs - try multiple sources
    # Primary: SuPar official releases
    # Fallback: Alternative sources if primary fails
    models = {
        "ptb.biaffine.dep.lstm.char.zip": [
            "https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dep.lstm.char.zip",
            "https://huggingface.co/yzhangcs/parser/resolve/main/ptb.biaffine.dep.lstm.char.zip"
        ],
        "ptb.crf.con.lstm.char.zip": [
            "https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.con.lstm.char.zip",
            "https://huggingface.co/yzhangcs/parser/resolve/main/ptb.crf.con.lstm.char.zip"
        ]
    }
    
    print("="*70)
    print("SuPar Model Downloader")
    print("="*70)
    print(f"\nModels will be downloaded to: {models_dir.absolute()}")
    print(f"Total size: ~665 MB (this may take several minutes)\n")
    
    success_count = 0
    for filename, urls in models.items():
        destination = models_dir / filename
        
        # Check if already exists
        if destination.exists():
            print(f"\n{filename} already exists. Skipping download...")
            unzip_model(destination, models_dir)
            success_count += 1
            continue
        
        # Try each URL until one works
        downloaded = False
        for url in urls:
            print(f"\nTrying to download {filename}...")
            if download_file(url, destination):
                downloaded = True
                break
            else:
                print(f"  URL failed, trying alternative...")
        
        if downloaded:
            # Unzip the downloaded file
            unzip_model(destination, models_dir)
            success_count += 1
        else:
            print(f"\n✗ Failed to download {filename} from all sources")
            print("Please download manually from one of these URLs:")
            for url in urls:
                print(f"  - {url}")
            print("\nOr visit: https://github.com/yzhangcs/parser/releases")
            print("Look for the model files in the v1.0.0 release.")
    
    print("\n" + "="*70)
    if success_count == len(models):
        print("✓ All models downloaded and extracted successfully!")
        print("\nThe models are ready to use. The extractor will load them from:")
        print(f"  {models_dir.absolute()}")
    else:
        print(f"⚠ Only {success_count}/{len(models)} models downloaded.")
        print("Please download the remaining models manually.")
    print("="*70)

if __name__ == "__main__":
    main()

