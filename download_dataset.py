"""
Automatic Dataset Downloader for Text Emotion Recognition

This script helps you download the emotion dataset automatically.

Usage:
  python download_dataset.py
"""

import os
import sys
import subprocess


def check_kaggle():
    """Check if Kaggle is installed and configured."""
    try:
        import kaggle
        print("✓ Kaggle module found")
        return True
    except ImportError:
        print("✗ Kaggle module not found")
        return False


def install_kaggle():
    """Install Kaggle CLI."""
    print("\nInstalling Kaggle CLI...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    print("✓ Kaggle CLI installed")


def check_credentials():
    """Check if Kaggle credentials are configured."""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_path):
        print(f"✓ Kaggle credentials found at {kaggle_path}")
        return True
    else:
        print(f"✗ Kaggle credentials not found at {kaggle_path}")
        return False


def download_dataset():
    """Download the emotion dataset."""
    dataset_dir = "datasets/text"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\nDownloading dataset to {dataset_dir}...")

    try:
        # Download using Kaggle CLI
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "praveengovi/emotions-dataset-for-nlp", "-p", dataset_dir],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ Dataset downloaded successfully")

            # Extract the zip file
            zip_file = os.path.join(dataset_dir, "emotions-dataset-for-nlp.zip")
            if os.path.exists(zip_file):
                print("\nExtracting dataset...")
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print("✓ Dataset extracted")

                # Check files
                files = os.listdir(dataset_dir)
                print(f"\nFiles in {dataset_dir}:")
                for f in files:
                    if f.endswith('.txt') or f.endswith('.csv'):
                        file_path = os.path.join(dataset_dir, f)
                        size = os.path.getsize(file_path)
                        print(f"  - {f} ({size:,} bytes)")

                return True
            else:
                print(f"✗ Zip file not found: {zip_file}")
                return False
        else:
            print(f"✗ Download failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ Kaggle CLI not found in PATH")
        return False
    except Exception as e:
        print(f"✗ Error during download: {e}")
        return False


def main():
    print("="*60)
    print("EMOTION DATASET DOWNLOADER")
    print("="*60)

    # Step 1: Check/Install Kaggle
    if not check_kaggle():
        response = input("\nKaggle module not found. Install it? (y/n): ").lower()
        if response == 'y':
            try:
                install_kaggle()
            except Exception as e:
                print(f"✗ Failed to install Kaggle: {e}")
                return
        else:
            print("\nPlease install Kaggle manually: pip install kaggle")
            return

    # Step 2: Check credentials
    if not check_credentials():
        print("\n" + "="*60)
        print("SETUP KAGGLE CREDENTIALS")
        print("="*60)
        print("\n1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Download kaggle.json")
        print("5. Move it to: " + os.path.expanduser("~/.kaggle/kaggle.json"))

        if os.name == 'nt':  # Windows
            print("\nOn Windows, run these commands:")
            print("  mkdir %USERPROFILE%\\.kaggle")
            print("  move Downloads\\kaggle.json %USERPROFILE%\\.kaggle\\")
        else:  # Linux/Mac
            print("\nOn Linux/Mac, run these commands:")
            print("  mkdir -p ~/.kaggle")
            print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("  chmod 600 ~/.kaggle/kaggle.json")

        print("\n" + "="*60)
        input("\nPress Enter after you've set up the credentials...")

        if not check_credentials():
            print("✗ Credentials still not found. Please set them up and try again.")
            return

    # Step 3: Download dataset
    print("\n" + "="*60)
    print("DOWNLOADING DATASET")
    print("="*60)

    if download_dataset():
        print("\n" + "="*60)
        print("✓ SUCCESS! Dataset downloaded and extracted.")
        print("="*60)
        print("\nNext steps:")
        print("1. Train the model:")
        print("   python model_training/train_lstm_text_model.py --csv datasets/text/train.txt --epochs 10")
        print("\n2. Run inference:")
        print('   python inference/text_lstm_inference.py "I am so happy!"')
        print("\n" + "="*60)
    else:
        print("\n" + "="*60)
        print("✗ FAILED to download dataset")
        print("="*60)
        print("\nAlternative: Manual download")
        print("1. Visit: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp")
        print("2. Click 'Download' button")
        print("3. Extract ZIP to: datasets/text/")
        print("="*60)


if __name__ == '__main__':
    main()
