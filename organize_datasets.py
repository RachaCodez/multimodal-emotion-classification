"""
Dataset Organization Script
Organizes downloaded datasets into the required folder structure:
- datasets/speech/{emotion}/*.wav
- datasets/images/{emotion}/*.jpg
- datasets/text/emotion_dataset.csv
"""

import os
import shutil
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent / "datasets"

# Target emotion labels (7 classes)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def organize_speech_tess():
    """Organize TESS speech dataset into emotion folders."""
    print("\n=== Organizing TESS Speech Dataset ===")
    
    tess_root = BASE_DIR / "TESS Toronto emotional speech set data"
    speech_dir = BASE_DIR / "speech"
    
    if not tess_root.exists():
        print(f"TESS folder not found at {tess_root}")
        return
    
    # Create emotion folders
    for emotion in EMOTIONS:
        (speech_dir / emotion).mkdir(parents=True, exist_ok=True)
    
    # Mapping from TESS folder names to our emotion labels
    emotion_mapping = {
        "angry": "angry",
        "disgust": "disgust", 
        "fear": "fear",
        "Fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "Sad": "sad",
        "pleasant_surprise": "surprise",
        "Pleasant_surprise": "surprise",
        "pleasant_surprised": "surprise",
    }
    
    file_count = 0
    
    # Process all TESS subfolders (OAF_*, YAF_*, and nested TESS folder)
    for folder in tess_root.iterdir():
        if folder.is_dir():
            # Extract emotion from folder name (e.g., "OAF_angry" -> "angry")
            folder_name = folder.name
            
            # Check if it's a nested TESS folder
            if folder_name == "TESS Toronto emotional speech set data":
                # Process nested folder recursively
                for nested_folder in folder.iterdir():
                    if nested_folder.is_dir():
                        process_tess_folder(nested_folder, speech_dir, emotion_mapping)
                continue
            
            process_tess_folder(folder, speech_dir, emotion_mapping)
    
    # Count files
    for emotion in EMOTIONS:
        count = len(list((speech_dir / emotion).glob("*.wav")))
        file_count += count
        print(f"  {emotion}: {count} files")
    
    print(f"Total speech files organized: {file_count}")


def process_tess_folder(folder, speech_dir, emotion_mapping):
    """Process a single TESS emotion folder."""
    folder_name = folder.name
    
    # Extract emotion from folder name
    emotion_key = None
    for key in emotion_mapping:
        if key.lower() in folder_name.lower():
            emotion_key = key
            break
    
    if emotion_key is None:
        print(f"  Skipping unknown folder: {folder_name}")
        return
    
    target_emotion = emotion_mapping[emotion_key]
    
    # Copy all .wav files
    for wav_file in folder.glob("*.wav"):
        # Create unique filename with prefix to avoid collisions
        prefix = folder_name.split("_")[0] if "_" in folder_name else "TESS"
        new_name = f"{prefix}_{wav_file.name}"
        target_path = speech_dir / target_emotion / new_name
        
        if not target_path.exists():
            shutil.copy2(wav_file, target_path)


def organize_images_fer2013():
    """Organize FER2013 image dataset - already structured correctly."""
    print("\n=== Organizing FER2013 Image Dataset ===")
    
    fer_root = BASE_DIR / "FER2013"
    images_dir = BASE_DIR / "images"
    
    if not fer_root.exists():
        print(f"FER2013 folder not found at {fer_root}")
        return
    
    # Create emotion folders
    for emotion in EMOTIONS:
        (images_dir / emotion).mkdir(parents=True, exist_ok=True)
    
    file_count = 0
    
    # FER2013 has train/test split with emotion subfolders
    for split in ["train", "test"]:
        split_dir = fer_root / split
        if not split_dir.exists():
            continue
        
        for emotion_folder in split_dir.iterdir():
            if emotion_folder.is_dir():
                emotion_name = emotion_folder.name.lower()
                
                if emotion_name not in EMOTIONS:
                    print(f"  Skipping unknown emotion folder: {emotion_name}")
                    continue
                
                # Copy all image files
                for img_file in emotion_folder.glob("*"):
                    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        # Add split prefix to avoid name collisions
                        new_name = f"{split}_{img_file.name}"
                        target_path = images_dir / emotion_name / new_name
                        
                        if not target_path.exists():
                            shutil.copy2(img_file, target_path)
                            file_count += 1
    
    # Count files per emotion
    for emotion in EMOTIONS:
        count = len(list((images_dir / emotion).glob("*")))
        print(f"  {emotion}: {count} files")
    
    print(f"Total image files organized: {file_count}")


def organize_text_emotion():
    """Organize text emotion dataset into CSV format."""
    print("\n=== Organizing Text Emotion Dataset ===")
    
    emotion_dataset_dir = BASE_DIR / "Emotion Dataset"
    text_dir = BASE_DIR / "text"
    
    if not emotion_dataset_dir.exists():
        print(f"Emotion Dataset folder not found at {emotion_dataset_dir}")
        return
    
    text_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from dataset labels to our emotion labels
    label_mapping = {
        "joy": "happy",
        "happiness": "happy",
        "happy": "happy",
        "sadness": "sad",
        "sad": "sad",
        "anger": "angry",
        "angry": "angry",
        "fear": "fear",
        "surprise": "surprise",
        "disgust": "disgust",
        "love": "happy",  # Map love to happy
        "neutral": "neutral",
    }
    
    all_data = []
    
    # Process train.txt, test.txt, val.txt
    for txt_file in ["train.txt", "test.txt", "val.txt"]:
        file_path = emotion_dataset_dir / txt_file
        if not file_path.exists():
            continue
        
        print(f"  Processing {txt_file}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Format: text;label
                if ";" in line:
                    parts = line.rsplit(";", 1)
                    if len(parts) == 2:
                        text, label = parts
                        label = label.lower().strip()
                        
                        # Map to our emotion labels
                        if label in label_mapping:
                            mapped_label = label_mapping[label]
                            all_data.append({"text": text.strip(), "label": mapped_label})
    
    # Write to CSV
    csv_path = text_dir / "emotion_dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Total text samples: {len(all_data)}")
    
    # Count per emotion
    emotion_counts = {}
    for item in all_data:
        label = item["label"]
        emotion_counts[label] = emotion_counts.get(label, 0) + 1
    
    for emotion in EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        print(f"  {emotion}: {count} samples")
    
    print(f"CSV saved to: {csv_path}")


def main():
    print("=" * 60)
    print("Dataset Organization Script")
    print("=" * 60)
    
    # Organize each dataset
    organize_speech_tess()
    organize_images_fer2013()
    organize_text_emotion()
    
    print("\n" + "=" * 60)
    print("Organization Complete!")
    print("=" * 60)
    print("\nFinal structure:")
    print("  datasets/speech/{angry,disgust,fear,happy,neutral,sad,surprise}/*.wav")
    print("  datasets/images/{angry,disgust,fear,happy,neutral,sad,surprise}/*.(jpg|png)")
    print("  datasets/text/emotion_dataset.csv")
    print("\nYou can now run the training scripts!")


if __name__ == "__main__":
    main()
