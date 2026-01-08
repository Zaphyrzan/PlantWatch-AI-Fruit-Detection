"""
Convert Classification Dataset to YOLO Object Detection Format

This script converts the banana_classification dataset (folder-based classification)
to YOLO format with bounding box annotations.

Since classification images don't have bounding boxes, this script assumes
the banana fills most of the frame and creates a centered bounding box.
"""

import os
import shutil
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = "banana_classification"
OUTPUT_DIR = "banana_yolo_dataset"

# Class mapping (folder name -> class id and YOLO label)
CLASS_MAPPING = {
    "unripe": (0, "Unripe Banana"),
    "ripe": (1, "Ripe Banana"),
    "overripe": (2, "Overripe Banana"),
    "rotten": (3, "Rotten Banana")
}

# Bounding box settings (normalized coordinates)
# Assuming banana fills ~80% of the image, centered
BBOX_WIDTH = 0.8   # 80% of image width
BBOX_HEIGHT = 0.8  # 80% of image height
BBOX_CENTER_X = 0.5  # Center of image
BBOX_CENTER_Y = 0.5  # Center of image


def create_directory_structure():
    """Create YOLO dataset directory structure."""
    dirs = [
        f"{OUTPUT_DIR}/train/images",
        f"{OUTPUT_DIR}/train/labels",
        f"{OUTPUT_DIR}/valid/images",
        f"{OUTPUT_DIR}/valid/labels",
        f"{OUTPUT_DIR}/test/images",
        f"{OUTPUT_DIR}/test/labels",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Created directory structure")


def create_yolo_label(class_id, cx=BBOX_CENTER_X, cy=BBOX_CENTER_Y, w=BBOX_WIDTH, h=BBOX_HEIGHT):
    """Create YOLO format label string."""
    return f"{class_id} {cx} {cy} {w} {h}"


def process_split(split_name):
    """Process a single split (train/valid/test)."""
    split_path = Path(SOURCE_DIR) / split_name
    
    if not split_path.exists():
        print(f"⚠️ Split '{split_name}' not found, skipping...")
        return 0
    
    count = 0
    for class_folder, (class_id, class_label) in CLASS_MAPPING.items():
        class_path = split_path / class_folder
        
        if not class_path.exists():
            print(f"  ⚠️ Class folder '{class_folder}' not found in {split_name}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        images = [f for f in class_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
        
        for img_path in images:
            try:
                # Verify image is valid
                with Image.open(img_path) as img:
                    img.verify()
                
                # Create unique filename
                new_filename = f"{class_folder}_{img_path.stem}"
                
                # Copy image
                dest_img = Path(OUTPUT_DIR) / split_name / "images" / f"{new_filename}{img_path.suffix}"
                shutil.copy2(img_path, dest_img)
                
                # Create label file
                label_content = create_yolo_label(class_id)
                dest_label = Path(OUTPUT_DIR) / split_name / "labels" / f"{new_filename}.txt"
                with open(dest_label, 'w') as f:
                    f.write(label_content)
                
                count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
    
    return count


def create_data_yaml():
    """Create the data.yaml file required for YOLO training."""
    yaml_content = f"""# Banana Ripeness Detection Dataset
# Converted from classification to YOLO object detection format

path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: {len(CLASS_MAPPING)}

# Class names
names:
"""
    for folder, (class_id, class_label) in sorted(CLASS_MAPPING.items(), key=lambda x: x[1][0]):
        yaml_content += f"  {class_id}: {class_label}\n"
    
    yaml_path = Path(OUTPUT_DIR) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created data.yaml at {yaml_path}")


def main():
    print("=" * 60)
    print("🍌 Banana Classification to YOLO Converter")
    print("=" * 60)
    
    # Check source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory '{SOURCE_DIR}' not found!")
        print("   Make sure the banana_classification folder is in the project root.")
        return
    
    # Create output structure
    create_directory_structure()
    
    # Process each split
    total = 0
    for split in ['train', 'valid', 'test']:
        print(f"\n📂 Processing {split}...")
        count = process_split(split)
        print(f"   Processed {count} images")
        total += count
    
    # Create data.yaml
    print()
    create_data_yaml()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"📊 Total images converted: {total}")
    print(f"📁 Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📄 Data config: {os.path.abspath(OUTPUT_DIR)}/data.yaml")
    print()
    print("📌 Next steps:")
    print("   1. Run the training script: python train_banana_model.py")
    print("   2. After training, copy best.pt to weights_3/")
    print("   3. Restart the application")
    print()
    print("⚠️ Note: The bounding boxes assume bananas fill ~80% of the frame.")
    print("   For better accuracy, manually annotate images in Roboflow.")


if __name__ == "__main__":
    main()
