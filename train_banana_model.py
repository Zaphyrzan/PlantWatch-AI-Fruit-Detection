"""
Train YOLO Model on Banana Ripeness Dataset

This script trains a YOLOv8 model on the converted banana dataset.
Run convert_to_yolo.py first to prepare the dataset.
"""

from ultralytics import YOLO
import os

# Configuration
DATASET_YAML = "banana_yolo_dataset/data.yaml"
MODEL_SIZE = "yolov8n.pt"  # Options: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt (accurate)
EPOCHS = 100
IMAGE_SIZE = 416  # Match the original dataset size
BATCH_SIZE = 16   # Reduce if you get GPU memory errors
PROJECT_NAME = "banana_training"


def main():
    print("=" * 60)
    print("🍌 Banana Ripeness YOLO Training")
    print("=" * 60)
    
    # Check dataset exists
    if not os.path.exists(DATASET_YAML):
        print(f"❌ Dataset not found at '{DATASET_YAML}'")
        print("   Run convert_to_yolo.py first!")
        return
    
    print(f"\n📊 Configuration:")
    print(f"   Model: {MODEL_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print()
    
    # Load pretrained model
    print("📥 Loading pretrained YOLOv8 model...")
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    print("\n🚀 Starting training...\n")
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        patience=20,          # Early stopping
        save=True,            # Save checkpoints
        plots=True,           # Generate training plots
        verbose=True,
        # Use CPU if no GPU available
        device='cpu'          # Change to 0 for GPU
    )
    
    # Training complete
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    best_model_path = f"runs/detect/{PROJECT_NAME}/weights/best.pt"
    print(f"\n📁 Best model saved at: {best_model_path}")
    
    print("\n📌 Next steps:")
    print("   1. Copy the model to weights_3/:")
    print(f"      copy \"{best_model_path}\" weights_3\\best.pt")
    print("   2. Restart the application:")
    print("      python app_4.py")
    print("   3. Test with a banana!")


if __name__ == "__main__":
    main()
