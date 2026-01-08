# 🍌 Banana Ripeness Detection System

> **UTM Artificial Intelligence Project 2025/2026**  
> Safuan Hakim & Wan Zafirzan

## Project Overview

An AI-powered web application that uses **YOLOv8 deep learning** to detect banana ripeness levels in real-time and provide harvestability recommendations. This system helps farmers and consumers quickly assess banana quality using just a camera.

### Problem Statement

Manual fruit ripeness assessment is time-consuming, subjective, and prone to human error. This project addresses the need for an automated, accurate, and real-time solution for banana ripeness classification.

### Objectives

1. Develop a real-time banana ripeness detection system using YOLOv8
2. Classify bananas into ripeness categories: **Unripe**, **Ripe**, **Overripe**, and **Rotten**
3. Provide actionable harvestability recommendations
4. Create an accessible web-based interface

---

## AI Theoretical Foundation

### Knowledge Representation

The system uses multiple knowledge representation techniques:

| Representation | Implementation |
|----------------|----------------|
| **Feature Vectors** | CNN extracts visual features (color, texture, shape) as numerical vectors |
| **Class Labels** | Categorical representation: `{0: unripe, 1: ripe, 2: overripe, 3: rotten}` |
| **Bounding Boxes** | Spatial representation: `[x1, y1, x2, y2]` coordinates |
| **Confidence Scores** | Probabilistic representation: 0.0 - 1.0 certainty values |
| **Rule-Based Logic** | Harvestability recommendations encoded as IF-THEN rules |

**Harvestability Knowledge Rules:**
```
IF ripeness = "ripe" AND confidence > 0.5 THEN status = "Harvestable"
IF ripeness = "unripe" THEN status = "Not Ready" AND recommendation = "Wait 3-5 days"
IF ripeness = "overripe" THEN status = "Past Optimal" AND recommendation = "Use for baking"
IF ripeness = "rotten" THEN status = "Not Consumable" AND recommendation = "Discard"
```

### State Space Representation

The banana ripeness detection system operates in a defined state space:

```
State Space S = {Initial State, Processing States, Goal States}

Initial State (S₀):
├── Camera: OFF
├── Model: Loaded
└── Detection: None

Processing States:
├── S₁: Camera Active, Frame Captured
├── S₂: Image Preprocessed (416×416 RGB)
├── S₃: Feature Extraction (CNN layers)
├── S₄: Object Detection (YOLO head)
└── S₅: Classification Complete

Goal States (Sₙ):
├── G₁: Banana detected → Ripeness classified → Recommendation displayed
└── G₂: No banana detected → "Point camera at banana" message
```

**State Transition Function:**
```
δ(S₀, capture_frame) → S₁
δ(S₁, preprocess) → S₂
δ(S₂, extract_features) → S₃
δ(S₃, detect_objects) → S₄
δ(S₄, classify) → S₅
δ(S₅, display_result) → G₁ or G₂
```

### Search Strategy

The YOLO model employs a **grid-based search** strategy:

| Component | Description |
|-----------|-------------|
| **Search Space** | 13×13 grid cells over input image |
| **Anchor Boxes** | 3 predefined aspect ratios per cell |
| **Objective Function** | Minimize: `L = L_box + L_obj + L_cls` |
| **Optimization** | Gradient descent with Adam optimizer |

### Machine Learning Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Input     │───▶│   Backbone  │───▶│    Neck     │───▶│    Head     │
│  (416×416)  │    │  (CSPNet)   │    │   (PANet)   │    │  (Detect)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │                  │
                          ▼                  ▼                  ▼
                   Feature Maps      Multi-scale         Predictions
                   (Low → High)       Fusion           (bbox, conf, cls)
```

### Inference Algorithm

```python
Algorithm: Banana Ripeness Detection
Input: Image frame I
Output: List of (class, confidence, bbox, recommendation)

1. Preprocess(I) → I' (resize, normalize)
2. Features ← CNN_Backbone(I')
3. MultiScale ← PANet(Features)
4. Predictions ← DetectionHead(MultiScale)
5. Detections ← NMS(Predictions, threshold=0.5)
6. FOR each detection d IN Detections:
      class ← argmax(d.class_probs)
      recommendation ← RuleEngine(class)
      OUTPUT(class, d.confidence, d.bbox, recommendation)
```

---

## Features

| Feature | Description |
|---------|-------------|
| 📷 **Real-time Detection** | Live camera feed with instant analysis |
| 🎯 **Ripeness Classification** | Detects unripe, ripe, overripe, and rotten bananas |
| 📊 **Confidence Score** | Shows detection accuracy percentage |
| 🌿 **Harvestability Guide** | Actionable recommendations based on ripeness |

### Ripeness Categories

| Status | Color | Recommendation |
|--------|-------|----------------|
| 🟢 **Ripe** | Green | Harvestable and Consumable - Ready for consumption |
| 🟡 **Unripe** | Yellow | Not Ready - Wait 3-5 days to ripen |
| 🔴 **Overripe** | Red | Past Optimal - Best for baking/smoothies |
| ⚫ **Rotten** | Gray | Not Consumable - Discard |

---

## Technology Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Computer Vision**: OpenCV
- **Dataset**: 13,478 banana images (Roboflow)

---

## Installation

### Prerequisites

- Python 3.8+
- Webcam/Camera (front and/or back camera supported)
- pip package manager

### Dependencies

This project requires the following Python packages:

| Package | Purpose |
|---------|----------|
| `Flask` | Web framework for the backend server |
| `ultralytics` | YOLOv8 deep learning model |
| `opencv-python` | Computer vision and image processing |
| `numpy` | Numerical computing |
| `Pillow` | Image manipulation |

### Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/Banana-Ripeness-Detection.git
cd Banana-Ripeness-Detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app_4.py
```

Open http://127.0.0.1:5000 in your browser.

> **📱 Mobile Access**: You can also access the app from your phone using the network URL (e.g., `http://192.168.x.x:5000`) when connected to the same network. The app supports switching between front and back cameras.

---

## Project Structure

```
Banana-Ripeness-Detection/
├── app_4.py                    # Main Flask application
├── train_banana_model.py       # Model training script
├── convert_to_yolo.py          # Dataset conversion utility
├── requirements.txt            # Python dependencies
├── templates/
│   ├── index.html              # Home page
│   └── fruit_detection.html    # Detection page
├── weights_3/
│   └── best.pt                 # Trained YOLO model
└── banana_yolo_dataset/        # Training dataset
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

---

## How It Works

1. **Image Capture**: Webcam captures frames in real-time
2. **Preprocessing**: Images are resized and normalized
3. **Detection**: YOLOv8 model identifies bananas and classifies ripeness
4. **Post-processing**: Results are filtered and formatted
5. **Display**: Ripeness status and recommendations shown to user

---

## Model Training

The model was trained on **13,478 banana images** across 4 classes:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='banana_yolo_dataset/data.yaml',
    epochs=100,
    imgsz=416,
    batch=16
)
```

### Training Results

| Metric | Value |
|--------|-------|
| mAP50 | 97.5% |
| Precision | 94.3% |
| Recall | 93.1% |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/fruit_detection` | GET | Camera detection page |
| `/detect_objects` | POST | Process image, return detection results |

---

## Project Progress

### Progress 1: Problem Definition & Research
- Identified the problem of manual banana ripeness assessment
- Conducted literature review on YOLO object detection
- Defined project scope and objectives
- Selected YOLOv8 as the detection framework

### Progress 2: Dataset Preparation & Model Training
- Collected 13,478 banana images from Roboflow
- Converted classification dataset to YOLO format
- Annotated images with bounding boxes
- Trained YOLOv8 model achieving 97.5% mAP50

### Progress 3: System Integration & Testing
- Developed Flask web application
- Implemented real-time camera detection
- Added harvestability recommendation engine
- Created responsive UI with bounding box visualization
- Conducted testing and validation

---

## Future Enhancements

- [ ] Mobile app version (Android/iOS)
- [ ] Disease detection capability
- [ ] Support for other fruits (mango, papaya, etc.)
- [ ] Historical data tracking
- [ ] Export reports as PDF

---

## References

- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Roboflow Dataset: https://roboflow.com/
- Flask Documentation: https://flask.palletsprojects.com/

---

## License

This project is licensed under the MIT License.

## Contributors

- **Safuan Hakim** - Developer
- **Wan Zafirzan** - Developer

---

*UTM Faculty of Computing • Artificial Intelligence Course • Semester 6 • 2025/2026*
