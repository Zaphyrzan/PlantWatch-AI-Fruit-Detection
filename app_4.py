from flask import Flask, render_template, request, Response, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)

# Load YOLO model for banana ripeness detection
banana_detection_model = YOLO("weights_3/best.pt")  # Path to YOLO model for banana detection


def get_harvestability_info(ripeness_status):
    """
    Returns harvestability information based on the ripeness status of the banana.
    """
    ripeness_lower = ripeness_status.lower()
    
    if ripeness_lower in ['ripe', 'ripen', 'yellow']:
        return {
            'status': 'Harvestable and Consumable',
            'recommendation': 'This banana is at its peak ripeness. Perfect for immediate consumption or sale.',
            'color': '#28a745'  # Green
        }
    elif ripeness_lower in ['unripe', 'raw', 'green', 'underripe']:
        return {
            'status': 'Not Ready for Harvest',
            'recommendation': 'This banana needs more time to ripen. Estimated time to ripeness: 3-5 days depending on storage conditions. Store at room temperature (20-25°C) to accelerate ripening.',
            'color': '#ffc107'  # Yellow/Warning
        }
    elif ripeness_lower in ['overripe', 'over-ripe', 'spoiled', 'rotten']:
        return {
            'status': 'Past Optimal Harvest Time',
            'recommendation': 'This banana is overripe. Not ideal for fresh consumption but can still be used for baking, smoothies, or compost. Consume within 1-2 days if still edible.',
            'color': '#dc3545'  # Red
        }
    else:
        return {
            'status': 'Unknown Ripeness',
            'recommendation': 'Unable to determine ripeness status. Please inspect manually.',
            'color': '#6c757d'  # Gray
        }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Receive image data from the client
    image_data = request.json['image_data'].split(',')[1]  # Remove the data URL prefix

    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)

    # Convert image bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Decode the image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Perform object detection using YOLO
    results = banana_detection_model(image)

    # Extract detection results
    detected_objects = []
    for result in results:
        boxes = result.boxes.xywh.cpu()  # xywh bbox list
        clss = result.boxes.cls.cpu().tolist()  # classes Id list
        names = result.names  # classes names list
        confs = result.boxes.conf.float().cpu().tolist()  # probabilities of classes

        for box, cls, conf in zip(boxes, clss, confs):
            class_name = names[cls]
            # Extract ripeness status (first word) from class name
            ripeness_status = class_name.split(' ')[0] if ' ' in class_name else class_name
            harvestability = get_harvestability_info(ripeness_status)
            
            detected_objects.append({
                'class': class_name, 
                'bbox': box.tolist(), 
                'confidence': conf,
                'harvestability': harvestability
            })

    return jsonify(detected_objects)


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # Read frame from camera
        if not success:
            break
        else:
            fruit_results = banana_detection_model(frame)
            for result in fruit_results:
                im_array = result.plot()
                im = Image.fromarray(im_array[..., ::-1])
                image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
