print("Script started")

import cv2
from ultralytics import YOLO
print("Imports done")

model = YOLO("yolov8n.pt")
print("Model loaded")

# Path to the traffic image (raw string avoids backslash problems on Windows)
image_path = r"C:/Users/bhakt/Desktop/New folder (2)/pic11.jpg"

# COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

def classify_traffic_density(detection_count):
    if detection_count <= 15:
        return "Low"
    elif detection_count <= 30:
        return "Medium"
    elif detection_count <= 50:
        return "High"
    else:
        return "Traffic Jam"

# Read the image
frame = cv2.imread(image_path)
print("Image loaded:", frame is not None)
if frame is None:
    raise FileNotFoundError(f"Could not read image at: {image_path}")

# Run YOLOv8 (vehicles only)
results = model(frame, classes=VEHICLE_CLASSES, conf=0.3)
print("Detection done")

boxes = results[0].boxes
detection_count = len(boxes)
traffic_status = classify_traffic_density(detection_count)

# Draw the counted bounding boxes
annotated = results[0].plot()
cv2.putText(annotated, f"Vehicles: {detection_count}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(annotated, f"Traffic Density: {traffic_status}", (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Print results and save BEFORE showing the window
print(f"Vehicles detected: {detection_count}")
print(f"Predicted Traffic Density: {traffic_status}")

cv2.imwrite("output_annotated.jpg", annotated)
print("Saved output_annotated.jpg")

# Resize for display if the image is too large
h, w = annotated.shape[:2]
if w > 1280:
    scale = 1280 / w
    annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

cv2.imshow("Traffic Density Prediction", annotated)
print("Window open: click it and press any key to close")
cv2.waitKey(0)
cv2.destroyAllWindows()