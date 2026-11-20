Traffic Density Prediction using YOLOv8

Detects vehicles in a traffic image using a pre-trained YOLOv8 model and classifies traffic density as Low, Medium, High or Traffic Jam based on the vehicle count.

Output
Features
Detects cars, motorcycles, buses and trucks (COCO classes 2, 3, 5, 7)
Draws bounding boxes with labels and confidence scores
Shows vehicle count and density level on the image
Saves the annotated output image

How It Works :
Load the image and run YOLOv8 (conf=0.3, vehicle classes only)
Count the detected vehicles
Classify density using count thresholds:

Vehicles	Density
0-15	  Low
16-30	  Medium
31-50	  High
50+	    Traffic Jam
Tech    Stack

Python, OpenCV, Ultralytics YOLOv8
