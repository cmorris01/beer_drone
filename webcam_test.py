# import statements
import cv2
import torch
from ultralytics import YOLO
import math

# load model
model = YOLO('/Users/coymorris/Desktop/beer_detector/trained_models/yolov8m_v1.pt')

# start webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

# detection loop
while True:
    success, img = cap.read()
    results = model(img, stream = True)

    # bounding boxes
    for r in results:
        boxes = r.boxes

        for box in boxes:

            # extract coordinates of bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", cls)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, str(cls), org, font, fontScale, color, thickness)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()