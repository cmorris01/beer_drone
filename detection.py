"""
detection.py
~~~~~~~~~~~~

A module to implement the trained YOLOv8 model. Functionalities include individual predictions,
webcam intialization with real time object detection, etc. (as implentation is called for)

"""

#### Libraries
# Standard library
import os
import math
from collections import defaultdict

# Third party libariess
from ultralytics import YOLO
import cv2
import torch

class Detection:
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])

    def start_webcam(self):
        # Start the webcam and set resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Width
        self.cap.set(4, 640)  # Height

    def perform_detection(self):
        # Detection loop
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = self.model(frame, conf=0.5)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.start_webcam()
        self.perform_detection()

if __name__ == "__main__":
        # Define the path to the model
    model_path = '/Users/coymorris/Desktop/beer_drone/experiments/exp1/best.pt'

    # Create an instance of the Detection class
    detection = Detection(model_path)

    # Run the detection
    detection.run()