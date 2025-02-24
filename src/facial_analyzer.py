import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

class FacialAnalyzer:
    def __init__(self):
        try:
            # Initialize face cascade classifier with error handling
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise Exception(f"Cascade file not found at {cascade_path}")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade classifier")
        except Exception as e:
            print(f"Error initializing facial analyzer: {str(e)}")
            self.face_cascade = None

    def capture_frame(self):
        """Capture a single frame from the webcam"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Could not access webcam")
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print("Failed to capture frame")
                return None

            return frame
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None

    def detect_faces(self, frame):
        """Detect faces in the frame"""
        if frame is None or self.face_cascade is None:
            return []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return []

    def analyze_emotion(self, frame):
        """Basic emotion analysis based on face detection"""
        if frame is None:
            return {'label': 'unknown', 'score': 0.0}

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return {'label': 'no_face', 'score': 0.0}

        # For demonstration, return a simplified emotion based on face position
        face_area = faces[0][2] * faces[0][3]  # width * height
        frame_area = frame.shape[0] * frame.shape[1]

        # Simple scoring based on face size relative to frame
        score = min(face_area / frame_area * 4, 1.0)

        return {'label': 'attentive', 'score': score}

    def draw_results(self, frame, faces, emotion_result):
        """Draw bounding boxes and emotion labels on the frame"""
        if frame is None or len(faces) == 0:
            return frame

        try:
            frame_copy = frame.copy()
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Add emotion label
                label = f"{emotion_result['label']}: {emotion_result['score']:.2f}"
                cv2.putText(
                    frame_copy,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
            return frame_copy
        except Exception as e:
            print(f"Error drawing results: {str(e)}")
            return frame

    def save_frame(self, frame):
        """Save the processed frame"""
        if frame is None:
            return None

        try:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(temp_dir, f"processed_frame_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
            return None