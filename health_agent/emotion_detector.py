import cv2
import numpy as np


class EmotionDetector:

    def __init__(self):
        pass

    def predict(self, face_roi):

        if face_roi is None:
            return {
                "emotion": "neutral",
                "confidence": 0.0
            }

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        contrast = np.std(gray)

        if brightness < 80:
            emotion = "sad"
            confidence = 0.65

        elif contrast > 60:
            emotion = "angry"
            confidence = 0.6

        else:
            emotion = "neutral"
            confidence = 0.7

        return {
            "emotion": emotion,
            "confidence": confidence
        }