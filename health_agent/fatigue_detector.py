import cv2
import numpy as np
import mediapipe as mp


class FatigueDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.eye_indices = [
            [33, 160, 158, 133, 153, 144],
            [362, 385, 387, 263, 373, 380]
        ]

        self.blink_count = 0
        self.frame_count = 0

    def _eye_aspect_ratio(self, eye_points):

        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))

        ear = (A + B) / (2.0 * C + 1e-6)

        return ear

    def predict(self, frame):

        if frame is None:
            return {
                "fatigue": 0.0,
                "blink_rate": 0.0
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "fatigue": 0.0,
                "blink_rate": 0.0
            }

        h, w = frame.shape[:2]

        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in self.eye_indices[0]
        ]

        right_eye = [
            (int(landmarks[i].x * w), int(landmarks[i].y * h))
            for i in self.eye_indices[1]
        ]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        self.frame_count += 1

        if ear < 0.21:
            self.blink_count += 1

        blink_rate = self.blink_count / (self.frame_count / 30 + 1e-6)

        fatigue = min(100.0, (1.0 - ear) * 120 + blink_rate * 10)

        return {
            "fatigue": round(float(fatigue), 2),
            "blink_rate": round(float(blink_rate), 2)
        }