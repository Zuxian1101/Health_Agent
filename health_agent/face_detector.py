import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]

        face_landmarks = results.multi_face_landmarks[0]

        landmarks = []

        xs = []
        ys = []

        for lm in face_landmarks.landmark:

            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks.append((x, y))

            xs.append(x)
            ys.append(y)

        bbox = (
            max(0, min(xs)),
            max(0, min(ys)),
            min(w, max(xs)),
            min(h, max(ys))
        )

        return {
            "bbox": bbox,
            "landmarks": landmarks
        }

    def get_face_roi(
        self,
        frame,
        face_data
    ):

        if face_data is None:
            return None

        x1, y1, x2, y2 = face_data["bbox"]

        return frame[y1:y2, x1:x2]

    def get_forehead_roi(
        self,
        frame,
        face_data
    ):

        if face_data is None:
            return None

        landmarks = face_data["landmarks"]

        forehead_ids = [
            10,
            67,
            103,
            109,
            338,
            297,
            332
        ]

        points = [
            landmarks[idx]
            for idx in forehead_ids
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))

        x2 = max(xs)
        y2 = max(ys)

        return frame[y1:y2, x1:x2]

    def draw_bbox(
        self,
        frame,
        face_data
    ):

        if face_data is None:
            return frame

        x1, y1, x2, y2 = face_data["bbox"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        return frame

    def draw_forehead(
        self,
        frame,
        face_data
    ):

        if face_data is None:
            return frame

        landmarks = face_data["landmarks"]

        forehead_ids = [
            10,
            67,
            103,
            109,
            338,
            297,
            332
        ]

        for idx in forehead_ids:

            x, y = landmarks[idx]

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

        return frame

    def close(self):

        self.face_mesh.close()