import cv2

from config import (
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS
)


class Camera:

    def __init__(
        self,
        camera_id=CAMERA_ID,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=FPS
    ):

        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_id}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

    def release(self):

        if self.cap is not None:
            self.cap.release()

    def get_resolution(self):

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        return width, height

    def get_fps(self):

        return self.cap.get(
            cv2.CAP_PROP_FPS
        )