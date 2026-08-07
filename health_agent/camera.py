"""Webcam wrapper with a real frame timestamp and clean shutdown."""

from __future__ import annotations

import time

import cv2

from config import CAMERA_ID, FPS, FRAME_HEIGHT, FRAME_WIDTH


class Camera:
    def __init__(
        self,
        camera_id: int = CAMERA_ID,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = FPS,
    ):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_id}. "
                "Check that no other app is using it and that camera "
                "permission is granted to your terminal / Python."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        """Returns (frame, timestamp) or (None, timestamp) on failure.

        The timestamp is taken at capture time, not at analysis time --
        rPPG frequency estimates depend on it being accurate.
        """
        ok, frame = self.cap.read()
        now = time.monotonic()
        return (frame if ok else None), now

    def get_resolution(self) -> tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    def release(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
