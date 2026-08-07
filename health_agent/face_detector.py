"""Single-pass face analysis.

The previous version ran two independent MediaPipe FaceMesh graphs (one here,
one inside FatigueDetector), i.e. two full model inferences per frame. This
module runs *one* FaceLandmarker and hands the result to every consumer, which
roughly halves per-frame cost and guarantees all modules see the same face.

It uses the MediaPipe Tasks API rather than the legacy `solutions.face_mesh`
because Tasks also returns the 52 blendshape coefficients that the expression
detector needs -- for free, from the same inference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import (
    FACE_LANDMARKER_MODEL,
    MIN_FACE_DETECTION_CONFIDENCE,
    MIN_FACE_PRESENCE_CONFIDENCE,
    MIN_FACE_TRACKING_CONFIDENCE,
)

# Midline + lateral points along the top of the forehead.
FOREHEAD_TOP_IDS = (10, 67, 297, 109, 338)
# Upper edge of each eyebrow -- the lower bound of the forehead patch.
EYEBROW_TOP_IDS = (105, 334, 63, 293)
# Horizontal inset applied to the forehead patch to keep hair out of the ROI.
FOREHEAD_X_INSET = 0.18
FOREHEAD_Y_INSET = 0.12

MIN_ROI_PIXELS = 100


@dataclass
class FaceResult:
    """Everything downstream modules need, from one inference."""

    landmarks: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    blendshapes: dict[str, float] = field(default_factory=dict)

    def points(self, indices) -> list[tuple[int, int]]:
        return [self.landmarks[i] for i in indices]


class FaceDetector:
    def __init__(self, model_path=FACE_LANDMARKER_MODEL):
        model_path = str(model_path)
        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
                min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MIN_FACE_TRACKING_CONFIDENCE,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - depends on local files
            raise RuntimeError(
                f"Could not load the FaceLandmarker model at {model_path}.\n"
                "Run `python download_model.py` first."
            ) from exc

        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    # ------------------------------------------------------------ detect --
    def detect(self, frame) -> FaceResult | None:
        """Run one inference. Returns None when no face is present."""
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None

        h, w = frame.shape[:2]
        pts = [
            (
                int(np.clip(lm.x * w, 0, w - 1)),
                int(np.clip(lm.y * h, 0, h - 1)),
            )
            for lm in result.face_landmarks[0]
        ]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            blendshapes = {
                c.category_name: float(c.score) for c in result.face_blendshapes[0]
            }

        return FaceResult(landmarks=pts, bbox=bbox, blendshapes=blendshapes)

    # --------------------------------------------------------------- ROIs --
    @staticmethod
    def get_face_roi(frame, face: FaceResult | None):
        if face is None or frame is None:
            return None
        x1, y1, x2, y2 = face.bbox
        if (x2 - x1) * (y2 - y1) < MIN_ROI_PIXELS:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def get_forehead_roi(frame, face: FaceResult | None):
        """Skin patch between the eyebrows and the hairline.

        Inset on all sides: the raw landmark hull includes hair and temple
        edges, and non-skin pixels are pure noise for rPPG.
        """
        if face is None or frame is None:
            return None

        h, w = frame.shape[:2]
        top_ys = [p[1] for p in face.points(FOREHEAD_TOP_IDS)]
        top_xs = [p[0] for p in face.points(FOREHEAD_TOP_IDS)]
        brow_ys = [p[1] for p in face.points(EYEBROW_TOP_IDS)]

        y1, y2 = min(top_ys), min(brow_ys)
        x1, x2 = min(top_xs), max(top_xs)
        if y2 <= y1 or x2 <= x1:
            return None

        dx = int((x2 - x1) * FOREHEAD_X_INSET)
        dy = int((y2 - y1) * FOREHEAD_Y_INSET)
        x1, x2 = max(0, x1 + dx), min(w, x2 - dx)
        y1, y2 = max(0, y1 + dy), min(h, y2 - dy)
        if y2 - y1 < 4 or x2 - x1 < 4:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size < MIN_ROI_PIXELS:
            return None
        return roi

    # ------------------------------------------------------------ drawing --
    @staticmethod
    def draw_bbox(frame, face: FaceResult | None, color=(0, 255, 0)):
        if face is None:
            return frame
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    @staticmethod
    def draw_forehead(frame, face: FaceResult | None, color=(255, 128, 0)):
        if face is None:
            return frame
        for x, y in face.points(FOREHEAD_TOP_IDS + EYEBROW_TOP_IDS):
            cv2.circle(frame, (x, y), 2, color, -1)
        return frame

    def close(self):
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
