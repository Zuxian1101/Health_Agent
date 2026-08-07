"""Drowsiness estimation from eye landmarks (PERCLOS).

Fixes relative to the first implementation:

1. **No second model.** The old class built its own MediaPipe FaceMesh, so
   every frame ran two full face models. It now consumes landmarks produced
   once by FaceDetector.
2. **The score was broken.** `fatigue = (1 - EAR) * 120 + blink_rate * 10`
   yields ~84 for a wide-awake person (open-eye EAR is about 0.3), and the
   agent fired at 75 -- so *everyone* was permanently reported as fatigued.
   The score now uses PERCLOS, the standard measure (Wierwille et al.):
   the fraction of time the eyes are closed over a rolling window.
3. **Blinks were frames, not blinks.** The old code incremented a counter on
   every frame with EAR below threshold, so a 3-frame blink counted as 3.
   Blinks are now edge-triggered with a minimum-duration debounce.
4. **Rate was cumulative, not rolling.** `blinks / (frames / 30)` measured
   the average since program start and hard-coded 30 fps. It is now a true
   rolling blinks-per-minute using real timestamps.
5. **Per-person calibration.** Absolute EAR varies a lot with eye shape,
   glasses and camera angle, so a fixed 0.21 cutoff is meaningless across
   users. A short calibration establishes each person's open-eye baseline.
"""

from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple, Sequence

import numpy as np

from config import (
    BLINK_MIN_FRAMES,
    BLINK_RATE_NORMAL,
    BLINK_RATE_WINDOW_SECONDS,
    BLINK_WEIGHT,
    EAR_CALIBRATION_SECONDS,
    EYE_CLOSED_RATIO,
    PERCLOS_ALERT,
    PERCLOS_DROWSY,
    PERCLOS_WEIGHT,
    PERCLOS_WINDOW_SECONDS,
)

EPS = 1e-6
LEFT_EYE_IDS = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_IDS = (362, 385, 387, 263, 373, 380)

# Fallback baseline if calibration never completes (typical open-eye EAR).
DEFAULT_EAR_BASELINE = 0.30


class FatigueResult(NamedTuple):
    fatigue: float          # 0-100
    perclos: float          # 0-1, fraction of window with eyes closed
    blink_rate: float       # blinks per minute
    ear: float
    eyes_closed: bool
    calibrating: bool
    valid: bool

    @property
    def display(self) -> str:
        if self.calibrating:
            return "calibrating..."
        if not self.valid:
            return "--"
        return f"{self.fatigue:.0f}"


def eye_aspect_ratio(points: Sequence[tuple[float, float]]) -> float:
    """EAR for six landmarks ordered [outer, top1, top2, inner, bot2, bot1]."""
    p = np.asarray(points, dtype=np.float64)
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c + EPS))


def _linear_map(value, in_lo, in_hi, out_lo, out_hi):
    if in_hi <= in_lo:
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    return out_lo + np.clip(t, 0.0, 1.0) * (out_hi - out_lo)


class FatigueDetector:
    def __init__(self, calibration_seconds: float = EAR_CALIBRATION_SECONDS):
        self.calibration_seconds = float(calibration_seconds)
        self._calibration_samples: list[float] = []
        self._calibration_start: float | None = None
        self._baseline: float | None = None

        self._closed_history: deque[tuple[float, bool]] = deque()
        self._blink_times: deque[float] = deque()
        self._closed_run = 0

    # ------------------------------------------------------------ public --
    @property
    def baseline(self) -> float:
        return self._baseline if self._baseline is not None else DEFAULT_EAR_BASELINE

    def reset(self):
        self.__init__(self.calibration_seconds)

    def update(self, landmarks=None, now: float | None = None) -> FatigueResult:
        """Feed the landmark list from FaceDetector (or None if no face)."""
        now = time.monotonic() if now is None else float(now)

        if landmarks is None:
            self._closed_run = 0
            self._trim(now)
            return self._score(now, ear=0.0, closed=False, valid=False)

        ear = 0.5 * (
            eye_aspect_ratio([landmarks[i] for i in LEFT_EYE_IDS])
            + eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE_IDS])
        )

        if self._calibrate(ear, now):
            return self._score(now, ear=ear, closed=False, valid=True, calibrating=True)

        closed = ear < self.baseline * EYE_CLOSED_RATIO
        self._closed_history.append((now, closed))
        self._register_blink(closed, now)
        self._trim(now)
        return self._score(now, ear=ear, closed=closed, valid=True)

    # ----------------------------------------------------------- internal --
    def _calibrate(self, ear: float, now: float) -> bool:
        """Returns True while still calibrating."""
        if self._baseline is not None:
            return False
        if self._calibration_start is None:
            self._calibration_start = now

        self._calibration_samples.append(ear)
        if now - self._calibration_start < self.calibration_seconds:
            return True

        if self._calibration_samples:
            # 75th percentile: robust to the blinks that occur during calibration.
            self._baseline = float(np.percentile(self._calibration_samples, 75))
        else:
            self._baseline = DEFAULT_EAR_BASELINE
        return False

    def _register_blink(self, closed: bool, now: float):
        if closed:
            self._closed_run += 1
            return
        # Rising edge: eyes just reopened. One blink, if it lasted long enough.
        if self._closed_run >= BLINK_MIN_FRAMES:
            self._blink_times.append(now)
        self._closed_run = 0

    def _trim(self, now: float):
        while (
            self._closed_history
            and now - self._closed_history[0][0] > PERCLOS_WINDOW_SECONDS
        ):
            self._closed_history.popleft()
        while (
            self._blink_times
            and now - self._blink_times[0] > BLINK_RATE_WINDOW_SECONDS
        ):
            self._blink_times.popleft()

    def _perclos(self) -> float:
        if not self._closed_history:
            return 0.0
        closed = sum(1 for _, c in self._closed_history if c)
        return closed / len(self._closed_history)

    def _blink_rate(self, now: float) -> float:
        if not self._closed_history:
            return 0.0
        span = now - self._closed_history[0][0]
        # Don't extrapolate a rate from a window that is barely open yet.
        if span < 10.0:
            return 0.0
        return len(self._blink_times) * 60.0 / span

    def _score(self, now, ear, closed, valid, calibrating=False) -> FatigueResult:
        perclos = self._perclos()
        blink_rate = self._blink_rate(now)

        if not valid or calibrating:
            return FatigueResult(
                0.0, perclos, blink_rate, round(ear, 4), closed, calibrating, valid
            )

        # PERCLOS drives the score: 0 -> 0, PERCLOS_ALERT -> 50, DROWSY -> 100.
        if perclos <= PERCLOS_ALERT:
            perclos_score = _linear_map(perclos, 0.0, PERCLOS_ALERT, 0.0, 50.0)
        else:
            perclos_score = _linear_map(
                perclos, PERCLOS_ALERT, PERCLOS_DROWSY, 50.0, 100.0
            )

        # Blink rate outside the normal band is a secondary signal. A rate of
        # exactly 0 usually means "window not populated yet", not "no blinks".
        lo, hi = BLINK_RATE_NORMAL
        if blink_rate <= 0.0:
            blink_score = 0.0
        elif blink_rate < lo:
            blink_score = _linear_map(blink_rate, lo, 0.0, 0.0, 100.0)
        elif blink_rate > hi:
            blink_score = _linear_map(blink_rate, hi, hi * 2.5, 0.0, 100.0)
        else:
            blink_score = 0.0

        fatigue = float(
            np.clip(PERCLOS_WEIGHT * perclos_score + BLINK_WEIGHT * blink_score, 0, 100)
        )
        return FatigueResult(
            fatigue=round(fatigue, 1),
            perclos=round(perclos, 4),
            blink_rate=round(blink_rate, 1),
            ear=round(ear, 4),
            eyes_closed=closed,
            calibrating=False,
            valid=True,
        )
