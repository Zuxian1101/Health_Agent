"""Real-time health-signal monitor.

Fixes relative to the first implementation:

* `work_time` was incremented once every 30 frames, i.e. it assumed exactly
  30 fps. With two FaceMesh models running the app was nowhere near 30 fps,
  so the "you have been sitting too long" timer drifted badly. It now uses
  a wall clock.
* One face inference per frame feeds every consumer (was two).
* Capture timestamps are passed through to the rPPG estimator instead of it
  calling time.time() at analysis time.
* Cleanup runs in a finally block, so Ctrl-C or a mid-loop exception still
  releases the camera and the model.
* An on-screen disclaimer. This is not a medical device.

Press ESC or q to quit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque

import cv2

from camera import Camera
from config import (
    CAMERA_ID,
    DISCLAIMER,
    FACE_LANDMARKER_MODEL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_NAME,
)
from expression_detector import ExpressionDetector
from face_detector import FaceDetector
from fatigue_detector import FatigueDetector
from health_agent import HealthAgent
from rppg_pos import POSHeartRateEstimator

GREEN = (0, 255, 0)
AMBER = (0, 190, 255)
GREY = (170, 170, 170)
RED = (0, 0, 255)


def draw_text(frame, text, y, color=GREEN, scale=0.7):
    cv2.putText(
        frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
        cv2.LINE_AA,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, default=CAMERA_ID)
    p.add_argument("--width", type=int, default=FRAME_WIDTH)
    p.add_argument("--height", type=int, default=FRAME_HEIGHT)
    p.add_argument("--duration", type=float, default=None,
                   help="stop after N seconds (default: run until ESC)")
    p.add_argument("--no-display", action="store_true",
                   help="headless; print a JSON summary on exit")
    p.add_argument("--mirror", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not FACE_LANDMARKER_MODEL.exists():
        print(
            f"Face model not found at {FACE_LANDMARKER_MODEL}\n"
            "Run:  python download_model.py",
            file=sys.stderr,
        )
        return 2

    cam = face = None
    agent = HealthAgent()
    try:
        cam = Camera(args.camera, args.width, args.height)
        face = FaceDetector()
        rppg = POSHeartRateEstimator()
        fatigue_detector = FatigueDetector()
        expression_detector = ExpressionDetector()

        started = time.monotonic()
        fps_window: deque[float] = deque(maxlen=30)
        active_advice: list = []
        advice_until = 0.0

        while True:
            frame, captured_at = cam.read()
            if frame is None:
                print("Camera returned no frame; stopping.", file=sys.stderr)
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            fps_window.append(captured_at)

            face_result = face.detect(frame)
            landmarks = face_result.landmarks if face_result else None
            forehead = FaceDetector.get_forehead_roi(frame, face_result)

            hr = rppg.update(forehead, now=captured_at)
            fat = fatigue_detector.update(landmarks, now=captured_at)
            expr = expression_detector.predict(
                face_result.blendshapes if face_result else None
            )

            # Wall clock, not a frame counter. This was the bug.
            work_minutes = (captured_at - started) / 60.0

            state = {
                "hr": hr.bpm,
                "hr_reliable": hr.reliable,
                "hr_snr_db": hr.snr_db,
                "expression": expr.expression if expr.valid else None,
                "fatigue": fat.fatigue if (fat.valid and not fat.calibrating) else None,
                "perclos": fat.perclos,
                "blink_rate": fat.blink_rate,
                "work_minutes": work_minutes,
                "face_present": face_result is not None,
            }

            new_advice = agent.analyze(state, now=captured_at)
            if new_advice:
                active_advice = new_advice
                advice_until = captured_at + 8.0
            elif captured_at > advice_until:
                active_advice = []

            if not args.no_display:
                FaceDetector.draw_bbox(frame, face_result)
                FaceDetector.draw_forehead(frame, face_result)

                hr_color = GREEN if hr.reliable else GREY
                draw_text(frame, f"HR       {hr.display} bpm  "
                                 f"(SNR {hr.snr_db:+.1f} dB)", 34, hr_color)
                draw_text(frame, f"Fatigue  {fat.display}"
                                 f"   PERCLOS {fat.perclos:.2f}"
                                 f"   {fat.blink_rate:.0f} blinks/min", 60,
                          AMBER if (fat.valid and fat.fatigue >= 70) else GREEN)
                draw_text(frame, f"Face     {expr.display}", 86, GREEN)

                fps = 0.0
                if len(fps_window) > 1:
                    span = fps_window[-1] - fps_window[0]
                    fps = (len(fps_window) - 1) / span if span > 0 else 0.0
                draw_text(frame, f"{fps:4.1f} fps   {work_minutes:.1f} min", 112, GREY)

                for i, adv in enumerate(active_advice):
                    draw_text(frame, f">> {adv.message}", 148 + i * 26, AMBER, 0.7)

                h = frame.shape[0]
                draw_text(frame, DISCLAIMER, h - 14, RED, 0.5)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.duration and (captured_at - started) >= args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.release()
        if face is not None:
            face.close()
        cv2.destroyAllWindows()

    print(json.dumps(agent.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
