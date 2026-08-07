"""PERCLOS must separate an alert user from a drowsy one."""

import pytest

from fatigue_detector import FatigueDetector, eye_aspect_ratio

LEFT = (33, 160, 158, 133, 153, 144)
RIGHT = (362, 385, 387, 263, 373, 380)


def _eye(ear, cx=0.0, w=30.0):
    h = ear * w
    return [(cx, 0.0), (cx + w * .3, -h / 2), (cx + w * .7, -h / 2),
            (cx + w, 0.0), (cx + w * .7, h / 2), (cx + w * .3, h / 2)]


def landmarks_with_ear(ear):
    lm = [(0.0, 0.0)] * 478
    for i, p in zip(LEFT, _eye(ear)):
        lm[i] = p
    for i, p in zip(RIGHT, _eye(ear, cx=100.0)):
        lm[i] = p
    return lm


def run(pattern, seconds=90.0, fps=30.0):
    d = FatigueDetector(calibration_seconds=5.0)
    t = 0.0
    r = None
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        r = d.update(landmarks_with_ear(pattern(t)), now=t)
    return d, r


def test_ear_formula():
    assert eye_aspect_ratio(_eye(0.30)) == pytest.approx(0.30, abs=1e-6)


def test_alert_user_scores_low():
    # 15 blinks/min, each ~130 ms
    _, r = run(lambda t: 0.10 if (t % 4.0) < 0.13 else 0.30)
    assert r.fatigue < 30, r
    assert 10 <= r.blink_rate <= 20


def test_drowsy_user_scores_high():
    # eyes closed ~35% of the time
    _, r = run(lambda t: 0.10 if (t % 3.0) < 1.05 else 0.30)
    assert r.fatigue > 70, r
    assert r.perclos > 0.30


def test_drowsy_scores_above_alert():
    _, alert = run(lambda t: 0.10 if (t % 4.0) < 0.13 else 0.30)
    _, drowsy = run(lambda t: 0.10 if (t % 3.0) < 1.05 else 0.30)
    assert drowsy.fatigue > alert.fatigue


def test_open_eyes_are_not_reported_as_fatigued():
    """The old formula gave ~84/100 for a wide-awake person."""
    _, r = run(lambda t: 0.30)
    assert r.fatigue < 20, f"wide-awake user scored {r.fatigue}"


def test_blinks_are_counted_as_events_not_frames():
    """A 6-frame blink is one blink, not six."""
    d = FatigueDetector(calibration_seconds=0.5)
    t = 0.0
    for _ in range(30):                      # calibrate on open eyes
        t += 1 / 30
        d.update(landmarks_with_ear(0.30), now=t)
    for _ in range(6):                       # one long blink
        t += 1 / 30
        d.update(landmarks_with_ear(0.10), now=t)
    for _ in range(10):
        t += 1 / 30
        d.update(landmarks_with_ear(0.30), now=t)
    assert len(d._blink_times) == 1


def test_single_frame_dropout_is_debounced():
    d = FatigueDetector(calibration_seconds=0.5)
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        d.update(landmarks_with_ear(0.30), now=t)
    t += 1 / 30
    d.update(landmarks_with_ear(0.10), now=t)      # single-frame glitch
    for _ in range(5):
        t += 1 / 30
        d.update(landmarks_with_ear(0.30), now=t)
    assert len(d._blink_times) == 0


def test_baseline_calibrates_to_the_person():
    d = FatigueDetector(calibration_seconds=2.0)
    t = 0.0
    for _ in range(120):                     # narrow-eyed user
        t += 1 / 30
        d.update(landmarks_with_ear(0.18), now=t)
    assert d.baseline == pytest.approx(0.18, abs=0.02)


def test_no_face_is_invalid_not_zero_fatigue():
    d = FatigueDetector()
    r = d.update(None, now=1.0)
    assert not r.valid
