"""The rPPG estimator must recover a known frequency and reject noise."""

import numpy as np
import pytest

from rppg_pos import POSHeartRateEstimator


def synth_roi_stream(bpm, seconds=12.0, fps=30.0, amplitude=1.2,
                     noise=0.8, jitter=0.004, seed=0):
    """Flat BGR patches modulated by a known pulse, with irregular timing."""
    rng = np.random.default_rng(seed)
    est = POSHeartRateEstimator()
    freq = bpm / 60.0
    base = np.array([110.0, 130.0, 160.0])      # B, G, R
    weights = np.array([0.4, 1.0, 0.25])        # green carries most signal
    t = 0.0
    result = None
    for _ in range(int(seconds * fps)):
        t += 1.0 / fps + rng.normal(0, jitter)
        px = base + weights * amplitude * np.sin(2 * np.pi * freq * t)
        px = px + rng.normal(0, noise, 3)
        result = est.update(np.tile(px, (12, 12, 1)), now=t)
    return result


@pytest.mark.parametrize("bpm", [50, 62, 75, 90, 110, 140])
def test_recovers_known_heart_rate(bpm):
    r = synth_roi_stream(bpm)
    assert r.reliable, f"{bpm} BPM signal was judged unreliable"
    assert abs(r.raw_bpm - bpm) < 3.0, f"got {r.raw_bpm}, want {bpm}"


def test_pure_noise_is_not_reported_as_reliable():
    rng = np.random.default_rng(7)
    est = POSHeartRateEstimator()
    t = 0.0
    r = None
    for _ in range(360):
        t += 1 / 30
        px = rng.normal(128, 8, 3)
        r = est.update(np.tile(px, (12, 12, 1)), now=t)
    assert not r.reliable
    assert r.snr_db < 2.0


def test_returns_none_before_the_window_fills():
    est = POSHeartRateEstimator()
    t = 0.0
    for _ in range(30):                      # only ~1 second of data
        t += 1 / 30
        r = est.update(np.full((8, 8, 3), 120.0), now=t)
    assert r.bpm is None
    assert not r.reliable


def test_estimate_stays_in_the_physiological_band():
    for bpm in (55, 80, 120):
        r = synth_roi_stream(bpm)
        assert 45.0 <= r.raw_bpm <= 180.0


def test_no_crash_on_low_frame_rate():
    """butter() used to raise when the effective rate fell below ~8 fps."""
    est = POSHeartRateEstimator()
    t = 0.0
    for _ in range(60):
        t += 1 / 6                            # 6 fps
        est.update(np.full((8, 8, 3), 120.0), now=t)   # must not raise


def test_handles_missing_roi():
    est = POSHeartRateEstimator()
    assert est.update(None, now=1.0).bpm is None
