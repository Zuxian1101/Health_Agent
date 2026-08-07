"""Remote photoplethysmography (rPPG) using the POS algorithm.

Reference: Wang et al., "Algorithmic Principles of Remote PPG",
IEEE TBME 64(7), 2017.

Fixes relative to the first implementation, in rough order of severity:

1. **Channel order.** OpenCV frames are BGR; the POS projection matrix is
   derived for RGB. The original code fed `roi.mean(axis=(0,1))` -- i.e.
   [B, G, R] -- straight into the matrix, silently swapping R and B and
   defeating the algorithm's motion robustness.
2. **Real POS.** POS is defined over short overlapping windows with
   overlap-add reconstruction. Normalising once across the whole 10 s buffer
   (as before) throws away the property that makes POS work.
3. **Non-uniform sampling.** Webcam frames do not arrive on a regular grid,
   but `np.fft` assumes they do. Samples are now resampled onto a uniform
   grid before any spectral analysis.
4. **Filter stability.** `butter()` raises if the normalised cutoff reaches
   1.0, which happened whenever the effective frame rate dropped below 8 fps.
   Resampling to a fixed rate plus an explicit guard removes that crash.
5. **Frequency resolution.** A 10 s window gives 0.1 Hz bins == 6 BPM
   quantisation. Zero-padding brings that under 1 BPM.
6. **Honesty.** The estimator now reports an SNR and a `reliable` flag
   instead of returning a confident-looking number for pure noise.
"""

from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, filtfilt, detrend

from config import (
    BANDPASS_ORDER,
    FFT_ZERO_PAD_FACTOR,
    HIGH_HR_HZ,
    HR_MIN_WINDOW_SECONDS,
    HR_SMOOTHING_ALPHA,
    HR_WINDOW_SECONDS,
    LOW_HR_HZ,
    MIN_HR_SNR_DB,
    POS_WINDOW_SECONDS,
    RESAMPLE_FPS,
)

EPS = 1e-9

# POS projection matrix, defined for channel order [R, G, B].
_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])

# Half-width of the band counted as "signal" around the peak, in Hz.
_SNR_HALF_BAND_HZ = 0.2


class HeartRateEstimate(NamedTuple):
    bpm: float | None       # smoothed estimate, None until one is available
    raw_bpm: float | None   # this window's estimate, unsmoothed
    snr_db: float
    reliable: bool
    effective_fps: float

    @property
    def display(self) -> str:
        if self.bpm is None:
            return "--"
        suffix = "" if self.reliable else "?"
        return f"{self.bpm:.0f}{suffix}"


class POSHeartRateEstimator:
    def __init__(
        self,
        window_seconds: float = HR_WINDOW_SECONDS,
        resample_fps: float = RESAMPLE_FPS,
    ):
        self.window_seconds = float(window_seconds)
        self.resample_fps = float(resample_fps)
        self._times: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self._smoothed_bpm: float | None = None

    # ------------------------------------------------------------ public --
    def reset(self):
        self._times.clear()
        self._rgb.clear()
        self._smoothed_bpm = None

    def update(self, roi, now: float | None = None) -> HeartRateEstimate:
        """Feed one forehead ROI (BGR, as returned by OpenCV)."""
        now = time.monotonic() if now is None else float(now)

        if roi is not None and getattr(roi, "size", 0) > 0:
            bgr_mean = np.asarray(roi, dtype=np.float64).mean(axis=(0, 1))
            # BGR -> RGB. This one line is the bug fix that matters most.
            self._rgb.append(bgr_mean[::-1].copy())
            self._times.append(now)

        self._trim(now)
        return self._estimate()

    # ----------------------------------------------------------- internal --
    def _trim(self, now: float):
        while self._times and now - self._times[0] > self.window_seconds:
            self._times.popleft()
            self._rgb.popleft()

    def _estimate(self) -> HeartRateEstimate:
        n = len(self._times)
        if n < 2:
            return self._result(None, 0.0, 0.0)

        times = np.fromiter(self._times, dtype=np.float64, count=n)
        span = times[-1] - times[0]
        effective_fps = (n - 1) / span if span > 0 else 0.0

        if span < HR_MIN_WINDOW_SECONDS:
            return self._result(None, 0.0, effective_fps)

        rgb = np.vstack(self._rgb)                      # (n, 3) in RGB order
        uniform, fs = self._resample(times, rgb)
        if uniform is None:
            return self._result(None, 0.0, effective_fps)

        pulse = self._pos(uniform, fs)
        pulse = self._bandpass(pulse, fs)
        if pulse is None:
            return self._result(None, 0.0, effective_fps)

        raw_bpm, snr_db = self._spectral_peak(pulse, fs)
        return self._result(raw_bpm, snr_db, effective_fps)

    def _resample(self, times: np.ndarray, rgb: np.ndarray):
        """Map irregular samples onto a uniform grid so the FFT is valid."""
        fs = self.resample_fps
        duration = times[-1] - times[0]
        count = int(duration * fs)
        if count < 32:
            return None, fs
        grid = np.linspace(times[0], times[-1], count, endpoint=True)
        out = np.column_stack(
            [np.interp(grid, times, rgb[:, c]) for c in range(3)]
        )
        return out, fs

    @staticmethod
    def _pos(rgb: np.ndarray, fs: float) -> np.ndarray:
        """POS with overlapping windows and overlap-add reconstruction."""
        n = len(rgb)
        win = max(16, int(round(POS_WINDOW_SECONDS * fs)))
        if win > n:
            win = n

        out = np.zeros(n, dtype=np.float64)
        for start in range(0, n - win + 1):
            end = start + win
            block = rgb[start:end].T                    # (3, win)
            mean = block.mean(axis=1, keepdims=True)
            normalised = block / (mean + EPS)
            s = _POS_PROJECTION @ normalised            # (2, win)

            std1 = s[1].std()
            alpha = (s[0].std() / (std1 + EPS)) if std1 > EPS else 0.0
            h = s[0] + alpha * s[1]
            out[start:end] += h - h.mean()
        return out

    @staticmethod
    def _bandpass(signal: np.ndarray, fs: float):
        nyquist = fs / 2.0
        low = LOW_HR_HZ / nyquist
        high = HIGH_HR_HZ / nyquist
        if not (0.0 < low < high < 1.0):
            return None

        b, a = butter(BANDPASS_ORDER, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if len(signal) <= padlen:
            return None
        return filtfilt(b, a, detrend(signal))

    @staticmethod
    def _spectral_peak(signal: np.ndarray, fs: float):
        n = len(signal)
        windowed = signal * np.hanning(n)
        nfft = int(2 ** np.ceil(np.log2(n * FFT_ZERO_PAD_FACTOR)))

        power = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

        band = (freqs >= LOW_HR_HZ) & (freqs <= HIGH_HR_HZ)
        if not band.any():
            return None, 0.0

        band_power = power[band]
        band_freqs = freqs[band]
        peak_freq = float(band_freqs[int(np.argmax(band_power))])

        # Signal = fundamental + first harmonic; noise = rest of the band.
        signal_mask = np.abs(band_freqs - peak_freq) <= _SNR_HALF_BAND_HZ
        signal_mask |= np.abs(band_freqs - 2 * peak_freq) <= _SNR_HALF_BAND_HZ
        sig = band_power[signal_mask].sum()
        noise = band_power[~signal_mask].sum()
        snr_db = 10.0 * np.log10((sig + EPS) / (noise + EPS))

        return peak_freq * 60.0, float(snr_db)

    def _result(self, raw_bpm, snr_db, effective_fps) -> HeartRateEstimate:
        reliable = raw_bpm is not None and snr_db >= MIN_HR_SNR_DB
        if reliable:
            if self._smoothed_bpm is None:
                self._smoothed_bpm = raw_bpm
            else:
                a = HR_SMOOTHING_ALPHA
                self._smoothed_bpm = a * raw_bpm + (1 - a) * self._smoothed_bpm

        bpm = round(self._smoothed_bpm, 1) if self._smoothed_bpm is not None else None
        return HeartRateEstimate(
            bpm=bpm,
            raw_bpm=round(raw_bpm, 1) if raw_bpm is not None else None,
            snr_db=round(float(snr_db), 2),
            reliable=bool(reliable),
            effective_fps=round(float(effective_fps), 1),
        )
