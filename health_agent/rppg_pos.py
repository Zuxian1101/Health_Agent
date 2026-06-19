import time

import cv2
import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt

from config import (
    HR_WINDOW_SECONDS,
    LOW_HR_HZ,
    HIGH_HR_HZ
)


class POSHeartRateEstimator:

    def __init__(
        self,
        window_seconds=HR_WINDOW_SECONDS
    ):

        self.window_seconds = window_seconds

        self.rgb_buffer = []

        self.time_buffer = []

    def update(self, roi):

        if roi is None:
            return None

        mean_rgb = roi.mean(axis=(0, 1))

        self.rgb_buffer.append(mean_rgb)

        self.time_buffer.append(time.time())

        self._trim_buffer()

        if len(self.rgb_buffer) < 100:
            return None

        return self._estimate_hr()

    def _trim_buffer(self):

        current_time = time.time()

        while (
            len(self.time_buffer) > 0
            and current_time - self.time_buffer[0]
            > self.window_seconds
        ):

            self.time_buffer.pop(0)
            self.rgb_buffer.pop(0)

    def _estimate_hr(self):

        rgb = np.array(
            self.rgb_buffer,
            dtype=np.float32
        )

        timestamps = np.array(
            self.time_buffer
        )

        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        if duration < 5:
            return None

        fs = len(timestamps) / duration

        signal = self._pos(rgb)

        signal = self._bandpass_filter(
            signal,
            fs
        )

        hr = self._fft_hr(
            signal,
            fs
        )

        return hr

    def _pos(self, rgb):

        rgb = rgb.T

        mean_color = np.mean(
            rgb,
            axis=1,
            keepdims=True
        )

        normalized = rgb / mean_color - 1

        projection = np.array([
            [0, 1, -1],
            [-2, 1, 1]
        ])

        s = projection @ normalized

        alpha = (
            np.std(s[0])
            / (
                np.std(s[1])
                + 1e-8
            )
        )

        pulse = (
            s[0]
            + alpha * s[1]
        )

        return pulse

    def _bandpass_filter(
        self,
        signal,
        fs
    ):

        nyquist = fs / 2

        low = LOW_HR_HZ / nyquist
        high = HIGH_HR_HZ / nyquist

        b, a = butter(
            3,
            [low, high],
            btype="band"
        )

        filtered = filtfilt(
            b,
            a,
            signal
        )

        return filtered

    def _fft_hr(
        self,
        signal,
        fs
    ):

        n = len(signal)

        fft = np.abs(
            np.fft.rfft(signal)
        )

        freqs = np.fft.rfftfreq(
            n,
            d=1 / fs
        )

        valid = np.logical_and(
            freqs >= LOW_HR_HZ,
            freqs <= HIGH_HR_HZ
        )

        if np.sum(valid) == 0:
            return None

        peak_freq = freqs[
            valid
        ][
            np.argmax(
                fft[valid]
            )
        ]

        hr = peak_freq * 60

        return round(
            float(hr),
            1
        )