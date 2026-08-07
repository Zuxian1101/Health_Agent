# Health_Agent

Real-time estimation of a few health-adjacent signals from a webcam:
heart rate (rPPG), facial expression, and drowsiness (PERCLOS) — with a
small rule engine that turns those into break reminders.

> **This is not a medical device.** Heart rate is inferred from subtle
> colour changes in webcam pixels, not measured. Expression is inferred from
> facial geometry, not from how you feel. Do not use any output here to make
> a health decision.

---

## Install

```bash
git clone https://github.com/ZuxianHe/Health_Agent.git
cd Health_Agent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python download_model.py        # one-time, ~3 MB face model
```

## Run

```bash
cd health_agent
python main.py
```

Press `ESC` or `q` to quit. A JSON session summary is printed on exit.

```
python main.py --camera 1          # pick a different webcam
python main.py --duration 60       # stop after a minute
python main.py --no-display        # headless, summary only
```

macOS will ask for camera permission for whichever terminal you launch from.

---

## What each module does

| Module | Signal | Method |
|---|---|---|
| `face_detector.py` | landmarks, blendshapes, ROIs | MediaPipe Tasks FaceLandmarker — **one** inference per frame, shared by everything downstream |
| `rppg_pos.py` | heart rate | POS (Wang et al. 2017) on the forehead patch, band-limited to 45–180 BPM, with an SNR-based reliability flag |
| `fatigue_detector.py` | drowsiness | PERCLOS over a 60 s window, plus rolling blink rate, against a per-person calibrated EAR baseline |
| `expression_detector.py` | facial expression | 52 blendshape coefficients grouped by the FACS Action Units of each prototypical expression |
| `health_agent.py` | advice | Rules with hysteresis and per-rule cooldowns |

All thresholds live in `config.py`.

## Calibration

The first ~5 seconds establish your open-eye EAR baseline — keep your eyes
normally open and face the camera. The heart rate needs ~6–10 seconds of
buffer before it reports anything, and shows `--` until then.

---

## Limits you should actually know about

**Heart rate**
- rPPG needs a still subject and stable, reasonably bright lighting. Head
  motion, talking, and flickering light all inject noise directly into the
  frequency band we care about.
- Accuracy degrades on darker skin tones and under low light — less signal
  reaches the sensor. This is a well-documented limitation of the whole
  method, not of this implementation.
- The reported SNR is your guide. Below ~2 dB the reading is greyed out and
  the rule engine ignores it. Trust that flag.
- Validated here only against synthetic signals (recovers a known frequency
  to within ~1 BPM). It has **not** been validated against a real pulse
  oximeter or ECG.

**Drowsiness**
- PERCLOS is a real, published measure, but the mapping from PERCLOS to a
  0–100 "fatigue" number is a convenience, not a clinical scale.
- Glasses, strong side lighting, and looking away all disturb EAR.

**Expression**
- Blendshapes describe what your face is *doing*, not what you *feel*. A
  polite smile and genuine delight produce the same signal. Expression is
  also culturally variable. Treat `expression` as "facial movement", and
  never as a claim about someone's emotional state.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The tests run offline — no camera, no model download. They verify the rPPG
recovers known synthetic frequencies, that PERCLOS separates alert from
drowsy, and that the rule engine's hysteresis and cooldowns hold.

## License

No license has been declared, which means default copyright applies and
others cannot legally reuse this. Add one (MIT is the usual choice for a
project like this) if that is not what you intend.
