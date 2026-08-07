"""Facial expression inference from MediaPipe blendshape coefficients.

Why this file replaces `emotion_detector.py`
--------------------------------------------
The previous detector classified "emotion" from image statistics::

    if brightness < 80:   emotion = "sad"
    elif contrast > 60:   emotion = "angry"
    else:                 emotion = "neutral"

That measures the *lighting*, not the person. Dim the room and everyone
becomes sad; wear a striped shirt and everyone becomes angry. There is no
causal path from those pixels to an emotional state, so the output was not
merely inaccurate -- it was meaningless.

This module instead reads the 52 blendshape coefficients that FaceLandmarker
already produces (no extra model, no extra inference) and combines them into
expression scores using the Action Units those blendshapes correspond to.

It is named *expression*, not *emotion*, on purpose. Blendshapes describe
what the face is doing (AU12 "lip corner puller"), not what the person feels.
A polite smile and genuine delight are the same signal here. Treat the output
as a facial-movement estimate, nothing more.
"""

from __future__ import annotations

from typing import NamedTuple

from config import EXPRESSION_MIN_SCORE

# Expression -> the blendshapes that make it up, with weights.
# Groupings follow the classic FACS Action Unit descriptions of each
# prototypical expression (Ekman & Friesen).
EXPRESSION_RULES: dict[str, dict[str, float]] = {
    # AU6 cheek raiser + AU12 lip corner puller
    "happy": {
        "mouthSmileLeft": 1.0,
        "mouthSmileRight": 1.0,
        "cheekSquintLeft": 0.5,
        "cheekSquintRight": 0.5,
    },
    # AU1 inner brow raiser + AU15 lip corner depressor
    "sad": {
        "mouthFrownLeft": 1.0,
        "mouthFrownRight": 1.0,
        "browInnerUp": 0.7,
        "mouthShrugLower": 0.3,
    },
    # AU4 brow lowerer + AU9 nose wrinkler + lip press
    "angry": {
        "browDownLeft": 1.0,
        "browDownRight": 1.0,
        "noseSneerLeft": 0.4,
        "noseSneerRight": 0.4,
        "mouthPressLeft": 0.3,
        "mouthPressRight": 0.3,
    },
    # AU1+2 brow raiser + AU5 upper lid raiser + AU26 jaw drop
    "surprised": {
        "browInnerUp": 0.8,
        "browOuterUpLeft": 0.8,
        "browOuterUpRight": 0.8,
        "eyeWideLeft": 0.6,
        "eyeWideRight": 0.6,
        "jawOpen": 1.0,
    },
    # AU1+2+5 with AU20 lip stretcher -- surprise plus horizontal mouth pull
    "fearful": {
        "browInnerUp": 0.8,
        "eyeWideLeft": 0.7,
        "eyeWideRight": 0.7,
        "mouthStretchLeft": 1.0,
        "mouthStretchRight": 1.0,
    },
    # AU9 nose wrinkler + AU10 upper lip raiser
    "disgusted": {
        "noseSneerLeft": 1.0,
        "noseSneerRight": 1.0,
        "mouthUpperUpLeft": 0.7,
        "mouthUpperUpRight": 0.7,
    },
}


class ExpressionResult(NamedTuple):
    expression: str
    score: float
    scores: dict[str, float]
    valid: bool

    @property
    def display(self) -> str:
        if not self.valid:
            return "--"
        if self.expression == "neutral":
            return "neutral"
        return f"{self.expression} ({self.score:.2f})"


class ExpressionDetector:
    """Stateless mapping from blendshapes to an expression label."""

    def __init__(self, min_score: float = EXPRESSION_MIN_SCORE):
        self.min_score = float(min_score)

    def predict(self, blendshapes: dict[str, float] | None) -> ExpressionResult:
        if not blendshapes:
            return ExpressionResult("neutral", 0.0, {}, valid=False)

        scores: dict[str, float] = {}
        for name, components in EXPRESSION_RULES.items():
            total_weight = sum(components.values())
            if total_weight <= 0:
                continue
            weighted = sum(
                blendshapes.get(key, 0.0) * weight
                for key, weight in components.items()
            )
            scores[name] = weighted / total_weight

        if not scores:
            return ExpressionResult("neutral", 0.0, {}, valid=False)

        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < self.min_score:
            return ExpressionResult("neutral", best_score, scores, valid=True)
        return ExpressionResult(best, round(best_score, 3), scores, valid=True)
