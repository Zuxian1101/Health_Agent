"""Deprecated. Kept only so old imports fail loudly instead of silently.

The brightness/contrast "emotion" heuristic that used to live here was
measuring room lighting, not facial expression. It has been removed rather
than tuned, because no amount of tuning makes those pixels mean anything.

Use `expression_detector.ExpressionDetector` instead.
"""

import warnings

from expression_detector import ExpressionDetector, ExpressionResult  # noqa: F401


class EmotionDetector(ExpressionDetector):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "EmotionDetector is deprecated and no longer performs "
            "brightness-based classification. Use ExpressionDetector, and note "
            "that it reports facial expression, not emotional state.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
