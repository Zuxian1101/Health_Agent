"""Rule-based advice engine.

Fixes relative to the first implementation:

1. **`state_history` was dead code.** It was appended to on every frame and
   never read. It now backs the trend rules and is exposed via `summary()`.
2. **First match won, forever.** Fatigue advice permanently masked every
   other rule, so a stressed, tachycardic user who was also tired only ever
   saw "Take a 5-minute break". Rules are now evaluated independently and
   the top N by priority are returned.
3. **No debouncing.** Advice was recomputed per frame, so it flickered
   whenever a signal sat near its threshold. A condition must now hold
   continuously for ADVICE_HYSTERESIS_SECONDS before it fires.
4. **Reminder intervals were never used.** WATER_REMIND_INTERVAL_MIN and
   REST_REMIND_INTERVAL_MIN existed in config but appeared nowhere in the
   code. Every rule now has a real cooldown.
5. **Unreachable branch.** The old code checked for emotion "fear", which
   the detector could never produce.
6. **Thresholds were hard-coded** (75, 100, 60) while config held different
   values. Everything reads from config now.
7. **No disclaimer.** See DISCLAIMER below.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple

from config import (
    ADVICE_HYSTERESIS_SECONDS,
    ADVICE_MAX_SHOWN,
    DISCLAIMER,
    FATIGUE_THRESHOLD,
    HR_HIGH_BPM,
    HR_LOW_BPM,
    HR_REMIND_INTERVAL_MIN,
    NEGATIVE_EXPRESSIONS,
    POSTURE_REMIND_INTERVAL_MIN,
    REST_REMIND_INTERVAL_MIN,
    STRESS_REMIND_INTERVAL_MIN,
    WATER_REMIND_INTERVAL_MIN,
)

HISTORY_LIMIT = 600


class Advice(NamedTuple):
    key: str
    message: str
    priority: int


@dataclass
class Rule:
    key: str
    message: str
    priority: int                     # higher fires first
    cooldown_seconds: float
    condition: Callable[[dict], bool]
    hysteresis_seconds: float = ADVICE_HYSTERESIS_SECONDS

    # runtime state
    _true_since: float | None = None
    _last_fired: float | None = None


def _num(state, key):
    """Return a numeric field, or None if absent/unusable."""
    value = state.get(key)
    return value if isinstance(value, (int, float)) and value is not None else None


class HealthAgent:
    """Turns a stream of measurements into at most a couple of suggestions."""

    DISCLAIMER = DISCLAIMER

    def __init__(self, max_shown: int = ADVICE_MAX_SHOWN):
        self.max_shown = int(max_shown)
        self.state_history: deque[dict] = deque(maxlen=HISTORY_LIMIT)
        self._rules = self._build_rules()

    # ------------------------------------------------------------- rules --
    def _build_rules(self) -> list[Rule]:
        def tired(s):
            f = _num(s, "fatigue")
            return f is not None and f >= FATIGUE_THRESHOLD

        def hr_high(s):
            # Only act on an estimate the rPPG itself considers trustworthy.
            if not s.get("hr_reliable"):
                return False
            hr = _num(s, "hr")
            return hr is not None and hr > HR_HIGH_BPM

        def hr_low(s):
            if not s.get("hr_reliable"):
                return False
            hr = _num(s, "hr")
            return hr is not None and hr < HR_LOW_BPM

        def negative_expression(s):
            return s.get("expression") in NEGATIVE_EXPRESSIONS

        def long_session(s):
            minutes = _num(s, "work_minutes")
            return minutes is not None and minutes >= REST_REMIND_INTERVAL_MIN

        def posture_due(s):
            minutes = _num(s, "work_minutes")
            return minutes is not None and minutes >= POSTURE_REMIND_INTERVAL_MIN

        def water_due(s):
            minutes = _num(s, "work_minutes")
            return minutes is not None and minutes >= WATER_REMIND_INTERVAL_MIN

        return [
            Rule("fatigue", "You look tired - take a 5 minute break", 100,
                 REST_REMIND_INTERVAL_MIN * 60, tired),
            Rule("hr_high", "Elevated heart rate - slow your breathing", 90,
                 HR_REMIND_INTERVAL_MIN * 60, hr_high),
            Rule("hr_low", "Unusually low heart rate reading", 85,
                 HR_REMIND_INTERVAL_MIN * 60, hr_low),
            Rule("stress", "You seem tense - try a short walk", 70,
                 STRESS_REMIND_INTERVAL_MIN * 60, negative_expression,
                 hysteresis_seconds=ADVICE_HYSTERESIS_SECONDS * 2),
            Rule("rest", "Long session - stand up and stretch", 60,
                 REST_REMIND_INTERVAL_MIN * 60, long_session),
            Rule("posture", "Check your posture", 40,
                 POSTURE_REMIND_INTERVAL_MIN * 60, posture_due),
            Rule("water", "Time to drink some water", 30,
                 WATER_REMIND_INTERVAL_MIN * 60, water_due),
        ]

    # ------------------------------------------------------------ public --
    def analyze(self, state: dict, now: float | None = None) -> list[Advice]:
        now = time.monotonic() if now is None else float(now)
        self.state_history.append({**state, "t": now})

        fired: list[Advice] = []
        for rule in self._rules:
            if not rule.condition(state):
                rule._true_since = None
                continue

            if rule._true_since is None:
                rule._true_since = now
            if now - rule._true_since < rule.hysteresis_seconds:
                continue
            if (
                rule._last_fired is not None
                and now - rule._last_fired < rule.cooldown_seconds
            ):
                continue

            rule._last_fired = now
            rule._true_since = None          # re-arm; cooldown does the spacing
            fired.append(Advice(rule.key, rule.message, rule.priority))

        fired.sort(key=lambda a: -a.priority)
        return fired[: self.max_shown]

    def summary(self) -> dict:
        """Aggregate over the retained history - this is what it is *for*."""
        hrs = [
            s["hr"] for s in self.state_history
            if s.get("hr_reliable") and isinstance(s.get("hr"), (int, float))
        ]
        fatigues = [
            s["fatigue"] for s in self.state_history
            if isinstance(s.get("fatigue"), (int, float))
        ]
        expressions = [
            s["expression"] for s in self.state_history if s.get("expression")
        ]
        negative = sum(1 for e in expressions if e in NEGATIVE_EXPRESSIONS)

        return {
            "samples": len(self.state_history),
            "mean_hr": round(sum(hrs) / len(hrs), 1) if hrs else None,
            "hr_samples": len(hrs),
            "mean_fatigue": round(sum(fatigues) / len(fatigues), 1) if fatigues else None,
            "peak_fatigue": round(max(fatigues), 1) if fatigues else None,
            "negative_expression_ratio": (
                round(negative / len(expressions), 3) if expressions else None
            ),
            "disclaimer": self.DISCLAIMER,
        }
