from health_agent import HealthAgent

BASE = {"fatigue": 0, "hr": None, "hr_reliable": False,
        "expression": "neutral", "work_minutes": 0}


def state(**kw):
    return {**BASE, **kw}


def test_advice_requires_the_condition_to_persist():
    """Hysteresis: a single spiky frame must not fire advice."""
    a = HealthAgent()
    assert a.analyze(state(fatigue=95), now=0.0) == []
    assert a.analyze(state(fatigue=95), now=1.0) == []


def test_advice_fires_after_hysteresis():
    a = HealthAgent()
    fired_at = None
    t = 0.0
    for _ in range(400):
        t += 0.1
        if a.analyze(state(fatigue=85), now=t) and fired_at is None:
            fired_at = t
    assert fired_at is not None
    assert 9.0 < fired_at < 12.0


def test_cooldown_prevents_repeat_spam():
    a = HealthAgent()
    t = 0.0
    fired = 0
    for _ in range(3 * 3600):               # 3 hours at 1 Hz
        t += 1.0
        fired += sum(1 for x in a.analyze(state(fatigue=85), now=t)
                     if x.key == "fatigue")
    assert fired == 2                        # 90 min interval


def test_multiple_rules_can_fire_not_just_the_first():
    a = HealthAgent()
    t = 0.0
    seen = set()
    for _ in range(2000):
        t += 1.0
        for x in a.analyze(state(fatigue=85, hr=120, hr_reliable=True,
                                 expression="angry",
                                 work_minutes=t / 60), now=t):
            seen.add(x.key)
    assert {"fatigue", "hr_high", "stress"} <= seen


def test_unreliable_heart_rate_is_ignored():
    a = HealthAgent()
    t = 0.0
    hits = 0
    for _ in range(2000):
        t += 1.0
        hits += sum(1 for x in a.analyze(state(hr=140, hr_reliable=False), now=t)
                    if x.key == "hr_high")
    assert hits == 0


def test_history_is_actually_used():
    a = HealthAgent()
    t = 0.0
    for _ in range(50):
        t += 1.0
        a.analyze(state(fatigue=42, hr=80, hr_reliable=True), now=t)
    s = a.summary()
    assert s["samples"] == 50
    assert s["mean_hr"] == 80.0
    assert s["mean_fatigue"] == 42.0
    assert "disclaimer" in s


def test_missing_signals_do_not_crash():
    a = HealthAgent()
    assert a.analyze({}, now=0.0) == []
    assert a.analyze({"fatigue": None, "hr": None}, now=1.0) == []
