from expression_detector import ExpressionDetector


def test_smile_reads_as_happy():
    d = ExpressionDetector()
    r = d.predict({"mouthSmileLeft": .9, "mouthSmileRight": .9,
                   "cheekSquintLeft": .6, "cheekSquintRight": .6})
    assert r.expression == "happy" and r.valid


def test_brow_down_reads_as_angry():
    d = ExpressionDetector()
    r = d.predict({"browDownLeft": .9, "browDownRight": .9,
                   "mouthPressLeft": .5, "mouthPressRight": .5})
    assert r.expression == "angry"


def test_resting_face_is_neutral():
    d = ExpressionDetector()
    r = d.predict({"mouthSmileLeft": .05, "browDownLeft": .08, "jawOpen": .03})
    assert r.expression == "neutral" and r.valid


def test_no_blendshapes_is_invalid():
    d = ExpressionDetector()
    assert not d.predict(None).valid
    assert not d.predict({}).valid


def test_brightness_no_longer_influences_anything():
    """The old detector called a dark image 'sad'. There is no path from
    pixel statistics to the output any more -- the input is blendshapes."""
    d = ExpressionDetector()
    assert d.predict({"mouthSmileLeft": .9, "mouthSmileRight": .9}).expression == "happy"
