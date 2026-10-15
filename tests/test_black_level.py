"""BlackLevelNormalization: precise mode estimate, no clipping by default."""

import numpy as np
import pytest

from correlative_imaging.pipeline.base import PipelineContext
from correlative_imaging.pipeline.preprocess import BlackLevelNormalization


def _frame(dtype=np.float32):
    """Background 100 ± 3 (integer counts) with 1 % of the pixels saturated."""
    rng = np.random.default_rng(0)
    a = np.round(rng.normal(100, 3, (512, 512)))
    bright = rng.random(a.shape) < 0.01
    a[bright] = 60000
    return a.astype(dtype)[None], ~bright


def _run(img, **kw):
    return BlackLevelNormalization(0, **kw).process(img, PipelineContext(channel_names=["a"]))


def test_mode_precise_despite_bright_pixels():
    img, _ = _frame()
    assert _run(img).info["black_level"] == pytest.approx(100, abs=1)


def test_mode_precise_on_non_integer_data():
    img, _ = _frame()
    img = img / 4.0 + 0.123                           # e.g. after binning by mean
    assert _run(img).info["black_level"] == pytest.approx(25.123, abs=0.25)


def test_background_not_clipped():
    img, bg = _frame()
    out = _run(img).image[0][bg]
    # integer counts: ~13 % sit exactly on the black level, the rest split evenly
    assert (out < 0).mean() == pytest.approx((out > 0).mean(), abs=0.02)
    assert out.mean() == pytest.approx(0, abs=0.1)


def test_integer_input_gives_float_without_wraparound():
    img, bg = _frame(np.uint16)
    out = _run(img).image
    assert out.dtype == np.float32
    assert (out[0][bg] < 0).any()
    assert out[0][bg].max() < 100                     # nothing wrapped to ~65000


def test_clip_true_reproduces_old_behaviour():
    img, bg = _frame()
    res = _run(img, clip=True)
    black = res.info["black_level"]
    expected = np.clip(img[0] - black, 0, None)
    np.testing.assert_allclose(res.image[0], expected)
    assert res.image[0][bg].mean() > 1.0              # one-sided clip biases the mean up


def test_old_json_fields_still_load():
    from correlative_imaging.pipeline.base import Step
    step = Step.from_dict({"type": "BlackLevelNormalization", "channel": 0,
                           "method": "mode", "percentile": 1.0, "bins": 256, "target": 0.0})
    assert step.clip is False
