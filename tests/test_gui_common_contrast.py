"""The shared auto-contrast rule (ci gui and ci reconstruct)."""
import numpy as np

from correlative_imaging.gui_common.contrast import contrast_rule


def test_black_point_is_the_minimum_white_point_the_999th_percentile():
    a = np.arange(10001, dtype=float)
    lo, hi = contrast_rule(a)
    assert lo == 0 and hi == np.percentile(a, 99.9)


def test_dark_channel_spans_at_least_100():
    lo, hi = contrast_rule(np.full(100, 7.0))
    assert (lo, hi) == (7.0, 107.0)


def test_non_finite_values_ignored_and_empty_is_safe():
    assert contrast_rule(np.array([np.nan, 5.0, np.inf, 20.0]))[0] == 5.0
    assert contrast_rule(np.array([np.nan])) == (0.0, 100.0)
