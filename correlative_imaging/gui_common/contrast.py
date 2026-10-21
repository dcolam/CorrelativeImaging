"""The one auto-contrast rule of the napari tools (ci gui and ci reconstruct): black point = the minimum, white point =
the 99.9th percentile but at least minimum + 100 (a dark channel is not stretched into noise). Pure numpy."""
from __future__ import annotations

import numpy as np


def contrast_rule(a: np.ndarray, min_span: float = 100.0) -> tuple:
    """(minimum, max(99.9th percentile, minimum + min_span)) over the finite values."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return 0.0, float(min_span)
    lo = float(a.min())
    return lo, max(float(np.percentile(a, 99.9)), lo + min_span)
