"""Un-projected (Z-stack) diagnostic crops."""

import numpy as np
import tifffile

from correlative_imaging.batch import _save_zstack_crop
from correlative_imaging.pipeline.preprocess import bin_array


def _stack():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, size=(3, 5, 40, 60), dtype=np.uint16)   # C, Z, Y, X


def test_crop_keeps_every_plane_and_raw_values(tmp_path):
    raw = _stack()
    out = tmp_path / "A01_hole_crop_zstack.tif"
    assert _save_zstack_crop(raw, (10, 20, 5, 25), 1, "mean", ["DAPI", "GFP", "RFP"],
                             ["blue", "green", "red"], 0.65, 2.0, out)
    with tifffile.TiffFile(out) as tf:
        arr = tf.asarray()
        ij = tf.imagej_metadata
    assert arr.shape == (5, 3, 10, 20)                       # Z, C, Y, X
    assert arr.dtype == np.uint16
    np.testing.assert_array_equal(arr, raw[:, :, 10:20, 5:25].transpose(1, 0, 2, 3))
    assert ij["spacing"] == 2.0 and ij["channels"] == 3 and ij["slices"] == 5


def test_binned_pipeline_grid_maps_back_to_raw(tmp_path):
    raw = _stack()
    out = tmp_path / "crop.tif"
    # bbox on a 2x-binned grid → raw rows 8:20, cols 4:24, then binned
    assert _save_zstack_crop(raw, (4, 10, 2, 12), 2, "mean", ["a", "b", "c"], [],
                             1.3, 2.0, out)
    arr = tifffile.imread(out)
    expected = bin_array(raw[:, :, 8:20, 4:24], 2, "mean").astype(np.uint16)
    np.testing.assert_array_equal(arr, expected.transpose(1, 0, 2, 3))


def test_2d_image_writes_nothing(tmp_path):
    out = tmp_path / "crop.tif"
    assert not _save_zstack_crop(np.zeros((2, 30, 30), np.uint16), (0, 10, 0, 10), 1,
                                 "mean", ["a", "b"], [], 1.0, 1.0, out)
    assert not out.exists()
