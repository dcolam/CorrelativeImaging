"""Which ROI selections get cropped diagnostic images."""

import numpy as np
import pytest


def _cfg(**kw):
    cfg = {"whole": False, "crops": True, "crop_pad_um": 0.0,
           "formats": {"jpg"}, "colors": [], "stamp_rois": False,
           "multichannel_tif": True, "include_bf": False}
    cfg.update(kw)
    return cfg


def _run(tmp_path, crop_selections):
    """Drive _save_well_diagnostics with two ROIs and return the files written."""
    from correlative_imaging.batch import _save_well_diagnostics
    from correlative_imaging.io.plate import WellInfo
    from correlative_imaging.pipeline.base import PipelineContext

    hole = np.zeros((64, 64), bool); hole[20:30, 20:30] = True
    background = np.ones((64, 64), bool)          # the whole frame

    ctx = PipelineContext(channel_names=["ch0"], pixel_size_um=1.0)
    ctx.masks.update({"roi_hole": hole, "roi_background": background})

    pl_dict = {"steps": [
        {"type": "LoadROI", "path": "x", "roi_name": "roi_hole"},
        {"type": "LoadROI", "path": "y", "roi_name": "roi_background"},
    ]}
    out = tmp_path / "diagnostics"
    _save_well_diagnostics(
        well=WellInfo(row="A", col=1, field=1),
        final_image=np.zeros((1, 64, 64), dtype="float32"),
        pixel_size_um=1.0, pl_dict=pl_dict, context=ctx,
        diag_cfg=_cfg(crop_selections=crop_selections, output_dir=str(out)),
    )
    return sorted(f.name for f in out.iterdir()) if out.is_dir() else []


def test_none_crops_every_selection(tmp_path):
    """Historical behaviour: no explicit choice means crop them all."""
    names = _run(tmp_path, None)
    assert any("roi_hole_crop" in n for n in names)
    assert any("roi_background_crop" in n for n in names)


def test_only_the_chosen_selection_is_cropped(tmp_path):
    names = _run(tmp_path, ["roi_hole"])
    assert any("roi_hole_crop" in n for n in names)
    assert not any("roi_background" in n for n in names)


def test_empty_choice_writes_no_crops(tmp_path):
    assert _run(tmp_path, []) == []


def test_unknown_selection_names_write_nothing(tmp_path):
    """A stale saved choice naming selections this run doesn't have must not
    silently fall back to cropping everything."""
    assert _run(tmp_path, ["roi_typo"]) == []


def test_crop_is_tight_around_the_roi(tmp_path):
    """The point of the hole crop: it is the ROI's bounding box, not the frame."""
    import tifffile
    names = _run(tmp_path, ["roi_hole"])
    tif = next(n for n in names if n.endswith("_crop_channels.tif"))
    arr = tifffile.imread(str(tmp_path / "diagnostics" / tif))
    assert arr.shape[-2:] == (10, 10)      # the 20:30 square, no padding


def test_selection_list_excludes_whole_image_selections(qapp):
    """A 'whole image' selection has no mask key — offering it for cropping
    would produce a 'crop' the size of the frame."""
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget, _ROISel

    w = CorrelativeImagingWidget()
    w._roi_tab.add_selection(_ROISel(label="Hole", source="well_class", class_name="hole"))
    keys = [k for k, _ in w._run_tab._roi_selection_list()]
    assert "roi_hole" in keys
    assert "" not in keys


def test_channel_colors_reach_the_tif_as_imagej_luts(tmp_path):
    """The colours chosen in the Channels tab were baked into the JPG composite
    only — the real per-channel TIF opened grayscale."""
    import tifffile
    from correlative_imaging.batch import _save_multichannel_tif

    planes = [np.zeros((8, 8), "float32"), np.ones((8, 8), "float32")]
    f = tmp_path / "x.tif"
    _save_multichannel_tif(planes, ["DAPI", "GFP"], None, f, colors=["blue", "green"])

    with tifffile.TiffFile(f) as t:
        md = t.imagej_metadata
    assert md["mode"] == "composite" and md["Labels"] == ["DAPI", "GFP"]
    assert [int(c.max()) for c in md["LUTs"][0]] == [0, 0, 255]     # blue
    assert [int(c.max()) for c in md["LUTs"][1]] == [0, 255, 0]     # green


def test_appended_bf_channel_is_gray(tmp_path):
    import tifffile
    from correlative_imaging.batch import _save_multichannel_tif

    planes = [np.zeros((8, 8), "float32"), np.ones((8, 8), "float32")]
    f = tmp_path / "y.tif"
    _save_multichannel_tif(planes, ["DAPI", "GFP"], np.ones((8, 8), "float32"),
                           f, colors=["blue", "green"])
    with tifffile.TiffFile(f) as t:
        luts = t.imagej_metadata["LUTs"]
    assert len(luts) == 3
    assert [int(c.max()) for c in luts[2]] == [255, 255, 255]


def test_no_colors_still_writes_a_valid_tif(tmp_path):
    import tifffile
    from correlative_imaging.batch import _save_multichannel_tif

    f = tmp_path / "z.tif"
    _save_multichannel_tif([np.zeros((8, 8), "float32")], ["ch0"], None, f)
    assert tifffile.imread(str(f)).shape == (8, 8)
