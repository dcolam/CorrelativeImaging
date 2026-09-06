"""Naming-scheme parsing and plate scanning across acquisition systems."""

import pytest

from correlative_imaging.io.naming import (
    BUILTIN_SCHEMES, NamingScheme, detect_scheme,
    load_custom_schemes, save_custom_schemes,
)
from correlative_imaging.io.plate import discover_plate_folders, scan_plate_folder


@pytest.fixture
def vsi_plate(tmp_path):
    """Olympus layout: flat folder, BF/FL distinguished only by serial order."""
    d = tmp_path / "PlateA"
    d.mkdir()
    for well in ("B10", "C2", "P24"):
        for serial in ("00001", "00002"):
            (d / f"__ROMK_18T39412_{well}-1_{serial}.vsi").touch()
    (d / "unrelated.vsi").touch()
    return d


@pytest.fixture
def ome_plates(tmp_path):
    """convert_new_md.py layout: role sub-folders under one folder per plate."""
    root = tmp_path / "converted"
    for ds in ("DATASET1", "DATASET2"):
        for role, sfx in (("brightfield", "bf"), ("fluorescence", "fl")):
            sub = root / ds / role
            sub.mkdir(parents=True)
            for well, idx in (("A01", "001"), ("B01", "025")):
                (sub / f"{ds}_{well}_id{idx}_s00_{sfx}.ome.tiff").touch()
    return root


# ── Olympus VSI must behave exactly as before the scheme refactor ──────

def test_vsi_pairs_by_serial_order(vsi_plate):
    wells = scan_plate_folder(vsi_plate)          # no scheme → historical default
    assert [w.well_id for w in wells] == ["B10", "C2", "P24"]
    assert all(w.is_complete for w in wells)
    assert wells[0].bf_serial == 1 and wells[0].fl_serial == 2
    assert wells[0].bf_path.name.endswith("00001.vsi")


def test_vsi_single_plate_folder(vsi_plate):
    assert list(discover_plate_folders(vsi_plate)) == ["PlateA"]
    assert list(discover_plate_folders(vsi_plate.parent)) == ["PlateA"]


def test_vsi_extension_override_still_honoured(tmp_path):
    d = tmp_path / "P"
    d.mkdir()
    (d / "__x_A01-1_00001.tif").touch()
    (d / "__x_A01-1_00002.tif").touch()
    assert scan_plate_folder(d) == []             # .vsi default finds nothing
    wells = scan_plate_folder(d, extension=".tif")
    assert len(wells) == 1 and wells[0].is_complete


# ── ImageXpress OME output ────────────────────────────────────────────

def test_ome_pairs_across_role_subfolders(ome_plates):
    wells = scan_plate_folder(ome_plates / "DATASET1", scheme="imagexpress_ome")
    assert [w.well_id for w in wells] == ["A1", "B1"]     # int column, matches the grid
    assert all(w.is_complete for w in wells)
    w = wells[0]
    assert "brightfield" in str(w.bf_path) and "fluorescence" in str(w.fl_path)


def test_ome_role_subfolders_are_not_mistaken_for_plates(ome_plates):
    """The trap: brightfield/ and fluorescence/ must never count as plates."""
    one = discover_plate_folders(ome_plates / "DATASET1", scheme="imagexpress_ome")
    assert list(one) == ["DATASET1"]
    many = discover_plate_folders(ome_plates, scheme="imagexpress_ome")
    assert list(many) == ["DATASET1", "DATASET2"]


def test_ome_role_read_from_filename_without_subfolders(tmp_path):
    d = tmp_path / "flat"
    d.mkdir()
    (d / "X_A01_id001_s00_bf.ome.tiff").touch()
    (d / "X_A01_id001_s00_fl.ome.tiff").touch()
    wells = scan_plate_folder(d, scheme="imagexpress_ome")
    assert len(wells) == 1 and wells[0].is_complete
    assert wells[0].bf_path.name.endswith("_bf.ome.tiff")


# ── Auto-detection ────────────────────────────────────────────────────

def test_detect_picks_the_right_scheme(vsi_plate, ome_plates):
    assert detect_scheme(vsi_plate)[0].scheme.key == "olympus_vsi"
    assert detect_scheme(ome_plates)[0].scheme.key == "imagexpress_ome"


def test_detect_reports_zero_rather_than_guessing(tmp_path):
    (tmp_path / "notes.txt").touch()
    assert all(s.n_parsed == 0 for s in detect_scheme(tmp_path))


# ── Custom schemes ────────────────────────────────────────────────────

@pytest.fixture
def token_scheme():
    return NamingScheme(
        key="mysystem", label="My system",
        pattern=r"^(?P<plate>[^_]+)_(?P<row>[A-P])(?P<col>\d+)_f(?P<field>\d+)_(?P<role>ch1|ch2)$",
        extensions=(".tif",), role_map={"ch1": "bf", "ch2": "fl"},
        plate_from="group",
    )


def test_custom_scheme_round_trip(tmp_path, token_scheme):
    f = tmp_path / "schemes.json"
    save_custom_schemes(f, {token_scheme.key: token_scheme})
    back = load_custom_schemes(f)["mysystem"]
    assert back.pattern == token_scheme.pattern
    assert back.role_map["ch2"] == "fl"
    assert back.plate_from == "group"


def test_plate_identity_from_a_filename_token(tmp_path, token_scheme):
    d = tmp_path / "flat"
    d.mkdir()
    for plate in ("PLATE7", "PLATE8"):
        for well in ("A01", "H12"):
            for ch in ("ch1", "ch2"):
                (d / f"{plate}_{well}_f1_{ch}.tif").touch()

    assert list(discover_plate_folders(d, scheme=token_scheme)) == ["PLATE7", "PLATE8"]
    wells = scan_plate_folder(d, scheme=token_scheme, plate_token="PLATE7")
    assert [w.well_id for w in wells] == ["A1", "H12"]
    assert all(w.is_complete for w in wells)
    # The token filter must not leak the other plate's files into this one.
    assert all("PLATE7" in w.bf_path.name for w in wells)


def test_builtin_schemes_compile():
    for key, sc in BUILTIN_SCHEMES.items():
        assert sc.key == key and sc.description


# ── Field numbering and ROI matching ──────────────────────────────────

def test_fields_are_1_based_on_every_scheme(tmp_path, vsi_plate):
    """_s00 is the first site, and so is -1 on the Olympus layout: both must
    surface as field 1, because ROI filenames embed the field number."""
    assert {w.field for w in scan_plate_folder(vsi_plate)} == {1}

    d = tmp_path / "ome"
    d.mkdir()
    (d / "X_A01_id001_s00_bf.ome.tiff").touch()
    (d / "X_A01_id001_s01_bf.ome.tiff").touch()
    wells = scan_plate_folder(d, scheme="imagexpress_ome")
    assert sorted(w.field for w in wells) == [1, 2]


def test_bf_pipeline_rois_are_found_under_a_non_default_scheme(tmp_path):
    """The BF pipeline names ROIs _<ROW><COL>-<field>_<kind>.roi from the
    WellInfo, so they must attach regardless of which scheme found the images."""
    plate = tmp_path / "P"
    (plate / "brightfield").mkdir(parents=True)
    (plate / "fluorescence").mkdir(parents=True)
    (plate / "brightfield" / "X_A01_id001_s00_bf.ome.tiff").touch()
    (plate / "fluorescence" / "X_A01_id001_s00_fl.ome.tiff").touch()
    rois = tmp_path / "rois"
    rois.mkdir()
    (rois / "_A1-1_hole.roi").touch()

    wells = scan_plate_folder(plate, scheme="imagexpress_ome", extra_roi_dirs=[rois])
    assert len(wells) == 1
    assert [p.name for p in wells[0].roi_paths] == ["_A1-1_hole.roi"]


def test_user_rois_named_like_the_images_are_found(tmp_path):
    """An ROI carrying the acquisition system's own naming, not this GUI's."""
    plate = tmp_path / "P"
    plate.mkdir()
    (plate / "X_A01_id001_s00_bf.ome.tiff").touch()
    (plate / "X_A01_id001_s00_fl.ome.tiff").touch()
    rois = tmp_path / "rois"
    rois.mkdir()
    (rois / "X_A01_id001_s00_hole.roi").touch()

    wells = scan_plate_folder(plate, scheme="imagexpress_ome", extra_roi_dirs=[rois])
    assert [p.name for p in wells[0].roi_paths] == ["X_A01_id001_s00_hole.roi"]


def test_ambiguous_roi_is_skipped_not_guessed(tmp_path):
    """Two fields in one well and an ROI naming no field: attaching it to an
    arbitrary field would silently analyse the wrong image."""
    plate = tmp_path / "P"
    plate.mkdir()
    for site in ("s00", "s01"):
        (plate / f"X_A01_id001_{site}_bf.ome.tiff").touch()
        (plate / f"X_A01_id001_{site}_fl.ome.tiff").touch()
    rois = tmp_path / "rois"
    rois.mkdir()
    (rois / "some_A01_thing.roi").touch()   # no field token

    wells = scan_plate_folder(plate, scheme="imagexpress_ome", extra_roi_dirs=[rois])
    assert len(wells) == 2
    assert all(not w.roi_paths for w in wells)


def test_image_named_rois_do_not_become_user_tags(tmp_path):
    """An ROI named after its image must not be mistaken for a user-tagged
    'well_existing' selection just because the stem contains '__'."""
    from correlative_imaging.io.naming import get_scheme
    scheme = get_scheme("imagexpress_ome")
    assert scheme.parse_text("X__run_A01_id001_s00_hole", tmp_path / "x.roi") is not None
    assert scheme.parse_text("myTissue__outline", tmp_path / "y.roi") is None
