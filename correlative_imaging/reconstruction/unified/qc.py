"""Step 1, checks per well, before anything is learnt.

  name       the barcode in the image file names differs from the plate folder (folders are trusted)
  few cells  the well holds far fewer surrounding cells than its plate: fewer than a quarter of the plate's median.
             Usually an imaging failure (lost focus, empty image) - or a well seeded without cells. The workflow cannot
             tell which: the user decides (experiment.json: exclude_wells or empty_wells). Until then such wells are
             left out of the learning but still called, with the flag.
  no hole    the image has no hole
  user       listed by the user (exclude_wells / empty_wells)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .experiment import matches


def check(T: dict, P: pd.DataFrame, e: dict) -> pd.DataFrame:
    im = T["images"][["plate_id", "image_id", "well_id", "filename", "file_barcode"]].copy()
    holes = set(zip(T["holes"].plate_id, T["holes"].image_id))
    field = P[(P.roi_mask == "whole_image") & (P.area_um2 >= 20)]
    n = field.groupby(["plate_id", "image_id"]).size().rename("n_objects")
    W = im.merge(n.reset_index(), on=["plate_id", "image_id"], how="left").fillna({"n_objects": 0})
    W["plate_median"] = W.groupby("plate_id").n_objects.transform("median")
    W["flag_name"] = W.file_barcode.notna() & (W.file_barcode != W.plate_id)
    W["flag_few_cells"] = W.n_objects < 0.25 * W.plate_median
    W["flag_no_hole"] = [(p, i) not in holes for p, i in zip(W.plate_id, W.image_id)]
    W["user_excluded"] = [matches(e.get("exclude_wells"), p, w) for p, w in zip(W.plate_id, W.well_id)]
    W["user_empty"] = [matches(e.get("empty_wells"), p, w) for p, w in zip(W.plate_id, W.well_id)]
    # learning uses wells that pass; calls skip only user-excluded wells and wells without a hole
    W["use_for_learning"] = ~W.flag_few_cells & ~W.user_excluded & ~W.user_empty
    W["call"] = ~W.user_excluded & ~W.flag_no_hole
    W["status"] = np.select(
        [W.user_excluded, W.flag_no_hole, W.user_empty, W.flag_few_cells],
        ["excluded (experiment.json)", "no hole", "empty well (experiment.json)", "few cells: check (focus? empty?)"], "ok")
    return W
