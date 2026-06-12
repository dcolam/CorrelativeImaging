"""BF-image pipeline steps: Z-projection and Ilastik-based ROI extraction."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import PipelineContext, Step, StepResult, register_step

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Z-projection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
@register_step
class ZProjection(Step):
    """Collapse a Z-stack channel to 2-D using a specified projection method.

    Parameters
    ----------
    channel:  Channel index to project.  -1 = all channels.
    method:   'min' | 'max' | 'mean' | 'sum'  (default 'min', best for BF).
    """
    channel: int = -1
    method: str = "min"

    @property
    def name(self) -> str:
        ch = "all" if self.channel == -1 else f"ch{self.channel}"
        return f"z_projection_{self.method}_{ch}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        if image.ndim != 4:
            return StepResult()   # already 2-D — nothing to do

        ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
        fn = ops.get(self.method, np.min)

        out = image.copy()
        channels = range(out.shape[0]) if self.channel == -1 else [self.channel]
        for c in channels:
            proj = fn(out[c], axis=0)   # (Z,Y,X) → (Y,X)
            out[c] = proj[np.newaxis]   # keep shape (1,Y,X) for one Z-slice

        # Squeeze the trivial Z axis so downstream steps see (C,Y,X)
        if out.shape[1] == 1:
            out = out[:, 0, :, :]

        return StepResult(image=out)


# ──────────────────────────────────────────────────────────────────────────────
# Ilastik ROI extraction
# ──────────────────────────────────────────────────────────────────────────────

def _find_ilastik() -> str | None:
    """Return the path to the ilastik executable, or None if not found."""
    # 1. Explicit env variable
    env = os.environ.get("ILASTIK_PATH")
    if env and Path(env).exists():
        return env
    # 2. Common install locations (Windows / Linux / macOS)
    candidates = [
        r"C:\Program Files\ilastik-1.4.2\ilastik.exe",
        r"C:\Program Files\ilastik-1.4.0\ilastik.exe",
        r"C:\Program Files\ilastik-1.3.3post3\ilastik.exe",
        "/usr/bin/ilastik",
        "/opt/ilastik/run_ilastik.sh",
        str(Path.home() / "ilastik" / "run_ilastik.sh"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # 3. PATH
    return shutil.which("ilastik") or shutil.which("run_ilastik.sh")


# Alias used by gui.py worker
_find_ilastik_exe = _find_ilastik


def _best_sub_roi(prob_map: np.ndarray, threshold: float = 0.5,
                  min_area: int = 500, min_circularity: float = 0.1) -> np.ndarray:
    """Return a binary mask containing only the best (largest × most circular)
    connected component from the thresholded probability map.

    Parameters
    ----------
    prob_map:        2-D float array of foreground probabilities.
    threshold:       Binarisation threshold (default 0.5).
    min_area:        Reject components smaller than this (pixels).
    min_circularity: Reject components below this circularity score.
    """
    from skimage.measure import label, regionprops

    binary = prob_map >= threshold
    labeled = label(binary)
    props = regionprops(labeled)

    if not props:
        log.warning("IlastikROI: no foreground components found at threshold %.2f", threshold)
        return np.zeros_like(binary, dtype=np.uint8)

    best = None
    best_score = -1.0
    for p in props:
        if p.area < min_area:
            continue
        # circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        circ = (4 * np.pi * p.area / p.perimeter ** 2) if p.perimeter > 0 else 0.0
        if circ < min_circularity:
            continue
        score = p.area * circ
        if score > best_score:
            best_score = score
            best = p.label

    if best is None:
        log.warning("IlastikROI: no component passed area/circularity filters; "
                    "returning largest component.")
        best = max(props, key=lambda p: p.area).label

    return (labeled == best).astype(np.uint8)


@dataclass
@register_step
class IlastikROI(Step):
    """Extract a cell/organoid ROI from a BF image using an Ilastik pixel classifier.

    Workflow
    --------
    1. Write the selected channel (2-D, already projected) to a temp TIFF.
    2. Call ``ilastik --headless`` to produce an HDF5 probability map.
    3. Threshold the foreground probability channel.
    4. Select the best connected component (largest × most circular).
    5. Store the binary mask in ``context.masks[roi_name]``.

    Parameters
    ----------
    ilp_path:        Path to the trained ``.ilp`` project file.
    channel:         Channel index of the brightfield image (default 0).
    roi_name:        Key used in ``context.masks`` (default ``'roi'``).
    ilastik_exe:     Path to the ilastik executable.  Empty = auto-detect.
    threshold:       Foreground probability threshold (default 0.5).
    fg_channel:      Which output channel of Ilastik is the foreground class
                     (0-indexed, default 1 — Ilastik labels 0=background, 1=foreground).
    min_area_px:     Minimum component area in pixels for best-ROI selection.
    min_circularity: Minimum circularity score (0–1) for best-ROI selection.
    """
    ilp_path: str
    channel: int = 0
    roi_name: str = "roi"
    ilastik_exe: str = ""
    threshold: float = 0.5
    fg_channel: int = 1
    min_area_px: int = 500
    min_circularity: float = 0.1

    @property
    def name(self) -> str:
        return f"ilastik_roi_{self.roi_name}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        import tifffile

        exe = self.ilastik_exe or _find_ilastik()
        if not exe:
            raise RuntimeError(
                "Ilastik executable not found. "
                "Set ILASTIK_PATH env variable or pass ilastik_exe parameter."
            )

        ilp = Path(self.ilp_path)
        if not ilp.exists():
            raise FileNotFoundError(f"Ilastik project not found: {ilp}")

        # Extract the channel — handle (C,Y,X) and (C,Z,Y,X)
        ch_data = image[self.channel]
        if ch_data.ndim == 3:
            # Still has Z — take min projection (best for BF)
            ch_data = ch_data.min(axis=0)

        # Normalise to uint8 for Ilastik input
        mn, mx = ch_data.min(), ch_data.max()
        if mx > mn:
            ch_u8 = ((ch_data - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            ch_u8 = np.zeros_like(ch_data, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_tiff  = tmp / "input.tif"
            out_h5   = tmp / "output.h5"

            tifffile.imwrite(str(in_tiff), ch_u8)

            cmd = [
                exe,
                "--headless",
                f"--project={ilp}",
                "--export_source=Probabilities",
                "--output_format=hdf5",
                f"--output_filename_format={out_h5}",
                str(in_tiff),
            ]

            log.info("Running Ilastik: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Ilastik failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
                )

            # Read HDF5 probability map
            import h5py
            with h5py.File(out_h5, "r") as f:
                # Ilastik exports under 'exported_data'; shape varies by version
                key = list(f.keys())[0]
                prob = f[key][()]   # shape: (Y, X, n_classes) or (1, Y, X, n_classes)

        # Squeeze batch/channel dims → (Y, X, n_classes)
        while prob.ndim > 3:
            prob = prob[0]

        fg_prob = prob[:, :, self.fg_channel].astype(np.float32)
        mask = _best_sub_roi(
            fg_prob,
            threshold=self.threshold,
            min_area=self.min_area_px,
            min_circularity=self.min_circularity,
        )

        log.info("IlastikROI '%s': mask coverage %.1f%%",
                 self.roi_name, mask.mean() * 100)

        return StepResult(masks={self.roi_name: mask})
