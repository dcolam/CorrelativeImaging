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

def select_z_range(
    arr: np.ndarray,
    z_start: int = 0,
    z_stop: int = 0,
    axis: int = 0,
    label: str = "",
) -> np.ndarray:
    """Restrict *arr* to a sub-stack along *axis* before projecting.

    The range is **1-based and inclusive**, matching how Fiji's "Make Substack"
    and the slice slider count planes: ``z_start=3, z_stop=7`` keeps planes 3–7,
    seven-minus-three-plus-one = 5 planes. ``0`` means unbounded, so
    ``z_start=0`` starts at the first plane and ``z_stop=0`` runs to the last —
    which makes "no range set" (0, 0) mean the whole stack, so pipelines saved
    before this option existed keep projecting exactly as they did.

    A range wider than the stack is clamped rather than raising: an 8-plane
    stack asked for 3–20 yields planes 3–8, logged once. An empty selection
    (start beyond the stack, or start > stop) is refused — silently projecting
    zero planes would produce an all-zero image that looks like a real result.
    """
    if arr.ndim <= axis or (not z_start and not z_stop):
        return arr
    n = arr.shape[axis]
    start = max(1, z_start or 1)
    stop = min(n, z_stop or n)
    if start > n or start > stop:
        raise ValueError(
            f"Z range {z_start or 1}–{z_stop or n} selects no planes "
            f"{('for ' + label) if label else ''}of a {n}-plane stack."
        )
    if (z_start and z_start > 1 and start != z_start) or (z_stop and stop != z_stop):
        log.warning("Z range %s–%s clamped to %d–%d (stack has %d planes)%s.",
                    z_start or 1, z_stop or n, start, stop, n,
                    f" for {label}" if label else "")
    if start == 1 and stop == n:
        return arr
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(start - 1, stop)
    return arr[tuple(sl)]


@dataclass
@register_step
class ZProjection(Step):
    """Collapse a Z-stack channel to 2-D using a specified projection method.

    Parameters
    ----------
    channel:  Channel index to project.  -1 = all channels.
    method:   'min' | 'max' | 'mean' | 'sum'  (default 'min', best for BF).
    z_start:  First Z plane to include, 1-based inclusive; 0 = first plane.
    z_stop:   Last Z plane to include, 1-based inclusive; 0 = last plane.
              ``z_start``/``z_stop`` default to 0/0 — the whole stack — so
              pipelines written before this option behave unchanged.
    """
    channel: int = -1
    method: str = "min"
    z_start: int = 0
    z_stop: int = 0

    @property
    def name(self) -> str:
        ch = "all" if self.channel == -1 else f"ch{self.channel}"
        rng = f"_z{self.z_start or 1}-{self.z_stop or 'end'}" if (self.z_start or self.z_stop) else ""
        return f"z_projection_{self.method}_{ch}{rng}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        if image.ndim != 4:
            return StepResult()   # already 2-D per channel — nothing to do

        ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
        fn = ops.get(self.method, np.min)

        # Z is axis 1 of (C, Z, Y, X): restrict to the sub-stack first, then
        # collapse. Projection is independent per channel, so a single
        # reduction over axis=1 handles all channels at once. (channel != -1 is
        # accepted for API symmetry but still collapses the whole stack —
        # leaving some channels 3-D and others 2-D would be an invalid
        # mixed-rank array.)
        sub = select_z_range(image, self.z_start, self.z_stop, axis=1)
        out = fn(sub, axis=1)
        return StepResult(image=out.astype(image.dtype, copy=False))


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
