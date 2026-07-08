"""Well-plate file discovery and BF/FL pairing for Olympus VSI plate experiments.

Naming convention assumed
-------------------------
Files follow the Olympus spinning-disk export pattern::

    __<experiment>_<plate>_<ROW><COL>-<field>_<serial>.<ext>

e.g.  ``__ROMK_18T39412_B10-1_00001.vsi``  (BF)
      ``__ROMK_18T39412_B10-1_00002.vsi``  (FL)

Pairing rule
------------
Files are grouped by well coordinate (row letter + column number + field).
Within each group they are sorted by serial number; the *lowest* serial is
treated as BF, the *next* as FL.

This covers both naming schemes encountered in practice:

* **Fixed per-well serials** — every well uses ``_00001`` (BF) / ``_00002`` (FL).
* **Continuous plate serials** — the microscope increments the counter across the
  whole plate, so well B2 might have ``_00001``/``_00002`` and well B3 has
  ``_00003``/``_00004``.  The sort-by-serial rule handles both identically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Matches  _<ROW><COL>-<field>_<serial>  at the end of a filename stem.
# Row is A-P (384-well plate); column 1-24; field and serial are integers.
_WELL_RE = re.compile(r"_([A-Pa-p])(\d{1,2})-(\d+)_(\d+)$")

# Lenient variant without serial — used for ROI files which often lack it.
_WELL_COORD_RE = re.compile(r"_([A-Pa-p])(\d{1,2})-(\d+)")


@dataclass
class WellInfo:
    """Paired BF + FL file paths for one well / field-of-view.

    Attributes
    ----------
    row:        Plate row letter (A-P), upper-cased.
    col:        Plate column number (1-24).
    field:      Field-of-view index within the well (usually 1).
    bf_path:    Path to the brightfield VSI (or other format) file.
    fl_path:    Path to the fluorescence VSI file.
    bf_serial:  Raw serial number extracted from the BF filename.
    fl_serial:  Raw serial number extracted from the FL filename.
    extra_paths: Any additional files beyond the expected pair.
    """
    row:        str
    col:        int
    field:      int
    bf_path:    Path | None = None
    fl_path:    Path | None = None
    bf_serial:  int | None = None
    fl_serial:  int | None = None
    extra_paths: list[Path] = field(default_factory=list)
    roi_paths:   list[Path] = field(default_factory=list)

    @property
    def well_id(self) -> str:
        """Human-readable well coordinate, e.g. ``'B10'``."""
        return f"{self.row}{self.col}"

    @property
    def is_complete(self) -> bool:
        """True when both BF and FL paths are present."""
        return self.bf_path is not None and self.fl_path is not None

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "incomplete"
        return (
            f"WellInfo({self.well_id} field={self.field} [{status}] "
            f"bf={self.bf_path.name if self.bf_path else None} "
            f"fl={self.fl_path.name if self.fl_path else None})"
        )


def scan_plate_folder(
    folder: str | Path,
    extension: str = ".vsi",
    contains: str = "",
    recursive: bool = False,
    extra_roi_dirs: list[str | Path] | None = None,
) -> list[WellInfo]:
    """Discover and pair BF/FL files in *folder*.

    Parameters
    ----------
    folder:     Root directory to scan.
    extension:  File extension to match (leading dot optional, case-insensitive).
    contains:   Optional substring that must appear in each filename.
    recursive:  When True, search subdirectories as well.
    extra_roi_dirs: Additional folders to search (non-recursively) for ROI
                    files, e.g. a BF-pipeline output ``rois/`` folder that
                    lives outside the plate/data folder.

    Returns
    -------
    List of :class:`WellInfo` sorted by (row, col, field).
    Wells with only one file (BF only) are included; ``fl_path`` will be ``None``.
    Files that do not match the well-coordinate pattern are silently skipped
    (logged at DEBUG level).
    """
    folder = Path(folder)
    if not extension.startswith("."):
        extension = f".{extension}"

    glob_fn = folder.rglob if recursive else folder.glob
    files: list[Path] = sorted(glob_fn(f"*{extension}"))
    if contains:
        files = [f for f in files if contains in f.name]

    # ── Parse filenames and group by (row, col, field) ──────────────────
    groups: dict[tuple[str, int, int], list[tuple[int, Path]]] = {}
    skipped = 0
    for path in files:
        m = _WELL_RE.search(path.stem)
        if not m:
            log.debug("No well coordinate in '%s' — skipping", path.name)
            skipped += 1
            continue
        row    = m.group(1).upper()
        col    = int(m.group(2))
        fov    = int(m.group(3))
        serial = int(m.group(4))
        groups.setdefault((row, col, fov), []).append((serial, path))

    if skipped:
        log.debug("Skipped %d files with no recognisable well coordinate", skipped)

    # ── Assign BF / FL by serial order within each well ─────────────────
    wells: list[WellInfo] = []
    for (row, col, fov), entries in sorted(groups.items()):
        entries.sort(key=lambda t: t[0])  # ascending serial → BF first
        w = WellInfo(row=row, col=col, field=fov)

        w.bf_serial, w.bf_path = entries[0]
        if len(entries) >= 2:
            w.fl_serial, w.fl_path = entries[1]
        if len(entries) > 2:
            w.extra_paths = [p for _, p in entries[2:]]
            log.warning(
                "Well %s%d field %d: found %d files (expected 2); "
                "using first as BF, second as FL.",
                row, col, fov, len(entries),
            )
        if not w.is_complete:
            log.warning(
                "Well %s%d field %d: only BF found, no FL counterpart.",
                row, col, fov,
            )

        wells.append(w)

    # ── Scan for ROI files and assign to wells ───────────────────────
    well_lookup = {(w.row, w.col, w.field): w for w in wells}
    roi_files: list[Path] = []
    for ext in (".roi", ".zip"):
        roi_files.extend(sorted(glob_fn(f"*{ext}")))
    for extra_dir in extra_roi_dirs or []:
        extra_dir = Path(extra_dir)
        if extra_dir.is_dir():
            # .tif/.tiff included here (but not in the main data-folder glob
            # above) because extra_roi_dirs is a dedicated ROI output folder
            # (e.g. the BF pipeline's rois/ dir) — never a folder containing
            # raw acquisition images, so there's no risk of misclassifying
            # a plate image as an ROI mask.
            for ext in (".roi", ".zip", ".tif", ".tiff"):
                roi_files.extend(sorted(extra_dir.glob(f"*{ext}")))
    if contains:
        roi_files = [f for f in roi_files if contains in f.name]

    for path in roi_files:
        m = _WELL_RE.search(path.stem) or _WELL_COORD_RE.search(path.stem)
        if not m:
            log.debug("No well coordinate in ROI file '%s' — skipping", path.name)
            continue
        row = m.group(1).upper()
        col = int(m.group(2))
        fov = int(m.group(3))
        w = well_lookup.get((row, col, fov))
        if w:
            w.roi_paths.append(path)
        else:
            log.debug("No matching well for ROI file '%s'", path.name)

    n_roi = sum(1 for w in wells if w.roi_paths)
    n_complete = sum(1 for w in wells if w.is_complete)
    log.info(
        "Plate scan: %d wells total, %d complete BF+FL pairs, %d incomplete, %d with ROI files.",
        len(wells), n_complete, len(wells) - n_complete, n_roi,
    )
    return wells


def read_well(
    well: WellInfo,
    load_bf: bool = True,
    load_fl: bool = True,
    scene: int = 0,
):
    """Load the BF and/or FL images for a well and return ``(bf_data, fl_data)``.

    Either element is ``None`` when the corresponding path is absent or loading
    was not requested.

    Parameters
    ----------
    well:       A :class:`WellInfo` returned by :func:`scan_plate_folder`.
    load_bf:    Whether to load the brightfield image.
    load_fl:    Whether to load the fluorescence image.
    scene:      Scene index passed to the underlying reader (default 0).

    Returns
    -------
    ``(bf: ImageData | None, fl: ImageData | None)``
    """
    from .reader import read_image  # local import to avoid circular deps

    bf_data = None
    fl_data = None

    if load_bf and well.bf_path is not None:
        log.debug("Loading BF for well %s: %s", well.well_id, well.bf_path.name)
        bf_data = read_image(well.bf_path, scene=scene)

    if load_fl and well.fl_path is not None:
        log.debug("Loading FL for well %s: %s", well.well_id, well.fl_path.name)
        fl_data = read_image(well.fl_path, scene=scene)

    return bf_data, fl_data
