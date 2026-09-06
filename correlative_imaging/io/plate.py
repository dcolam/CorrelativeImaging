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
from dataclasses import dataclass, field, replace
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


def _roi_coordinate(path: Path, scheme) -> tuple[str, int, int] | None:
    """(row, col, field) for an ROI sidecar file, or None.

    Tried in order: this module's own ROI convention, then the plate scheme's
    pattern (so ROIs named after the images they were drawn on are found on any
    acquisition system), then the lenient field-less variant.
    """
    stem = path.stem
    m = _WELL_RE.search(stem) or _WELL_COORD_RE.search(stem)
    if m:
        return m.group(1).upper(), int(m.group(2)), int(m.group(3))
    rec = scheme.parse_text(stem, path)
    if rec is not None:
        return rec.row, rec.col, rec.field
    return None


def scan_plate_folder(
    folder: str | Path,
    extension: str = ".vsi",
    contains: str = "",
    recursive: bool = False,
    extra_roi_dirs: list[str | Path] | None = None,
    scheme=None,
    plate_token: str | None = None,
) -> list[WellInfo]:
    """Discover and pair BF/FL files in *folder*.

    Parameters
    ----------
    folder:     Root directory to scan.
    extension:  File extension to match (leading dot optional, case-insensitive).
                Ignored when *scheme* is given — the scheme carries its own
                extensions, which may be several (e.g. ``.ome.tiff``/``.ome.tif``).
    contains:   Optional substring that must appear in each filename.
    recursive:  When True, search subdirectories as well.
    extra_roi_dirs: Additional folders to search (non-recursively) for ROI
                    files, e.g. a BF-pipeline output ``rois/`` folder that
                    lives outside the plate/data folder.
    scheme:     A :class:`~correlative_imaging.io.naming.NamingScheme` (or its
                key) describing this acquisition system's file layout. Defaults
                to the Olympus VSI convention, i.e. the historical behaviour.
    plate_token: For schemes with ``plate_from="group"``, keep only files whose
                captured plate token equals this. Ignored otherwise.

    Returns
    -------
    List of :class:`WellInfo` sorted by (row, col, field).
    Wells with only one file (BF only) are included; ``fl_path`` will be ``None``.
    Files that do not match the scheme are silently skipped (logged at DEBUG level).
    """
    from .naming import BF, FL, get_scheme

    folder = Path(folder)
    scheme = get_scheme(scheme)
    if extension and scheme.key == "olympus_vsi":
        # Back-compat: callers that only pass `extension` still steer the scan.
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension.lower() != scheme.extensions[0]:
            scheme = replace(scheme, extensions=(extension,), builtin=False)

    # ── Parse filenames and group by (row, col, field) ──────────────────
    groups: dict[tuple[str, int, int], list] = {}
    skipped = 0
    for path in scheme.iter_files(folder, recursive=recursive):
        if contains and contains not in path.name:
            continue
        rec = scheme.parse(path, folder)
        if rec is None:
            log.debug("No well coordinate in '%s' — skipping", path.name)
            skipped += 1
            continue
        if plate_token is not None and rec.plate is not None and rec.plate != plate_token:
            continue
        groups.setdefault(rec.key, []).append(rec)

    if skipped:
        log.debug("Skipped %d files with no recognisable well coordinate", skipped)

    # ── Assign BF / FL ──────────────────────────────────────────────────
    wells: list[WellInfo] = []
    for (row, col, fov), recs in sorted(groups.items()):
        # Stable order: by serial when the scheme captures one, else by name.
        recs.sort(key=lambda r: (r.serial if r.serial is not None else 0, r.path.name))
        w = WellInfo(row=row, col=col, field=fov)

        if scheme.role_rule == "group":
            bf = [r for r in recs if r.role == BF]
            fl = [r for r in recs if r.role == FL]
            unknown = [r for r in recs if r.role is None]
            # A file the scheme could place on the plate but not label falls back
            # to filling whichever slot is still empty, cheapest-first: this is
            # what makes a half-labelled folder usable instead of half-empty.
            for r in unknown:
                (bf if not bf else fl).append(r)
            if bf:
                w.bf_path, w.bf_serial = bf[0].path, bf[0].serial
            if fl:
                w.fl_path, w.fl_serial = fl[0].path, fl[0].serial
            extras = bf[1:] + fl[1:]
            if extras:
                w.extra_paths = [r.path for r in extras]
                log.warning(
                    "Well %s%d field %d: %d extra file(s) beyond one BF + one FL.",
                    row, col, fov, len(extras),
                )
        else:  # serial_order — lowest serial is BF, next is FL
            w.bf_serial, w.bf_path = recs[0].serial, recs[0].path
            if len(recs) >= 2:
                w.fl_serial, w.fl_path = recs[1].serial, recs[1].path
            if len(recs) > 2:
                w.extra_paths = [r.path for r in recs[2:]]
                log.warning(
                    "Well %s%d field %d: found %d files (expected 2); "
                    "using first as BF, second as FL.",
                    row, col, fov, len(recs),
                )
        if not w.is_complete:
            log.warning(
                "Well %s%d field %d: only BF found, no FL counterpart.",
                row, col, fov,
            )

        wells.append(w)

    # ── Scan for ROI files and assign to wells ───────────────────────
    well_lookup = {(w.row, w.col, w.field): w for w in wells}
    glob_fn = folder.rglob if recursive else folder.glob
    roi_files: list[Path] = list(sorted(glob_fn("*.roi")))
    for extra_dir in extra_roi_dirs or []:
        extra_dir = Path(extra_dir)
        if extra_dir.is_dir():
            # .tif/.tiff included here (but not in the main data-folder glob
            # above) because extra_roi_dirs is a dedicated ROI output folder
            # (e.g. the BF pipeline's rois/ dir) — never a folder containing
            # raw acquisition images, so there's no risk of misclassifying
            # a plate image as an ROI mask.
            for ext in (".roi", ".tif", ".tiff"):
                roi_files.extend(sorted(extra_dir.glob(f"*{ext}")))
    if contains:
        roi_files = [f for f in roi_files if contains in f.name]

    # ROI files may be named by this GUI's own convention (see
    # _roi_filename_for_well: _<ROW><COL>-<field>_<kind>) or, for ROIs the user
    # brought along, by the same convention as the images — so try the active
    # scheme too rather than only the Olympus pattern.
    by_well: dict[tuple[str, int], list[WellInfo]] = {}
    for w in wells:
        by_well.setdefault((w.row, w.col), []).append(w)

    for path in roi_files:
        coord = _roi_coordinate(path, scheme)
        if coord is None:
            log.debug("No well coordinate in ROI file '%s' — skipping", path.name)
            continue
        row, col, fov = coord
        w = well_lookup.get((row, col, fov))
        if w is None:
            # Field numbering differs between the ROI name and the images (or
            # the ROI names no field at all). One well for this coordinate is
            # unambiguous, so use it; several means we cannot tell which.
            candidates = by_well.get((row, col), [])
            if len(candidates) == 1:
                w = candidates[0]
            elif candidates:
                log.debug("ROI file '%s': %d fields at %s%d — cannot tell which.",
                          path.name, len(candidates), row, col)
        if w:
            w.roi_paths.append(path)
        else:
            log.debug("No matching well for ROI file '%s'", path.name)

    n_roi = sum(1 for w in wells if w.roi_paths)
    n_complete = sum(1 for w in wells if w.is_complete)
    log.info(
        "Plate scan (%s): %d wells total, %d complete BF+FL pairs, %d incomplete, "
        "%d with ROI files.",
        scheme.key, len(wells), n_complete, len(wells) - n_complete, n_roi,
    )
    return wells


def discover_plate_folders(
    root: str | Path,
    extension: str = ".vsi",
    contains: str = "",
    scheme=None,
) -> dict[str, Path]:
    """Find one or more plate export folders under *root*.

    A "plate folder" is any directory that holds at least one file the *scheme*
    can place on a plate, within that scheme's search depth.

    Three layouts are handled:

    * **Single plate** — *root* itself holds the well files. Returns
      ``{root.name: root}``. Its sub-folders are then *not* considered
      separately, so a scheme whose files live in ``brightfield/`` and
      ``fluorescence/`` sub-folders yields one plate, not two.
    * **Multiple plates** — *root* is a parent whose immediate subdirectories
      are each one plate's export folder. Returns one entry per matching
      subfolder, keyed by that subfolder's own name. Directories named as role
      folders by the scheme (``brightfield``/``fluorescence``) are never
      treated as plates.
    * **Several plates in one folder** — only for schemes with
      ``plate_from="group"``, where the plate identity is a token in the path
      rather than the folder. Returns one entry per distinct token, all
      pointing at the same folder; pass the token to
      :func:`scan_plate_folder` as ``plate_token`` to scan just that plate.

    Duplicate subfolder names (rare, but possible if plates were exported
    under differently-located parents with the same folder name) are
    disambiguated with a numeric suffix and logged.
    """
    from .naming import _plate_candidates, get_scheme
    from dataclasses import replace as _replace

    root = Path(root)
    scheme = get_scheme(scheme)
    if extension and scheme.key == "olympus_vsi":
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension.lower() != scheme.extensions[0]:
            scheme = _replace(scheme, extensions=(extension,), builtin=False)

    def _matching(d: Path) -> list[Path]:
        files = list(scheme.iter_files(d))
        if contains:
            files = [f for f in files if contains in f.name]
        return [f for f in files if scheme.parse(f, d) is not None]

    candidates = [d for d in _plate_candidates(root, scheme) if _matching(d)]

    # Plate identity carried in the path rather than the folder.
    if scheme.plate_from == "group":
        plates: dict[str, Path] = {}
        for d in candidates:
            for f in _matching(d):
                rec = scheme.parse(f, d)
                if rec and rec.plate:
                    plates.setdefault(rec.plate, d)
        if plates:
            log.info("Plate-token scan: %d plate(s) found under %s", len(plates), root)
            return dict(sorted(plates.items()))
        # No token captured anywhere — fall through to folder identity.

    if not candidates:
        return {}
    if candidates == [root]:
        return {root.name or str(root): root}

    plates = {}
    seen: dict[str, int] = {}
    for d in candidates:
        name = d.name
        if name in seen:
            seen[name] += 1
            key = f"{name} ({seen[name]})"
            log.warning("Duplicate plate folder name '%s' — disambiguating as '%s'", name, key)
        else:
            seen[name] = 0
            key = name
        plates[key] = d

    if plates:
        log.info("Multi-plate scan: %d plate folder(s) found under %s", len(plates), root)
    return plates


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
