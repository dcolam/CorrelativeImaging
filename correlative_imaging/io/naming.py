"""Pluggable file-naming schemes for plate scans.

Different acquisition systems name their files differently, but the pipeline
only ever needs the same five facts out of a path:

    plate · row · column · field · role (brightfield or fluorescence)

A :class:`NamingScheme` says how to extract those from one system's layout. The
scanner (:mod:`correlative_imaging.io.plate`) is scheme-driven, so supporting a
new microscope means adding a scheme — or, in the GUI, writing one by hand when
the convention is not recognised — rather than touching the scanner.

Where the regex is applied
--------------------------
Each scheme's ``pattern`` is matched against the file path **relative to the
plate folder, in POSIX form, with the scheme's matched extension stripped**::

    plate folder:  .../converted/DATASET1
    file:          .../converted/DATASET1/brightfield/PNC_A01_id001_s00_bf.ome.tiff
    matched text:  "brightfield/PNC_A01_id001_s00_bf"

Sub-folders are therefore part of the match, which is what lets a scheme read
the role from a ``brightfield/`` directory instead of the filename. For a flat
folder the matched text is just the file stem, so a filename-only regex works
unchanged.

Named groups the parser understands (all optional except row/col):

``row``     plate row letter(s), A–P on a 384-well plate.
``col``     plate column number; int-cast, so ``A01`` and ``A1`` are the same well.
``field``   field-of-view / site index within the well. Defaults to 1.
``role``    token naming the role, resolved through ``role_map``.
``role_dir`` fallback role token (typically the sub-folder), used when ``role``
            is absent or unmapped.
``plate``   plate identifier carried *in the path* — only consulted by schemes
            with ``plate_from="group"``.
``serial``  acquisition serial number, used by ``role_rule="serial_order"``.

Roles are the two the pipeline knows about: ``"bf"`` and ``"fl"``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger(__name__)

BF, FL = "bf", "fl"

# Tokens that name a role, however a given system spells it. Used as the default
# role_map and by the custom-scheme editor in the GUI.
DEFAULT_ROLE_MAP: dict[str, str] = {
    "bf": BF, "brightfield": BF, "bright_field": BF, "bright-field": BF,
    "trans": BF, "transmitted": BF, "tl": BF, "dia": BF, "phase": BF,
    "fl": FL, "fluorescence": FL, "fluo": FL, "fluorescent": FL,
}


@dataclass(frozen=True)
class ParsedFile:
    """One file resolved to its place on a plate."""
    path:   Path
    row:    str
    col:    int
    field:  int = 1
    role:   str | None = None      # "bf" | "fl" | None (undecided → serial order)
    plate:  str | None = None      # plate token from the path, if the scheme captures one
    serial: int | None = None

    @property
    def well_id(self) -> str:
        # Matches _PlateGrid's key format: int column, so "A01" → "A1".
        return f"{self.row}{self.col}"

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.row, self.col, self.field)


@dataclass
class NamingScheme:
    """How one acquisition system's file layout maps onto a plate.

    Attributes
    ----------
    key:          Stable identifier, used in saved settings.
    label:        Human-readable name shown in the GUI.
    extensions:   Extensions to scan, longest-match-first (so ``.ome.tiff``
                  wins over ``.tif``). Matched case-insensitively.
    depth:        How deep below the plate folder to look for image files.
                  1 = files sit directly in the plate folder; 2 = one level of
                  sub-folders (e.g. ``brightfield/``, ``fluorescence/``).
    pattern:      Regex with the named groups documented in the module docstring,
                  applied with ``re.search`` to the plate-relative path.
    role_rule:    ``"group"``  — role comes from the ``role``/``role_dir`` group.
                  ``"serial_order"`` — within a well, the lowest ``serial`` is BF
                  and the next is FL (the Olympus convention).
    role_map:     Token → ``"bf"``/``"fl"``. Compared lower-cased.
    plate_from:   ``"folder"`` — one plate per folder (the usual case).
                  ``"group"``  — the ``plate`` regex group identifies the plate,
                  so several plates may share one folder.
    field_offset: Added to the captured ``field``. Set to 1 for systems that
                  number sites from zero (``_s00``), so every scheme reports
                  1-based fields and ``WellInfo.field`` means the same thing
                  everywhere — ROI filenames embed it, so a mismatch would put
                  a well's ROIs out of reach.
    role_dirs:    Sub-folder names that hold role-split files. Such folders are
                  never mistaken for plates of their own.
    description:  One-line explanation shown under the scheme picker.
    """
    key:         str
    label:       str
    pattern:     str
    extensions:  tuple[str, ...] = (".tif", ".tiff")
    depth:       int = 1
    role_rule:   str = "group"
    role_map:    dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_MAP))
    plate_from:  str = "folder"
    field_offset: int = 0
    role_dirs:   tuple[str, ...] = ()
    description: str = ""
    builtin:     bool = False

    def __post_init__(self) -> None:
        if self.role_rule not in ("group", "serial_order"):
            raise ValueError(f"role_rule must be 'group' or 'serial_order', got {self.role_rule!r}")
        if self.plate_from not in ("folder", "group"):
            raise ValueError(f"plate_from must be 'folder' or 'group', got {self.plate_from!r}")
        self.extensions = tuple(
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.extensions
        )
        # Longest first so ".ome.tiff" is stripped whole rather than leaving ".ome".
        self.extensions = tuple(sorted(self.extensions, key=len, reverse=True))
        try:
            self._rx = re.compile(self.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"scheme {self.key!r}: invalid pattern — {exc}") from exc

    # ── path matching ────────────────────────────────────────────────

    def matches_extension(self, path: Path) -> str | None:
        """Return the extension this scheme claims for *path*, or None."""
        low = path.name.lower()
        for ext in self.extensions:
            if low.endswith(ext):
                return ext
        return None

    def match_text(self, path: Path, plate_dir: Path) -> str | None:
        """The text the pattern is matched against — see the module docstring."""
        ext = self.matches_extension(path)
        if ext is None:
            return None
        try:
            rel = path.relative_to(plate_dir)
        except ValueError:
            rel = Path(path.name)
        text = rel.as_posix()
        return text[: -len(ext)]

    def parse(self, path: Path, plate_dir: Path) -> ParsedFile | None:
        """Resolve one file to a :class:`ParsedFile`, or None if it doesn't match."""
        text = self.match_text(path, plate_dir)
        if text is None:
            return None
        return self.parse_text(text, path)

    def parse_text(self, text: str, path: Path | None = None) -> ParsedFile | None:
        """Apply the pattern to arbitrary text, bypassing the extension check.

        Used for ROI sidecar files, which follow the same convention as the
        images they were drawn on but carry a ``.roi``/``.tif`` extension the
        scheme does not claim.
        """
        m = self._rx.search(text)
        if not m:
            return None
        g = m.groupdict()
        try:
            col = int(g["col"])
        except (KeyError, TypeError, ValueError):
            return None
        row = (g.get("row") or "").upper()
        if not row:
            return None
        return ParsedFile(
            path=path if path is not None else Path(text), row=row, col=col,
            field=_int_or(g.get("field"), 1 - self.field_offset) + self.field_offset,
            role=self._role(g),
            plate=(g.get("plate") or None) if self.plate_from == "group" else None,
            serial=_int_or(g.get("serial"), None),
        )

    def _role(self, g: dict) -> str | None:
        if self.role_rule != "group":
            return None
        for token in (g.get("role"), g.get("role_dir")):
            if token:
                mapped = self.role_map.get(str(token).lower())
                if mapped:
                    return mapped
        return None

    # ── file discovery ───────────────────────────────────────────────

    def iter_files(self, plate_dir: Path, recursive: bool = False):
        """Yield candidate image files under *plate_dir*, honouring ``depth``
        (or the whole tree when *recursive* is set by the user)."""
        plate_dir = Path(plate_dir)
        if not plate_dir.is_dir():
            return
        max_depth = 64 if recursive else max(1, self.depth)
        yield from _walk(plate_dir, max_depth, self.matches_extension)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("builtin", None)
        d["extensions"] = list(self.extensions)
        d["role_dirs"] = list(self.role_dirs)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NamingScheme":
        d = dict(d)
        d.pop("builtin", None)
        d["extensions"] = tuple(d.get("extensions") or (".tif",))
        d["role_dirs"] = tuple(d.get("role_dirs") or ())
        allowed = {f for f in cls.__dataclass_fields__ if f != "builtin"}
        return cls(**{k: v for k, v in d.items() if k in allowed})


def _int_or(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _walk(root: Path, max_depth: int, accept, _depth: int = 1):
    """Depth-limited file walk; *accept* is called with each path."""
    try:
        entries = sorted(root.iterdir())
    except (OSError, PermissionError) as exc:
        log.debug("Cannot list %s: %s", root, exc)
        return
    for p in entries:
        if p.is_file():
            if accept(p):
                yield p
        elif p.is_dir() and _depth < max_depth:
            yield from _walk(p, max_depth, accept, _depth + 1)


# ──────────────────────────────────────────────────────────────────────
# Built-in schemes
# ──────────────────────────────────────────────────────────────────────

# Olympus spinning-disk VSI export — the original convention, unchanged:
#   __ROMK_18T39412_B10-1_00001.vsi   (BF, lowest serial)
#   __ROMK_18T39412_B10-1_00002.vsi   (FL, next serial)
# Files sit flat in the plate folder and carry no role token, so BF/FL is
# decided by serial order within the well.
OLYMPUS_VSI = NamingScheme(
    key="olympus_vsi",
    label="Olympus VSI (spinning disk)",
    pattern=r"_(?P<row>[A-Pa-p])(?P<col>\d{1,2})-(?P<field>\d+)_(?P<serial>\d+)$",
    extensions=(".vsi",),
    depth=1,
    role_rule="serial_order",
    description="Flat folder, _<ROW><COL>-<field>_<serial>; BF = lowest serial, FL = next.",
    builtin=True,
)

# Output of md_convert/convert_new_md.py — ImageXpress data converted to
# OME-TIFF, split into brightfield/ and fluorescence/ sub-folders:
#   brightfield/PNC_n3_18T40670_A01_id001_s00_bf.ome.tiff
# The role is taken from the _bf/_fl suffix, falling back to the sub-folder,
# so the files still classify correctly if moved out of their folders.
IMAGEXPRESS_OME = NamingScheme(
    key="imagexpress_ome",
    label="ImageXpress OME-TIFF (converted)",
    pattern=(
        r"^(?:(?P<role_dir>[^/]+)/)?"
        r".*?_(?P<row>[A-Pa-p])(?P<col>\d{1,2})_id\d+_s(?P<field>\d+)"
        r"(?:_(?P<role>[A-Za-z]+))?$"
    ),
    extensions=(".ome.tiff", ".ome.tif"),
    depth=2,
    role_rule="group",
    field_offset=1,      # _s00 is the first site — report it as field 1
    role_dirs=("brightfield", "fluorescence"),
    description="convert_new_md.py output: <name>_<WELL>_id<NNN>_s<NN>_bf|fl.ome.tiff "
                "in brightfield/ + fluorescence/ sub-folders.",
    builtin=True,
)

# Deliberately permissive fallback: any format, role from a bf/fl token anywhere
# in the path (filename suffix or sub-folder), well from a _<ROW><COL> token.
# Scores lower than the specific schemes on their own data, so auto-detect only
# lands here when nothing else fits.
GENERIC_BF_FL = NamingScheme(
    key="generic_bf_fl",
    label="Generic (BF/FL token in path)",
    pattern=(
        r"^(?:(?P<role_dir>[^/]+)/)?"
        r".*?[_/-](?P<row>[A-Pa-p])(?P<col>\d{1,2})(?:[-_](?P<field>\d+))?"
        r"(?:[^/]*?[_-](?P<role>bf|fl|brightfield|fluorescence))?$"
    ),
    extensions=(".ome.tiff", ".ome.tif", ".tif", ".tiff", ".czi", ".lif", ".nd2", ".vsi"),
    depth=2,
    role_rule="group",
    role_dirs=("brightfield", "fluorescence", "bf", "fl"),
    description="Best-effort: well from a _<ROW><COL> token, role from a bf/fl "
                "token in the filename or sub-folder.",
    builtin=True,
)

BUILTIN_SCHEMES: dict[str, NamingScheme] = {
    s.key: s for s in (OLYMPUS_VSI, IMAGEXPRESS_OME, GENERIC_BF_FL)
}

DEFAULT_SCHEME = OLYMPUS_VSI


def get_scheme(key_or_scheme) -> NamingScheme:
    """Accept a scheme, a scheme key, or None (→ the default)."""
    if key_or_scheme is None:
        return DEFAULT_SCHEME
    if isinstance(key_or_scheme, NamingScheme):
        return key_or_scheme
    try:
        return BUILTIN_SCHEMES[str(key_or_scheme)]
    except KeyError:
        raise ValueError(
            f"unknown naming scheme {key_or_scheme!r}; "
            f"known: {', '.join(BUILTIN_SCHEMES)}"
        ) from None


# ──────────────────────────────────────────────────────────────────────
# Auto-detection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SchemeScore:
    """How well one scheme explains the files under a folder."""
    scheme:    NamingScheme
    n_files:   int      # files with an extension this scheme claims
    n_parsed:  int      # of those, how many yielded a well coordinate
    n_roles:   int      # of those, how many resolved to a bf/fl role
    samples:   list[ParsedFile] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.n_parsed / self.n_files if self.n_files else 0.0

    @property
    def role_rate(self) -> float:
        return self.n_roles / self.n_parsed if self.n_parsed else 0.0

    @property
    def score(self) -> float:
        """Rank: how many files it explains, then whether roles are explicit,
        then a nudge toward specific schemes over the generic fallback."""
        if not self.n_parsed:
            return 0.0
        explicit = 1.0 if self.scheme.role_rule == "serial_order" else self.role_rate
        specific = 0.9 if self.scheme.key == "generic_bf_fl" else 1.0
        return (self.match_rate * 0.7 + explicit * 0.3) * specific

    def __repr__(self) -> str:
        return (f"SchemeScore({self.scheme.key}: {self.n_parsed}/{self.n_files} files, "
                f"{self.n_roles} with role, score={self.score:.2f})")


def score_scheme(root: Path, scheme: NamingScheme, limit: int = 400) -> SchemeScore:
    """Try *scheme* against up to *limit* files under *root* (which may be a
    plate folder or a parent of several)."""
    root = Path(root)
    plate_dirs = _plate_candidates(root, scheme) or [root]
    n_files = n_parsed = n_roles = 0
    samples: list[ParsedFile] = []
    for pdir in plate_dirs:
        for path in scheme.iter_files(pdir):
            n_files += 1
            rec = scheme.parse(path, pdir)
            if rec is not None:
                n_parsed += 1
                if rec.role or scheme.role_rule == "serial_order":
                    n_roles += 1
                if len(samples) < 12:
                    samples.append(rec)
            if n_files >= limit:
                break
        if n_files >= limit:
            break
    return SchemeScore(scheme, n_files, n_parsed, n_roles, samples)


def detect_scheme(root: Path, schemes=None, limit: int = 400) -> list[SchemeScore]:
    """Score every candidate scheme against *root*, best first.

    Returns every score (including zero-match ones) so the GUI can show *why*
    a scheme was or wasn't chosen, rather than silently picking one.
    """
    schemes = list(schemes) if schemes is not None else list(BUILTIN_SCHEMES.values())
    scores = [score_scheme(root, s, limit) for s in schemes]
    scores.sort(key=lambda s: (s.score, s.n_parsed), reverse=True)
    return scores


def _plate_candidates(root: Path, scheme: NamingScheme) -> list[Path]:
    """Directories under *root* that look like plate folders for *scheme*.

    A directory qualifies when it holds matching files within the scheme's
    depth. ``root`` itself is tested first: if it qualifies, it is the single
    plate and its sub-folders are not considered separately — this is what
    stops ``brightfield/`` and ``fluorescence/`` from being reported as two
    plates of their own.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    if _holds_files(root, scheme):
        return [root]
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.lower() in {r.lower() for r in scheme.role_dirs}:
            continue
        if _holds_files(d, scheme):
            out.append(d)
    return out


def _holds_files(d: Path, scheme: NamingScheme) -> bool:
    for _ in scheme.iter_files(d):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Persistence for user-written schemes
# ──────────────────────────────────────────────────────────────────────

def load_custom_schemes(path: Path) -> dict[str, NamingScheme]:
    """Load user-defined schemes from a JSON file; ``{}`` if absent or invalid."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read custom naming schemes from %s: %s", path, exc)
        return {}
    out: dict[str, NamingScheme] = {}
    for entry in raw.get("schemes", []):
        try:
            s = NamingScheme.from_dict(entry)
        except (TypeError, ValueError) as exc:
            log.warning("Skipping invalid custom scheme %r: %s", entry.get("key"), exc)
            continue
        out[s.key] = s
    return out


def save_custom_schemes(path: Path, schemes: dict[str, NamingScheme]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schemes": [s.to_dict() for s in schemes.values() if not s.builtin]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
