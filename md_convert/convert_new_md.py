"""Convert the *new* ImageXpress single-file output to OME-TIFFs.

The new microscope writes one TIFF per (timepoint, well, site, channel, z-slice),
flat inside ``.../experiment_z_stack/timepoint<T>/``:

    <name>_t<T>_<well>_s<site>_w<chan>_z<z>.tif
    e.g.  PNC_n3_18T40670_t0_A01_s0_w0_z0.tif   (well = row A, col 01)

The old ``zmb_md_converter`` regex only matches the classic MetaXpress layout
(``TimePoint_1/ZStep_3/Name_B05_s1_w2.tif``) and treats ``z0`` as a projection —
so it silently parses nothing here. This script parses the NEW pattern, groups by
(name, position, site), stacks channels × z-planes into a ``(T, C, Z, Y, X)``
array, pulls physical calibration + channel names from the MetaSeries TIFF tags,
and writes one OME-TIFF per well/site (Fiji- and bioio-readable).

Only needs numpy + tifffile (both already in the env). No dask/xarray.

Usage
-----
    # 1) confirm what metadata your files actually carry (run this first):
    python convert_new_md.py --inspect "D:\\...\\timepoint0"

    # 2) convert a whole timepoint folder (or the experiment_z_stack folder):
    python convert_new_md.py "D:\\...\\experiment_z_stack" "D:\\...\\ome_out"

    # overrides if calibration is missing from the tags:
    python convert_new_md.py IN OUT --dxy 0.65 --dz 2.0 --unit um

Brightfield / fluorescence split
--------------------------------
Channels are classified from the MetaSeries ``_IllumSetting_`` tag and written
to separate sub-folders::

    OUT/brightfield/<name>_<well>_s00_bf.ome.tiff
    OUT/fluorescence/<name>_<well>_s00_fl.ome.tiff

The kind is marked both by the sub-folder and by the ``_bf`` / ``_fl`` filename
suffix (rename either with --bf-subdir/--fl-subdir, --bf-suffix/--fl-suffix), so
the files stay self-describing once moved out of their folder.
The classification of every channel is printed on each run. If the tag on this
instrument uses a word the default token list does not cover, override it::

    --bf-channels 0,3          # explicit w-indices (never guesses)
    --bf-pattern "trans|dia"   # case-insensitive regex on _IllumSetting_
    --no-split                 # old behaviour: one file with all channels
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

# <name>_t<T>_<well>_s<site>_w<chan>_z<z>.tif   (name may contain underscores;
# well is a plate coordinate like A01 / B05 / P20)
_PATTERN = re.compile(
    r"^(?P<name>.+)_t(?P<t>\d+)_(?P<well>[A-Za-z]+\d+)_s(?P<site>\d+)"
    r"_w(?P<w>\d+)_z(?P<z>\d+)\.tif{1,2}$",
    re.IGNORECASE,
)

# MetaSeries PlaneInfo keys we try to read (all optional — missing → fallback).
_META_KEYS = (
    "spatial-calibration-x", "spatial-calibration-y", "spatial-calibration-units",
    "_IllumSetting_", "wavelength", "Exposure Time", "_MagSetting_",
    "stage-label", "Z Step", "ImageXpress Micro Z", "z-position",
)


def parse_name(path: Path) -> dict | None:
    m = _PATTERN.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "name": d["name"], "t": int(d["t"]), "well": d["well"].upper(),
        "site": int(d["site"]), "w": int(d["w"]), "z": int(d["z"]), "path": path,
    }


def read_metaseries_metadata(path: Path) -> dict:
    """Best-effort MetaSeries PlaneInfo metadata; ``{}`` if not a MetaSeries TIFF
    or on any error (so conversion still runs without calibration)."""
    try:
        with tifffile.TiffFile(path) as tif:
            if not getattr(tif, "is_metaseries", False):
                return {}
            plane = tif.metaseries_metadata.get("PlaneInfo", {})
        out = {k: plane[k] for k in _META_KEYS if k in plane}
        for k in plane:                       # keep any *Intensity keys too
            if k.endswith("Intensity"):
                out[k] = plane[k]
        return out
    except Exception:
        return {}


def _stage_z(meta: dict):
    for k in ("ImageXpress Micro Z", "z-position"):
        if k in meta:
            try:
                return float(meta[k])
            except (TypeError, ValueError):
                pass
    return None


def collect(input_dir: Path) -> dict:
    """Walk *input_dir* and group parsed files by (name, well, site)."""
    groups: dict = defaultdict(list)
    n_seen = n_ok = 0
    for p in input_dir.rglob("*.tif*"):
        n_seen += 1
        rec = parse_name(p)
        if rec is None:
            continue
        n_ok += 1
        groups[(rec["name"], rec["well"], rec["site"])].append(rec)
    print(f"Scanned {n_seen} .tif files; {n_ok} matched the new naming pattern; "
          f"{len(groups)} well/site groups.")
    if n_ok == 0 and n_seen:
        print("  ⚠ No files matched — check the pattern against a real filename.",
              file=sys.stderr)
    return groups


# Words that appear in the MetaSeries ``_IllumSetting_`` of a transmitted-light
# channel. Matched case-insensitively as substrings; override with --bf-pattern
# or --bf-channels when this instrument names the channel differently.
_BF_TOKENS = (
    "trans", "brightfield", "bright field", "bright-field",
    "tl", "dia", "phase", "dic", "widefield tl",
)


def is_brightfield(name: str, w: int, bf_channels=None, bf_pattern=None) -> bool:
    """Classify one channel as brightfield.

    *bf_channels* (an explicit set of ``w`` indices) wins outright — use it when
    the acquisition setting is known, since it cannot misfire. Otherwise
    *bf_pattern* (a case-insensitive regex) or, failing that, the default token
    list is matched against the channel's ``_IllumSetting_`` name.
    """
    if bf_channels is not None:
        return w in bf_channels
    if bf_pattern is not None:
        return re.search(bf_pattern, name, re.IGNORECASE) is not None
    low = name.lower()
    # "tl"/"dia" as whole words only, so e.g. "mCherry" or "Cardio" don't match.
    return any(
        (re.search(rf"\b{re.escape(tok)}\b", low) if len(tok) <= 3 else tok in low)
        for tok in _BF_TOKENS
    )


def _channel_name(meta: dict, w: int) -> str:
    for k in ("_IllumSetting_",):
        if meta.get(k):
            return str(meta[k])
    if meta.get("wavelength"):
        return f"w{w}_{meta['wavelength']}"
    return f"w{w}"


def build_group(recs: list, dxy=None, dz=None, unit="um") -> tuple:
    """Assemble one (name,pos,site) group into a (T,C,Z,Y,X) array + OME metadata.

    Returns ``(array, ome_metadata, well_label, w_indices)``. *w_indices* is the
    filename ``w`` number of each C-plane, index-aligned with both the C axis and
    ``ome['Channel']['Name']`` — the caller slices all three together when
    splitting brightfield from fluorescence."""
    ts = sorted({r["t"] for r in recs})
    ws = sorted({r["w"] for r in recs})
    zs = sorted({r["z"] for r in recs})
    by = {(r["t"], r["w"], r["z"]): r["path"] for r in recs}

    # shape from the first available image
    first = recs[0]["path"]
    sample = tifffile.imread(first)
    ny, nx = sample.shape[-2:]
    dtype = sample.dtype
    arr = np.zeros((len(ts), len(ws), len(zs), ny, nx), dtype=dtype)

    missing = 0
    for ti, t in enumerate(ts):
        for ci, w in enumerate(ws):
            for zi, z in enumerate(zs):
                path = by.get((t, w, z))
                if path is None:
                    missing += 1
                    continue
                arr[ti, ci, zi] = tifffile.imread(path)

    # --- metadata (one representative tif per channel) ---
    meta0 = read_metaseries_metadata(first)
    # stage-label is empty on this instrument → use the well from the filename.
    well = str(meta0.get("stage-label") or recs[0]["well"])
    px_unit = str(meta0.get("spatial-calibration-units") or unit)
    if px_unit == "um":
        px_unit = "µm"
    dx = dxy if dxy is not None else _flt(meta0.get("spatial-calibration-x"))
    dy = dxy if dxy is not None else _flt(meta0.get("spatial-calibration-y"))

    # z-step: user override → stage-Z deltas → "Z Step" tag → 1
    zstep = dz
    if zstep is None and len(zs) > 1:
        zpos = []
        for z in zs:
            p = by.get((ts[0], ws[0], z))
            zval = _stage_z(read_metaseries_metadata(p)) if p else None
            zpos.append(zval)
        if all(v is not None for v in zpos):
            zstep = round(float(np.mean(np.diff(zpos))), 4)
    if zstep is None:
        zstep = _flt(meta0.get("Z Step")) or 1.0
    zstep = abs(zstep) or 1.0

    channel_names = []
    for w in ws:
        p = by.get((ts[0], w, zs[0]))
        channel_names.append(_channel_name(read_metaseries_metadata(p) if p else {}, w))

    ome = {
        "axes": "TCZYX",
        "Channel": {"Name": channel_names},
    }
    if dx:
        ome.update(PhysicalSizeX=float(dx), PhysicalSizeXUnit=px_unit,
                   PhysicalSizeY=float(dy or dx), PhysicalSizeYUnit=px_unit)
    ome.update(PhysicalSizeZ=float(zstep), PhysicalSizeZUnit=px_unit)

    if missing:
        print(f"  ⚠ {well}: {missing} missing (t,c,z) planes filled with zeros.")
    return arr, ome, well, ws


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def inspect(input_dir: Path, bf_channels=None, bf_pattern=None) -> None:
    """Print one representative file per channel with its MetaSeries metadata and
    the brightfield/fluorescence verdict, so channel naming can be confirmed
    before a full run."""
    reps: dict[int, Path] = {}
    for p in sorted(input_dir.rglob("*.tif*")):
        rec = parse_name(p)
        if rec is None:
            continue
        reps.setdefault(rec["w"], p)
    if not reps:
        print("No files matching the new naming pattern found under", input_dir)
        return

    print(f"{len(reps)} channel(s) found under {input_dir}\n")
    for w, path in sorted(reps.items()):
        meta = read_metaseries_metadata(path)
        name = _channel_name(meta, w)
        kind = "brightfield" if is_brightfield(name, w, bf_channels, bf_pattern) \
            else "fluorescence"
        print(f"w{w}  {path.name}\n     channel name: {name!r}  →  {kind}")
        try:
            with tifffile.TiffFile(path) as tif:
                if not getattr(tif, "is_metaseries", False):
                    print("     (not a MetaSeries TIFF — no PlaneInfo tags)")
                    continue
                plane = tif.metaseries_metadata.get("PlaneInfo", {})
                for k in sorted(plane):
                    print(f"       {k}: {plane[k]}")
        except Exception as e:
            print(f"     (could not read metadata: {e})")
        print()


def _parse_bf_channels(spec):
    if spec is None:
        return None
    try:
        return {int(x) for x in spec.replace(",", " ").split()}
    except ValueError:
        raise SystemExit(f"--bf-channels: expected w-indices like '0,3', got {spec!r}")


def _write(out: Path, arr, ome, tag: str, prefix: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    big = arr.nbytes >= 2**32 - 2**25          # per-subset, not per-group
    with tifffile.TiffWriter(out, ome=True, bigtiff=big) as tw:
        tw.write(arr, photometric="minisblack", metadata=ome)
    print(f"{prefix} {tag:<12} {out.parent.name}/{out.name}  "
          f"T{arr.shape[0]} C{arr.shape[1]} Z{arr.shape[2]} "
          f"{arr.shape[3]}×{arr.shape[4]} {arr.dtype}  ch={ome['Channel']['Name']}")


def _subset(arr, ome, keep: list[int]) -> tuple:
    """Slice the C axis and the channel-name list with the same mask."""
    sub_ome = dict(ome)
    sub_ome["Channel"] = {"Name": [ome["Channel"]["Name"][i] for i in keep]}
    return arr[:, keep], sub_ome


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="folder to scan recursively")
    ap.add_argument("output", type=Path, nargs="?", help="output folder for OME-TIFFs")
    ap.add_argument("--inspect", action="store_true",
                    help="print one file per channel + metadata + BF/FL verdict, then exit")
    ap.add_argument("--dxy", type=float, default=None, help="pixel size override (units)")
    ap.add_argument("--dz", type=float, default=None, help="z-step override (units)")
    ap.add_argument("--unit", default="um", help="fallback calibration unit (default um)")
    ap.add_argument("--bf-channels", default=None, metavar="0,3",
                    help="explicit brightfield w-indices; overrides name matching")
    ap.add_argument("--bf-pattern", default=None, metavar="REGEX",
                    help="case-insensitive regex on the channel name marking brightfield")
    ap.add_argument("--no-split", action="store_true",
                    help="write one file with all channels (no brightfield/ + fluorescence/)")
    ap.add_argument("--bf-subdir", default="brightfield", help="brightfield sub-folder name")
    ap.add_argument("--fl-subdir", default="fluorescence", help="fluorescence sub-folder name")
    ap.add_argument("--bf-suffix", default="bf", help="brightfield filename suffix (default bf)")
    ap.add_argument("--fl-suffix", default="fl", help="fluorescence filename suffix (default fl)")
    args = ap.parse_args(argv)

    bf_channels = _parse_bf_channels(args.bf_channels)

    if args.inspect:
        inspect(args.input, bf_channels, args.bf_pattern)
        return 0
    if args.output is None:
        ap.error("output folder is required (unless --inspect)")

    args.output.mkdir(parents=True, exist_ok=True)
    groups = collect(args.input)
    if not groups:
        return 1

    seen: set = set()   # distinct channel layouts already reported
    for i, (key, recs) in enumerate(sorted(groups.items()), 1):
        name, _well, site = key
        arr, ome, well, ws = build_group(recs, args.dxy, args.dz, args.unit)
        names = ome["Channel"]["Name"]
        prefix = f"[{i}/{len(groups)}]"
        stem = f"{name}_{well}_s{site:02d}"

        if args.no_split:
            _write(args.output / f"{stem}.ome.tiff", arr, ome, "", prefix)
            continue

        flags = [is_brightfield(n, w, bf_channels, args.bf_pattern)
                 for n, w in zip(names, ws)]
        # Groups may differ in channel layout (e.g. two experiments under one
        # input folder), so report each distinct layout the first time it shows up.
        sig = tuple(zip(ws, names, flags))
        if sig not in seen:
            seen.add(sig)
            print("Channel classification:" if len(seen) == 1
                  else f"Channel classification ({name} {well} s{site:02d}):")
            for n, w, bf in zip(names, ws, flags):
                print(f"  w{w} {n!r} → {'brightfield' if bf else 'fluorescence'}")
            print()

        bf_idx = [c for c, bf in enumerate(flags) if bf]
        fl_idx = [c for c, bf in enumerate(flags) if not bf]
        if not bf_idx:
            print(f"{prefix} {well} s{site:02d}: no brightfield channel — "
                  f"writing fluorescence only.")
        if not fl_idx:
            print(f"{prefix} {well} s{site:02d}: no fluorescence channel — "
                  f"writing brightfield only.")
        if bf_idx:
            sub, sub_ome = _subset(arr, ome, bf_idx)
            _write(args.output / args.bf_subdir / f"{stem}_{args.bf_suffix}.ome.tiff",
                   sub, sub_ome, "brightfield", prefix)
        if fl_idx:
            sub, sub_ome = _subset(arr, ome, fl_idx)
            _write(args.output / args.fl_subdir / f"{stem}_{args.fl_suffix}.ome.tiff",
                   sub, sub_ome, "fluorescence", prefix)

    print(f"\nDone → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
