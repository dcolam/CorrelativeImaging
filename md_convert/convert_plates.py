"""Convert a folder of plates in parallel — one worker per plate folder.

Expected layout (new ImageXpress output)::

    pnc_n3-5_mdx_raw/
      PNC_n3_18T40665/                              ← one plate
        PNC_n3_18T40665_20260903_195035/            ← acquisition
          experiment_z_stack/timepoint0/*.tif       ← converted
          experiment/  autofocus/                   ← ignored

Each plate is converted with ``convert_new_md`` into a mirrored folder next to
the input (``*_mdx_raw`` → ``*_converted``, otherwise ``<input>_converted``)::

    pnc_n3-5_converted/
      PNC_n3_18T40665/brightfield/*.ome.tiff
      PNC_n3_18T40665/fluorescence/*.ome.tiff
      PNC_n3_18T40665/convert.log

If a plate holds more than one acquisition, each goes to its own sub-folder
``<plate>/<acquisition>/``.

Usage
-----
    python convert_plates.py /mnt/z/.../pnc_n3-5_mdx_raw
    python convert_plates.py IN OUT --jobs 4          # explicit output, cap workers
    python convert_plates.py IN --bf-channels 0       # other options → convert_new_md
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import convert_new_md


def _default_output(input_dir: Path) -> Path:
    name = input_dir.name
    stem = name[: -len("_mdx_raw")] if name.endswith("_mdx_raw") else name
    return input_dir.parent / f"{stem}_converted"


def _jobs_for_plate(plate: Path, out: Path, subdir: str) -> list[tuple[Path, Path]]:
    """(source, destination) pairs: one per acquisition holding *subdir*."""
    srcs = sorted(p for p in plate.glob(f"*/{subdir}") if p.is_dir())
    if len(srcs) == 1:
        return [(srcs[0], out)]
    return [(s, out / s.parent.name) for s in srcs]


def _convert_plate(plate: str, pairs: list[tuple[Path, Path]],
                   extra: list[str]) -> tuple[str, int, float]:
    """Run convert_new_md for every acquisition of one plate, logging to convert.log."""
    t0 = time.time()
    rc = 0
    for src, dst in pairs:
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "convert.log", "w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"source: {src}")
            try:
                rc = convert_new_md.main([str(src), str(dst), *extra]) or rc
            except SystemExit as e:         # argparse errors
                rc = e.code if isinstance(e.code, int) else 1
            except Exception:
                traceback.print_exc()
                rc = 1
    return plate, rc, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="folder containing one sub-folder per plate")
    ap.add_argument("output", type=Path, nargs="?",
                    help="output root (default: sibling '<name>_converted')")
    ap.add_argument("--jobs", type=int, default=None,
                    help="max parallel workers (default: one per plate)")
    ap.add_argument("--subdir", default="experiment_z_stack",
                    help="acquisition sub-folder to convert (default experiment_z_stack)")
    args, extra = ap.parse_known_args(argv)

    out_root = args.output or _default_output(args.input)
    work = {}
    for plate in sorted(p for p in args.input.iterdir() if p.is_dir()):
        pairs = _jobs_for_plate(plate, out_root / plate.name, args.subdir)
        if pairs:
            work[plate.name] = pairs
        else:
            print(f"  skipping {plate.name}: no */{args.subdir} folder", file=sys.stderr)
    if not work:
        print(f"No plates with */{args.subdir} found in {args.input}", file=sys.stderr)
        return 1

    jobs = min(args.jobs or len(work), len(work))
    print(f"{len(work)} plate(s), {jobs} worker(s) → {out_root}"
          + (f"\npassing through: {' '.join(extra)}" if extra else ""))

    failed = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_convert_plate, name, pairs, extra)
                   for name, pairs in work.items()]
        for fut in as_completed(futures):
            name, rc, secs = fut.result()
            status = "ok" if rc == 0 else f"FAILED (exit {rc}) — see convert.log"
            print(f"  {name}: {status}  {secs / 60:.1f} min")
            if rc != 0:
                failed.append(name)

    if failed:
        print(f"\n{len(failed)} plate(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nDone → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
