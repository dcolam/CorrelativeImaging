"""Command line for the downstream analyzer (prototype; no install needed, run from the repo root):

  python -m correlative_imaging.downstream run --root <output folder> --run-tag <run_..._HHMMSS> --out <dir>
  python -m correlative_imaging.downstream run --dev-v5 <workflow_comparison dir> --v5-db <romk/imaging/v5> --out <dir>
      [--col-data plate_map.csv] [--config cfg.json] [--wells 18T39412_N11 ...] [--workers 8] [--reference tiers.csv]
  python -m correlative_imaging.downstream gui [--root <output folder>] [--results <results folder>]
  python -m correlative_imaging.downstream fit-models --labels labels.csv --out models/<name>.json [--config cfg.json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m correlative_imaging.downstream")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score, refine and call every well of a run; write tables + viewer")
    r.add_argument("--root"); r.add_argument("--run-tag")
    r.add_argument("--dev-v5"); r.add_argument("--v5-db")
    r.add_argument("--out", required=True); r.add_argument("--config")
    r.add_argument("--col-data", help="CSV with plate_id, well_id + experiment columns (display + evaluation only)")
    r.add_argument("--wells", nargs="*"); r.add_argument("--workers", type=int, default=8)
    r.add_argument("--reference", help="reference calls CSV (plate_id, well_id, tier, call) to compare against")
    r.add_argument("--title", default="Patched-cell identity")
    g = sub.add_parser("gui", help="open the GUI (run, settings, review, results)")
    g.add_argument("--root"); g.add_argument("--results")
    f = sub.add_parser("fit-models", help="fit P(cell) / P(neighbour) on hand-labelled particles")
    f.add_argument("--labels", required=True); f.add_argument("--out", required=True); f.add_argument("--config")
    a = ap.parse_args(argv)
    if a.cmd == "gui":
        from .gui import launch
        raise SystemExit(launch(a.root, a.results))
    cfg = load_config(a.config)

    if a.cmd == "fit-models":
        from .models import fit_models
        m = fit_models(a.labels, list(cfg.channels.values()))
        m.save(a.out)
        print(f"P(cell) cv AUC {m.spec['cell']['cv_auc']:.3f} | P(neighbour) cv AUC {m.spec['nb']['cv_auc']:.3f} -> {a.out}")
        return

    from .pipeline import run_downstream
    from .sources import DevV5Source, RunSource
    src = DevV5Source(a.dev_v5, a.v5_db, cfg) if a.dev_v5 else RunSource(a.root, a.run_tag, cfg)
    cd = None
    if a.col_data:
        from .evaluate import read_col_data
        cd = read_col_data(a.col_data)
    res = run_downstream(src, cfg, a.out, wells=a.wells, workers=a.workers, col_data=cd, title=a.title)
    print(f"wrote {a.out}: {len(res['calls'])} wells, {len(res['cells'])} cells, viewer at {Path(a.out) / 'viewer/index.html'}")
    if cd is not None and {"Condition", "Induction"} <= set(cd.columns):
        from .evaluate import report
        ref = None
        if a.reference:
            ref = pd.read_csv(a.reference, dtype={"plate_id": str})[["plate_id", "well_id", "tier", "call"]]
            ref["tier"] = ref.tier.replace({"ROMK only": "marker only"})
        txt = report(res["calls"], cd, ref)
        (Path(a.out) / "evaluation.txt").write_text(txt, encoding="utf-8")
        print(txt)


if __name__ == "__main__":
    main()
