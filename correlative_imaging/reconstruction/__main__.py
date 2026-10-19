"""Command line of the particle-identity tool (run from the repo root, no install needed):

  python -m correlative_imaging.reconstruction gui [--results <output folder>]
  python -m correlative_imaging.reconstruction detect --root <run output folder> --run-tag <tag> --out <dir> [--config romk]
  python -m correlative_imaging.reconstruction detect --spec source.json --out <dir> [--config romk]
  python -m correlative_imaging.reconstruction classify --out <dir> [--model <file.joblib>] [--config ...]
  python -m correlative_imaging.reconstruction train --results <dir> [<dir> ...] --labels ci_labels.db --model <name>
        [--arm self-train] [--rounds 3]
  python -m correlative_imaging.reconstruction import-labels --csv labels.csv --labels ci_labels.db --dataset <name>
        --names GFP=488nm mScarlet=561nm ... [--wells-csv wells.csv]

`--config` takes a JSON file or a preset name (identity/presets/*.json).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MODELS, load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m correlative_imaging.reconstruction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gui")
    g.add_argument("--results")
    d = sub.add_parser("detect", help="stage 1: particles, reference, features (no model needed)")
    d.add_argument("--root"); d.add_argument("--run-tag")
    d.add_argument("--spec", help="JSON source spec instead of --root/--run-tag (e.g. a viewer-crop source)")
    d.add_argument("--out", required=True); d.add_argument("--config"); d.add_argument("--wells", nargs="*")
    d.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("classify", help="stage 2: classify, reconstruct, identity call")
    c.add_argument("--out", required=True); c.add_argument("--model"); c.add_argument("--config")
    c.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train", help="fit the particle classifier on hand labels (+ all particles)")
    t.add_argument("--results", nargs="+", required=True); t.add_argument("--labels", required=True)
    t.add_argument("--model", required=True); t.add_argument("--arm"); t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--arms", nargs="+", help="arms to compare (default: all)")
    i = sub.add_parser("import-labels", help="import a first labelling session (CSV)")
    i.add_argument("--csv", required=True); i.add_argument("--labels", required=True)
    i.add_argument("--dataset", required=True); i.add_argument("--names", nargs="+", required=True)
    i.add_argument("--wells-csv")
    a = ap.parse_args(argv)

    if a.cmd == "gui":
        from .gui.app import launch
        raise SystemExit(launch(a.results))
    if a.cmd == "detect":
        import json
        from .pipeline import detect_run, source_from_spec
        cfg = load_config(a.config)
        spec = json.loads(Path(a.spec).read_text()) if a.spec else dict(kind="run", root=a.root, run_tag=a.run_tag)
        r = detect_run(source_from_spec(spec, cfg), cfg, a.out, a.wells, a.workers, source_spec=spec)
        print(f"{len(r['wells'])} wells, {int(r['particles'].pool.sum())} particles in the holes -> {a.out}")
    elif a.cmd == "classify":
        from .learn import ParticleModel
        from .pipeline import Results, classify_run
        cfg = load_config(a.config) if a.config else Results(a.out).config
        model = ParticleModel.load(a.model) if a.model else None
        r = classify_run(a.out, cfg, model, a.workers)
        print(r["calls"].call.value_counts().to_string())
    elif a.cmd == "train":
        from .labels import LabelStore
        from .learn import assemble, train
        T, tg, rep = assemble(a.results, LabelStore(a.labels))
        print(rep.to_string(index=False))
        from .learn import ARMS
        if rep.attrs.get("feature_config_differs"):
            print("WARNING: the results folders were detected with different feature settings")
        res = train(T, tg, arms=a.arms or ARMS, rounds=a.rounds, final_arm=a.arm, progress=print,
                    feature_config=rep.attrs.get("feature_config"))
        pd.set_option("display.width", 200)
        print("grouped CV (choose by this):\n", res.cv.round(3).to_string(index=False))
        print("frozen test wells (report only):\n", res.test.round(3).to_string(index=False))
        for n in res.notes:
            print("note:", n)
        path = Path(a.model)
        path = path if path.suffix else MODELS / f"{a.model}.joblib"
        res.model.save(path)
        print(f"saved {res.model.spec['arm']} -> {path}")
    elif a.cmd == "import-labels":
        from .labels import LabelStore
        names = dict(n.split("=", 1) for n in a.names)
        wells = None
        if a.wells_csv:
            w = pd.read_csv(a.wells_csv, dtype=str)
            wells = set(zip(w.plate_id, w.well_id))
        print(LabelStore(a.labels).import_legacy(a.csv, a.dataset, names, wells))


if __name__ == "__main__":
    main()
