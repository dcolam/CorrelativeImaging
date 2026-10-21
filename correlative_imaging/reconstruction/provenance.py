"""How a results folder was made: its history, the commands that make it again, and a methods text.

workflow.json   append-only, one event per stage (detect, classify, edit, train): the settings that stage ran with
                (run.json keeps only the latest), its inputs (source, well subset, label library, model file + sha1,
                hashes of the hand edits and cell labels) and the software (package versions, git commit, dirty or
                not). Written at the end of each stage; a failed write never fails the stage.
commands()      the command lines that make the results again from the same inputs, and the files they need
                (write_reproduce() saves them into <results>/reproduce/ on request).
methods()       a methods paragraph built only from what the results folder records: the settings each stage ran
                with, how the thresholds were obtained per well, the model's provenance and the identity calibration.
                Nothing it says is assumed; what was not recorded (older results) is said to be not recorded.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

EVENTS = "workflow.json"
PACKAGES = ["numpy", "scipy", "scikit-image", "scikit-learn", "pandas", "napari"]
CLI = "python -m correlative_imaging.reconstruction"


# ------------------------------------------------------------------------------------------------------------------
# software and hashes

def _git(repo: Path) -> dict:
    """Commit and dirty flag from the .git folder (read only; git itself only for the dirty check, with a timeout)."""
    out = {}
    g = repo / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = g / ref
            if p.exists():
                out["commit"] = p.read_text().strip()
            else:                                            # packed refs
                for line in (g / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        out["commit"] = line.split(" ", 1)[0]
            out["branch"] = ref.rsplit("/", 1)[-1]
        else:
            out["commit"] = head
    except OSError:
        return out
    try:
        import subprocess
        r = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out["dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return out


def software() -> dict:
    from importlib import metadata
    import platform
    import sys
    v = {}
    for p in PACKAGES:
        try:
            v[p] = metadata.version(p)
        except Exception:
            pass
    repo = Path(__file__).resolve().parents[2]
    from .. import __version__
    return dict(correlative_imaging=__version__, python=sys.version.split()[0], platform=platform.platform(),
                packages=v, git=_git(repo))


def sha1_file(path) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha1_obj(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# events

def events(out_dir) -> list:
    try:
        return list(json.loads((Path(out_dir) / EVENTS).read_text()).get("events", []))
    except (OSError, ValueError, AttributeError):
        return []


def record(out_dir, step: str, **info):
    """Append one event (never raises: provenance must not fail a stage)."""
    try:
        d = Path(out_dir)
        ev = dict(step=step, t=datetime.datetime.now().isoformat(timespec="seconds"), software=software(), **info)
        all_ = events(d) + [ev]
        tmp = d / (EVENTS + ".tmp")
        tmp.write_text(json.dumps(dict(events=all_), indent=2, default=str))
        os.replace(tmp, d / EVENTS)
    except Exception:
        pass


def last(out_dir, step: str) -> dict | None:
    ev = [e for e in events(out_dir) if e.get("step") == step]
    return ev[-1] if ev else None


# ------------------------------------------------------------------------------------------------------------------
# configs as the stages ran

def _complete(cfg_dict: dict) -> dict:
    """A stored config as a full, loadable Config (keys of older runs filled as they implicitly were)."""
    from .config import LEGACY_FEATURE_VALUES, Config
    c = Config()
    for k, v in LEGACY_FEATURE_VALUES.items():
        if k not in cfg_dict:
            setattr(c, k, v)
    for k, v in (cfg_dict or {}).items():
        if hasattr(c, k):
            setattr(c, k, v)
    return json.loads(c.to_json())


def stage_configs(out_dir) -> tuple[dict, dict, list]:
    """(detect config, classify config, notes on what was not recorded)."""
    run = json.loads((Path(out_dir) / "run.json").read_text())
    notes = []
    d, c = last(out_dir, "detect"), last(out_dir, "classify")
    if d is None:
        notes.append("the Detect settings were not recorded (results made before workflow.json): the settings "
                     "stored in run.json are assumed for Detect")
    if c is None and run.get("classified"):
        notes.append("the Classify settings were not recorded separately: the settings stored in run.json are used")
    detect_cfg = _complete(d["config"] if d else run.get("config", {}))
    classify_cfg = _complete(c["config"] if c else run.get("config", {}))
    return detect_cfg, classify_cfg, notes


# ------------------------------------------------------------------------------------------------------------------
# commands

def _q(p) -> str:
    s = str(p)
    return f'"{s}"' if any(ch in s for ch in ' &()') or "\\" in s else s


def commands(out_dir, where: str | None = None) -> tuple[str, dict]:
    """(text, files): the command lines that make these results again, and the files they read ({name: content}).
    where: the folder the files will be saved in (default <results>/reproduce)."""
    out = Path(out_dir)
    run = json.loads((out / "run.json").read_text())
    rdir = Path(where) if where else out / "reproduce"
    detect_cfg, classify_cfg, notes = stage_configs(out)
    d, c = last(out, "detect"), last(out, "classify")
    new = f"{out}_reproduced"
    files = {"source.json": json.dumps(run.get("source") or {}, indent=2),
             "detect_config.json": json.dumps(detect_cfg, indent=2)}
    lines = ["# Make these results again (same inputs: the images / databases of the source, the label library, "
             "the model file).", f"# results: {out}", ""]
    if not run.get("source"):
        notes.append("run.json holds no source specification: give --root/--run-tag of the original run to detect")
    det = [CLI, "detect", "--spec", _q(rdir / "source.json"), "--config", _q(rdir / "detect_config.json"),
           "--out", _q(new)]
    if d and d.get("wells"):
        det += ["--wells"] + list(d["wells"])
    lib = (d or {}).get("labels") or (c or {}).get("labels")
    if (d or {}).get("n_edited_wells") or run.get("n_edited_wells"):
        if lib:
            det += ["--labels", _q(lib)]
        else:
            notes.append("Detect applied hand edits from a label library that was not recorded: add --labels <library>")
    lines += ["# 1  Detect: particles, field reference, features", " ".join(det), ""]
    if run.get("classified"):
        files["classify_config.json"] = json.dumps(classify_cfg, indent=2)
        cls = [CLI, "classify", "--out", _q(new), "--config", _q(rdir / "classify_config.json")]
        arm = (run.get("model") or {}).get("arm")
        mfile = (c or {}).get("model_file")
        if arm and arm != "preview":
            if mfile:
                cls += ["--model", _q(mfile)]
            else:
                from .config import Config
                mp = Config(model=classify_cfg.get("model", "")).model_path()
                cls += ["--model", _q(mp) if mp else "<model file>"]
                notes.append("the model file was not recorded: the one named in the settings is used")
        if (c or {}).get("n_cell_labels"):
            cls += ["--labels", _q(c.get("labels") or "<label library>")]
        lines += ["# 2  Classify: particle classes, reconstruction, colour call" +
                  (" (Detect preview, no trained model)" if arm == "preview" else ""), " ".join(cls), ""]
        spec = run.get("model") or {}
        if arm and arm != "preview":
            t = last(out, "train")
            lines += ["# How the model was made (reproduces only with the same labels: labels sha1 "
                      f"{spec.get('labels_sha1', '?')[:12]}, {spec.get('n_train', '?')} training labels):",
                      "# " + " ".join([CLI, "train", "--results", _q(out), "--labels",
                                       _q((t or {}).get("labels") or lib or "<label library>"),
                                       "--model", "<name>", "--arm", str(arm)]), ""]
    if classify_cfg.get("path_map") or detect_cfg.get("path_map"):
        notes.append("path_map in the settings maps stored paths to this computer's drives: adjust it on another one")
    if notes:
        lines += ["# Notes:"] + [f"#  - {n}" for n in notes]
    return "\n".join(lines) + "\n", files


def write_reproduce(out_dir) -> Path:
    """Save the commands and the files they read into <results>/reproduce/ (with a README of the inputs)."""
    out = Path(out_dir)
    rdir = out / "reproduce"
    rdir.mkdir(exist_ok=True)
    text, files = commands(out, rdir)
    for name, content in files.items():
        (rdir / name).write_text(content)
    (rdir / "commands.txt").write_text(text)
    c, d = last(out, "classify") or {}, last(out, "detect") or {}
    readme = ["Inputs that must come along to make these results again:",
              f"  source (images / databases): see source.json",
              f"  label library: {c.get('labels') or d.get('labels') or 'not recorded'}"
              + (f" (sha1 {sha1_file(c.get('labels') or d.get('labels'))})" if (c.get('labels') or d.get('labels'))
                 and sha1_file(c.get('labels') or d.get('labels')) else ""),
              f"  model file: {c.get('model_file') or 'none (Detect preview) or not recorded'}"
              + (f" (sha1 {c.get('model_sha1')})" if c.get("model_sha1") else ""),
              "", "Software of the last Classify (or Detect):",
              json.dumps((c or d).get("software", {}) or "not recorded", indent=2)]
    (rdir / "README.txt").write_text("\n".join(readme) + "\n")
    return rdir


# ------------------------------------------------------------------------------------------------------------------
# methods text

DISPLAY_ONLY = {"display": "names and colours on screen", "path_map": "where this computer finds the files",
                "view_R": "how much of the frame is shown when there is no hole crop"}

ARM_TEXT = {
    "legacy": "a logistic regression on four features of the reference workflow (SNR, area, share inside the hole, "
              "hole fill)",
    "hand": "gradient-boosted trees on all features, trained on the hand labels",
    "spread": "label spreading over a k-nearest-neighbour graph of all particles, followed by a kNN classifier",
    "self-train": "gradient-boosted trees trained on the hand labels plus confident pseudo-labels of unlabelled "
                  "particles (self-training)",
    "self-train+rules": "gradient-boosted trees trained on the hand labels plus confident pseudo-labels "
                        "(self-training), started from colour-blind rules as weak labels",
}


class _Used(dict):
    """A config dict that remembers which settings the text used (the test checks every setting is covered)."""

    def __init__(self, *a):
        super().__init__(*a)
        self.used = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)


def _n(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _n_features(out: Path, features: list) -> int | str:
    """How many of the features exist in the stored particles (older results may hold fewer than today's list)."""
    try:
        import pyarrow.parquet as pq
        names = set(pq.read_schema(out / "particles.parquet").names)
        return sum(f in names for f in features)
    except Exception:
        return len(features)


def methods(out_dir, _used: set | None = None) -> str:
    """The methods paragraph (plain text, one paragraph per stage)."""
    from .features import FEATURES
    from .identify import MIN_CELLS, MIN_USER, S0
    out = Path(out_dir)
    run = json.loads((out / "run.json").read_text())
    dcfg, ccfg, notes = stage_configs(out)
    D, C = _Used(dcfg), _Used(ccfg)
    chans = run.get("channels", [])
    name = lambda ch: (ccfg.get("display", {}).get(ch, {}) or {}).get("name") or ch
    P = []

    # detection
    try:
        W = pd.read_csv(out / "wells.csv")
        n_w = len(W)
        notes_w = W.get("notes", pd.Series("", index=W.index)).fillna("").astype(str)
        n_trw = int(notes_w.str.contains("whole-image threshold computed").sum())
        n_trh = int(notes_w.str.contains("hole threshold computed").sum())
        crop = int(notes_w.str.contains("viewer crop").sum())
    except Exception:
        n_w, n_trw, n_trh, crop = run.get("n_wells", "?"), 0, 0, 0
    src = (run.get("source") or {}).get("kind", "run")
    thr_src = ("the CorrelativeImaging run's databases (its whole-image threshold and its threshold within the hole "
               "selection, with the method that run was configured with)" if src == "run" else
               "the source's wells table (the thresholds of the workflow that made the crops)")
    used_ch = D["channels"]
    subset = bool(used_ch) and set(used_ch) != set(chans)
    t = (f"Particle detection. In each of {n_w} wells, particles were detected in every fluorescence channel "
         f"({', '.join(name(c) for c in chans)}"
         + (f"; channels used: {', '.join(name(c) for c in used_ch)}" if subset else "")
         + ") at two thresholds: a stringent one (the whole-image threshold) and a lenient one (the threshold "
         "computed on the hole pixels), each applied to the whole detected area so that a cell crossing the rim of "
         f"the hole stays whole. Thresholds were taken from {thr_src}")
    if n_trw or n_trh:
        t += (f"; where they were missing they were recomputed with the triangle method (whole-image threshold in "
              f"{n_trw} wells, hole threshold in {n_trh} wells)")
    t += (f". Objects smaller than {_n(D['min_object_px'])} pixels were removed before labelling, and a particle "
          f"had to cover at least {_n(D['min_inside_um2'])} µm² of the hole. Each particle was described by "
          f"{_n_features(out, FEATURES)} features measured within ±{_n(D['work_R'])} hole radii of the hole (neighbourhood "
          f"features within ±{_n(D['context_R'])} hole radii), with brightness in units of a typical field cell "
          f"(per plate and channel, the median signal of whole-image objects of at least "
          f"{_n(D['field_min_um2'])} µm²). Peaks were counted on lightly smoothed images "
          f"(σ = {_n(D['smooth_px'])} px) as h-maxima at least {_n(D['peak_h'])} hole-noise SDs high.")
    if D["split_particles"]:
        t += (f" Particles touching the hole and at most {_n(D['split_max_x_hole'])} hole areas large were split "
              f"where the saddle between two peaks fell below {_n(D['split_dip_ratio'])} of the lower peak.")
    else:
        t += " Particles were not split automatically at Detect."
    if crop:
        t += f" In {crop} wells the images were viewer crops, so neighbours beyond the crop are cut."
    if run.get("n_edited_wells") or run.get("edited_wells"):
        t += f" Hand edits (cuts, merges, splits) were applied in {run.get('n_edited_wells', 0)} wells."
    P.append(t + f" In total {run.get('n_pool', '?')} particles (both thresholds counted) touched a hole.")

    # classification
    spec = run.get("model") or {}
    arm = spec.get("arm")
    if not run.get("classified"):
        P.append("Classification. Not run yet.")
    elif arm == "preview":
        C["model"]
        P.append("Particle classification. No trained classifier was used: particles got the provisional classes of "
                 "the Detect preview (rule-based: inside the hole / neighbour / debris / noise).")
    else:
        counts = spec.get("train_counts") or {}
        t = (f"Particle classification. Every particle was classified into {', '.join(spec.get('classes') or [])} "
             f"by {ARM_TEXT.get(arm, arm)}, trained on {spec.get('n_train', '?')} hand-labelled particles"
             + (f" ({', '.join(f'{k} {v}' for k, v in counts.items())})" if counts else "")
             + (f" from the dataset(s) {', '.join(spec.get('datasets') or [])}" if spec.get("datasets") else "")
             + (f" (model file {Path(C['model']).name})" if C["model"] else "")
             + ". Labels in the frozen test wells were never used for training.")
        test = (spec.get("test") or {}).get(arm) or {}
        if test.get("macro_f1") is not None:
            t += (f" On {spec.get('n_test', '?')} particles labelled blind in the frozen test wells, the macro F1 "
                  f"was {test['macro_f1']:.2f}.")
        cv = (spec.get("cv") or {}).get(arm) or {}
        if cv.get("macro_f1") is not None and cv.get("macro_f1") == cv.get("macro_f1"):
            t += f" Cross-validation grouped by well gave a macro F1 of {cv['macro_f1']:.2f}."
        if spec.get("sklearn"):
            t += f" (scikit-learn {spec['sklearn']}.)"
        P.append(t)

    # reconstruction
    if run.get("classified"):
        idc = run.get("identity_channels") or []
        markers = C["markers"]
        kept = "neighbours, debris and plugs" if "plug" in (spec.get("classes") or []) else "neighbours and debris"
        t = (f"Cell reconstruction. Per hole, a signal map was built as the maximum over the identity channels "
             f"({', '.join(name(c) for c in idc) or 'all'}) of the hole-background-subtracted signal in hole-noise "
             f"SDs. Particles with P(cell) + P(merged) of at least {_n(C['p_seed'])} seeded cells; {kept} were "
             "kept apart. Basins grew by watershed from peaks at least "
             f"{_n(C['peak_h'])} SDs high over pixels at least {_n(C['grow_floor'])} SDs above the hole background.")
        if C["split_dips"]:
            t += (f" Neighbouring basins were joined unless the saddle between them fell below {_n(C['dip_ratio'])} "
                  "of the lower peak; a cell basin and a neighbour basin were never joined.")
        else:
            t += " Connected bright areas were not split at signal dips."
        t += (f" Seeds of different channels closer than {_n(C['cell_gap_um'])} µm were joined. A cell kept the "
              f"pixels of its basins with at least {_n(C['grow_frac'])} of its seed's median signal, within "
              f"{_n(C['grow_margin_um'])} µm of the hole, connected to the seed; above {_n(C['grow_max_x_hole'])} "
              "hole areas the relative cut was raised in steps until the cell fitted.")
        if markers:
            t += f" Marker channels ({', '.join(name(c) for c in markers)}) were measured but did not shape the cells."
        P.append(t)

        # colour call
        try:
            info = json.loads((out / "identity.json").read_text())
        except (OSError, ValueError):
            info = {}
        per = []
        for ch in idc:
            i = info.get(ch, {})
            if i.get("user"):
                per.append(f"{name(ch)}: logistic fit of your judged cells ({i['user'].get('n_pos')} positive, "
                           f"{i['user'].get('n_neg')} negative)")
            elif i.get("ashman_d") is not None:
                per.append(f"{name(ch)}: mixture, Ashman's D = {i['ashman_d']:.2f}"
                           + (" (weak separation)" if i.get("weak_separation") else ""))
            elif i:
                per.append(f"{name(ch)}: fewer than {MIN_CELLS} cells, P from the body SNR")
        bg = C["identity_bg"]
        t = ("Colour call. Each cell's body mean in every channel, minus the "
             + ("hole background (median of the hole pixels outside every object)" if bg == "hole"
                else "frame background")
             + ", was divided by the typical field cell of that plate and channel. Per identity channel, "
             f"P(positive) came from a two-component Gaussian mixture of asinh(n / {S0}) over all reconstructed "
             f"cells (at least {MIN_CELLS} cells), or, once at least {MIN_USER} positive and {MIN_USER} negative "
             "cells were judged by hand outside the test wells, from a logistic fit of those judgements"
             + (f" ({'; '.join(per)})" if per else "")
             + f". Mixtures with Ashman's D below {_n(C['min_mixture_sep'])} were flagged. The probability of every "
             "combination of identity channels was the product of the per-channel probabilities; the most likely "
             f"combination was called when its probability was at least {_n(C['p_call'])}, otherwise the cell was "
             "'uncertain'. The well's call is its best cell (P(cell) × share of its body inside the hole).")
        P.append(t)
        try:
            calls = pd.read_csv(out / "calls.csv")
            st = calls["status"].value_counts().to_dict() if "status" in calls else {}
            unc = int((calls["call"].astype(str) == "uncertain").sum()) if "call" in calls else 0
            P.append("Result. " + ", ".join(f"{v} wells {k}" for k, v in st.items())
                     + (f"; {unc} calls uncertain." if unc else "."))
        except Exception:
            pass

    # software
    ev = last(out, "classify") or last(out, "detect")
    if ev and ev.get("software"):
        s = ev["software"]
        pk = s.get("packages", {})
        g = s.get("git", {})
        P.append("Software. CorrelativeImaging " + s.get("correlative_imaging", "?")
                 + (f" (git {g['commit'][:10]}{', with uncommitted changes' if g.get('dirty') else ''})"
                    if g.get("commit") else "")
                 + f", Python {s.get('python', '?')}, " + ", ".join(f"{k} {v}" for k, v in pk.items()) + ".")
    else:
        notes.append("the software versions were not recorded (results made before workflow.json)")
    if notes:
        P.append("Not recorded: " + "; ".join(notes) + ".")
    if _used is not None:
        _used |= D.used | C.used
    return "\n\n".join(P) + "\n"
