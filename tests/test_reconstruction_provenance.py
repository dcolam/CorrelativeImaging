"""Provenance (reconstruction/provenance.py): history, reproduce commands and the methods text, on hand-made results
folders. The generated commands are parsed by the real command line; the methods text must cover every setting."""
from __future__ import annotations

import dataclasses
import json
import shlex

import pandas as pd
import pytest

from correlative_imaging.reconstruction import provenance as pv
from correlative_imaging.reconstruction.config import Config


def _cfg(**kw):
    c = json.loads(Config().to_json())
    c.update(kw)
    return c


def _results(d, model=None, classified=True, cfg=None, source=True):
    d.mkdir(parents=True, exist_ok=True)
    run = dict(dataset="ds", channels=["488nm", "561nm"], config=cfg or _cfg(), detected="2026-10-21T10:00:00",
               n_wells=3, n_pool=12, n_edited_wells=0,
               source=dict(kind="run", root="C:\\data\\my run", run_tag="exp_20261021_101010") if source else {})
    if classified:
        run.update(classified="2026-10-21T11:00:00", model=model or dict(arm="preview"),
                   identity_channels=["488nm", "561nm"])
        pd.DataFrame(dict(status=["one cell", "no cell", "one cell"], call=["488nm", "no cell", "uncertain"])) \
            .to_csv(d / "calls.csv", index=False)
        (d / "identity.json").write_text(json.dumps({"488nm": dict(ashman_d=3.1, weak_separation=False),
                                                     "561nm": dict(user=dict(n_pos=7, n_neg=9))}))
    (d / "run.json").write_text(json.dumps(run))
    pd.DataFrame(dict(plate_id="P", well_id=["A01", "A02", "A03"],
                      notes=["", "488nm: whole-image threshold computed (triangle)",
                             "561nm: hole threshold computed (triangle)"])).to_csv(d / "wells.csv", index=False)
    return d


def test_old_results_say_what_was_not_recorded(tmp_path):
    d = _results(tmp_path / "r")
    text = pv.methods(d)
    assert "Not recorded:" in text and "Detect settings were not recorded" in text
    assert "recomputed with the triangle method (whole-image threshold in 1 wells, hole threshold in 1 wells)" in text
    assert "Detect preview" in text or "No trained classifier" in text
    cmd, files = pv.commands(d)
    assert "were not recorded" in cmd and "detect_config.json" in files


def test_record_appends_with_software(tmp_path):
    d = _results(tmp_path / "r")
    pv.record(d, "detect", config=_cfg(), wells=["P_A01"])
    pv.record(d, "classify", config=_cfg(dip_ratio=0.5))
    ev = pv.events(d)
    assert [e["step"] for e in ev] == ["detect", "classify"]
    s = ev[0]["software"]
    assert s["packages"].get("numpy") and "python" in s
    pv.record(tmp_path / "missing" / "x", "detect")              # never raises


def test_stage_configs_kept_apart(tmp_path):
    d = _results(tmp_path / "r", cfg=_cfg(dip_ratio=0.5))
    pv.record(d, "detect", config=_cfg(min_object_px=40))
    pv.record(d, "classify", config=_cfg(dip_ratio=0.5))
    dc, cc, notes = pv.stage_configs(d)
    assert dc["min_object_px"] == 40 and cc["dip_ratio"] == 0.5 and not notes
    text = pv.methods(d)
    assert "smaller than 40 pixels" in text and "below 0.5 of the lower peak" in text


def _parse(line, monkeypatch, calls):
    """Run a generated command line through the real CLI with the heavy work replaced."""
    import correlative_imaging.reconstruction.learn as L
    import correlative_imaging.reconstruction.pipeline as P
    from correlative_imaging.reconstruction.__main__ import main
    monkeypatch.setattr(P, "source_from_spec", lambda spec, cfg: calls.append(("source", spec, cfg)) or object())
    monkeypatch.setattr(P, "detect_run", lambda *a, **k: calls.append(("detect", a, k)) or
                        dict(wells=[1], particles=pd.DataFrame(dict(pool=[True]))))
    monkeypatch.setattr(P, "classify_run", lambda *a, **k: calls.append(("classify", a, k)) or
                        dict(calls=pd.DataFrame(dict(call=["x"]))))
    monkeypatch.setattr(L.ParticleModel, "load", classmethod(lambda cls, p: calls.append(("model", p)) or "M"))
    argv = shlex.split(line, posix=False)
    argv = [a.strip('"') for a in argv]
    assert argv[:3] == ["python", "-m", "correlative_imaging.reconstruction"]
    main(argv[3:])


def test_commands_parse_and_load(tmp_path, monkeypatch):
    model = dict(arm="hand", classes=["cell"], n_train=40, labels_sha1="abc", test={}, cv={})
    d = _results(tmp_path / "r", model=model, cfg=_cfg(dip_ratio=0.5, model="m.joblib"))
    (tmp_path / "m.joblib").write_bytes(b"x")
    pv.record(d, "detect", config=_cfg(min_object_px=40), wells=["P_A01", "P_A02"], n_edited_wells=2,
              labels=str(tmp_path / "lib.db"))
    pv.record(d, "classify", config=_cfg(dip_ratio=0.5, model="m.joblib"), model_file=str(tmp_path / "m.joblib"),
              labels=str(tmp_path / "lib.db"), n_cell_labels=4)
    rdir = pv.write_reproduce(d)
    assert {p.name for p in rdir.iterdir()} >= {"source.json", "detect_config.json", "classify_config.json",
                                                "commands.txt", "README.txt"}
    lines = [l for l in (rdir / "commands.txt").read_text().splitlines() if l.startswith("python")]
    assert len(lines) == 2
    calls = []
    for line in lines:
        if " classify " in line:
            (tmp_path / "lib.db").write_bytes(b"")                    # the library must exist for --labels
            import correlative_imaging.reconstruction.labels as LB
            monkeypatch.setattr(LB.LabelStore, "cell_frame", lambda self, ds: pd.DataFrame())
            monkeypatch.setattr(LB.LabelStore, "test_wells", lambda self, ds: set())
            import correlative_imaging.reconstruction.pipeline as P
            monkeypatch.setattr(P, "Results", lambda out: type("R", (), {"dataset": "ds"})())
        _parse(line, monkeypatch, calls)
    det = next(c for c in calls if c[0] == "detect")
    assert det[1][3] == ["P_A01", "P_A02"] and det[2]["labels_path"].endswith("lib.db")
    assert det[1][1].min_object_px == 40                              # the Detect-time settings
    cls = next(c for c in calls if c[0] == "classify")
    assert cls[1][1].dip_ratio == 0.5                                 # the Classify-time settings
    assert any(c[0] == "model" and str(c[1]).endswith("m.joblib") for c in calls)
    assert "--arm hand" in (rdir / "commands.txt").read_text()       # how the model was made


def test_preview_has_no_model_and_old_source_is_flagged(tmp_path):
    d = _results(tmp_path / "r", source=False)
    text, files = pv.commands(d)
    assert "--model" not in text and "Detect preview" in text
    assert "no source specification" in text


def test_methods_cover_every_setting(tmp_path):
    d = _results(tmp_path / "r", model=dict(arm="hand", classes=["cell", "noise"], n_train=40,
                                            test={"hand": {"macro_f1": 0.81}}, cv={}),
                 cfg=_cfg(split_particles=True, split_dips=True, markers=["561nm"], model="m.joblib"))
    used = set()
    text = pv.methods(d, _used=used)
    all_fields = {f.name for f in dataclasses.fields(Config)}
    missing = all_fields - used - set(pv.DISPLAY_ONLY)
    assert not missing, f"settings the methods text never mentions: {missing}"
    assert "macro F1 was 0.81" in text and "Marker channels (561nm)" in text
    assert "Ashman's D = 3.10" in text and "logistic fit of your judged cells (7 positive, 9 negative)" in text


def test_text_follows_the_settings(tmp_path):
    a = pv.methods(_results(tmp_path / "a", cfg=_cfg(dip_ratio=0.6)))
    b = pv.methods(_results(tmp_path / "b", cfg=_cfg(dip_ratio=0.45)))
    assert "below 0.6 of the lower peak" in a and "below 0.45 of the lower peak" in b
    c = pv.methods(_results(tmp_path / "c", cfg=_cfg(split_dips=False)))
    assert "saddle between them" not in c and "not split at signal dips" in c


def test_not_classified_and_unknown_never_raise(tmp_path):
    d = _results(tmp_path / "r", classified=False)
    assert "Not run yet" in pv.methods(d)
    assert "classify" not in pv.commands(d)[0].split("# Notes")[0].split("Detect:")[1]
