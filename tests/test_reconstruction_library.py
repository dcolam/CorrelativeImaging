"""Label library per results folder (reconstruction/library.py): a new run starts clean, older folders keep theirs."""
from __future__ import annotations

import hashlib
import json

import pytest

from correlative_imaging.reconstruction import library as L
from correlative_imaging.reconstruction.labels import LabelStore


def _md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def _shared(tmp_path, dataset="run_x"):
    st = LabelStore(tmp_path / "ci_labels.db")                     # the project library beside the results
    st.set(dataset, "P1", "A01", "c1", "any", 1, 1, "cell")
    st.set(dataset, "P1", "A02", "c1", "any", 2, 2, "noise")
    st.seed_test_wells(dataset, [("P1", "A05")])
    st.add_cut(dataset, "P1", "A01", [(0, 0), (5, 5)])
    st.set("other_run", "P1", "A01", "c1", "any", 3, 3, "debris")
    return tmp_path / "ci_labels.db"


def test_probe_is_read_only(tmp_path):
    lib = _shared(tmp_path)
    before = _md5(lib)
    info = L.probe(lib, "run_x")
    assert info == dict(labels=2, test_wells=1, cells=0, edits=1)
    assert L.probe(lib, "nothing") == dict(labels=0, test_wells=0, cells=0, edits=0)
    assert _md5(lib) == before


def test_new_folder_is_own_and_clean_even_with_the_same_dataset_beside(tmp_path):
    _shared(tmp_path)
    new = tmp_path / "reconstruction_run_x_v2"
    L.remember(new, "own")                                          # what Detect writes into a folder it creates
    r = L.resolve(new, "run_x")
    assert r["kind"] == "own" and r["decided"] and r["path"] == new / "ci_labels.db"


def test_older_folder_gets_the_choice_with_keep_as_default(tmp_path):
    _shared(tmp_path)
    old = tmp_path / "reconstruction_run_x"
    old.mkdir()
    r = L.resolve(old, "run_x")
    assert not r["decided"] and r["kind"] == "shared" and r["shared"]["labels"] == 2
    L.remember(old, "shared", r["path"])
    r2 = L.resolve(old, "run_x")
    assert r2["decided"] and r2["kind"] == "shared" and r2["path"] == (tmp_path / "ci_labels.db").resolve()
    assert L.resolve(tmp_path / "unrelated", "run_y")["kind"] == "own"   # nothing beside for its dataset


def test_copy_takes_only_this_dataset_and_leaves_the_source_untouched(tmp_path):
    lib = _shared(tmp_path)
    before = _md5(lib)
    old = tmp_path / "reconstruction_run_x"
    old.mkdir()
    n = L.copy_dataset(lib, L.own_path(old), "run_x")
    assert n["particle_labels"] == 2 and n["test_wells"] == 1 and n["particle_cuts"] == 1
    st = LabelStore(L.own_path(old))
    assert len(st.frame("run_x")) == 2 and len(st.frame("other_run")) == 0
    assert st.edits("run_x")                                         # the hand cut came along
    assert _md5(lib) == before


def test_detect_marks_only_folders_it_creates(tmp_path, monkeypatch):
    import correlative_imaging.reconstruction.pipeline as P
    monkeypatch.setattr(P, "_detect_run", lambda *a, **k: dict(wells=[1]))
    monkeypatch.setattr("correlative_imaging.reconstruction.provenance.record", lambda *a, **k: None)
    new = tmp_path / "fresh"
    P.detect_run(None, P.Config(), new)
    assert L.decision(new)["kind"] == "own"
    old = tmp_path / "old"
    old.mkdir()
    (old / "run.json").write_text("{}")                              # re-detecting an existing folder
    P.detect_run(None, P.Config(), old)
    assert L.decision(old) is None                                   # its choice is not made for it


def test_model_found_in_the_results_folder_first(tmp_path):
    from correlative_imaging.reconstruction.config import MODELS, Config
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mine.joblib").write_bytes(b"x")
    assert Config(model="mine.joblib").model_path(tmp_path) == tmp_path / "models" / "mine.joblib"
    assert Config(model="other.joblib").model_path(tmp_path) == MODELS / "other.joblib"
    assert Config(model="").model_path(tmp_path) is None


def test_unreadable_library_beside_is_never_silently_replaced(tmp_path):
    (tmp_path / "ci_labels.db").write_bytes(b"not a database at all, but the user's file")
    old = tmp_path / "reconstruction_run_x"
    old.mkdir()
    r = L.resolve(old, "run_x")
    assert not r["decided"] and "error" in r["shared"]
