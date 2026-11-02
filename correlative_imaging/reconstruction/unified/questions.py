"""The first pass's questions: where the data of this dataset cannot decide on its own, it asks you - only there, with
the reason and how many answers it needs. Technical questions only (what is a cell here); what kind of cell it is
comes from your labels (labels.py).

  wells      wells with far fewer surrounding cells than their plate: lost focus, or no cells seeded?
             -> exclude / empty (a "no cell" reference) / keep
  particle   hole particles: cell, several cells, neighbour reaching in, plug/halo, debris. Asked when there are no
             "no cell" examples yet (the aperture can glow as brightly as a dim cell); afterwards only on the
             particles the shape model is unsure about (P 0.3-0.7). Feeds the shape model
  size       surrounding cells: one cell? Asked when the size of one cell body is unclear (wide spread). Feeds the cell
             size (the median core area of the cells you call "one cell")
  channel    per channel, holes near its cut (0.5-1.5x) and its dim calls: positive? Feeds that channel's cut
  clump      holes at least twice the typical hole cell: one cell or several? Feeds the clump rule

Answers are kept in <out>/interventions/answers.csv (question, item key, answer, time) and applied on the next run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["question", "plate_id", "well_id", "image_id", "obj"]
CHOICES = {
    "wells": ["exclude", "empty", "keep"],
    "particle": ["cell", "several cells", "neighbour", "plug / halo", "debris", "unsure"],
    "size": ["one cell", "several cells", "part of a cell", "not a cell", "unsure"],
    "channel": ["positive", "negative", "not a cell", "unsure"],
    "clump": ["one cell", "several cells", "not a cell", "unsure"],
}
CELL = {"cell", "several cells"}
NOT_CELL = {"neighbour", "plug / halo", "debris"}


class Answers:
    """interventions/answers.csv; every set() is written at once."""

    def __init__(self, out: Path):
        self.f = Path(out) / "interventions" / "answers.csv"
        self.f.parent.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.f, dtype=str).fillna("") if self.f.exists() else \
            pd.DataFrame(columns=KEYS + ["answer", "t"])

    @staticmethod
    def key(question, plate_id="", well_id="", image_id="", obj="") -> tuple:
        return tuple(str(v) for v in (question, plate_id, well_id, image_id, obj))

    def set(self, k: tuple, answer: str):
        m = (self.df[KEYS].astype(str).apply(tuple, axis=1) == k) if len(self.df) else pd.Series(dtype=bool)
        self.df = self.df[~m] if len(m) else self.df
        if answer:
            self.df = pd.concat([self.df, pd.DataFrame([dict(zip(KEYS, k), answer=answer,
                                                              t=time.strftime("%Y-%m-%d %H:%M:%S"))])])
        self.df.to_csv(self.f, index=False)

    def get(self, question: str) -> pd.DataFrame:
        return self.df[self.df.question == question] if len(self.df) else self.df

    def lookup(self) -> dict:
        return {tuple(r[k] for k in KEYS): r["answer"] for r in self.df.to_dict("records")}


# ------------------------------------------------------------------------------------------------- apply (run.py)
def wells(A: Answers) -> dict:
    return {(r["plate_id"], r["well_id"]): r["answer"] for r in A.get("wells").to_dict("records")}


def particles(A: Answers) -> dict:
    """{(plate, well, ch, level, lab): 1 cell / 0 not a cell} for the shape model."""
    out = {}
    for r in A.get("particle").to_dict("records"):
        ch, level, lab = r["obj"].split("|")
        if r["answer"] in CELL:
            out[(r["plate_id"], r["well_id"], ch, level, int(lab))] = 1
        elif r["answer"] in NOT_CELL:
            out[(r["plate_id"], r["well_id"], ch, level, int(lab))] = 0
    return out


def sizes(A: Answers, cores: pd.DataFrame) -> dict:
    """{channel: median core area of the cells you called one cell} (>= 5 per channel)."""
    a = A.get("size")
    if not len(a) or cores is None or not len(cores):
        return {}
    one = a[a.answer == "one cell"]
    C = cores.assign(image_id=cores.image_id.astype(str), obj=cores.channel + ":" + cores.label.astype(str))
    m = C.merge(one, on=["plate_id", "image_id", "obj"])
    return {c: float(g.area_core.median()) for c, g in m.groupby("channel") if len(g) >= 5}


def channels(A: Answers) -> tuple[dict, dict]:
    """({(plate, well): {channel: True/False}}, {(plate, well): "no"} for "not a cell")."""
    marks, no = {}, {}
    for r in A.get("channel").to_dict("records"):
        k = (r["plate_id"], r["well_id"])
        if r["answer"] in ("positive", "negative"):
            marks.setdefault(k, {})[r["obj"]] = r["answer"] == "positive"
        elif r["answer"] == "not a cell":
            no[k] = "no"
    return marks, no


def clumps(A: Answers) -> dict:
    return {(r["plate_id"], r["well_id"]): r["answer"] for r in A.get("clump").to_dict("records")}


# ------------------------------------------------------------------------------------------- open questions (GUI)
def _sample(D: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return D.sample(min(n, len(D)), random_state=seed) if len(D) else D


def open_questions(out: Path, n_items: int = 48) -> list[dict]:
    """Per question: id, title, why, choices, needed (text), answered, state (open / done / optional / not needed),
    items (DataFrame: kind, plate_id, well_id, image_id, obj, caption; not yet answered first)."""
    out = Path(out)
    S = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    nm = S.get("names", {})
    A = Answers(out)
    look = A.lookup()
    H = pd.read_csv(out / "hole_calls.csv", dtype={"plate_id": str, "well_id": str}) if (out / "hole_calls.csv").exists() \
        else pd.DataFrame()
    Q = []

    def answered(q):
        return A.get(q)

    def items(D, q, cap):
        D = D.assign(question=q)
        for k in ("image_id", "obj"):
            D[k] = D[k].astype(str) if k in D else ""
        D["answer"] = [look.get(tuple(str(r[k]) for k in KEYS), "") for r in D.to_dict("records")]
        D["caption"] = cap(D) if len(D) else []
        return D.sort_values("answer", key=lambda s: s != "", kind="stable").reset_index(drop=True)

    # wells with few cells
    C = pd.read_csv(out / "checks.csv", dtype={"plate_id": str, "well_id": str}) if (out / "checks.csv").exists() \
        else pd.DataFrame()
    few = C[C.status.astype(str).str.startswith("few cells")] if len(C) else C
    if len(few):
        a = answered("wells")
        it = items(few.assign(kind="well", obj="", image_id=few.image_id.astype(str)), "wells",
                   lambda D: [f"{p} {w}: {n} cells (plate median {m:.0f})" for p, w, n, m in
                              zip(D.plate_id, D.well_id, D.n_objects, D.plate_median)])
        left = int((it.answer == "").sum())
        Q.append(dict(id="wells", title="Wells with few cells: lost focus, or no cells seeded?",
                      why=f"{len(few)} wells have under a quarter of their plate's median number of surrounding cells.",
                      choices=CHOICES["wells"], needed="all", answered=len(a),
                      state="open" if left else "done", items=it))
    # particles: cell or not
    fp = out / "hole_particles.parquet"
    if fp.exists():
        P = pd.read_parquet(fp)
        P["image_id"] = P.image_id.astype(str) if "image_id" in P else ""
        P["obj"] = P.ch.astype(str) + "|" + P.level.astype(str) + "|" + P.lab.astype(str)
        P["kind"] = "particle"
        sh = S.get("particle_shape", {})
        a = answered("particle")
        n_cell, n_not = int(a.answer.isin(CELL).sum()), int(a.answer.isin(NOT_CELL).sum())
        cap = lambda D: [f"{nm.get(c, c)} particle, {u:.0f} µm²" + (f", P(cell) {p:.2f}" if np.isfinite(p) else "")
                         for c, u, p in zip(D.ch, D.in_um2.fillna(0), D.p_cell.astype(float).fillna(np.nan))]
        if not sh.get("used"):
            # no examples yet: particles of every channel across the brightness range
            P = P.assign(_b=P.groupby("ch").hsnr.transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)))
            D = pd.concat([_sample(g, max(n_items // max(P.ch.nunique() * 4, 1), 3), seed=i)
                           for i, (_, g) in enumerate(P.groupby(["ch", "_b"]))])
            need = n_cell < 15 or n_not < 15
            Q.append(dict(id="particle", title="Cell or not? (particles in the hole)",
                          why="No examples of holes without a cell yet, so a plug or the aperture's glow can pass for "
                              "a dim cell. Your answers teach the shape model what a cell looks like here.",
                          choices=CHOICES["particle"], needed=f"15 cell, 15 not a cell (now {n_cell}, {n_not})",
                          answered=len(a), state="open" if need else "done", items=items(D, "particle", cap)))
        else:
            U = P[(P.p_cell >= 0.3) & (P.p_cell <= 0.7)]
            share = len(U) / max(len(P), 1)
            Q.append(dict(id="particle", title="Cell or not? (particles the shape model is unsure about)",
                          why=f"Shape model AUC {sh.get('auc', float('nan')):.2f}; {len(U)} particles "
                              f"({100 * share:.0f} %) have P(cell) 0.3-0.7.",
                          choices=CHOICES["particle"], needed="20 answers", answered=len(a),
                          state="open" if share >= 0.05 and len(a) < 20 else ("done" if len(a) >= 20 else "optional"),
                          items=items(_sample(U, n_items), "particle", cap)))
    # size of one cell
    fc = out / "cell_cores.parquet"
    Cc = pd.read_parquet(fc) if fc.exists() else pd.DataFrame()
    if len(Cc) and "image_id" in Cc:
        Cc["image_id"] = Cc.image_id.astype(str)
        Cc["obj"] = Cc.channel.astype(str) + ":" + Cc.label.astype(str)
        spread = {c: float((g.area_core.quantile(.75) - g.area_core.quantile(.25)) / g.area_core.median())
                  for c, g in Cc.groupby("channel") if len(g) >= 10}
        a = answered("size")
        wide = {c: s for c, s in spread.items() if s > 0.6}
        D = pd.concat([_sample(g, max(n_items // max(Cc.channel.nunique(), 1), 6), seed=1)
                       for _, g in Cc.groupby("channel")]).assign(kind="field")
        one = a[a.answer == "one cell"]
        Q.append(dict(id="size", title="Is this one cell? (surrounding cells)",
                      why=("The size of one cell body is unclear in " + ", ".join(
                          f"{nm.get(c, c)} (spread {s:.1f})" for c, s in wide.items()) + ".") if wide else
                      "The size of one cell body is clear; answer only if the learnt size looks wrong.",
                      choices=CHOICES["size"], needed="10 'one cell' per channel", answered=len(a),
                      state=("open" if len(one) < 10 * len(wide) else "done") if wide else "optional",
                      items=items(D, "size", lambda D: [f"{nm.get(c, c)}, core {u:.0f} µm²"
                                                         for c, u in zip(D.channel, D.area_core)])))
    # per channel: positive?
    if len(H):
        V = H[H.status_cell == "cell"] if "status_cell" in H else H
        for c in S.get("colour", []):
            x = V.get(f"x_{c}")
            if x is None:
                continue
            near = V[(x >= 0.5) & (x <= 1.5)]
            dim = V[V.pattern.astype(str) == f"{nm.get(c, c)}? (dim)"]
            D = pd.concat([_sample(dim, n_items // 2, 2), _sample(near, n_items, 3)]).drop_duplicates(
                ["plate_id", "well_id"]).head(n_items)
            a = answered("channel")
            a = a[a.obj == c]
            pos, neg = int((a.answer == "positive").sum()), int((a.answer == "negative").sum())
            need = (len(near) + len(dim) >= 10) and (pos < 5 or neg < 5)
            Q.append(dict(id=f"channel:{c}", question="channel", channel=c,
                          title=f"Is {nm.get(c, c)} positive? (hole cells near its cut)",
                          why=f"{len(near)} hole cells lie within 0.5-1.5x of the {nm.get(c, c)} cut, {len(dim)} are dim "
                              f"calls. Your answers set where {nm.get(c, c)} starts.",
                          choices=CHOICES["channel"], needed=f"5 positive, 5 negative (now {pos}, {neg})",
                          answered=len(a), state="open" if need else ("done" if pos >= 5 and neg >= 5 else "optional"),
                          items=items(D.assign(kind="hole", obj=c, image_id=D.get("image_id", "")), "channel",
                                      lambda D: [f"{p} {w}: {pt}" for p, w, pt in zip(D.plate_id, D.well_id, D.pattern)])))
        # clumps
        cl = V[V["flags"].astype(str).str.contains("possible clump")] if "flags" in V else V.iloc[:0]
        if len(cl):
            a = answered("clump")
            Q.append(dict(id="clump", title="One cell or several? (large hole cells)",
                          why=f"{len(cl)} hole cells are at least twice the typical hole cell.",
                          choices=CHOICES["clump"], needed="10 answers", answered=len(a),
                          state="open" if len(cl) >= 10 and len(a) < 10 else ("done" if len(a) >= 10 else "optional"),
                          items=items(_sample(cl, n_items, 4).assign(kind="hole", obj=""), "clump",
                                      lambda D: [f"{p} {w}: {b:.1f} cell bodies" for p, w, b in
                                                 zip(D.plate_id, D.well_id, D.cells_in_body)])))
    for q in Q:
        q.setdefault("question", q["id"])
    return Q
