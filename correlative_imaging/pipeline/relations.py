"""How the channels relate to each other inside a selection (e.g. the hole): what is a copy of another channel
(bleed-through, crosstalk) and what is the channel's own signal.

Every channel is put on a signal-to-noise scale first: (value - local background) / noise. The noise is the robust
SD (1.4826 x MAD) of the whole image outside all particles (every channel's whole-image threshold mask grown by
``margin_um``, clipped pixels left out) - the same pixels as IntensityMeasurement's ``whole_image:outside_particles``.
The background is the median of a band ``band_um`` outside the selection, outside all particles (wider band, then
the whole image, when too few pixels are left).

Inside the selection each channel is fitted as a non-negative copy of the others (no intercept, so a uniform fill
of a cell is never absorbed), every copy bounded by a plausible crosstalk: at most ``max_copy`` of the other
channel's raw intensity (``max_copy_pairs`` per pair). What the copy does not explain is the channel's own signal:
the **residual**, in SNR units. A bleed-through looks like a 1:1 mirror of the other channel and leaves ~0; a cell
of its own leaves a positive residual.

Output, one row per object x channel x other channel (objects: the selection itself, label 0, and every particle of
every channel inside it - the same labels as ParticleAnalysis of this selection):

  copy_fraction        the fitted copy of the other channel over the whole selection (raw intensity ratio)
  slope, pearson_r     within the object: the channel against the other channel, pixel by pixel (slope as a raw
                       intensity ratio)
  texture_r            the same correlation after removing the smooth part (Gaussian ``texture_sigma_um``): do the
                       fine structures (specks, edges) coincide - typical of a mirror
  explained_fraction   share of the channel's signal in the object explained by the copy of all others
  residual_*_snr       the own signal in the object (median, 75th percentile); median_snr = the channel itself
  background, background_source, noise   what the SNR scale used
(explained_fraction, residual_*, median_snr and the background columns belong to the channel; they repeat on its
rows for every other channel.)

The residual image of every channel is returned as ``residual_ch{c}@{selection}`` (float32, SNR; 0 outside the
selection and on clipped pixels) for the preview and the diagnostic crops; :func:`topography_contours` draws the
channels' contour lines inside the selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.optimize import lsq_linear

from .analyze import _robust_sd, foreground_mask
from .base import PipelineContext, Step, StepResult, register_step
from .saturation import any_saturated
from .segment import _project

RESIDUAL_PREFIX = "residual_ch"
SUMMARY_KEY = "channel_relations"     # context.metadata[SUMMARY_KEY][selection] -> per-channel summary (diagnostics)
PARTICLES_ROI_KEY = "particles_roi"   # context.metadata: channel -> selection of its current particles_ch{c}


def residual_key(channel: int, roi_mask: str) -> str:
    return f"{RESIDUAL_PREFIX}{channel}@{roi_mask or 'whole_image'}"


def parse_residual_key(key: str) -> tuple[int, str] | None:
    """(channel, selection) of a residual image key, else None."""
    if not key.startswith(RESIDUAL_PREFIX) or "@" not in key:
        return None
    ch, roi = key[len(RESIDUAL_PREFIX):].split("@", 1)
    return (int(ch), roi) if ch.isdigit() else None


def _band(sel: np.ndarray, outside: np.ndarray, px: float, inner_um: float, outer_um: float) -> np.ndarray:
    um = lambda d: max(int(round(d / px)), 1)
    return binary_dilation(sel, iterations=um(outer_um)) & ~binary_dilation(sel, iterations=um(inner_um)) & outside


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


@dataclass
@register_step
class ChannelRelation(Step):
    """Own signal vs copy of the other channels inside a selection (see the module docstring).

    Parameters
    ----------
    roi_mask:        the selection (empty = whole image).
    channels:        channel indices to relate; None = all.
    max_copy:        upper bound of any copy, as a raw intensity ratio (0.05 = 5 %).
    max_copy_pairs:  per-pair bounds ``[into, from, max]``, e.g. ``[[0, 1, 0.02]]``: channel 0 holds at most 2 % of
                     channel 1.
    band_um:         background band ``[inner, outer]`` µm outside the selection.
    margin_um:       particles are grown by this much before the noise / band pixels are taken.
    texture_sigma_um: the smooth part removed for ``texture_r``.
    """
    roi_mask: str = ""
    channels: list[int] | None = None
    max_copy: float = 0.05
    max_copy_pairs: list[list[float]] = field(default_factory=list)
    band_um: list[float] = field(default_factory=lambda: [2.0, 6.0])
    margin_um: float = 1.0
    texture_sigma_um: float = 1.0
    z_projection: str = "max"

    @property
    def name(self) -> str:
        return f"channel_relation{'_' + self.roi_mask if self.roi_mask else ''}"

    def bound(self, into: int, frm: int) -> float:
        for p in self.max_copy_pairs:
            if int(p[0]) == into and int(p[1]) == frm:
                return float(p[2])
        return float(self.max_copy)

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        px = context.pixel_size_um
        chans = list(self.channels) if self.channels else list(range(image.shape[0]))
        planes = {c: _project(image[c], self.z_projection).astype(np.float64) for c in chans}
        shape = planes[chans[0]].shape
        roi = context.masks.get(self.roi_mask) if self.roi_mask else None
        sel = roi.astype(bool) if roi is not None else np.ones(shape, bool)
        roi_name = self.roi_mask or "whole_image"
        if not sel.any() or len(chans) < 2:
            return StepResult(measurements=pd.DataFrame(), info={"n_objects": 0})

        # noise and background pixels: outside every channel's whole-image threshold mask (+ margin), unclipped
        sat = any_saturated(context, shape)
        sat = sat if sat is not None else np.zeros(shape, bool)
        fg = foreground_mask(context, "", shape) | sat
        r = int(round(self.margin_um / px)) if self.margin_um > 0 else 0
        if r > 0 and fg.any():
            fg = binary_dilation(fg, iterations=r)
        outside = ~fg
        if roi is not None:
            inner, outer = self.band_um
            band, source = _band(sel, outside, px, inner, outer), f"band {inner:g}-{outer:g} um"
            if band.sum() < 50:     # everything around the selection is segmented or clipped: look further out
                band, source = _band(sel, outside, px, inner, 2 * outer), f"band {inner:g}-{2 * outer:g} um"
            if band.sum() < 50:
                band, source = outside, "whole image outside particles"
        else:
            band, source = outside, "whole image outside particles"
        bg, noise = {}, {}
        for c in chans:
            ob = planes[c][outside] if outside.any() else planes[c].ravel()
            n = _robust_sd(ob)
            noise[c] = n if n > 0 else (float(ob.std()) or 1.0)
            bv = planes[c][band] if band.any() else ob
            bg[c] = float(np.median(bv))

        # work in the selection's box (+ room for the texture filter)
        ys, xs = np.nonzero(sel)
        pad = int(np.ceil(4 * self.texture_sigma_um / px)) + 1
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, shape[1])
        box = (slice(y0, y1), slice(x0, x1))
        s_box = sel[box]
        ok = s_box & ~sat[box]
        snr = {c: (planes[c][box] - bg[c]) / noise[c] for c in chans}
        sig = self.texture_sigma_um / px
        tex = {c: snr[c] - gaussian_filter(snr[c], sig) for c in chans}

        # bounded non-negative copy of the other channels, fitted over the selection
        coef, resid = {}, {}
        for c in chans:
            others = [o for o in chans if o != c]
            y = snr[c][ok]
            A = np.column_stack([snr[o][ok] for o in others])
            ub = np.array([max(self.bound(c, o), 0.0) * noise[o] / noise[c] for o in others])
            if y.size >= len(others) and ub.max() > 0:
                k = lsq_linear(A, y, bounds=(np.zeros(len(others)), np.maximum(ub, 1e-12))).x
                k = np.where(ub > 0, k, 0.0)
            else:
                k = np.zeros(len(others))
            coef[c] = {o: float(k[i]) for i, o in enumerate(others)}
            rb = snr[c] - sum(k[i] * snr[o] for i, o in enumerate(others))
            resid[c] = np.where(ok, rb, 0.0)

        # objects: the selection, and every particle of this selection's ParticleAnalysis
        objects: list[tuple[str, int, np.ndarray]] = [("selection", 0, ok)]
        owner = context.metadata.get(PARTICLES_ROI_KEY, {})
        names = context.channel_names
        cname = lambda c: names[c] if c < len(names) else f"ch{c}"
        for c in chans:
            lab = context.masks.get(f"particles_ch{c}")
            if lab is None or owner.get(c) != self.roi_mask:
                continue
            lb = lab[box] * s_box
            for v in np.unique(lb[ok]):
                if v:
                    objects.append((cname(c), int(v), ok & (lb == v)))

        rows = []
        for obj_ch, obj_lab, m in objects:
            n_px = int(m.sum())
            per_c = {}
            for c in chans:
                y = snr[c][m]
                ss = float((y * y).sum())
                rr = resid[c][m]
                per_c[c] = dict(
                    explained_fraction=(1.0 - float((rr * rr).sum()) / ss) if ss > 0 else float("nan"),
                    residual_median_snr=float(np.median(rr)) if n_px else float("nan"),
                    residual_p75_snr=float(np.percentile(rr, 75)) if n_px else float("nan"),
                    median_snr=float(np.median(y)) if n_px else float("nan"),
                )
            for c in chans:
                for o in chans:
                    if o == c:
                        continue
                    a, b = snr[o][m], snr[c][m]
                    va = float(((a - a.mean()) ** 2).sum()) if n_px else 0.0
                    slope = (float(((a - a.mean()) * (b - b.mean())).sum()) / va * noise[c] / noise[o]
                             if va > 0 else float("nan"))
                    rows.append(dict(
                        roi_mask=roi_name, object_channel=obj_ch, object_label=obj_lab,
                        channel=cname(c), other_channel=cname(o), n_px=n_px,
                        copy_fraction=coef[c][o] * noise[c] / noise[o],
                        slope=slope, pearson_r=_corr(b, a), texture_r=_corr(tex[c][m], tex[o][m]),
                        **per_c[c],
                        background=bg[c], background_source=source, noise=noise[c],
                    ))

        masks = {}
        for c in chans:
            full = np.zeros(shape, np.float32)
            full[box] = resid[c]
            masks[residual_key(c, self.roi_mask)] = full
        summary = {cname(c): dict(copy={cname(o): coef[c][o] * noise[c] / noise[o] for o in coef[c]},
                                  residual_median_snr=next(r["residual_median_snr"] for r in rows
                                                           if r["object_label"] == 0 and r["channel"] == cname(c)),
                                  explained_fraction=next(r["explained_fraction"] for r in rows
                                                          if r["object_label"] == 0 and r["channel"] == cname(c)),
                                  background=bg[c], noise=noise[c])
                   for c in chans}
        context.metadata.setdefault(SUMMARY_KEY, {})[roi_name] = summary
        return StepResult(measurements=pd.DataFrame(rows), masks=masks,
                          info={"n_objects": len(objects), "background_source": source, "summary": summary})


def topography_contours(plane: np.ndarray, sel: np.ndarray, n_levels: int = 7,
                        exclude: np.ndarray | None = None) -> list[tuple[float, np.ndarray]]:
    """Contour lines of ``plane`` inside the selection, on the channel's own range there (2nd-98th percentile of
    its unclipped pixels -> 0..1, levels evenly spaced in between). Returns ``[(level, (N, 2) row/col array)]`` in
    full-frame coordinates."""
    from skimage.measure import find_contours

    sel = sel.astype(bool)
    if not sel.any():
        return []
    use = sel & ~exclude if exclude is not None else sel
    if not use.any():
        return []
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = max(ys.min() - 1, 0), ys.max() + 2, max(xs.min() - 1, 0), xs.max() + 2
    p, s = plane[y0:y1, x0:x1].astype(float), sel[y0:y1, x0:x1]
    lo, hi = np.percentile(plane[use], [2, 98])
    a = np.clip((p - lo) / max(hi - lo, 1e-9), 0, 1)
    out = []
    for lv in np.linspace(0, 1, n_levels + 2)[1:-1]:
        for cnt in find_contours(a, lv, mask=s):
            if len(cnt) >= 3:
                out.append((float(lv), cnt + [y0, x0]))
    return out
