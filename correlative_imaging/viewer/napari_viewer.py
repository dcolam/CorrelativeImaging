"""Napari-based interactive preview for the pipeline.

napari is an optional dependency.  Import guard is applied at function call
time so the rest of the package works without it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from correlative_imaging.diagnostics import auto_contrast_limits

if TYPE_CHECKING:
    from correlative_imaging.io import ImageData
    from correlative_imaging.pipeline import Pipeline

log = logging.getLogger(__name__)


def _require_napari():
    try:
        import napari
        return napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for the preview mode. "
            "Install with: pip install 'correlative-imaging[viewer]'"
        ) from exc


class NapariViewer:
    """Wraps a napari Viewer and provides helpers for adding microscopy layers.

    Usage
    -----
    ::

        viewer = NapariViewer()
        viewer.show_image(image_data)         # all channels as separate layers
        viewer.show_mask("ROI", roi_mask)     # binary or label overlay
        napari.run()                          # enter event loop
    """

    def __init__(self, title: str = "Correlative Imaging"):
        napari = _require_napari()
        self._viewer = napari.Viewer(title=title)

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def show_image(self, image_data: ImageData, group: str = "raw", projection: str = "max",
                   blending: str = "additive") -> None:
        """Add every channel as a separate napari Image layer.

        projection: 'min' | 'max' | 'mean' | 'sum'  (Z-projection method; no-op for 2-D images).
        blending:   napari blending mode. Use 'translucent' (not the default
                    'additive') when this image will be shown alongside
                    another — e.g. brightfield next to fluorescence — since
                    additively summing a bright, near-full-frame BF layer
                    with FL channels washes the whole view out to white.
        """
        from napari.utils.colormaps import ensure_colormap

        colormaps = ["gray", "green", "red", "cyan", "magenta", "yellow"]
        mip = image_data.project(projection)   # (C, Y, X) or (Y, X)
        if mip.ndim == 2:
            mip = mip[np.newaxis]        # ensure (C, Y, X) shape

        for i, ch_name in enumerate(image_data.channel_names):
            cmap = colormaps[i % len(colormaps)]
            self._viewer.add_image(
                mip[i],
                name=f"{group}/{ch_name}",
                colormap=cmap,
                blending=blending,
                scale=[image_data.pixel_size_um, image_data.pixel_size_um],
                contrast_limits=auto_contrast_limits(mip[i]),
            )

    def show_mask(
        self,
        name: str,
        mask: np.ndarray,
        is_labels: bool | None = None,
        pixel_size_um: float = 1.0,
    ) -> None:
        """Add a mask as a Labels or Image layer."""
        if is_labels is None:
            # Treat as labels if it looks like a label image (integers > 1)
            is_labels = int(mask.max()) > 1

        scale = [pixel_size_um, pixel_size_um]
        if is_labels:
            self._viewer.add_labels(mask.astype(int), name=name, scale=scale)
        else:
            data = mask.astype(float)
            self._viewer.add_image(
                data,
                name=name,
                colormap="red",
                blending="additive",
                opacity=0.4,
                scale=scale,
                contrast_limits=auto_contrast_limits(data),
            )

    def show_measurements(
        self,
        df,
        name: str = "measurements",
        pixel_size_um: float = 1.0,
    ) -> None:
        """Overlay particle centroids as Points layer."""
        if df is None or df.empty:
            return
        if "centroid_row" in df.columns and "centroid_col" in df.columns:
            points = df[["centroid_row", "centroid_col"]].values * pixel_size_um
            self._viewer.add_points(
                points,
                name=name,
                size=5,
                face_color="yellow",
                border_color="black",
            )

    # ------------------------------------------------------------------
    # High-level pipeline preview
    # ------------------------------------------------------------------

    def run_pipeline_interactive(
        self,
        image_data: ImageData,
        pipeline: Pipeline,
    ) -> None:
        """Execute the pipeline step-by-step and add a layer for each result.

        .. deprecated::
            Runs on the calling thread — blocks the UI if called from a Qt slot.
            Use the ``thread_worker``-based preview in ``gui.RunTab`` instead.
        """
        from correlative_imaging.pipeline.base import PipelineContext

        self.show_image(image_data, group="raw")

        context = PipelineContext(
            channel_names=image_data.channel_names,
            pixel_size_um=image_data.pixel_size_um,
            z_step_um=image_data.z_step_um,
        )

        def _on_step(step, result, current_image):
            step_name = step.name
            if result.image is not None:
                # Show max-projection of each channel after this step
                mip = current_image.max(axis=1) if current_image.ndim == 4 else current_image
                for i, ch in enumerate(image_data.channel_names):
                    self._viewer.add_image(
                        mip[i],
                        name=f"{step_name}/{ch}",
                        visible=False,
                        blending="additive",
                        scale=[image_data.pixel_size_um, image_data.pixel_size_um],
                        contrast_limits=auto_contrast_limits(mip[i]),
                    )
            for mask_name, mask in result.masks.items():
                self.show_mask(
                    f"{step_name}/{mask_name}",
                    mask,
                    pixel_size_um=image_data.pixel_size_um,
                )
            if result.measurements is not None:
                self.show_measurements(
                    result.measurements,
                    name=f"{step_name}/centroids",
                    pixel_size_um=image_data.pixel_size_um,
                )

        pipeline.run(image_data.data, context, on_step=_on_step)

    @property
    def viewer(self):
        return self._viewer


def show_pipeline_preview(image_data: ImageData, pipeline: Pipeline) -> None:
    """Convenience function: open napari, run pipeline, enter event loop."""
    napari = _require_napari()
    v = NapariViewer()
    v.run_pipeline_interactive(image_data, pipeline)
    napari.run()
