"""Multi-format microscopy image reader built on bioio."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_java_configured = False


def _ensure_java_for_bioformats() -> None:
    """Point scyjava at an already-installed JDK instead of letting it download one.

    bioio-bioformats starts a JVM via scyjava, which by default *always*
    downloads its own pinned Java build into the user's cache dir — even if a
    perfectly good JDK is already installed. On locked-down Windows machines
    (e.g. corporate VMs) Group Policy commonly blocks *executing* binaries
    from user-writable folders like AppData, so that download-and-run step
    fails even though the download itself succeeds. Reusing a JDK already
    installed under Program Files sidesteps that restriction entirely.
    """
    global _java_configured
    if _java_configured:
        return
    _java_configured = True

    if "JAVA_HOME" not in os.environ:
        java_exe = shutil.which("java")
        if java_exe:
            # java.exe normally lives at <JAVA_HOME>/bin/java(.exe)
            java_home = Path(java_exe).resolve().parent.parent
            os.environ["JAVA_HOME"] = str(java_home)
            log.debug("JAVA_HOME not set — using detected Java at %s", java_home)

    try:
        import scyjava.config
        # "auto": prefer JAVA_HOME / system java, only download if none found.
        scyjava.config.set_java_constraints(fetch="auto")
    except ImportError:
        pass

# Map suffix → explicit reader class to avoid bioio's trial-and-error detection.
# bioio's default auto-detection tries ome-tiff first for any .tif file, which
# produces a noisy WARNING for every plain TIFF.  Pinning the reader silences it.
def _reader_for_suffix(suffix: str, name_lower: str) -> Any | None:
    """Return the preferred bioio reader class for a given file extension, or None
    to let bioio auto-detect (used for less common formats)."""
    if name_lower.endswith(".ome.tiff") or name_lower.endswith(".ome.tif"):
        try:
            from bioio_ome_tiff import Reader
            return Reader
        except ImportError:
            return None
    if suffix in {".tif", ".tiff"}:
        try:
            from bioio_tifffile import Reader
            return Reader
        except ImportError:
            return None
    if suffix == ".czi":
        try:
            from bioio_czi import Reader
            return Reader
        except ImportError:
            return None
    if suffix == ".lif":
        try:
            from bioio_lif import Reader
            return Reader
        except ImportError:
            return None
    if suffix == ".nd2":
        try:
            from bioio_nd2 import Reader
            return Reader
        except ImportError:
            return None
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
        try:
            from bioio_imageio import Reader
            return Reader
        except ImportError:
            return None
    if suffix in bioformats_extensions:
        try:
            from bioio_bioformats import Reader
            return Reader
        except ImportError:
            return None
    return None  # let bioio figure it out

# Extensions handled natively by bioio plugins (no Java required)
supported_extensions: set[str] = {
    ".tif", ".tiff",           # bioio-tifffile
    ".ome.tif", ".ome.tiff",   # bioio-ome-tiff
    ".czi",                    # bioio-czi  (Zeiss)
    ".lif",                    # bioio-lif  (Leica)
    ".nd2",                    # bioio-nd2  (Nikon)
    ".zarr",                   # bioio-ome-zarr
    ".png", ".jpg", ".jpeg",   # bioio-imageio
    ".bmp", ".gif",
}

# Additional formats require bioio-bioformats (Java/Maven)
bioformats_extensions: set[str] = {
    ".vsi", ".scn",  # Olympus/Leica whole slide
    ".ics", ".ids",  # Andor
    ".lsm",          # Zeiss LSM (old)
    ".oif", ".oib",  # Olympus
    ".zvi",          # Zeiss AxioVision
    ".lei",          # Leica (old)
}


@dataclass
class ImageData:
    """Canonical image container used throughout the pipeline.

    ``data`` has shape ``(C, Z, Y, X)`` for Z-stacks or ``(C, Y, X)`` for 2-D.
    Channel axis is always first for easy per-channel indexing.
    """

    data: np.ndarray
    channel_names: list[str]
    pixel_size_um: float = 1.0   # XY physical pixel size in micrometers
    z_step_um: float = 1.0       # Z step in micrometers (only meaningful for Z-stacks)
    source_path: Path | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def is_zstack(self) -> bool:
        return self.data.ndim == 4

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.data.shape[-2], self.data.shape[-1]

    def channel(self, idx: int | str) -> np.ndarray:
        """Return the array for a single channel (Z, Y, X) or (Y, X)."""
        if isinstance(idx, str):
            idx = self.channel_names.index(idx)
        return self.data[idx]

    def max_project(self, channel: int | str | None = None) -> np.ndarray:
        """Max-intensity projection along Z.  Returns (C, Y, X) or (Y, X)."""
        if not self.is_zstack:
            return self.data if channel is None else self.channel(channel)
        if channel is None:
            return self.data.max(axis=1)
        return self.channel(channel).max(axis=0)


def read_image(path: str | Path, scene: int = 0) -> ImageData:
    """Read any supported microscopy format and return an :class:`ImageData`.

    Parameters
    ----------
    path:   Path to the image file.
    scene:  Scene / series index for multi-scene files (default 0).

    Raises
    ------
    ImportError  if the required bioio reader plugin is not installed.
    ValueError   if the format is not recognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    all_known = supported_extensions | bioformats_extensions
    if suffix not in all_known:
        log.warning("Extension %s not in known list — attempting bioio anyway", suffix)

    if suffix in bioformats_extensions:
        log.warning(
            "%s requires bioio-bioformats (Java). "
            "Install with: pip install bioio-bioformats",
            suffix,
        )
        _ensure_java_for_bioformats()

    try:
        from bioio import BioImage
    except ImportError as exc:
        raise ImportError(
            "bioio is required for image reading. "
            "Install with: pip install bioio bioio-tifffile bioio-czi bioio-lif bioio-nd2"
        ) from exc

    reader_cls = _reader_for_suffix(suffix, name_lower)
    if reader_cls is not None:
        # bioio_bioformats.Reader (and some others) don't accept `scene` in
        # __init__ — BioImage forwards the kwarg there and it blows up.
        # Create without scene, then set it via the public API.
        img = BioImage(str(path), reader=reader_cls)
        if scene != 0:
            img.set_scene(scene)
    else:
        img = BioImage(str(path), scene=scene)

    # Always get a (C, Z, Y, X) array; squeeze T=0
    try:
        data = img.get_image_data("CZYX", T=0)
    except Exception:
        # Fallback for 2-D images without Z axis
        data_tczyx = img.xarray_dask_data.compute().values  # (T, C, Z, Y, X)
        data = data_tczyx[0, :, 0, :, :]  # -> (C, Y, X)

    # If Z has only 1 slice collapse it to 2-D for convenience
    if data.ndim == 4 and data.shape[1] == 1:
        data = data[:, 0, :, :]

    pixel_sizes = img.physical_pixel_sizes
    px_y = float(pixel_sizes.Y) if pixel_sizes.Y is not None else 1.0
    px_z = float(pixel_sizes.Z) if pixel_sizes.Z is not None else 1.0

    channel_names: list[str] = img.channel_names or [
        f"ch{i}" for i in range(data.shape[0])
    ]

    metadata = {
        "dims": img.dims.order,
        "shape": list(img.shape),
        "scenes": img.scenes,
        "current_scene": scene,
    }

    return ImageData(
        data=data.astype(np.float32),
        channel_names=list(channel_names),
        pixel_size_um=px_y,
        z_step_um=px_z,
        source_path=path,
        metadata=metadata,
    )
