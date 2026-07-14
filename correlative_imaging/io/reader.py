"""Multi-format microscopy image reader built on bioio."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_java_configured = False

# Bio-Formats starts its JVM lazily on the *first* read — concurrent threads
# racing that one-time startup (relevant once batch runs are parallelized;
# see batch.WellBatchRunner) can crash the process. Serialize only the reads
# that happen before the JVM is confirmed running; once ready, subsequent
# Bio-Formats reads proceed unlocked/concurrently like any other format.
_bf_construct_lock = threading.Lock()
_bf_jvm_ready = False


def _construct_bioformats_image(bioimage_cls, path: Path, reader_cls):
    global _bf_jvm_ready
    if _bf_jvm_ready:
        return bioimage_cls(str(path), reader=reader_cls)
    with _bf_construct_lock:
        # Java setup (JAVA_HOME, scyjava constraints) must finish under this
        # same lock, not before it — otherwise a second thread can see
        # _java_configured already True (set at the top of that function,
        # before the slow work) and start the JVM before setup completes.
        _ensure_java_for_bioformats()
        img = bioimage_cls(str(path), reader=reader_cls)
        _bf_jvm_ready = True
        return img

# jpype's native bridge (used by scyjava/bffile to embed the JVM) lags behind
# the newest JDK releases — brand-new majors (e.g. 21+) have been observed to
# crash the process with no traceback (native access restrictions tightened
# each release; see JEP 472). Stick to well-established LTS versions.
_JAVA_MIN_COMPATIBLE = 8
_JAVA_MAX_COMPATIBLE = 17


def _java_major_version(java_exe: str) -> int | None:
    try:
        out = subprocess.run(
            [java_exe, "-version"], capture_output=True, text=True, timeout=5
        ).stderr
    except OSError:
        return None
    m = re.search(r'version "1\.(\d+)', out) or re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else None


def _find_compatible_java() -> str | None:
    """Search PATH and common Windows install roots for a Java known to work
    with jpype, preferring the newest version within the compatible range.
    """
    exe_name = "java.exe" if os.name == "nt" else "java"
    search_dirs = list(dict.fromkeys(os.environ.get("PATH", "").split(os.pathsep)))
    for root in (r"C:\Program Files", r"C:\Program Files (x86)"):
        try:
            search_dirs += [str(p / "bin") for p in Path(root).iterdir() if p.is_dir()]
        except OSError:
            pass

    seen: set[str] = set()
    best: tuple[int, str] | None = None
    for d in search_dirs:
        exe = str(Path(d) / exe_name)
        if exe in seen or not Path(exe).exists():
            continue
        seen.add(exe)
        major = _java_major_version(exe)
        if major is None:
            continue
        if _JAVA_MIN_COMPATIBLE <= major <= _JAVA_MAX_COMPATIBLE:
            if best is None or major > best[0]:
                best = (major, exe)
    return best[1] if best else None


def _select_java_home() -> Path | None:
    """Pick a JAVA_HOME, preferring the bundled jdk4py JDK (self-contained,
    version-pinned, same wheel-install mechanism as PyQt6/numpy) over
    whatever happens to be installed on the host.

    Returns ``None`` if no *version-checked* compatible JDK can be found —
    callers must treat that as fatal, not silently proceed. jpype's native
    bridge has been observed to crash the whole process with zero Python
    traceback when handed an incompatible JDK (e.g. a brand-new major); an
    unchecked ``shutil.which("java")`` fallback used to risk exactly that.
    """
    try:
        from jdk4py import JAVA_HOME as bundled_home
        return Path(bundled_home)
    except ImportError:
        pass

    java_exe = _find_compatible_java()
    if java_exe:
        # java.exe normally lives at <JAVA_HOME>/bin/java(.exe)
        return Path(java_exe).resolve().parent.parent
    return None


def _ensure_java_for_bioformats() -> None:
    """Point scyjava at a suitable JDK instead of letting it download one.

    bioio-bioformats starts a JVM via scyjava, which by default *always*
    downloads its own pinned Java build into the user's cache dir — even if a
    perfectly good JDK is already installed. On locked-down Windows machines
    (e.g. corporate VMs) Group Policy commonly blocks *executing* binaries
    from user-writable folders like AppData, so that download-and-run step
    fails even though the download itself succeeds. Installing the optional
    ``jdk4py`` dependency (see the ``bioformats`` extra) sidesteps this
    entirely by bundling a known-good JDK inside the wheel itself; falling
    back to a host-installed JDK otherwise.
    """
    global _java_configured
    if _java_configured:
        return

    java_home = _select_java_home()
    if java_home is None:
        # Do NOT let this fall through to scyjava's own default behavior:
        # it will either try to auto-download a JDK (blocked by Group
        # Policy on locked-down Windows machines — the original failure
        # mode this module works around) or hand an unvalidated host JDK
        # to jpype, which can crash the process natively with no
        # traceback. Fail loudly in Python instead, with an actionable fix.
        raise ImportError(
            "No compatible Java runtime found for Bio-Formats (.vsi/.czi/.lif/etc). "
            "Install the bundled JDK: pip install -e \".[bioformats]\" "
            "(installs jdk4py — a self-contained JDK 17, no system Java needed)."
        )
    _java_configured = True

    os.environ["JAVA_HOME"] = str(java_home)
    # Some Java-discovery paths (e.g. jgo) scan PATH rather than
    # JAVA_HOME — put our chosen version first so it wins either way.
    os.environ["PATH"] = str(java_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    log.debug("Using Java at %s", java_home)

    try:
        import scyjava.config
        # "auto": prefer JAVA_HOME / system java, only download if none found.
        scyjava.config.set_java_constraints(fetch="auto")
        # Bio-Formats' own reader logging (via bffile/SLF4J) is extremely
        # chatty at INFO level — every parsed file tag gets printed, which can
        # bury the actual error when something goes wrong. Must be set before
        # the JVM starts; DebugTools.setRootLevel (called after JVM start)
        # would be the alternative but risks racing bffile's own startup.
        scyjava.config.add_option("-Dorg.slf4j.simpleLogger.defaultLogLevel=warn")
    except ImportError:
        pass

    # Belt-and-suspenders: bffile also mirrors Java log records through this
    # Python logger, independent of the JVM-side SLF4J level.
    logging.getLogger("bffile").setLevel(logging.WARNING)

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

    def project(self, method: str = "max", channel: int | str | None = None) -> np.ndarray:
        """Z-projection using the given method.  Returns (C, Y, X) or (Y, X).

        method: 'min' | 'max' | 'mean' | 'sum'  (no-op if not a Z-stack).
        """
        if not self.is_zstack:
            return self.data if channel is None else self.channel(channel)
        ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
        fn = ops.get(method, np.max)
        if channel is None:
            return fn(self.data, axis=1)
        return fn(self.channel(channel), axis=0)

    def max_project(self, channel: int | str | None = None) -> np.ndarray:
        """Max-intensity projection along Z.  Returns (C, Y, X) or (Y, X)."""
        return self.project("max", channel)


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
        if suffix in bioformats_extensions:
            # Java setup happens inside _construct_bioformats_image, under the
            # same lock as the first JVM-starting construction (see there for why).
            img = _construct_bioformats_image(BioImage, path, reader_cls)
        else:
            img = BioImage(str(path), reader=reader_cls)
        if scene != 0:
            img.set_scene(scene)
    else:
        # Only warn when the format GENUINELY needs the Bio-Formats reader and
        # it isn't installed (reader_cls is None because the import failed) —
        # the read below will then fail, so make the real cause obvious rather
        # than logging a scary "requires bioio-bioformats" line on every
        # successful read (which it isn't — a working install lands in the
        # branch above and stays silent).
        if suffix in bioformats_extensions:
            log.warning(
                "%s requires the 'bioio-bioformats' reader, which is not "
                "installed. Install with: pip install bioio-bioformats",
                suffix,
            )
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
