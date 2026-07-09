"""Core pipeline abstractions: Step, Pipeline, PipelineContext, StepResult."""

from __future__ import annotations

import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Registry populated by @register_step decorator
_STEP_REGISTRY: dict[str, type[Step]] = {}


def register_step(cls: type) -> type:
    _STEP_REGISTRY[cls.__name__] = cls
    return cls


# ------------------------------------------------------------------
# Runtime context shared across all steps in one pipeline execution
# ------------------------------------------------------------------

@dataclass
class PipelineContext:
    channel_names: list[str]
    pixel_size_um: float = 1.0
    z_step_um: float = 1.0
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    # Source file path for each entry in `masks`, when the mask came from a
    # file (LoadROI) rather than auto-detection — lets downstream steps
    # (ParticleAnalysis, IntensityMeasurement) record *which file* was
    # actually used, not just the user-chosen label for the selection.
    mask_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def channel_index(self, name_or_idx: int | str) -> int:
        if isinstance(name_or_idx, int):
            return name_or_idx
        return self.channel_names.index(name_or_idx)


# ------------------------------------------------------------------
# Result returned by each step
# ------------------------------------------------------------------

@dataclass
class StepResult:
    """What a Step produces.  All fields are optional."""
    image: np.ndarray | None = None            # modified (C,Z,Y,X) or (C,Y,X) array
    measurements: pd.DataFrame | None = None   # tabular measurements
    masks: dict[str, np.ndarray] = field(default_factory=dict)  # named binary/label masks
    mask_paths: dict[str, str] = field(default_factory=dict)    # source file per mask, if any
    info: dict[str, Any] = field(default_factory=dict)          # scalar stats / metadata


# ------------------------------------------------------------------
# Step base class
# ------------------------------------------------------------------

class Step(ABC):
    """A single processing step.  Subclasses must be @dataclass-decorated."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult: ...

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {"type": self.__class__.__name__}
        d.update(dataclasses.asdict(self))  # works because all subclasses are @dataclass
        return d

    @staticmethod
    def from_dict(data: dict) -> Step:
        data = data.copy()
        step_type = data.pop("type")
        if step_type not in _STEP_REGISTRY:
            raise KeyError(
                f"Unknown step type '{step_type}'. "
                f"Known: {sorted(_STEP_REGISTRY)}"
            )
        return _STEP_REGISTRY[step_type](**data)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class Pipeline:
    """Ordered sequence of Steps with JSON serialization and step-by-step execution."""

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: list[Step] = []

    # -- Building -----------------------------------------------------

    def add(self, step: Step) -> Pipeline:
        self.steps.append(step)
        return self   # allow chaining: pipeline.add(...).add(...)

    # -- Running ------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        context: PipelineContext,
        on_step: Callable[[Step, StepResult, np.ndarray], None] | None = None,
    ) -> tuple[np.ndarray, list[StepResult]]:
        """Execute all steps sequentially.

        ``on_step`` is called after each step with (step, result, current_image).
        Use it to drive a live viewer or progress display.
        Returns (final_image, list_of_StepResults).
        """
        results: list[StepResult] = []
        current = image.copy()
        for step in self.steps:
            log.debug("Running step: %s", step.name)
            result = step.process(current, context)
            if result.image is not None:
                current = result.image
            context.masks.update(result.masks)
            context.mask_paths.update(result.mask_paths)
            results.append(result)
            if on_step:
                on_step(step, result, current)
        return current, results

    # -- Serialization ------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
        }
        Path(path).write_text(json.dumps(data, indent=2))
        log.info("Pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        data = json.loads(Path(path).read_text())
        p = cls(name=data["name"])
        for step_data in data["steps"]:
            p.steps.append(Step.from_dict(step_data))
        return p

    def __repr__(self) -> str:
        steps_repr = "\n  ".join(repr(s) for s in self.steps)
        return f"Pipeline(name={self.name!r})\n  {steps_repr}"
