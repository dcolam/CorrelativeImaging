"""The reconstruction's workflow strip (the shared gui_common.workflow_strip): Detect → Label → Train → Classify →
Results, states from workflow.workflow_state."""
from __future__ import annotations

from ...gui_common.workflow_strip import STATE_COLOR, STATE_TEXT, StepTile  # noqa: F401  (re-exported)
from ...gui_common.workflow_strip import WorkflowStrip as _Strip
from ..workflow import STEPS, TITLES, next_step

ACTION = {"detect": "Run Detect", "label": "Open the labeller", "train": "Train + classify",
          "classify": "Run Classify", "results": "Show results"}


class WorkflowStrip(_Strip):
    def __init__(self, app, theme: dict):
        super().__init__({k: TITLES[k] for k in STEPS}, ACTION, next_step, lambda k: app.show_step(k),
                         lambda k: app.run_step(k), theme,
                         loop_text="Label, Train and Classify form a loop: label → train + classify → label the new "
                                   "suggestions",
                         busy_fn=lambda: getattr(app, "_job_running", False), last="results")
