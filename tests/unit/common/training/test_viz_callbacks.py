from typing import Any

import torch

from larry.common.training.viz_callbacks import VizCallback


class RecordingViz(VizCallback):
    def __init__(self, viz_steps: int) -> None:
        self.viz_steps = viz_steps
        self.steps: list[int] = []

    def viz(self, step: int) -> None:
        self.steps.append(step)


def test_viz_runs_on_completed_steps_so_the_last_step_and_checkpoints_line_up() -> None:
    callback = RecordingViz(viz_steps=2000)

    for step in range(100_000):
        callback.on_step_end(torch.tensor(0.0), step)

    assert callback.steps[0] == 2000
    assert callback.steps[-1] == 100_000
    assert len(callback.steps) == 50


def test_every_viz_callback_inherits_the_completed_step_trigger() -> None:
    def subclasses(cls: type[Any]) -> set[type[Any]]:
        return {sub for direct in cls.__subclasses__() for sub in {direct, *subclasses(direct)}}

    import larry.image.training.vae_viz_callbacks
    import larry.multimodal.training.multimodal_viz_callbacks
    import larry.voice.training.voice_viz_callbacks  # noqa: F401

    assert all(sub.on_step_end is VizCallback.on_step_end for sub in subclasses(VizCallback))
