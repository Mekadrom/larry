from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator, debug_launcher
from accelerate.state import AcceleratorState

from larry.common.training.trainers import TrainerBase, TrainState


def test_tokens_seen_survives_an_accelerate_checkpoint_round_trip(tmp_path: Path) -> None:
    saving = Accelerator(cpu=True)
    state = TrainState()
    saving.register_for_checkpointing(state)
    state.global_step, state.tokens_seen = 12, 34567
    saving.save_state(str(tmp_path / "checkpoint"))

    loading = Accelerator(cpu=True)
    restored = TrainState()
    loading.register_for_checkpointing(restored)
    loading.load_state(str(tmp_path / "checkpoint"))

    assert (restored.global_step, restored.tokens_seen) == (12, 34567)


def _count_and_sync_on_each_process() -> None:
    accelerator = Accelerator(cpu=True)
    trainer = SimpleNamespace(accelerator=accelerator, train_state=TrainState(), _pending_tokens=0)
    rank = accelerator.process_index

    TrainerBase.count_tokens(trainer, 10 * (rank + 1))
    TrainerBase.count_tokens(trainer, 1)
    first = TrainerBase.sync_tokens_seen(trainer)

    TrainerBase.count_tokens(trainer, 100 if rank == 0 else 0)
    second = TrainerBase.sync_tokens_seen(trainer)
    third = TrainerBase.sync_tokens_seen(trainer)

    assert (first, second, third) == (32, 132, 132), (rank, first, second, third)


def test_tokens_seen_sums_every_process_and_agrees_on_all_of_them() -> None:
    # accelerate keeps process-wide state; one left by an earlier single-process Accelerator would leak into the fork
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        debug_launcher(_count_and_sync_on_each_process, num_processes=2)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
