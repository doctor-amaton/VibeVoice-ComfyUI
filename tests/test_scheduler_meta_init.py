import pytest
import torch

from vvembed.schedule.dpm_solver import DPMSolverMultistepScheduler, _safe_move_to_cpu


def test_scheduler_init_in_meta_context_does_not_copy_meta_tensor():
    # Reproduces transformers v5 init path where tensors may be created on meta.
    try:
        with torch.device("meta"):
            scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=10,
                beta_schedule="linear",
            )
    except NotImplementedError as exc:
        pytest.fail(f"Scheduler should not attempt meta->cpu copy during init: {exc}")

    assert scheduler is not None


def test_safe_move_to_cpu_ignores_meta_copy_error_message():
    class FakeTensor:
        is_meta = False

        def to(self, *_args, **_kwargs):
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    fake = FakeTensor()
    moved = _safe_move_to_cpu(fake)
    assert moved is fake


def test_scheduler_set_timesteps_after_meta_init_does_not_fail():
    with torch.device("meta"):
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=10,
            beta_schedule="linear",
        )

    # This previously failed by trying to convert meta tensors to numpy.
    scheduler.set_timesteps(num_inference_steps=4, device="cpu")
    assert scheduler.timesteps.shape[0] == 4
