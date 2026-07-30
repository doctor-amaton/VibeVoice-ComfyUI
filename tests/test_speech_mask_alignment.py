import torch

from vvembed.modular.modeling_vibevoice_inference import (
    _apply_speech_embeds,
    _inject_inputs_embeds_if_missing,
    _resolve_prefill_speech_input_mask,
)


def test_apply_speech_embeds_trims_long_mask_without_crash():
    # Reproduces runtime error shape pattern: mask seq_len > embeds seq_len.
    inputs_embeds = torch.zeros(1, 208, 4)
    speech_input_mask = torch.zeros(1, 209, dtype=torch.bool)
    speech_input_mask[0, 0] = True
    speech_embeds = torch.ones(1, 4)

    out = _apply_speech_embeds(inputs_embeds, speech_input_mask, speech_embeds)
    assert out.shape == (1, 208, 4)
    assert torch.allclose(out[0, 0], torch.ones(4))


def test_apply_speech_embeds_handles_slot_embed_count_mismatch():
    inputs_embeds = torch.zeros(1, 4, 3)
    speech_input_mask = torch.tensor([[True, True, True, False]])
    speech_embeds = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    out = _apply_speech_embeds(inputs_embeds, speech_input_mask, speech_embeds)
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(out[0, 1], torch.tensor([2.0, 2.0, 2.0]))
    # Third slot remains unchanged since only two embeds were available.
    assert torch.allclose(out[0, 2], torch.tensor([0.0, 0.0, 0.0]))


def test_inject_inputs_embeds_if_missing_does_not_override_input_ids():
    model_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}
    embeds = torch.zeros(1, 1, 3)

    out = _inject_inputs_embeds_if_missing(model_inputs, embeds)
    assert "inputs_embeds" not in out
    assert out["input_ids"] is not None


def test_inject_inputs_embeds_if_missing_when_input_ids_none():
    model_inputs = {"input_ids": None}
    embeds = torch.zeros(1, 1, 3)

    out = _inject_inputs_embeds_if_missing(model_inputs, embeds)
    assert "inputs_embeds" in out
    assert out["input_ids"] is None
    assert out["inputs_embeds"].shape == (1, 1, 3)


def test_resolve_prefill_speech_input_mask_prefers_input_ids_diffusion_tokens():
    model_input_ids = torch.tensor([[10, 42, 42, 11]])
    stale_mask = torch.tensor([[False, True, False, False]])

    resolved = _resolve_prefill_speech_input_mask(model_input_ids, stale_mask, speech_diffusion_id=42)
    assert resolved.tolist() == [[False, True, True, False]]


def test_resolve_prefill_speech_input_mask_falls_back_without_input_ids():
    stale_mask = torch.tensor([[False, True, False]])

    resolved = _resolve_prefill_speech_input_mask(None, stale_mask, speech_diffusion_id=42)
    assert torch.equal(resolved, stale_mask)
