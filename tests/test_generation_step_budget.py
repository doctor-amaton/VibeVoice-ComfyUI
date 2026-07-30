import torch

from vvembed.modular.modeling_vibevoice_inference import _compute_text_token_lengths


def test_text_lengths_exclude_speech_placeholder_tokens():
    # attention has 10 valid tokens
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    # 6 of those are speech prompt placeholders
    speech_input_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)

    lengths = _compute_text_token_lengths(attention_mask, speech_input_mask)
    assert lengths.tolist() == [4]


def test_text_lengths_align_when_speech_mask_is_longer():
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    speech_input_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool)

    lengths = _compute_text_token_lengths(attention_mask, speech_input_mask)
    assert lengths.tolist() == [1]


def test_text_lengths_without_speech_mask_uses_attention_mask():
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

    lengths = _compute_text_token_lengths(attention_mask, None)
    assert lengths.tolist() == [3, 5]
