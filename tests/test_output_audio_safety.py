import torch


def test_filter_none_speech_outputs_before_concat():
    speech_tensors = [None, torch.ones(1, 10), None, torch.ones(1, 6)]
    valid = [t for t in speech_tensors if isinstance(t, torch.Tensor)]
    assert len(valid) == 2
    out = torch.cat(valid, dim=-1)
    assert out.shape[-1] == 16
