from vvembed.processor.vibevoice_processor import VibeVoiceProcessor


class _DummyTokenizer:
    speech_start_id = 1001
    speech_end_id = 1002
    speech_diffusion_id = 1003
    pad_id = 0

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        # Return a deterministic, non-empty token list for all calls
        return [11, 12, 13]


def test_system_prompt_encoding_disables_special_tokens():
    tok = _DummyTokenizer()
    proc = VibeVoiceProcessor(tokenizer=tok, audio_processor=None)

    proc._process_single("Speaker 1: hello", voice_samples=None)

    # First encode call is for system prompt and must avoid tokenizer-added special tokens.
    assert tok.calls[0][0] == proc.system_prompt
    assert tok.calls[0][1] is False
