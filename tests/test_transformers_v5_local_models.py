from pathlib import Path

import pytest


pytestmark = [pytest.mark.v5, pytest.mark.local_model]


def _skip_if_not_v5(is_transformers_v5_plus):
    if not is_transformers_v5_plus:
        pytest.skip("These checks target transformers v5+ only.")


def _require_local_tests_enabled(run_local_model_tests):
    if not run_local_model_tests:
        pytest.skip(
            "Local model tests are disabled. Set VIBEVOICE_RUN_LOCAL_MODEL_TESTS=1 to enable."
        )


def _require_path(path: Path, label: str):
    assert path is not None, (
        f"Could not find {label}. Check env vars and local model placement."
    )


def test_can_load_local_vibevoice_model_from_pretrained(
    is_transformers_v5_plus,
    run_local_model_tests,
    models_root,
    vibevoice_model_dir,
):
    _skip_if_not_v5(is_transformers_v5_plus)
    _require_local_tests_enabled(run_local_model_tests)

    _require_path(models_root, "models root")
    _require_path(vibevoice_model_dir, "a local VibeVoice model directory")

    from vvembed.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        str(vibevoice_model_dir),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    assert model is not None


def test_can_load_local_vibevoice_processor_with_qwen_tokenizer(
    is_transformers_v5_plus,
    run_local_model_tests,
    models_root,
    vibevoice_model_dir,
    qwen_tokenizer_dir,
):
    _skip_if_not_v5(is_transformers_v5_plus)
    _require_local_tests_enabled(run_local_model_tests)

    _require_path(models_root, "models root")
    _require_path(vibevoice_model_dir, "a local VibeVoice model directory")
    _require_path(qwen_tokenizer_dir, "Qwen tokenizer directory")

    from vvembed.processor.vibevoice_processor import VibeVoiceProcessor

    processor = VibeVoiceProcessor.from_pretrained(
        str(vibevoice_model_dir),
        trust_remote_code=True,
        local_files_only=True,
        language_model_pretrained_name=str(qwen_tokenizer_dir),
    )
    assert processor is not None
