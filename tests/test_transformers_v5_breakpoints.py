import inspect

import pytest


pytestmark = pytest.mark.v5


def _skip_if_not_v5(is_transformers_v5_plus):
    if not is_transformers_v5_plus:
        pytest.skip("These checks target transformers v5+ only.")


def test_embedded_modeling_modules_import_on_v5(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    # These imports are currently known to be fragile because they use
    # internal transformers modules/paths that changed in v5.
    import vvembed.modular.modeling_vibevoice as _mv  # noqa: F401
    import vvembed.modular.modeling_vibevoice_inference as _mvi  # noqa: F401


def test_embedded_tokenizer_module_import_on_v5(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    import vvembed.modular.modular_vibevoice_text_tokenizer as _tok  # noqa: F401


def test_generation_mixin_contract_used_by_vibevoice(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    from transformers.generation import GenerationMixin

    required_methods = [
        "_prepare_generation_config",
        "_prepare_special_tokens",
        "_prepare_generated_length",
        "_prepare_cache_for_generation",
        "prepare_inputs_for_generation",
        "_update_model_kwargs_for_generation",
    ]

    missing = [name for name in required_methods if not hasattr(GenerationMixin, name)]
    assert not missing, f"GenerationMixin API changed or removed in v5: {missing}"


def test_prepare_cache_for_generation_signature(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    from transformers.generation import GenerationMixin

    sig = inspect.signature(GenerationMixin._prepare_cache_for_generation)
    # VibeVoice currently has branching logic for old/new signatures.
    # If this changes again in v5+, this test catches it before runtime.
    param_names = list(sig.parameters.keys())

    assert len(param_names) in {6, 7}, (
        "Unexpected _prepare_cache_for_generation signature; "
        f"got {param_names}."
    )


def test_tie_weights_accepts_v5_kwargs(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    from vvembed.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
    from vvembed.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    for cls in (VibeVoiceForConditionalGeneration, VibeVoiceForConditionalGenerationInference):
        sig = inspect.signature(cls.tie_weights)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        has_recompute_mapping = "recompute_mapping" in sig.parameters
        assert has_kwargs or has_recompute_mapping, (
            f"{cls.__name__}.tie_weights must accept v5 kwargs like recompute_mapping"
        )


def test_prepare_generation_config_v5_signature(is_transformers_v5_plus):
    _skip_if_not_v5(is_transformers_v5_plus)

    from transformers.generation import GenerationMixin

    sig = inspect.signature(GenerationMixin._prepare_generation_config)
    params = list(sig.parameters.values())

    # v5-style contract: (self, generation_config, **kwargs)
    assert len(params) >= 3
    assert params[1].name == "generation_config"
    assert params[2].kind == inspect.Parameter.VAR_KEYWORD
