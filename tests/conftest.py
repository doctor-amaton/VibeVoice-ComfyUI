import os
from pathlib import Path
from typing import Optional

import pytest
from packaging import version
import transformers


MIN_V5 = version.parse("5.0.0")


def _has_model_files(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    names = {p.name for p in folder.iterdir()}
    has_config = (folder / "config.json").exists()
    if not has_config:
        return False

    has_weights = (
        "pytorch_model.bin" in names
        or "model.safetensors" in names
        or "pytorch_model.bin.index.json" in names
        or "model.safetensors.index.json" in names
        or any(name.startswith("pytorch_model-") and name.endswith(".bin") for name in names)
        or any(name.startswith("model-") and name.endswith(".safetensors") for name in names)
    )
    return has_weights


def _find_model_dir_recursive(folder: Path, max_depth: int = 4, depth: int = 0) -> Optional[Path]:
    if depth >= max_depth:
        return None
    if _has_model_files(folder):
        return folder

    for child in folder.iterdir():
        if child.name.startswith(".") or child.name in {"loras", "__pycache__"}:
            continue
        if child.is_dir():
            result = _find_model_dir_recursive(child, max_depth=max_depth, depth=depth + 1)
            if result is not None:
                return result
    return None


def _resolve_models_root() -> Optional[Path]:
    candidates = []

    env_dir = os.environ.get("VIBEVOICE_TEST_MODELS_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    comfy_root = os.environ.get("COMFYUI_ROOT")
    if comfy_root:
        candidates.append(Path(comfy_root).expanduser() / "models" / "vibevoice")

    candidates.append(Path.home() / "ComfyUI" / "models" / "vibevoice")

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def _resolve_qwen_tokenizer_dir(models_root: Path) -> Optional[Path]:
    # Priority 1: models/vibevoice/tokenizer
    tokenizer_dir = models_root / "tokenizer"
    required = ["tokenizer_config.json", "vocab.json", "merges.txt"]
    if tokenizer_dir.exists() and all((tokenizer_dir / f).exists() for f in required):
        return tokenizer_dir

    # Priority 2: models/vibevoice/models--Qwen--Qwen2.5-1.5B/snapshots/[hash]
    qwen_cache_dir = models_root / "models--Qwen--Qwen2.5-1.5B" / "snapshots"
    if qwen_cache_dir.exists():
        for child in sorted(qwen_cache_dir.iterdir()):
            if child.is_dir() and (child / "tokenizer_config.json").exists():
                return child

    # Priority 3: standard HF cache locations
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-1.5B" / "snapshots"
    if hf_cache.exists():
        for child in sorted(hf_cache.iterdir()):
            if child.is_dir() and (child / "tokenizer_config.json").exists():
                return child

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_home_cache = Path(hf_home) / "hub" / "models--Qwen--Qwen2.5-1.5B" / "snapshots"
        if hf_home_cache.exists():
            for child in sorted(hf_home_cache.iterdir()):
                if child.is_dir() and (child / "tokenizer_config.json").exists():
                    return child

    return None


@pytest.fixture(scope="session")
def transformers_version():
    return version.parse(transformers.__version__)


@pytest.fixture(scope="session")
def is_transformers_v5_plus(transformers_version):
    return transformers_version >= MIN_V5


@pytest.fixture(scope="session")
def models_root() -> Optional[Path]:
    return _resolve_models_root()


@pytest.fixture(scope="session")
def vibevoice_model_dir(models_root: Optional[Path]) -> Optional[Path]:
    if models_root is None:
        return None

    preferred = os.environ.get("VIBEVOICE_TEST_MODEL_FOLDER")
    if preferred:
        preferred_base = models_root / preferred
        result = _find_model_dir_recursive(preferred_base)
        if result:
            return result

    for child in sorted(models_root.iterdir()):
        if child.name.startswith(".") or child.name == "loras":
            continue
        if child.is_dir():
            result = _find_model_dir_recursive(child)
            if result:
                return result
    return None


@pytest.fixture(scope="session")
def qwen_tokenizer_dir(models_root: Optional[Path]) -> Optional[Path]:
    if models_root is None:
        return None
    return _resolve_qwen_tokenizer_dir(models_root)


@pytest.fixture(scope="session")
def run_local_model_tests() -> bool:
    return os.environ.get("VIBEVOICE_RUN_LOCAL_MODEL_TESTS", "0") == "1"
