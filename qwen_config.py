"""Release names shared by setup, the Qwen runner, and CPU-only tests."""

import math
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv-qwen"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MODEL_DIR = ROOT / "models" / "qwen-edit-2511"
BASE_REPO = "Qwen/Qwen-Image-Edit-2511"
BASE_DIR = MODEL_DIR / "base"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_NAME = "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"
LORA_DIR = MODEL_DIR / "lightning"
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
GGUF_NAME = "qwen-image-edit-2511-Q4_K_M.gguf"
GGUF_DIR = MODEL_DIR / "gguf"

# Shift=3, matching the Lightning author's generate_with_diffusers.py.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3),
    "max_image_seq_len": 8192, "max_shift": math.log(3),
    "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "invert_sigmas": False, "stochastic_sampling": False,
    "time_shift_type": "exponential", "use_dynamic_shifting": True,
    "use_beta_sigmas": False, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}

DEFAULT_PROMPT = (
    "Edit the input photo so the student is naturally holding one black smartphone "
    "and looking down at its screen. Keep the same person, hairstyle, school uniform, "
    "desk, chair, classroom, camera angle and lighting. Change only the hands and "
    "phone as needed. Use realistic hand anatomy and natural photographic details."
)


def missing_models():
    required = [BASE_DIR / "model_index.json", BASE_DIR / "transformer" / "config.json",
                BASE_DIR / "text_encoder" / "config.json",
                BASE_DIR / "text_encoder" / "model.safetensors.index.json",
                BASE_DIR / "vae" / "config.json", BASE_DIR / "processor" / "preprocessor_config.json",
                BASE_DIR / "tokenizer" / "tokenizer_config.json", LORA_DIR / LORA_NAME,
                GGUF_DIR / GGUF_NAME]
    missing = [str(path) for path in required if not path.is_file()]
    index = BASE_DIR / "text_encoder" / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        missing.extend(str(BASE_DIR / "text_encoder" / name) for name in sorted(set(weights.values()))
                       if not (BASE_DIR / "text_encoder" / name).is_file())
    for component in ("text_encoder", "vae"):
        if not any((BASE_DIR / component).glob("*.safetensors")):
            missing.append(str(BASE_DIR / component / "*.safetensors"))
    return missing
