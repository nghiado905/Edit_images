"""Set up the VOSR -> Qwen-Image-Edit-2511 Lightning pipeline for a 12 GB GPU."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import venv

from model_downloads import download_release, selected_release
from qwen_config import (BASE_DIR, BASE_REPO, GGUF_DIR, GGUF_NAME, GGUF_REPO,
                         LORA_DIR, LORA_NAME, LORA_REPO, MODEL_DIR, PYTHON, ROOT, VENV)


def run(command):
    print("+", subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    subprocess.run([str(arg) for arg in command], check=True)


def download_models(plan_only=False):
    releases = [
        (selected_release(BASE_REPO, [
            "model_index.json", "README.md", "scheduler/*", "tokenizer/*", "processor/*",
            "text_encoder/*", "vae/*", "transformer/config.json",
        ]), BASE_DIR),
        (selected_release(LORA_REPO, [LORA_NAME, "README.md"]), LORA_DIR),
        (selected_release(GGUF_REPO, [GGUF_NAME, "README.md"]), GGUF_DIR),
    ]
    total = sum(file.get("size", 0) for release, _ in releases for file in release["files"])
    remaining = sum(
        max(0, file.get("size", 0) - ((directory / file["rfilename"]).stat().st_size
                                    if (directory / file["rfilename"]).is_file() else 0))
        for release, directory in releases for file in release["files"]
    )
    print(f"Selected model files: {total / 1024**3:.2f} GiB; remaining: {remaining / 1024**3:.2f} GiB", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "download_plan.json").write_text(
        json.dumps([dict(release, directory=str(directory)) for release, directory in releases], indent=2),
        encoding="utf-8",
    )
    if plan_only:
        return
    if shutil.disk_usage(ROOT).free < remaining + 5 * 1024**3:
        raise RuntimeError("Not enough free disk space for the selected Qwen files plus a 5 GiB reserve.")
    for release, directory in releases:
        download_release(release, directory)
    print("Qwen models ready. Run: python qwen_lightning_edit.py --seed 42", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.download_only or args.plan_only:
        download_models(plan_only=args.plan_only)
        return
    import torch

    print(f"Base Python: {sys.executable}; torch: {torch.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run setup with the working CUDA gen_img Python.")
    if not PYTHON.exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(VENV)
    run([PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "--prefer-binary",
         "--cache-dir", ROOT / ".cache" / "pip", "-r", ROOT / "requirements-qwen-lightning.txt"])
    run([PYTHON, "-c", "import torch, gguf, peft; from diffusers import QwenImageEditPlusPipeline, "
         "QwenImageTransformer2DModel, GGUFQuantizationConfig; print('Qwen dependencies OK; CUDA:',torch.cuda.is_available())"])
    if not args.skip_download:
        download_models()
    print("Qwen setup complete.", flush=True)


if __name__ == "__main__":
    main()
