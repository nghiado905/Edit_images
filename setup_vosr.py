"""Install VOSR inference separately, sharing the working gen_img CUDA PyTorch."""

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "VOSR"
VENV = ROOT / ".venv-vosr"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MODEL_REPO = "CSWRY/VOSR"
# Pin the upstream release inspected for this integration.
MODEL_REVISION = "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3"


def run(command, **kwargs):
    print("+", subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def download_weights(model):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    checkpoint_root = REPO / "preset" / "ckpts"
    config_file = hf_hub_download(
        MODEL_REPO, f"{model}/args.json", revision=MODEL_REVISION,
        local_dir=checkpoint_root,
    )
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    patterns = [f"{model}/*", "README.md"]
    if config["ae_type"] == "qwen":
        patterns += ["Qwen-Image-vae-2d/*"]
    else:
        patterns += ["stable-diffusion-2-1-base/vae/*", "sd21_lwdecoder.pth"]
    # Download only the chosen encoder's weights, not every DINOv2 size.
    encoder = {"dinov2b": "vitb14", "dinov2l": "vitl14", "dinov2g": "vitg14"}[config["enc_type"]]
    dino_code = "torch_cache/facebookresearch_dinov2_main"
    patterns += [f"torch_cache/checkpoints/dinov2_{encoder}_pretrain.pth",
                 f"{dino_code}/*.py", f"{dino_code}/LICENSE*", f"{dino_code}/MODEL_CARD.md"]
    # The Xet extension inherited from newer gen_img installations can stall on
    # large files with this upstream Hub version. Windows curl supports the same
    # official HTTPS URLs, resumes partial files, and bounds stalled transfers.
    curl = shutil.which("curl.exe") if os.name == "nt" else None
    large_files = []
    if curl:
        info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
        large_files = [f for f in info.siblings if f.size and f.size >= 64 * 1024**2
                       and any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in patterns)]
    snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_dir=checkpoint_root,
        allow_patterns=patterns,
        ignore_patterns=["**/__pycache__/*"] + [f.rfilename for f in large_files], max_workers=4,
    )
    for file in large_files:
        destination = checkpoint_root / file.rfilename
        if destination.is_file() and destination.stat().st_size == file.size:
            print(f"Already downloaded: {file.rfilename}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{quote(file.rfilename, safe='/')}"
        print(f"Downloading {file.rfilename} ({file.size / 1024**3:.2f} GiB) via HTTPS...", flush=True)
        if not partial.is_file() or partial.stat().st_size < file.size:
            run([curl, "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                 "--retry-delay", "2", "--connect-timeout", "20", "--speed-limit", "1024",
                 "--speed-time", "120", "--continue-at", "-", "--output", partial, url])
        if partial.stat().st_size != file.size:
            raise RuntimeError(f"Incomplete download: {file.rfilename}; rerun setup to resume.")
        if file.lfs and file.lfs.sha256:
            with partial.open("rb") as downloaded:
                digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
            if digest != file.lfs.sha256:
                raise RuntimeError(f"Checksum mismatch: {partial}. Remove this partial file and rerun setup.")
        partial.replace(destination)
        print(f"Verified: {file.rfilename}", flush=True)
    print(f"VOSR checkpoint ready: {checkpoint_root / model}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["VOSR2", "VOSR_0.5B_os", "VOSR_1.4B_os"], default="VOSR2")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    if args.download_only:
        download_weights(args.model)
        return

    import torch
    import torchvision

    print(f"Base Python: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Activate the gen_img environment with working CUDA before setup.")
    if not (REPO / "inference_vosr_onestep.py").is_file():
        if REPO.exists():
            raise RuntimeError(f"{REPO} exists but is not a complete VOSR checkout.")
        run(["git", "clone", "https://github.com/cswry/VOSR.git", REPO])
    if not PYTHON.exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(VENV)
    run([PYTHON, "-m", "pip", "install", "--disable-pip-version-check", "--prefer-binary",
         "--cache-dir", ROOT / ".cache" / "pip",
         "-r", ROOT / "requirements-vosr-inference.txt"])
    run([PYTHON, "-c", "import torch, torchvision, timm, einops, fairscale, cv2; "
         "from diffusers import AutoencoderKL; print('VOSR dependencies OK; CUDA:', torch.cuda.is_available())"])
    if not args.skip_download:
        run([PYTHON, Path(__file__), "--download-only", "--model", args.model])
    print("Setup complete. Run: python test_gen.py", flush=True)


if __name__ == "__main__":
    main()
