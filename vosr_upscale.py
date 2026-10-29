"""Run upstream VOSR in a subprocess, releasing its GPU memory before editing."""

import math
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "VOSR"
DEFAULT_PYTHON = ROOT / ".venv-vosr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def upscale_image(input_path, output_dir, *, scale=4, model="VOSR2", python=None, tile_size=512, seed=42, stage_label="[1/2]"):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    executable = Path(python or DEFAULT_PYTHON).resolve()
    checkpoint = REPO / "preset" / "ckpts" / model
    if scale < 1:
        raise ValueError("VOSR scale must be a positive integer.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not executable.is_file() or not (checkpoint / "args.json").is_file():
        raise RuntimeError("VOSR is not ready. Run: python setup_vosr.py --model " + model)
    if not any(checkpoint.rglob("*.safetensors")):
        raise RuntimeError(f"VOSR weights missing in {checkpoint}. Run python setup_vosr.py --model {model}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.png"
    # DiT uses 2x2 patches on 8x-downsampled latents. Pad before inference and
    # crop afterward so non-multiple-of-16 sizes do not break patch embedding.
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    multiple = 16 // math.gcd(16, scale)
    pad_w, pad_h = (-width) % multiple, (-height) % multiple
    padded = Image.new("RGB", (width + pad_w, height + pad_h))
    padded.paste(image, (0, 0))
    if pad_w:
        padded.paste(image.crop((width - 1, 0, width, height)).resize((pad_w, height)), (width, 0))
    if pad_h:
        padded.paste(padded.crop((0, height - 1, padded.width, height)).resize((padded.width, pad_h)), (0, height))

    env = os.environ.copy()
    # Upstream has @torch.compile decorators; ordinary CUDA works on Windows
    # without installing Linux Triton or incompatible xformers binaries.
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["XFORMERS_DISABLED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    with tempfile.TemporaryDirectory(prefix="vosr-", dir=output_dir) as temporary:
        staging = Path(temporary)
        staged_input = staging / "input.png"
        staged_output = staging / "result"
        padded.save(staged_input)
        command = [str(executable), str(REPO / "inference_vosr_onestep.py"),
                   "-c", str(checkpoint), "-i", str(staged_input), "-o", str(staged_output),
                   "-u", str(scale), "--tile_size", str(tile_size),
                   "--vae_tile_size", "512", "--seed", str(seed), "--force_rerun"]
        print(f"{stage_label} VOSR {model}: {width}x{height} -> {width * scale}x{height * scale}", flush=True)
        subprocess.run(command, cwd=REPO, env=env, check=True)
        generated = staged_output / "input.png"
        # Upstream catches per-image exceptions and can still exit with code 0.
        if not generated.is_file():
            raise RuntimeError("VOSR produced no image. See the traceback above; edit was not started.")
        with Image.open(generated) as result:
            expected = (padded.width * scale, padded.height * scale)
            if result.size != expected:
                raise RuntimeError(f"Unexpected VOSR output size: {result.size}, expected {expected}")
            result.crop((0, 0, width * scale, height * scale)).save(output_path)
    print(f"Upscaled image saved: {output_path}", flush=True)
    return output_path
