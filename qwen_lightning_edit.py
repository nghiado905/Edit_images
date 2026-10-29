"""Image crop -> VOSR 4x -> Qwen Image Edit 2511 + Lightning 8-step LoRA."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image, ImageOps

from qwen_config import DEFAULT_PROMPT, PYTHON, ROOT, missing_models
from vosr_upscale import upscale_image

DEFAULT_INPUT = ROOT / "09_49_50_09_50_30_8b7d902c_0.0000_a45b5d0388044edd961b47a9d763f625.jpg"

CAMERA_ANGLES = (
    ("front", "Create a front camera view at eye level."),
    ("three_quarter", "Create a three-quarter camera view from the front-left side at eye level."),
    ("side", "Create a side-profile camera view from the student's right side at eye level."),
)


def prepare_image(path, max_side=1024):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width = max(64, (image.width + 15) // 16 * 16)
    height = max(64, (image.height + 15) // 16 * 16)
    # Repeat edge pixels instead of stretching the crop or adding a dark border.
    padded = Image.new("RGB", (width, height))
    padded.paste(image, (0, 0))
    if width > image.width:
        padded.paste(image.crop((image.width - 1, 0, image.width, image.height)).resize(
            (width - image.width, image.height)), (image.width, 0))
    if height > image.height:
        padded.paste(padded.crop((0, image.height - 1, width, image.height)).resize(
            (width, height - image.height)), (0, image.height))
    return padded


def run_worker(stage, job_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["DIFFUSERS_GGUF_CUDA_KERNELS"] = "false"
    command = [str(PYTHON), "-u", str(ROOT / "qwen_lightning_worker.py"),
               "--stage", stage, "--job", str(job_path)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, action="append", default=[], help="Optional additional reference image; repeat for more")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=" ")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--three-angles", action="store_true",
                        help="Generate front, three-quarter and side views using consecutive seeds")
    parser.add_argument("--edit-max-side", type=int, default=1024, help="0 keeps the entire upscale; larger images use more VRAM")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--vosr-model", choices=["VOSR2", "VOSR_0.5B_os", "VOSR_1.4B_os"], default="VOSR2")
    parser.add_argument("--vosr-python", type=Path)
    parser.add_argument("--skip-upscale", action="store_true")
    parser.add_argument("--upscale-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check setup without loading GPU models")
    args = parser.parse_args()
    if args.check:
        missing = missing_models()
        print(f"Qwen Python: {PYTHON} ({'found' if PYTHON.is_file() else 'missing'})")
        print("Qwen model files: " + ("ready" if not missing else "missing:\n" + "\n".join(missing)))
        return 0 if PYTHON.is_file() and not missing else 1
    for path in [args.input, *args.reference]:
        if not path.is_file():
            parser.error(f"Image not found: {path}")
    if args.upscale < 1 or args.edit_max_side < 0 or (0 < args.edit_max_side < 64):
        parser.error("Use --upscale >= 1 and --edit-max-side 0 or >= 64")
    if args.steps < 1 or args.true_cfg_scale < 1 or not args.prompt.strip():
        parser.error("Use --steps >= 1, --true-cfg-scale >= 1 and a nonempty prompt")
    if args.skip_upscale and args.upscale_only:
        parser.error("--skip-upscale and --upscale-only cannot be combined")
    if not args.upscale_only and (not PYTHON.is_file() or missing_models()):
        parser.error("Qwen setup is incomplete. Run: python setup_qwen_lightning.py")

    started = time.perf_counter()
    run_dir = args.output_dir.resolve() / datetime.now().strftime("qwen_edit_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    image_path = args.input.resolve()
    if not args.skip_upscale:
        image_path = upscale_image(image_path, run_dir / "upscaled", scale=args.upscale,
                                   model=args.vosr_model, python=args.vosr_python, stage_label="[1/3]")
    timings = {"upscale_seconds": round(time.perf_counter() - started, 2)}
    if args.upscale_only:
        print(f"Upscale complete: {image_path}")
        return 0

    edit_image = prepare_image(image_path, args.edit_max_side)
    edit_path = run_dir / "edit_input.png"
    edit_image.save(edit_path)
    references = [str(edit_path)]
    for index, source in enumerate(args.reference, start=1):
        path = run_dir / f"reference_{index}.png"
        prepare_image(source, args.edit_max_side).save(path)
        references.append(str(path))
    variants = CAMERA_ANGLES if args.three_angles else (("default", ""),)
    outputs = []
    for index, (angle_name, angle_instruction) in enumerate(variants, start=1):
        variant_dir = run_dir / f"angle_{index}_{angle_name}" if args.three_angles else run_dir
        variant_dir.mkdir(exist_ok=True)
        prompt = f"{args.prompt.rstrip()} {angle_instruction}".strip()
        seed = args.seed + index - 1
        job = {
            "images": references, "prompt": prompt, "negative_prompt": args.negative_prompt,
            "width": edit_image.width, "height": edit_image.height,
            "steps": args.steps, "true_cfg_scale": args.true_cfg_scale, "seed": seed,
            "camera_angle": angle_name,
            "embeddings": str(variant_dir / "prompt_embeddings.safetensors"),
            "output": str(variant_dir / f"qwen_lightning_{seed}.png"),
        }
        job_path = variant_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        label = f"Angle {index}/{len(variants)} ({angle_name})"
        print(f"[2/3] {label}: encoding Qwen image + edit instruction...", flush=True)
        stage_started = time.perf_counter()
        run_worker("encode", job_path)
        job_timings = {"encode_seconds": round(time.perf_counter() - stage_started, 2)}
        if not Path(job["embeddings"]).is_file():
            raise RuntimeError("Qwen encoder did not produce embeddings; generation was not started.")
        print(f"[3/3] {label}: Qwen GGUF 4-bit + Lightning LoRA, "
              f"{args.steps}-step editing...", flush=True)
        stage_started = time.perf_counter()
        run_worker("edit", job_path)
        job_timings["edit_seconds"] = round(time.perf_counter() - stage_started, 2)
        if not Path(job["output"]).is_file():
            raise RuntimeError("Qwen generation exited without producing an image.")
        job["timings"] = job_timings
        job_path.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(job["output"])
    timings["total_seconds"] = round(time.perf_counter() - started, 2)
    print(f"Completed in {timings['total_seconds']:.1f}s. Outputs:\n" + "\n".join(outputs), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
