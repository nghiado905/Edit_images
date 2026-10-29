
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps

from vosr_upscale import upscale_image

# ================= CONFIG =================
MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

INPUT_PATH = r"D:\SDS\gen_img\09_49_50_09_50_30_8b7d902c_0.0000_a45b5d0388044edd961b47a9d763f625.jpg"
OUTPUT_DIR = "output"

EDIT_MAX_SIDE = 1024
UPSCALE = 4
STRENGTH = 0.65
GUIDANCE_SCALE = 7.5
NUM_STEPS = 35

SEEDS = [42, 123, 456]

# ================= PROMPT =================
PROMPT = """
RAW photo, photorealistic CCTV surveillance photograph,
a student sitting naturally at a classroom desk,
the same student from the reference image,

(student holding a black smartphone in one hand:1.4),
(student looking down at the smartphone screen:1.3),
(fingers naturally wrapped around the smartphone:1.3),
realistic hand position,
natural arm posture,
phone screen facing the student,
student actively using a mobile phone,

preserve original facial appearance,
preserve original hairstyle,
preserve original clothing,
preserve original body proportions,
preserve original classroom environment,
preserve the same desk and chair,
preserve original camera perspective,
preserve original lighting and shadows,

realistic human anatomy,
natural hands and fingers,
consistent perspective,
realistic shadows,
photographic details,
natural image texture
"""

NEGATIVE_PROMPT = """
cartoon, anime, illustration, painting,
CGI, 3d render, artificial image,
plastic skin, unrealistic skin,

extra fingers, missing fingers,
six fingers, deformed hands,
extra hands, extra arms,
disconnected limbs, distorted anatomy,

floating smartphone,
multiple smartphones,
oversized smartphone,
phone fused with hand,
incorrect phone perspective,

different person,
changed face,
different clothes,
changed hairstyle,
modified background,
additional people,

blurry, low quality,
watermark, text, artifacts,
unrealistic lighting
"""

def prepare_edit_image(path, max_side):
    """Keep aspect ratio; only pad right/bottom to the VAE's multiple of 8."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width = max(64, (image.width + 7) // 8 * 8)
    height = max(64, (image.height + 7) // 8 * 8)
    return ImageOps.expand(image, (0, 0, width - image.width, height - image.height), fill=(20, 20, 20))


def main():
    parser = argparse.ArgumentParser(description="Crop -> VOSR upscale -> Realistic Vision image editing")
    parser.add_argument("--input", type=Path, default=Path(INPUT_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / OUTPUT_DIR)
    parser.add_argument("--upscale", type=int, default=UPSCALE)
    parser.add_argument("--vosr-model", choices=["VOSR2", "VOSR_0.5B_os", "VOSR_1.4B_os"], default="VOSR2")
    parser.add_argument("--vosr-python", type=Path)
    parser.add_argument("--skip-upscale", action="store_true", help="Edit an already-upscaled input")
    parser.add_argument("--upscale-only", action="store_true")
    parser.add_argument("--edit-max-side", type=int, default=EDIT_MAX_SIDE, help="0 keeps the full VOSR resolution")
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--local-files-only", action="store_true", help="Require the edit model to be cached already")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image not found: {args.input}")
    if args.upscale < 1 or args.edit_max_side < 0 or (0 < args.edit_max_side < 64):
        parser.error("Use --upscale >= 1 and --edit-max-side 0 or >= 64")
    if args.steps < 1 or not 0 < args.strength <= 1 or int(args.steps * args.strength) < 1:
        parser.error("Use --steps >= 1, 0 < --strength <= 1 and steps * strength >= 1")
    if args.upscale_only and args.skip_upscale:
        parser.error("--upscale-only and --skip-upscale cannot be combined")

    import torch

    print(f"Python: {sys.executable}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run using the gen_img Python interpreter.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    # Each run has its own directory, so earlier generated images are preserved.
    run_dir = args.output_dir.resolve() / datetime.now().strftime("vosr_edit_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    edit_source = args.input.resolve()
    if not args.skip_upscale:
        edit_source = upscale_image(
            edit_source, run_dir / "upscaled", scale=args.upscale,
            model=args.vosr_model, python=args.vosr_python,
        )
    if args.upscale_only:
        print(f"Completed upscale in {time.perf_counter() - started:.1f}s: {edit_source}", flush=True)
        return

    # VOSR's process has exited: its models no longer occupy GPU memory.
    from diffusers import StableDiffusionImg2ImgPipeline

    image = prepare_edit_image(edit_source, args.edit_max_side)
    image.save(run_dir / "edit_input.png")
    print(f"[2/2] Loading edit model; input size: {image.width}x{image.height}", flush=True)
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, local_files_only=True,
        )
    except OSError:
        if args.local_files_only:
            raise
        print("Edit model is not fully cached; downloading missing files...", flush=True)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    for seed in args.seeds:
        print(f"Generating seed {seed}...", flush=True)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt, negative_prompt=args.negative_prompt,
                image=image, strength=args.strength,
                guidance_scale=GUIDANCE_SCALE, num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
        output_path = run_dir / f"student_phone_{seed}.png"
        result.save(output_path)
        print(f"Saved: {output_path}", flush=True)

    print(f"Completed in {time.perf_counter() - started:.1f}s. Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
