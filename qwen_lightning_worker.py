"""Separate Qwen encoder/edit processes keep a 20B edit model practical on 12GB VRAM."""

import argparse
import json
from pathlib import Path

from PIL import Image

from qwen_config import BASE_DIR, GGUF_DIR, GGUF_NAME, LIGHTNING_SCHEDULER, LORA_DIR, LORA_NAME


def load_images(job):
    images = []
    for path in job["images"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def encode(job):
    import psutil
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.hooks import apply_group_offloading
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions
    from safetensors.torch import save_file

    available = psutil.virtual_memory().available / 1024**3
    if available < 20:
        raise RuntimeError(f"Qwen text encoder needs about 20 GiB available system RAM; currently {available:.1f} GiB.")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(BASE_DIR), transformer=None, vae=None, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    # Qwen calls its token embedding from the outer vision-language wrapper,
    # before language_model.forward. Leaf hooks follow those direct calls too.
    apply_group_offloading(
        pipe.text_encoder, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        offload_type="leaf_level", use_stream=False,
    )
    condition_images = []
    for image in load_images(job):
        width, height = calculate_dimensions(CONDITION_IMAGE_SIZE, image.width / image.height)
        condition_images.append(pipe.image_processor.resize(image, height, width))
    tensors = {}
    prompts = [("prompt", job["prompt"])]
    if job["true_cfg_scale"] > 1:
        prompts.append(("negative", job["negative_prompt"]))
    with torch.inference_mode():
        for name, prompt in prompts:
            embeddings, mask = pipe.encode_prompt(prompt=prompt, image=condition_images, device=torch.device("cuda"))
            tensors[f"{name}_embeds"] = embeddings.detach().cpu().contiguous()
            if mask is None:
                mask = torch.ones(embeddings.shape[:2], dtype=torch.int64, device=embeddings.device)
            tensors[f"{name}_mask"] = mask.detach().cpu().contiguous()
    save_file(tensors, job["embeddings"])
    print(f"Image-conditioned prompt embeddings saved: {job['embeddings']}", flush=True)


def edit(job):
    import torch
    from diffusers import (FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig,
                           QwenImageEditPlusPipeline, QwenImageTransformer2DModel)
    from safetensors.torch import load_file

    transformer = QwenImageTransformer2DModel.from_single_file(
        str(GGUF_DIR / GGUF_NAME), config=str(BASE_DIR), subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    )
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(BASE_DIR), transformer=transformer, text_encoder=None,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER),
        dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    )
    # Keep LoRA separate: fusing floating-point LoRA updates into packed GGUF
    # integer weights would invalidate the quantized representation.
    pipe.load_lora_weights(str(LORA_DIR), weight_name=LORA_NAME, adapter_name="lightning")
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        offload_type="block_level", num_blocks_per_group=1, use_stream=False,
    )
    pipe.vae.to("cuda")
    pipe.vae.enable_tiling()
    tensors = load_file(job["embeddings"], device="cuda")
    inputs = {
        "image": load_images(job), "prompt_embeds": tensors["prompt_embeds"],
        "prompt_embeds_mask": tensors["prompt_mask"],
        "height": job["height"], "width": job["width"],
        "num_inference_steps": job["steps"], "true_cfg_scale": job["true_cfg_scale"],
        "generator": torch.Generator(device="cpu").manual_seed(job["seed"]),
    }
    if job["true_cfg_scale"] > 1:
        inputs.update(negative_prompt_embeds=tensors["negative_embeds"],
                      negative_prompt_embeds_mask=tensors["negative_mask"])
    with torch.inference_mode():
        output = pipe(**inputs).images[0]
    if output.size != (job["width"], job["height"]):
        raise RuntimeError(f"Unexpected Qwen output size: {output.size}")
    output.save(job["output"])
    print(f"Qwen edited image saved: {job['output']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["encode", "edit"], required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen requires CUDA. Recreate .venv-qwen from the working gen_img environment.")
    print(f"Qwen {args.stage}: {torch.cuda.get_device_name(0)}, BF16 + CPU offload", flush=True)
    torch.cuda.reset_peak_memory_stats()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    (encode if args.stage == "encode" else edit)(job)
    print(f"Peak CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
