# VOSR -> Qwen-Image-Edit-2511 Lightning

This is a separate pipeline. The existing `test_gen.py` Realistic Vision pipeline
is still available.

```text
crop -> VOSR 4x -> Qwen image/prompt encoder -> Qwen 2511 GGUF + Lightning 8 steps
```

The Lightning repository contains an adapter; it is not a standalone edit pipeline.
This implementation uses the official Qwen 2511 text encoder, processor and VAE,
Unsloth's Q4_K_M quantization of the Qwen 2511 transformer, and the original
`lightx2v/Qwen-Image-Edit-2511-Lightning` BF16 LoRA. The flow-matching scheduler uses
the author's shift=3 settings, with 8 steps and `true_cfg_scale=1` by default.
The LoRA remains separate from the quantized weights.

## Setup

Use the working CUDA environment:

```powershell
conda activate gen_img
python setup_qwen_lightning.py
```

Setup creates `.venv-qwen/`, sharing CUDA PyTorch from `gen_img`, and downloads only
the selected files to `models/qwen-edit-2511/`. It does not download the roughly
40GB BF16 transformer. The selected files total about 29 GiB / 31 GB. Downloads
resume and large files are SHA-256 checked. See `download_plan.json` for the exact
selected releases. `--plan-only` lists download size; `--skip-download` installs
dependencies only. VOSR uses its existing `.venv-vosr/` and checkpoint.

The RTX 3060 has 12GB VRAM. Encoder and denoising run in different processes to
release CPU/GPU memory between stages. The encoder uses leaf CPU offload to follow
Qwen's direct embedding calls; the transformer uses block CPU offload and the VAE
uses tiling. Keep at least approximately 20 GiB system RAM available. CPU offload
and GGUF dequantization trade inference time for lower VRAM usage; 8 steps does
not imply the same speed as the smaller Realistic Vision model.

## Run

```powershell
python qwen_lightning_edit.py --input "D:\SDS\gen_img\your_crop.jpg" --seed 42
```

The default input is the existing crop in this workspace. To avoid an IDE picking
the old `dots_ocr` interpreter:

```powershell
& "C:\Users\MINHNGHIA\miniconda3\envs\gen_img\python.exe" .\qwen_lightning_edit.py --seed 42
```

Give Qwen an edit instruction, rather than a Stable Diffusion list of style tags:

```powershell
python qwen_lightning_edit.py --prompt "Change the shirt to blue. Keep the same person, pose, classroom and lighting."
```

One input image is sufficient. Its VOSR upscale is the first reference image.
Optional additional references can be passed with repeated `--reference` flags;
describe the relationship between image 1 and image 2 in the prompt.

```powershell
python qwen_lightning_edit.py --input crop.jpg --reference person_reference.jpg --prompt "Use image 1 as the scene and image 2 as the appearance reference."
```

Useful options:

```powershell
python qwen_lightning_edit.py --check
python qwen_lightning_edit.py --upscale-only
python qwen_lightning_edit.py --input already_upscaled.png --skip-upscale
python qwen_lightning_edit.py --edit-max-side 768
python qwen_lightning_edit.py --edit-max-side 0
```

Output is saved in a new `output/qwen_edit_<timestamp>/` directory:

- `upscaled/`: the full VOSR result.
- `edit_input.png`: the actual Qwen input, preserving aspect ratio with edge padding to 16 pixels.
- `job.json`: prompt, dimensions, seed and generation parameters.
- `prompt_embeddings.safetensors`: image-conditioned Qwen embeddings.
- `qwen_lightning_<seed>.png`: the edited image.

The longest edit side defaults to at most 1024 pixels. `--edit-max-side 0` retains
the full upscale but needs more VRAM. The input remains an image reference; Qwen
Edit Plus does not use Stable Diffusion's `strength` parameter. A negative prompt
only applies when `--true-cfg-scale` is greater than 1; keep the recommended default
of 1 with this 4-step Lightning adapter.

Sources:

- https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF
- https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning/blob/main/generate_with_diffusers.py

## Verified run

Tested on the RTX 3060 12GB with the workspace crop, seed 42, 8 steps, and a
992x512 edit input. End-to-end time was **233.8 seconds**, including model loading:
79.57s for VOSR, 44.10s for image/prompt encoding, and 110.01s for Qwen loading and
editing. The four denoising steps themselves took about 63s. Time depends on disk
cache, available RAM and other GPU work.

Sample: `output/qwen_edit_20261029_112736_363407/qwen_lightning_42.png`.
Per-stage timing is saved in that run's `job.json`. The encoder and edit workers
reported peak CUDA allocations of 1.20 and 1.37 GiB respectively; these figures
exclude VOSR, the CUDA driver and other processes.
