# VOSR upscale before image editing

The pipeline in `test_gen.py` runs:

`input crop -> VOSR one-step (4x) -> Realistic Vision img2img -> PNG`

VOSR runs once per input, before the edit model loads. Its subprocess exits and
releases VRAM before the edit model uses CUDA. Your existing prompts and seeds
remain the defaults. Every run writes a new directory under `output/`.

## Setup on this machine

Use the working `gen_img` environment, not `dots_ocr`:

```powershell
conda activate gen_img
python setup_vosr.py
```

Setup reuses the existing `VOSR/` checkout and creates `.venv-vosr/` with access to
the working CUDA torch/torchvision from `gen_img`. Other VOSR inference packages
are installed into that local environment, leaving `gen_img` unchanged. Windows
inference disables upstream `torch.compile` and xformers; Triton and training-only
dependencies are not installed. Keep the base `gen_img` environment available.

The default checkpoint is **VOSR2**, the upstream 1.4B one-step model. Only the
selected checkpoint, its VAE, and the selected DINOv2 encoder are downloaded from
`CSWRY/VOSR` into `VOSR/preset/ckpts/`. The first setup needs internet and several
GB of storage. First inference may also fetch the upstream DINOv2 code/weights.
On Windows, large weights use resumable HTTPS downloads through `curl.exe` and
are checked against the release's SHA-256 hashes. This avoids an observed stall
in the Xet downloader. Rerunning setup resumes `.partial` downloads.

## Run

```powershell
python test_gen.py
python test_gen.py --input "D:\SDS\gen_img\your_crop.jpg" --seeds 42
```

To avoid an IDE choosing `dots_ocr`, explicitly select the interpreter:

```powershell
& "C:\Users\MINHNGHIA\miniconda3\envs\gen_img\python.exe" .\test_gen.py
```

Useful options:

```powershell
# Test only the VOSR stage.
python test_gen.py --upscale-only

# Keep the upscaled resolution for editing (more VRAM/time).
python test_gen.py --edit-max-side 0 --seeds 42

# Edit an upscaled PNG again without rerunning VOSR.
python test_gen.py --input "output\your_run\upscaled\your_crop.png" --skip-upscale

# Smaller VOSR checkpoint, if needed.
python setup_vosr.py --model VOSR_0.5B_os
python test_gen.py --vosr-model VOSR_0.5B_os
```

`--upscale` defaults to 4. The full upscale is saved to `upscaled/`. Before editing,
the longest side defaults to at most 1024 pixels to limit VRAM on the RTX 3060.
The input keeps its aspect ratio and is padded to a multiple of 8, rather than
being squeezed into 512x512. `edit_input.png` records the actual edit input.
Use `--edit-max-side 0` to retain the entire upscale, or `--edit-max-side 768` for less VRAM.

The default edit strength remains 0.65. Reduce it with `--strength 0.35` for less
change. The default still generates all three existing seeds; use `--seeds 42` for
one output. VOSR's restored details and the edit output are generated details, not
an identity-preserving reconstruction guarantee.

The edit model first loads from its local Hugging Face cache, avoiding unnecessary
network checks on subsequent runs. `--local-files-only` prevents downloading a
missing edit model.

References: https://github.com/cswry/VOSR and https://huggingface.co/CSWRY/VOSR.
