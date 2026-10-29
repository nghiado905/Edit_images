
---
license: apache-2.0
pipeline_tag: image-to-image
tags:
  - image-super-resolution
  - image-restoration
  - diffusion
  - one-step
---

<div align="center">

# VOSR Checkpoints

### Vision-Only Generative Model for Real-World Image Super-Resolution

[![Paper](https://img.shields.io/badge/arXiv-2604.03225-b31b1b.svg)](https://arxiv.org/abs/2604.03225)
[![Source](https://img.shields.io/badge/GitHub-cswry%2FVOSR-181717?logo=github)](https://github.com/cswry/VOSR)
[![Conference](https://img.shields.io/badge/CVPR-2026-2563eb)](https://arxiv.org/abs/2604.03225)
[![License](https://img.shields.io/badge/License-Apache--2.0-16a34a)](https://github.com/cswry/VOSR/blob/main/LICENSE)

Không cần prompt · Khôi phục ảnh thực tế · Suy luận one-step

</div>

---

## Giới thiệu

VOSR là mô hình super-resolution dùng hoàn toàn thông tin thị giác từ ảnh độ
phân giải thấp. Mô hình không phụ thuộc vào text prompt và được thiết kế riêng
cho bài toán khôi phục ảnh thực tế.

Thư mục này chứa checkpoint cục bộ phục vụ pipeline:

```text
ảnh đầu vào -> VOSR upscale -> ảnh độ phân giải cao -> Qwen Image Edit
```

## Checkpoint đang sử dụng

| Thành phần | Vai trò | Dung lượng gần đúng |
|---|---|---:|
| `VOSR2/` | Mô hình VOSR 2.0 one-step 1.4B | 5.2 GiB |
| `Qwen-Image-vae-2d/` | VAE 2D dùng để mã hóa và giải mã ảnh | 170 MiB |
| `torch_cache/` | Mã nguồn và trọng số DINOv2 được cache cục bộ | Tùy phiên bản |

`VOSR2` là lựa chọn mặc định và được khuyến nghị cho chất lượng phục hồi chi
tiết tốt. Có thể dùng model nhỏ hơn nếu GPU hoặc RAM bị giới hạn.

## Cấu trúc thư mục

```text
preset/ckpts/
├── VOSR2/
│   ├── args.json
│   └── checkpoints/
│       └── ema_model.safetensors
├── Qwen-Image-vae-2d/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── torch_cache/
└── README.md
```

Tên file và vị trí thư mục cần được giữ nguyên để script inference tìm thấy
checkpoint mà không cần tải lại.

## Chạy inference trực tiếp

Từ thư mục gốc của VOSR:

```powershell
cd D:\SDS\gen_img\VOSR

& "D:\SDS\gen_img\.venv-vosr\Scripts\python.exe" `
  .\inference_vosr_onestep.py `
  -c .\preset\ckpts\VOSR2 `
  -i D:\path\input `
  -o D:\path\output `
  -u 4 `
  --tile_size 512 `
  --vae_tile_size 512 `
  --seed 42 `
  --force_rerun
```

Trong đó:

- `-c`: thư mục checkpoint.
- `-i`: ảnh hoặc thư mục ảnh đầu vào.
- `-o`: thư mục lưu kết quả.
- `-u 4`: upscale chiều rộng và chiều cao lên 4 lần.
- `--tile_size 512`: chia ảnh thành tile để giảm VRAM.
- `--vae_tile_size 512`: bật xử lý VAE theo tile.
- `--seed 42`: seed để tái lập kết quả.

## Dùng trong pipeline Qwen Edit

Từ `D:\SDS\gen_img`:

```powershell
python .\qwen_lightning_edit.py `
  --input "D:\path\anh_goc.jpg" `
  --reference "D:\path\anh_tham_chieu.jpg" `
  --vosr-model VOSR2 `
  --upscale 4 `
  --edit-max-side 1024 `
  --steps 8 `
  --seed 42 `
  --prompt "Describe the requested edit here."
```

Pipeline upscale ảnh gốc bằng VOSR trước, sau đó mới chuyển ảnh đã tăng độ phân
giải cùng ảnh tham chiếu sang Qwen Image Edit.

## Kiểm tra checkpoint

Các file tối thiểu cần tồn tại:

```text
VOSR2/args.json
VOSR2/checkpoints/ema_model.safetensors
Qwen-Image-vae-2d/config.json
Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors
```

Nếu thiếu checkpoint, chạy script thiết lập từ workspace:

```powershell
cd D:\SDS\gen_img
python .\setup_vosr.py
```

## Lưu ý

> [!IMPORTANT]
> Checkpoint có dung lượng lớn. Không đưa các file `*.safetensors`, cache hoặc
> model đã tải vào Git. Chỉ theo dõi mã nguồn, cấu hình và tài liệu.

> [!TIP]
> Với GPU 12 GB VRAM, nên giữ `tile_size=512`. Nếu hết VRAM, giảm tile xuống
> `256`; tốc độ sẽ chậm hơn nhưng mức sử dụng bộ nhớ thấp hơn.

## Nguồn

- Paper: [VOSR: Vision-Only Generative Model for Image Super-Resolution](https://arxiv.org/abs/2604.03225)
- Source code: [cswry/VOSR](https://github.com/cswry/VOSR)
- License: [Apache License 2.0](https://github.com/cswry/VOSR/blob/main/LICENSE)
