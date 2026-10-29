"""Small real-CUDA check: PEFT LoRA + GGUF 4-bit + block offload, no model weights."""

from collections import OrderedDict


def main():
    import gguf
    import numpy as np
    import torch
    from diffusers.hooks import apply_group_offloading
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter
    from peft import LoraConfig, inject_adapter_in_model

    source = np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32)
    quant_type = gguf.GGMLQuantizationType.Q4_0
    packed = gguf.quants.quantize(source, quant_type)
    linear = GGUFLinear(32, 32, compute_dtype=torch.bfloat16)
    linear.weight = GGUFParameter(torch.from_numpy(packed), quant_type=quant_type)
    model = torch.nn.Sequential(OrderedDict(proj=linear))
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]), model)
    with torch.no_grad():
        model.proj.lora_A["default"].weight.fill_(0.1)
        model.proj.lora_B["default"].weight.fill_(0.1)
    apply_group_offloading(model, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                           offload_type="block_level", num_blocks_per_group=1, use_stream=False)
    with torch.inference_mode():
        result = model(torch.ones((1, 32), device="cuda", dtype=torch.bfloat16))
    assert result.shape == (1, 32) and torch.isfinite(result).all()
    assert model.proj.base_layer.weight.quant_type == quant_type
    assert model.proj.base_layer.weight.device.type == "cpu"
    print("GGUF 4-bit + PEFT LoRA + CUDA block offload OK")


if __name__ == "__main__":
    main()
