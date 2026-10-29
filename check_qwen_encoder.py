"""Check real Qwen vision/text leaf offload with tiny random weights on CUDA."""


def main():
    import torch
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
    from qwen_config import BASE_DIR

    config = Qwen2_5_VLConfig.from_pretrained(str(BASE_DIR / "text_encoder"), local_files_only=True)
    text = config.text_config
    text.hidden_size, text.intermediate_size = 64, 128
    text.num_attention_heads, text.num_key_value_heads = 4, 2
    text.num_hidden_layers, text.max_window_layers = 2, 2
    text.layer_types = ["full_attention", "full_attention"]
    text.rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [2, 3, 3]}
    text.use_cache = False
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 2, 64, 128
    vision.num_heads, vision.out_hidden_size = 4, 64
    vision.fullatt_block_indexes = [1]
    encoder = Qwen2_5_VLForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    processor = Qwen2VLProcessor.from_pretrained(str(BASE_DIR / "processor"), local_files_only=True)
    pipe = QwenImageEditPlusPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(), vae=None, transformer=None,
        text_encoder=encoder, tokenizer=processor.tokenizer, processor=processor,
    )
    apply_group_offloading(
        encoder, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
        offload_type="leaf_level", use_stream=False,
    )
    with torch.inference_mode():
        embeds, mask = pipe.encode_prompt(prompt="Change the shirt to blue.",
                                          image=[Image.new("RGB", (112, 112), "white")],
                                          device=torch.device("cuda"))
    assert embeds.shape[0] == 1 and embeds.shape[-1] == 64 and torch.isfinite(embeds).all()
    assert all(parameter.device.type == "cpu" for parameter in encoder.parameters())
    print(f"Qwen image-conditioned encoding + language/vision leaf offload OK: {tuple(embeds.shape)}")


if __name__ == "__main__":
    main()
