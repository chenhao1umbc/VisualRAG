"""
Inference script for the LoRA fine-tuned HunyuanOCR model.

Usage:
    python inference_lora.py \
        --lora_path ./hunyuanocr-lora-output/final \
        --image /path/to/image.jpg \
        --prompt "Extract all text from the image."
"""

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from peft import PeftModel


def clean_repeated_substrings(text):
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


def main():
    parser = argparse.ArgumentParser(description="Inference with LoRA fine-tuned HunyuanOCR")
    parser.add_argument("--base_model", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="Extract all text from the image.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=False)

    base_model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.base_model,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=image, return_tensors="pt")

    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(clean_repeated_substrings(output[0]))


if __name__ == "__main__":
    main()
