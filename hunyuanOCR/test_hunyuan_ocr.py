#%%
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch
import os
import json
from pathlib import Path

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
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

def process_image(img_path, processor, model):
    """Process a single image with HunyuanOCR model"""
    try:
        image = Image.open(img_path)

        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": (
                        "检测并识别图片中的文字，将文本坐标格式化输出。"
                    )},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)

        if "input_ids" in inputs:
            input_ids = inputs.input_ids
        else:
            input_ids = inputs.inputs

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        output_text = clean_repeated_substrings(output_text)
        return output_text
    except Exception as e:
        return f"Error processing image: {str(e)}"

def main():
    model_name_or_path = "tencent/HunyuanOCR"
    test_images_dir = "/Users/hc/Downloads/CA_Samples/jpg_output"
    output_dir = "ocr_results"

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model loaded successfully!")

    # Process all images
    results = {}
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.jpg')])

    print(f"\nFound {len(image_files)} images to process")

    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_images_dir, img_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")

        ocr_text = process_image(img_path, processor, model)
        results[img_file] = ocr_text

        # Save individual result
        output_file = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        print(f"Saved result to: {output_file}")

    # Save all results as JSON
    json_output_file = os.path.join(output_dir, "ocr_results.json")
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nAll results saved to: {json_output_file}")

if __name__ == "__main__":
    main()

# %%
