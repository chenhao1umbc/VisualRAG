"""
Shared dataset utilities for HunyuanOCR fine-tuning scripts.

Used by both finetune_lora.py and finetune_duallora.py.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    """
    Loads training data from a JSONL file.

    Each sample becomes a conversation:
        system: ""
        user:   [image] + prompt
        assistant: response    <- this is what the model learns to generate

    Labels are masked so the model only computes loss on the assistant response,
    not on the image tokens or the user prompt.

    Expected JSONL schema per line:
        {"image": "dataset/edgar/images/foo.jpg", "prompt": "...", "response": "..."}

    Note: `image` values are project-root-relative paths, not relative to `data_dir`.
    They are used directly as-is (not joined with data_dir).
    """

    def __init__(self, data_dir: str, processor, max_length: int = 4096) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.samples: list[dict] = []
        jsonl_path = os.path.join(data_dir, "train.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} training samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img_path = sample["image"]
        image = Image.open(img_path).convert("RGB")

        # Build full conversation (system + user + assistant response)
        full_messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
            {"role": "assistant", "content": sample["response"]},
        ]
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Build prompt-only part to find where assistant response starts
        prompt_messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[full_text],
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Labels: -100 for prompt tokens (ignored by loss), real ids for response
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels[attention_mask == 0] = -100

        result = {}
        for k, v in inputs.items():
            result[k] = v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v
        result["labels"] = labels
        return result


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    collated: dict = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated
