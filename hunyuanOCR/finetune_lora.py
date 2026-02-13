"""
LoRA fine-tuning script for HunyuanOCR.

Requirements (install on your remote GPU server):
    pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
    pip install peft accelerate bitsandbytes pillow torch

Usage:
    python finetune_lora.py --data_dir ./sample_data

Dataset structure:
    data_dir/
        train.jsonl
        images/
            receipt_001.jpg
            document_001.jpg
            ...

Each line in train.jsonl has 3 fields:
    {
        "image": "images/receipt_001.jpg",   <- relative path to image
        "prompt": "...",                      <- the task instruction (see PROMPT GUIDE below)
        "response": "..."                    <- the expected model output
    }

=== PROMPT GUIDE (from HunyuanOCR official instructions) ===

Task 1: Text Spotting (detect + recognize text with bounding boxes)
    Prompt:   "Detect and recognize text in the image, and output the text coordinates
               in a formatted manner."
    Response: "<ref>COFFEE SHOP</ref><quad>(120,50),(480,120)</quad>\n
               <ref>OPEN 7AM-9PM</ref><quad>(150,140),(430,200)</quad>"
    Note:     Coordinates are normalized to [0, 1000] range.

Task 2: Document Parsing (full page -> markdown)
    Prompt:   "Extract all information from the main body of the document image and
               represent it in markdown format, ignoring headers and footers. Tables
               should be expressed in HTML format, formulas in the document should be
               represented using LaTeX format, and the parsing should be organized
               according to the reading order."
    Response: "## Title\n\nParagraph text...\n\n<table>...</table>\n\n$$formula$$"

Task 3: General Text Extraction
    Prompt:   "Extract the text in the image."
    Response: "All visible text content in reading order..."

Task 4: Formula Recognition
    Prompt:   "Identify the formula in the image and represent it using LaTeX format."
    Response: "$$E = mc^2$$"

Task 5: Table Parsing
    Prompt:   "Parse the table in the image into HTML."
    Response: "<table><tr><th>Name</th><th>Age</th></tr>...</table>"

Task 6: Information Extraction (structured fields -> JSON)
    Prompt:   "Extract the content of the fields: ['name','company','phone','email']
               from the image and return it in JSON format."
    Response: '{"name": "John", "company": "Acme", "phone": "555-0123", "email": "j@a.com"}'

Task 7: Video Subtitle Extraction
    Prompt:   "Extract the subtitles from the image."
    Response: "First line of subtitle\nSecond line of subtitle"

Task 8: Translation
    Prompt:   "First extract the text, then translate the text content into English."
    Response: "[parsing]\nOriginal text...\n\n[translation]\nTranslated text..."
"""

import argparse
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


class OCRDataset(Dataset):
    """
    Loads training data from a JSONL file.

    Each sample becomes a conversation:
        system: ""
        user:   [image] + prompt
        assistant: response    <- this is what the model learns to generate

    Labels are masked so the model only computes loss on the assistant response,
    not on the image tokens or the user prompt.
    """

    def __init__(self, data_dir, processor, max_length=4096):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        jsonl_path = os.path.join(data_dir, "train.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples)} training samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_dir, sample["image"])
        image = Image.open(img_path).convert("RGB")

        # === Build the full conversation (system + user + assistant) ===
        # This is the COMPLETE sequence including the ground-truth response.
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

        # === Build the prompt-only part (system + user, no assistant response) ===
        # Used to determine where the assistant response starts, so we can mask
        # the prompt portion in the labels (set to -100 = ignored by loss).
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

        # === Tokenize the full sequence ===
        inputs = self.processor(
            text=[full_text],
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # === Tokenize prompt-only to measure its length ===
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # === Build labels: -100 for prompt tokens, real ids for response tokens ===
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        # Also mask padding tokens
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels[attention_mask == 0] = -100

        result = {}
        for k, v in inputs.items():
            result[k] = v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 1 else v
        result["labels"] = labels
        return result


def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated


def find_target_modules(model):
    """Find all linear layer names in the LLM (model.layers.*) for LoRA.

    HunyuanOCR structure:
        model.layers.*  -> LLM (Hunyuan-0.5B) — LoRA targets here
        vit.*           -> Vision Transformer — frozen, skip
        lm_head         -> output head — skip
    """
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers."):
            short_name = name.split(".")[-1]
            target_modules.add(short_name)
    print(f"Found LoRA target modules: {target_modules}")
    return list(target_modules)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune HunyuanOCR")
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset dir containing train.jsonl and images/")
    parser.add_argument("--output_dir", type=str, default="./hunyuanocr-lora-output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    print("Loading model...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze the vision encoder (vit.*) — only fine-tune the LLM via LoRA
    for name, param in model.named_parameters():
        if name.startswith("vit."):
            param.requires_grad = False

    # Configure LoRA on all linear layers in the language model
    target_modules = find_target_modules(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = OCRDataset(args.data_dir, processor, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1) * args.gradient_accumulation_steps
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    model.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

        avg = epoch_loss / len(dataloader) * args.gradient_accumulation_steps
        print(f"Epoch {epoch+1} finished. Avg loss: {avg:.4f}")

    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
