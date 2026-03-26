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
import functools
import itertools
import os

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from hunyuanOCR.dataset import OCRDataset, collate_fn


def _lr_warmup_decay(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """LambdaLR multiplier: linear warmup then linear decay to 0."""
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    remaining = total_steps - current_step
    decay_steps = total_steps - warmup_steps
    return max(0.0, remaining / max(1, decay_steps))


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_epoch_indices(
    dataset_size: int, epoch: int, skip: int = 0, seed: int = 42
) -> list[int]:
    """Return shuffled dataset indices for one epoch, skipping the first `skip` items.

    Uses a fixed seed per epoch so the shuffle is reproducible across restarts.
    """
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=g).tolist()[skip:]


def find_target_modules(model: torch.nn.Module) -> list[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune HunyuanOCR")
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset dir containing train.jsonl and images/",
    )
    parser.add_argument("--output_dir", type=str, default="./hunyuanocr-lora-output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from (e.g. lora-output-v2/checkpoint-600)",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1048576,
        help="Max image pixels fed to the vision encoder (default 1M = 1024 image tokens). "
        "Reduce to prevent sequence-length OOM on MPS.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear LR warmup (default: 0.05)",
    )
    parser.add_argument(
        "--replay_data_dir",
        type=str,
        default=None,
        help="Path to replay dataset dir (train.jsonl + images/). "
        "Every --replay_every optimizer steps, one batch is substituted from this set.",
    )
    parser.add_argument(
        "--replay_every",
        type=int,
        default=4,
        help="Substitute a replay batch every N optimizer steps (default: 4)",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"Device: {device}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    processor.image_processor.max_pixels = args.max_pixels
    print(
        f"Image processor max_pixels set to {args.max_pixels} (~{args.max_pixels // 1024} image tokens max)"
    )

    print("Loading model...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(device)

    # Freeze the vision encoder (vit.*) — only fine-tune the LLM via LoRA
    for name, param in model.named_parameters():
        if name.startswith("vit."):
            param.requires_grad = False

    # Configure LoRA — either resume from checkpoint or create fresh adapters
    resume_step = 0
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
        ckpt_name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        resume_step = int(ckpt_name.split("-")[-1])
        print(
            f"Resuming from checkpoint: {args.resume_from_checkpoint} (step {resume_step})"
        )
    else:
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

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    print("Loading dataset...")
    dataset = OCRDataset(args.data_dir, processor, max_length=args.max_length)

    replay_iter = None
    if args.replay_data_dir is not None:
        print(f"Loading replay dataset from {args.replay_data_dir}...")
        replay_dataset = OCRDataset(
            args.replay_data_dir, processor, max_length=args.max_length
        )
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        replay_iter = itertools.cycle(replay_loader)
        print(
            f"Replay buffer: {len(replay_dataset)} samples, injecting every {args.replay_every} optimizer steps"
        )

    steps_per_epoch = len(dataset) // (
        args.batch_size * args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    start_epoch = resume_step // steps_per_epoch
    skip_in_epoch = resume_step % steps_per_epoch

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        functools.partial(
            _lr_warmup_decay, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )

    # Advance scheduler to match resume point (cheap — no gradients)
    for _ in range(resume_step):
        scheduler.step()

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step = resume_step

    for epoch in range(start_epoch, args.epochs):
        skip = skip_in_epoch if epoch == start_epoch else 0
        indices = _make_epoch_indices(len(dataset), epoch, skip=skip)
        dataloader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        epoch_loss = 0.0
        epoch_accum_steps = 0

        for step, batch in enumerate(dataloader):
            if (
                replay_iter is not None
                and global_step > 0
                and global_step % args.replay_every == 0
            ):
                batch = next(replay_iter)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += outputs.loss.detach().item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_accum_steps += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch + 1}/{args.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    model.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

        avg = epoch_loss / max(epoch_accum_steps * args.gradient_accumulation_steps, 1)
        print(f"Epoch {epoch + 1} finished. Avg loss: {avg:.4f}")

    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
