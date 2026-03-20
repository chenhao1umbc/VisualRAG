"""
Dual LoRA fine-tuning script for HunyuanOCR (FinDocOCR project).

Trains HunyuanOCR with Dual LoRA adapters on all four target features:
strikethrough, underline, colored text, and complex financial tables.
Includes replay buffer mixing to prevent catastrophic forgetting.

Requirements:
    pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
    pip install accelerate pillow torch

Usage:
    python finetune_duallora.py \\
        --data_dir dataset/train \\
        --replay_data_dir dataset/synthetic/replay \\
        --output_dir ./duallora-output \\
        --dual_lora_rank 16 \\
        --dual_lora_alpha 16.0 \\
        --lr_magnitude 2e-4 \\
        --lr_direction 2e-5

Dataset structure (both --data_dir and --replay_data_dir):
    dir/
        train.jsonl    <- {"image": "images/foo.jpg", "prompt": "...", "response": "..."}
        images/
"""

import argparse
import functools
import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

from hunyuanOCR.dataset import OCRDataset, collate_fn
from hunyuanOCR.dual_lora import DualLoraLinear, apply_dual_lora


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


# LLM target modules (from notes/target_layers.md — all 7 types, 1.86% at r=16)
TARGET_MODULES_LLM: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Combined LLM+ViT target modules (adds dense_h_to_4h / dense_4h_to_h; 3.60% at r=16)
TARGET_MODULES_LLM_VIT: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "dense_h_to_4h",
    "dense_4h_to_h",
]


def _collect_dual_lora_modules(model: nn.Module) -> list[DualLoraLinear]:
    return [m for _, m in model.named_modules() if isinstance(m, DualLoraLinear)]


def _set_warmup_scale(model: nn.Module, step: int, warmup_steps: int, final_scales: dict[int, float]) -> None:
    """Linearly scale DualLoraLinear.scale from 0 → final over warmup_steps."""
    if warmup_steps <= 0:
        return
    factor = min(1.0, step / warmup_steps)
    for mod_id, mod in ((id(m), m) for m in _collect_dual_lora_modules(model)):
        mod.scale = final_scales[mod_id] * factor


def _build_optimizer(model: nn.Module, lr_magnitude: float, lr_direction: float, weight_decay: float) -> torch.optim.AdamW:
    """Build AdamW with separate param groups for magnitude (A,B) and direction (C,D)."""
    magnitude_params: list[nn.Parameter] = []
    direction_params: list[nn.Parameter] = []
    for mod in _collect_dual_lora_modules(model):
        magnitude_params.extend([mod.A, mod.B])
        direction_params.extend([mod.C, mod.D])
    return torch.optim.AdamW(
        [
            {"params": magnitude_params, "lr": lr_magnitude},
            {"params": direction_params, "lr": lr_direction},
        ],
        weight_decay=weight_decay,
    )


def _save_dual_lora_adapters(model: nn.Module, path: str) -> None:
    """Save only the Dual LoRA adapter weights (A, B, C, D) to a compact checkpoint."""
    adapters: dict[str, dict] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, DualLoraLinear):
            adapters[name] = {
                "A": mod.A.data.cpu(),
                "B": mod.B.data.cpu(),
                "C": mod.C.data.cpu(),
                "D": mod.D.data.cpu(),
                "r1": mod.r1,
                "r2": mod.r2,
                "alpha": mod.alpha,
            }
    torch.save(adapters, path)
    print(f"Saved {len(adapters)} Dual LoRA adapters to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual LoRA fine-tune HunyuanOCR")
    # Inherited from finetune_lora.py
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to main FinDocOCR dataset dir (train.jsonl + images/)")
    parser.add_argument("--output_dir", type=str, default="./duallora-output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_pixels", type=int, default=1048576,
                        help="Max image pixels fed to the vision encoder (default 1M = 1024 image tokens). "
                             "Reduce to prevent sequence-length OOM on MPS.")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    # Dual LoRA specific
    parser.add_argument("--dual_lora_rank", type=int, default=16,
                        help="Rank r for both magnitude and direction groups (r1=r2=r)")
    parser.add_argument("--dual_lora_alpha", type=float, default=16.0,
                        help="Scaling factor alpha for Dual LoRA")
    parser.add_argument("--lr_magnitude", type=float, default=2e-4,
                        help="Learning rate for magnitude group (A, B matrices)")
    parser.add_argument("--lr_direction", type=float, default=2e-5,
                        help="Learning rate for direction group (C, D matrices)")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Linear warm-up steps scaling adapter contribution 0→1")
    parser.add_argument("--include_vit", action="store_true",
                        help="Also apply Dual LoRA to ViT encoder layers (LLM+ViT ablation, ~3.60%% params at r=16)")
    # Replay buffer
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total steps used for linear LR warmup (default: 0.05)")
    parser.add_argument("--replay_data_dir", type=str, default=None,
                        help="Path to SynFinTabs replay dataset dir (optional)")
    parser.add_argument("--replay_every", type=int, default=8,
                        help="Substitute a replay batch every N optimizer steps")
    args = parser.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    processor.image_processor.max_pixels = args.max_pixels
    print(f"Image processor max_pixels set to {args.max_pixels} (~{args.max_pixels // 1024} image tokens max)")

    device = _get_device()
    print(f"Device: {device}")
    print("Loading model...")
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(device)

    # Freeze all parameters; apply_dual_lora will unfreeze adapters per replaced layer
    for param in model.parameters():
        param.requires_grad = False

    target_modules = TARGET_MODULES_LLM_VIT if args.include_vit else TARGET_MODULES_LLM
    print(f"Applying Dual LoRA (rank={args.dual_lora_rank}, alpha={args.dual_lora_alpha}, targets={'LLM+ViT' if args.include_vit else 'LLM-only'})...")
    apply_dual_lora(
        model,
        target_modules=target_modules,
        rank=args.dual_lora_rank,
        alpha=args.dual_lora_alpha,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Store final scales for warm-up restoration
    dual_modules = _collect_dual_lora_modules(model)
    final_scales: dict[int, float] = {id(m): m.scale for m in dual_modules}
    if args.warmup_steps > 0:
        for m in dual_modules:
            m.scale = 0.0

    print("Building optimizer (two param groups: magnitude + direction)...")
    optimizer = _build_optimizer(
        model,
        lr_magnitude=args.lr_magnitude,
        lr_direction=args.lr_direction,
        weight_decay=0.01,
    )

    print(f"Loading main dataset from {args.data_dir}...")
    main_dataset = OCRDataset(args.data_dir, processor, max_length=args.max_length)
    main_loader = DataLoader(
        main_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    replay_iter = None
    if args.replay_data_dir is not None:
        print(f"Loading replay dataset from {args.replay_data_dir}...")
        replay_dataset = OCRDataset(args.replay_data_dir, processor, max_length=args.max_length)
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        replay_iter = itertools.cycle(replay_loader)
        print(f"Replay buffer: {len(replay_dataset)} samples, injecting every {args.replay_every} optimizer steps")

    total_steps = len(main_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps_lr = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(_lr_warmup_decay, warmup_steps=warmup_steps_lr, total_steps=total_steps)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(main_loader):
            # Inject replay batch instead of main batch every replay_every optimizer steps
            if replay_iter is not None and global_step > 0 and global_step % args.replay_every == 0:
                batch = next(replay_iter)

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
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

                # Linear warm-up: scale adapter contribution from 0 → final
                _set_warmup_scale(model, global_step, args.warmup_steps, final_scales)

                if global_step % args.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1) * args.gradient_accumulation_steps
                    lr_mag = optimizer.param_groups[0]["lr"]
                    lr_dir = optimizer.param_groups[1]["lr"]
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR mag: {lr_mag:.2e} | LR dir: {lr_dir:.2e}"
                    )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    _save_dual_lora_adapters(
                        model, os.path.join(ckpt_dir, "dual_lora_adapters_only.pt")
                    )
                    print(f"Saved checkpoint to {ckpt_dir}")

        avg = epoch_loss / len(main_loader) * args.gradient_accumulation_steps
        print(f"Epoch {epoch+1} finished. Avg loss: {avg:.4f}")

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    _save_dual_lora_adapters(model, os.path.join(final_dir, "dual_lora_adapters_only.pt"))
    print(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
