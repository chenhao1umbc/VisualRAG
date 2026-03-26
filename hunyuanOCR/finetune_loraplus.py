"""
LoRA+ fine-tuning script for HunyuanOCR.

LoRA+ (Hayou et al., 2024) improves on standard LoRA by using separate learning
rates for the A and B adapter matrices. A matrices are initialized from a Gaussian
and updated at the base learning rate; B matrices are initialized to zero and
updated at a higher rate (lr * loraplus_lr_ratio, default 16×). This asymmetry
better matches the effective gradient magnitudes and improves convergence.

Usage:
    python hunyuanOCR/finetune_loraplus.py \\
        --data_dir dataset/train_medium \\
        --output_dir loraplus-output \\
        --epochs 3 \\
        --gradient_checkpointing \\
        --max_length 2048 \\
        --max_pixels 1048576 \\
        --loraplus_lr_ratio 16.0

Dataset structure (same as finetune_lora.py):
    data_dir/
        train.jsonl   <- {"image": "...", "prompt": "...", "response": "..."}
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
    """Return shuffled dataset indices for one epoch, skipping the first `skip` items."""
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=g).tolist()[skip:]


def find_target_modules(model: torch.nn.Module) -> list[str]:
    """Find all linear layer names in the LLM (model.layers.*) for LoRA+."""
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.startswith("model.layers."):
            target_modules.add(name.split(".")[-1])
    print(f"Found LoRA+ target modules: {target_modules}")
    return list(target_modules)


def _build_loraplus_optimizer(
    model: torch.nn.Module,
    lr: float,
    lr_ratio: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW with LoRA+ param groups: A matrices at lr, B matrices at lr * lr_ratio."""
    a_params: list[torch.nn.Parameter] = []
    b_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            b_params.append(param)
        else:
            a_params.append(param)
    print(
        f"LoRA+ param groups: {len(a_params)} A-group params (lr={lr:.2e}), "
        f"{len(b_params)} B-group params (lr={lr * lr_ratio:.2e})"
    )
    return torch.optim.AdamW(
        [
            {"params": a_params, "lr": lr},
            {"params": b_params, "lr": lr * lr_ratio},
        ],
        weight_decay=weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA+ fine-tune HunyuanOCR")
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanOCR")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset dir containing train.jsonl and images/",
    )
    parser.add_argument("--output_dir", type=str, default="./loraplus-output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--loraplus_lr_ratio",
        type=float,
        default=16.0,
        help="LR multiplier for LoRA B matrices relative to A matrices (default: 16.0)",
    )
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
        help="Path to checkpoint dir to resume from (e.g. loraplus-output/checkpoint-600)",
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

    for name, param in model.named_parameters():
        if name.startswith("vit."):
            param.requires_grad = False

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

    optimizer = _build_loraplus_optimizer(
        model,
        lr=args.lr,
        lr_ratio=args.loraplus_lr_ratio,
        weight_decay=0.01,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        functools.partial(
            _lr_warmup_decay, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )

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
                    lrs = scheduler.get_last_lr()
                    print(
                        f"Epoch {epoch + 1}/{args.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR_A: {lrs[0]:.2e} | LR_B: {lrs[1]:.2e}"
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
