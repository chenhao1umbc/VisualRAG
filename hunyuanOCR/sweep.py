"""
Hyperparameter sweep generator for Dual LoRA fine-tuning.

Sweep grid (from project plan):
  rank         ∈ {8, 16, 32}
  LR ratio     ∈ {0.1, 1.0, 10.0}  (lr_magnitude / lr_direction)
  lr_magnitude = 2e-4 (fixed)
  lr_direction ∈ {2e-3, 2e-4, 2e-5}
  alpha        = rank (by convention)

Total: 9 configurations.

Usage:
    # Print all 9 configs without launching anything
    python hunyuanOCR/sweep.py --dry_run --data_dir ./dataset/train

    # Launch all 9 runs sequentially
    python hunyuanOCR/sweep.py --launch \\
        --data_dir ./dataset/train \\
        --output_dir ./sweep_output \\
        --replay_data_dir ./dataset/replay
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Fixed sweep constants
_LR_MAGNITUDE: float = 2e-4
_RANKS: list[int] = [8, 16, 32]
_LR_RATIOS: list[float] = [0.1, 1.0, 10.0]
_LR_DIRECTIONS: list[float] = [2e-3, 2e-4, 2e-5]  # ratio = lr_magnitude / lr_direction


def _run_name(rank: int, ratio: float) -> str:
    """Return a deterministic run identifier used as subdirectory name."""
    ratio_str = f"{ratio:g}"
    return f"r{rank}_ratio{ratio_str}"


def generate_configs() -> list[dict]:
    """
    Return all 9 sweep configurations as argparse-compatible dicts.

    Keys match finetune_duallora.py CLI argument names (without leading --).
    """
    configs: list[dict] = []
    for rank in _RANKS:
        for ratio, lr_dir in zip(_LR_RATIOS, _LR_DIRECTIONS):
            configs.append({
                "run_name": _run_name(rank, ratio),
                "dual_lora_rank": rank,
                "dual_lora_alpha": float(rank),
                "lr_magnitude": _LR_MAGNITUDE,
                "lr_direction": lr_dir,
                "lr_ratio": ratio,
            })
    return configs


def print_sweep_table(configs: list[dict]) -> None:
    """Print a formatted table of all sweep configurations."""
    header = f"{'#':<4} {'run_name':<18} {'rank':<6} {'alpha':<7} {'lr_mag':<10} {'lr_dir':<10} {'ratio':<6}"
    print(header)
    print("-" * len(header))
    for i, cfg in enumerate(configs):
        print(
            f"{i+1:<4} "
            f"{cfg['run_name']:<18} "
            f"{cfg['dual_lora_rank']:<6} "
            f"{cfg['dual_lora_alpha']:<7g} "
            f"{cfg['lr_magnitude']:<10.2e} "
            f"{cfg['lr_direction']:<10.2e} "
            f"{cfg['lr_ratio']:<6g}"
        )
    print(f"\nTotal: {len(configs)} configurations")


def _build_command(
    cfg: dict,
    data_dir: str,
    output_dir: str,
    extra_args: list[str],
) -> list[str]:
    """Build the subprocess command for one sweep configuration."""
    script = str(Path(__file__).parent / "finetune_duallora.py")
    run_output_dir = os.path.join(output_dir, cfg["run_name"])
    cmd = [
        sys.executable, script,
        "--data_dir", data_dir,
        "--output_dir", run_output_dir,
        "--dual_lora_rank", str(cfg["dual_lora_rank"]),
        "--dual_lora_alpha", str(cfg["dual_lora_alpha"]),
        "--lr_magnitude", str(cfg["lr_magnitude"]),
        "--lr_direction", str(cfg["lr_direction"]),
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual LoRA hyperparameter sweep")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry_run", action="store_true", default=True,
                      help="Print sweep table without launching any runs (default)")
    mode.add_argument("--launch", action="store_true", default=False,
                      help="Launch all configurations sequentially via subprocess")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Main training data directory (passed to finetune_duallora.py)")
    parser.add_argument("--output_dir", type=str, default="./sweep_output",
                        help="Root output directory; each run goes into output_dir/<run_name>")
    parser.add_argument("--replay_data_dir", type=str, default=None,
                        help="Replay buffer data directory (optional, passed through)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for all runs (passed through)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size for all runs (passed through)")

    args = parser.parse_args()

    configs = generate_configs()

    # Build extra passthrough args
    extra_args: list[str] = []
    if args.replay_data_dir is not None:
        extra_args += ["--replay_data_dir", args.replay_data_dir]
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]

    print_sweep_table(configs)

    if not args.launch:
        print("\n[dry_run] No runs launched. Pass --launch to execute.")
        return

    print(f"\n[launch] Starting {len(configs)} runs into {args.output_dir}/\n")
    for i, cfg in enumerate(configs):
        run_label = f"[{i+1}/{len(configs)}] {cfg['run_name']}"
        print(f"\n{'='*60}")
        print(f"{run_label}")
        print(f"{'='*60}")

        cmd = _build_command(cfg, args.data_dir, args.output_dir, extra_args)
        print("Command:", " ".join(cmd))

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"[WARNING] {run_label} exited with code {result.returncode}. Continuing to next run.")

    print(f"\n[launch] All {len(configs)} configurations complete.")


if __name__ == "__main__":
    main()
