"""
Evaluation script for fine-tuned HunyuanOCR checkpoints.

Loads a model checkpoint, runs inference on a held-out eval set, and computes
per-feature metrics (Strikethrough F1, Underline F1, Color Accuracy, Table TEDS)
using hunyuanOCR.eval.metrics.evaluate_batch.

If the checkpoint directory contains `dual_lora_adapters_only.pt`, Dual LoRA
adapter weights are applied on top of the base model; otherwise the checkpoint
is loaded as a full model.

Usage:
    python hunyuanOCR/eval/run_eval.py \\
        --checkpoint_dir ./duallora-output/final \\
        --eval_dir ./dataset/eval \\
        --output_file eval_results.json
"""

import argparse
import json
import os
import signal
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

from hunyuanOCR.dual_lora import DualLoraLinear, apply_dual_lora
from hunyuanOCR.eval.metrics import evaluate_batch


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_ADAPTER_FILENAME = "dual_lora_adapters_only.pt"
_SAMPLE_TIMEOUT_SECS = 180


class _SampleTimeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise _SampleTimeout()


_FEATURE_PROMPTS: dict[str, str] = {
    "tables":        "Convert this table image to HTML with colspan and rowspan attributes.",
    "strikethrough": "Transcribe this text. Use ~~text~~ for strikethrough text.",
    "underline":     "Transcribe this text. Use <u>text</u> for underlined text.",
    "color":         'Transcribe this text. Use <span style="background-color:NAME;">text</span> for highlighted text.',
    "general":       "Transcribe this image to text.",
}


def _find_eval_jsonl(eval_dir: str) -> str:
    """Return path to eval JSONL, checking common names in priority order."""
    for name in ("eval_bootstrap.jsonl", "eval.jsonl", "train.jsonl"):
        path = os.path.join(eval_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No eval_bootstrap.jsonl, eval.jsonl, or train.jsonl found in {eval_dir}"
    )


def _load_eval_records(eval_dir: str, eval_jsonl: str | None = None) -> list[dict]:
    """Load eval records and normalise field names from eval_bootstrap schema.

    If eval_jsonl is given it is used directly; otherwise the JSONL is located
    by scanning eval_dir for well-known filenames.
    """
    jsonl_path = eval_jsonl if eval_jsonl is not None else _find_eval_jsonl(eval_dir)
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    for rec in records:
        if "image" not in rec and "image_path" in rec:
            rec["image"] = rec.pop("image_path")
        if "response" not in rec and "ground_truth" in rec:
            rec["response"] = rec["ground_truth"]
        if "prompt" not in rec:
            feature = rec.get("feature", "general")
            rec["prompt"] = _FEATURE_PROMPTS.get(feature, _FEATURE_PROMPTS["general"])

    print(f"Loaded {len(records)} eval samples from {jsonl_path}")
    return records


def _inject_dual_lora_adapters(model: torch.nn.Module, adapters_path: str) -> None:
    """
    Load Dual LoRA adapter weights from a .pt checkpoint and apply them to the model.

    The checkpoint contains {layer_full_name: {A, B, C, D, r1, r2, alpha}}.
    This function installs DualLoraLinear wrappers via apply_dual_lora (using
    the rank/alpha from the checkpoint) and injects the saved tensors.
    """
    adapters: dict[str, dict] = torch.load(adapters_path, map_location="cpu")
    if not adapters:
        print(f"[warn] Adapter file {adapters_path} is empty; skipping.")
        return

    # Infer target_modules, rank, alpha from the first entry
    first = next(iter(adapters.values()))
    rank: int = first["r1"]
    alpha: float = first["alpha"]
    target_modules: list[str] = list({name.split(".")[-1] for name in adapters})

    print(f"Applying Dual LoRA: rank={rank}, alpha={alpha}, targets={target_modules}")

    # Freeze all params then install DualLoraLinear wrappers
    for param in model.parameters():
        param.requires_grad = False
    apply_dual_lora(model, target_modules=target_modules, rank=rank, alpha=alpha)

    # Inject saved A, B, C, D tensors
    named_modules = dict(model.named_modules())
    loaded = 0
    for layer_name, weights in adapters.items():
        mod = named_modules.get(layer_name)
        if mod is None or not isinstance(mod, DualLoraLinear):
            print(f"[warn] Could not find DualLoraLinear at '{layer_name}'; skipping.")
            continue
        mod.A.data.copy_(weights["A"])
        mod.B.data.copy_(weights["B"])
        mod.C.data.copy_(weights["C"])
        mod.D.data.copy_(weights["D"])
        loaded += 1

    print(f"Injected weights into {loaded}/{len(adapters)} Dual LoRA layers.")


def _load_model(checkpoint_dir: str, base_model_path: str = "tencent/HunyuanOCR") -> tuple:
    """
    Load processor and model from a checkpoint directory.

    Returns (processor, model). Three checkpoint types are handled:
    - PEFT LoRA (adapter_config.json present): loads clean base model from base_model_path,
      applies adapter via PeftModel.from_pretrained.
    - Dual LoRA (dual_lora_adapters_only.pt present): loads clean base model from base_model_path,
      injects Dual LoRA adapter weights.
    - Full model: loads directly from checkpoint_dir.
    """
    device = _get_device()
    print(f"Device: {device}")

    peft_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    dual_lora_path = os.path.join(checkpoint_dir, _ADAPTER_FILENAME)

    if os.path.exists(peft_config_path):
        # PEFT LoRA adapter — load clean base model then apply adapter
        print(f"Found adapter_config.json — PEFT LoRA. Base model: {base_model_path}")
        processor = AutoProcessor.from_pretrained(base_model_path, use_fast=False)
        base_model = HunYuanVLForConditionalGeneration.from_pretrained(
            base_model_path,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        ).to(device)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model.eval()
    elif os.path.exists(dual_lora_path):
        # Dual LoRA adapter — load clean base model then inject custom adapters
        print(f"Found {_ADAPTER_FILENAME} — Dual LoRA. Base model: {base_model_path}")
        processor = AutoProcessor.from_pretrained(base_model_path, use_fast=False)
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            base_model_path,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        _inject_dual_lora_adapters(model, dual_lora_path)
    else:
        # Full model checkpoint
        print(f"Loading processor from {checkpoint_dir}...")
        processor = AutoProcessor.from_pretrained(checkpoint_dir, use_fast=False)
        print(f"Loading model from {checkpoint_dir}...")
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            checkpoint_dir,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        print(f"No adapters found — using full checkpoint as-is.")

    return processor, model


def _run_inference_single(
    sample: dict,
    processor,
    model: torch.nn.Module,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Run inference on a single eval sample. Returns the decoded prediction string."""
    img_path = sample["image"]  # project-root-relative; do not join with eval_dir
    image = Image.open(img_path).convert("RGB")

    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text],
        images=image,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(_SAMPLE_TIMEOUT_SECS)
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1
            )
        signal.alarm(0)
    except _SampleTimeout:
        print(f"[warn] Sample timed out after {_SAMPLE_TIMEOUT_SECS}s; returning empty prediction.")
        signal.signal(signal.SIGALRM, old_handler)
        return ""
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)

    input_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[0][input_len:]
    return processor.decode(trimmed, skip_special_tokens=True)


def _print_summary(results: dict) -> None:
    """Print a human-readable summary table of evaluation results."""
    overall = results["overall"]
    by_feature = results["by_feature"]

    print("\n" + "=" * 60)
    print(f"Checkpoint : {results['checkpoint']}")
    print(f"Samples    : {results['n_samples']}")
    print("=" * 60)
    print(f"\n{'Feature':<18} {'Metric':<22} {'Value':>8}")
    print("-" * 50)

    feature_metric_map = {
        "strikethrough": [("strike_f1", "Strikethrough F1")],
        "underline":     [("underline_f1", "Underline F1")],
        "color":         [("color_acc", "Color Accuracy"), ("color_boundary_f1", "Color Boundary F1")],
        "tables":        [("teds", "Table TEDS")],
    }
    for feature, metrics in feature_metric_map.items():
        feat_results = by_feature.get(feature, {})
        for key, label in metrics:
            val = feat_results.get(key)
            val_str = f"{val:.4f}" if val is not None else "  N/A"
            print(f"  {feature:<16} {label:<22} {val_str:>8}")

    print("-" * 50)
    for key in ("cer", "wer"):
        val = overall.get(key)
        val_str = f"{val:.4f}" if val is not None else "  N/A"
        print(f"  {'overall':<16} {key.upper():<22} {val_str:>8}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HunyuanOCR checkpoint on the FinDocOCR eval set")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Path to eval data directory containing eval.jsonl (or train.jsonl) and images/")
    parser.add_argument("--output_file", type=str, default="eval_results.json",
                        help="Path to write JSON results (default: eval_results.json)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens to generate per sample (default: 512)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1; >1 not yet optimised for variable image sizes)")
    parser.add_argument("--base_model_path", type=str, default="tencent/HunyuanOCR",
                        help="Base model path/ID used when checkpoint_dir is a PEFT adapter (default: tencent/HunyuanOCR)")
    parser.add_argument("--eval_jsonl", type=str, default=None,
                        help="Direct path to eval JSONL file (overrides --eval_dir auto-discovery)")
    parser.add_argument("--partial_file", type=str, default=None,
                        help="Path to incremental JSONL for crash-resume (appended per sample; resumable on restart)")
    parser.add_argument("--max_pixels", type=int, default=1048576,
                        help="Max image pixels fed to the vision encoder (default 1M). Reduce to prevent OOM.")
    args = parser.parse_args()

    processor, model = _load_model(args.checkpoint_dir, base_model_path=args.base_model_path)
    processor.image_processor.max_pixels = args.max_pixels
    print(f"Image processor max_pixels set to {args.max_pixels} (~{args.max_pixels // 1024} image tokens max)")
    device = _get_device()
    eval_records = _load_eval_records(args.eval_dir, eval_jsonl=args.eval_jsonl)

    # Load partial results if resuming
    result_records: list[dict] = []
    skip_count = 0
    if args.partial_file and os.path.exists(args.partial_file):
        with open(args.partial_file, encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if line:
                    result_records.append(json.loads(line))
        skip_count = len(result_records)
        print(f"Resuming from partial file: {skip_count} samples already done, skipping.")

    # Track per-feature running counts for progress logging
    feature_counts: dict[str, int] = defaultdict(int)
    for rec in result_records:
        feature_counts[rec["feature"]] += 1

    partial_fh = open(args.partial_file, "a", encoding="utf-8") if args.partial_file else None

    print(f"\nRunning inference on {len(eval_records)} samples...")
    for i, sample in enumerate(eval_records):
        if i < skip_count:
            continue

        prediction = _run_inference_single(
            sample, processor, model, device, args.max_new_tokens
        )
        feature = sample.get("feature", "general")
        feature_counts[feature] += 1
        rec = {
            "prediction": prediction,
            "ground_truth": sample.get("response", ""),
            "feature": feature,
        }
        result_records.append(rec)
        if partial_fh is not None:
            partial_fh.write(json.dumps(rec) + "\n")
            partial_fh.flush()

        if (i + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or (i + 1) == len(eval_records):
            counts_str = "  ".join(f"{f}={n}" for f, n in sorted(feature_counts.items()))
            print(f"  [{i+1}/{len(eval_records)}]  {counts_str}")

    if partial_fh is not None:
        partial_fh.close()

    # Overall metrics
    overall_metrics = evaluate_batch(result_records)

    # Per-feature metrics
    by_feature: dict[str, dict] = {}
    records_by_feature: dict[str, list[dict]] = defaultdict(list)
    for rec in result_records:
        records_by_feature[rec["feature"]].append(rec)
    for feature, recs in records_by_feature.items():
        by_feature[feature] = evaluate_batch(recs)

    results = {
        "checkpoint": str(Path(args.checkpoint_dir).resolve()),
        "n_samples": len(result_records),
        "overall": overall_metrics,
        "by_feature": by_feature,
    }

    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults written to {output_path}")

    _print_summary(results)


if __name__ == "__main__":
    main()
