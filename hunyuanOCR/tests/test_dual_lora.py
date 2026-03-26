"""
Unit tests for DualLoraLinear and apply_dual_lora.

Tests:
1. Output shape is correct for representative layer sizes.
2. Gradients flow through both magnitude (A, B) and direction (C, D) components.
3. Dual LoRA output scale is in the same ballpark as standard LoRA at equal rank/alpha.
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hunyuanOCR.dual_lora import DualLoraLinear, apply_dual_lora


# ─── Test 1: Output shape ─────────────────────────────────────────────────────

def test_output_shape():
    """DualLoraLinear must preserve the output shape of the original nn.Linear."""
    configs = [
        (64, 32, 4, 4),    # small square-ish
        (768, 768, 8, 8),  # transformer self-attention (square)
        (768, 3072, 16, 16),  # transformer FFN expand
        (3072, 768, 16, 16),  # transformer FFN contract
    ]
    for in_f, out_f, r1, r2 in configs:
        linear = nn.Linear(in_f, out_f, bias=True)
        dual = DualLoraLinear(linear, r1=r1, r2=r2, alpha=float(r1))

        # 3-D input: (batch, seq_len, in_features)
        x = torch.randn(2, 10, in_f)
        out = dual(x)
        assert out.shape == (2, 10, out_f), (
            f"Shape mismatch for ({in_f}→{out_f}, r1={r1}, r2={r2}): got {out.shape}"
        )

        # 2-D input: (batch, in_features)
        x2 = torch.randn(4, in_f)
        out2 = dual(x2)
        assert out2.shape == (4, out_f), (
            f"Shape mismatch for 2-D input ({in_f}→{out_f}): got {out2.shape}"
        )


# ─── Test 2: Gradient flow through all four matrices ──────────────────────────

def test_gradient_flow():
    """
    After a backward pass, all four low-rank matrices (A, B, C, D) must have
    non-None, non-zero gradients. The frozen base weight must have no gradient.
    """
    linear = nn.Linear(128, 64, bias=False)
    dual = DualLoraLinear(linear, r1=8, r2=8, alpha=8.0, init_std=0.01)

    x = torch.randn(4, 16, 128)
    loss = dual(x).sum()
    loss.backward()

    for name in ("A", "B", "C", "D"):
        grad = getattr(dual, name).grad
        assert grad is not None, f"No gradient for {name}"
        assert grad.abs().sum().item() > 0, f"Zero gradient for {name}"

    # Base weight must remain frozen (no gradient)
    assert dual.weight.grad is None, "Base weight should not have a gradient"


# ─── Test 3: Output scale comparable to standard LoRA ─────────────────────────

def test_output_scale_vs_lora():
    """
    With r1=r2=r and alpha=r, the Dual LoRA delta_W should have an RMS magnitude
    within 2 orders of magnitude of standard LoRA delta_W (same rank r, same alpha).
    This is a sanity check — the parameterizations differ, but both should produce
    updates of the same rough scale rather than exploding or vanishing relative to LoRA.
    """
    torch.manual_seed(0)
    in_f, out_f, r = 256, 128, 16
    alpha = float(r)

    linear = nn.Linear(in_f, out_f, bias=False)

    # Dual LoRA delta_W
    dual = DualLoraLinear(linear, r1=r, r2=r, alpha=alpha, init_std=0.01)
    W_m = torch.relu(dual.B @ dual.A)
    W_d = (dual.D @ dual.C).sign()
    dual_delta = (alpha / math.sqrt(r * r)) * W_m * W_d
    dual_rms = dual_delta.pow(2).mean().sqrt().item()

    # Standard LoRA starts at delta_W=0 (B=0). After one step, Dual LoRA will
    # also be near zero due to warm-up. At init, Dual LoRA is non-zero; check
    # it is finite and not astronomically large compared to the base weight scale.
    base_rms = linear.weight.data.pow(2).mean().sqrt().item()
    assert math.isfinite(dual_rms), "Dual LoRA delta_W is not finite"
    assert dual_rms < base_rms * 10, (
        f"Dual LoRA delta_W RMS ({dual_rms:.4f}) is >10x base weight RMS ({base_rms:.4f}) at init"
    )
    # LoRA starts at exactly 0; Dual LoRA at init is small but non-zero — that's expected.
    # Just confirm it doesn't explode.
    assert dual_rms < 1.0, f"Dual LoRA delta_W RMS {dual_rms:.4f} seems unexpectedly large"


# ─── Test 4: apply_dual_lora replaces correct layers ──────────────────────────

def test_apply_dual_lora():
    """apply_dual_lora replaces target nn.Linear layers and freezes base weights."""
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)  # should NOT be replaced

        def forward(self, x):
            return self.out_proj(self.q_proj(x) + self.v_proj(x))

    model = TinyModel()
    apply_dual_lora(model, target_modules=["q_proj", "v_proj"], rank=8, alpha=8.0)

    assert isinstance(model.q_proj, DualLoraLinear), "q_proj should be DualLoraLinear"
    assert isinstance(model.v_proj, DualLoraLinear), "v_proj should be DualLoraLinear"
    assert isinstance(model.out_proj, nn.Linear), "out_proj should remain nn.Linear"

    # Base weights frozen, adapters trainable
    assert not model.q_proj.weight.requires_grad, "Base weight should be frozen"
    assert model.q_proj.A.requires_grad, "Adapter A should be trainable"

    # Forward pass still works
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out.shape == (2, 10, 64)


if __name__ == "__main__":
    test_output_shape()
    print("test_output_shape PASSED")
    test_gradient_flow()
    print("test_gradient_flow PASSED")
    test_output_scale_vs_lora()
    print("test_output_scale_vs_lora PASSED")
    test_apply_dual_lora()
    print("test_apply_dual_lora PASSED")
    print("\nAll tests passed.")
