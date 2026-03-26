"""
Dual LoRA adapter for HunyuanOCR fine-tuning.

Reference: Xu et al., "Dual LoRA: Towards Effective Parameter-Efficient
Fine-Tuning", arXiv:2512.03402, 2025.

Weight update: ΔW = (α / √(r₁·r₂)) · ReLU(B·A) ⊙ Sign(D·C)
  - Magnitude group: A ∈ R^{r₁×k}, B ∈ R^{d×r₁}  →  ReLU(B·A) ∈ R^{d×k}
  - Direction group:  C ∈ R^{r₂×k}, D ∈ R^{d×r₂}  →  Sign(D·C) ∈ R^{d×k}
  - STE used for Sign backward: gradient clipped to [-1, 1]
  - All four matrices initialized from N(0, σ²); no zero-init (would block ReLU gradients)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SignSTE(torch.autograd.Function):
    """Sign function with Straight-Through Estimator for backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1.0, 1.0)


def _sign_ste(x: torch.Tensor) -> torch.Tensor:
    return _SignSTE.apply(x)


class DualLoraLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with Dual LoRA adapters.

    The base weight is frozen. Only A, B, C, D are trainable.

    Args:
        linear:  The original nn.Linear layer (its weight is frozen in-place).
        r1:      Rank of the magnitude group.
        r2:      Rank of the direction group.
        alpha:   Scaling factor (analogous to LoRA alpha).
        init_std: Std of Gaussian init for all four low-rank matrices.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r1: int,
        r2: int,
        alpha: float,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.scale = alpha / math.sqrt(r1 * r2)

        # Frozen base weight and optional bias
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        device = linear.weight.device

        # Magnitude group: ReLU(B @ A),  A: (r1, k),  B: (d, r1)
        self.A = nn.Parameter(torch.empty(r1, self.in_features, device=device))
        self.B = nn.Parameter(torch.empty(self.out_features, r1, device=device))

        # Direction group: Sign(D @ C),  C: (r2, k),  D: (d, r2)
        self.C = nn.Parameter(torch.empty(r2, self.in_features, device=device))
        self.D = nn.Parameter(torch.empty(self.out_features, r2, device=device))

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        for param in (self.A, self.B, self.C, self.D):
            nn.init.normal_(param, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)

        # Magnitude gate: non-negative, zeros out poorly-adapted directions
        W_m = F.relu(self.B @ self.A)  # (d, k)

        # Direction gate: binarized ±1 via STE
        W_d = _sign_ste(self.D @ self.C)  # (d, k)

        delta_W = self.scale * (W_m * W_d)  # (d, k)
        return base + F.linear(x, delta_W.to(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r1={self.r1}, r2={self.r2}, alpha={self.alpha}"
        )


def apply_dual_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
    init_std: float = 0.01,
) -> nn.Module:
    """
    Replace all nn.Linear layers whose name ends with any entry in
    `target_modules` with DualLoraLinear adapters (r1=r2=rank).

    The base weight of each replaced layer is frozen (requires_grad=False);
    all other model parameters are unaffected — the caller must freeze those
    separately (e.g. ViT encoder weights).

    Args:
        model:           The model to modify (mutated in place).
        target_modules:  List of module-name suffixes to target, e.g.
                         ["q_proj", "v_proj", "fc1", "fc2"].
        rank:            Dual LoRA rank (r1 = r2 = rank).
        alpha:           Scaling factor passed to DualLoraLinear.
        init_std:        Std of Gaussian init for low-rank matrices.

    Returns:
        The mutated model (same object).
    """
    target_set = set(target_modules)
    replacements: list[tuple[nn.Module, str, DualLoraLinear]] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in target_set:
            continue
        dual = DualLoraLinear(module, r1=rank, r2=rank, alpha=alpha, init_std=init_std)
        # Resolve parent and attribute name
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child_name = model, full_name
        replacements.append((parent, child_name, dual))

    for parent, child_name, dual in replacements:
        setattr(parent, child_name, dual)

    return model


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count) for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
