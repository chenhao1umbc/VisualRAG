# Target Layer Decision for Dual LoRA

**Model**: HunYuanVLForConditionalGeneration (`tencent/HunyuanOCR`)
**Total parameters**: 996.2M (≈ 996M)
**Source**: layer inventory derived from `AutoConfig.from_pretrained` + `torch.device("meta")` instantiation (no weights downloaded).

---

## 1. Layer Inventory

### LLM Decoder — `model.layers.*` (24 layers)

| Leaf name   | in_features | out_features | Layers |
|-------------|-------------|--------------|--------|
| `q_proj`    | 1024        | 2048         | 24     |
| `k_proj`    | 1024        | 1024         | 24     |
| `v_proj`    | 1024        | 1024         | 24     |
| `o_proj`    | 2048        | 1024         | 24     |
| `gate_proj` | 1024        | 3584         | 24     |
| `up_proj`   | 1024        | 3584         | 24     |
| `down_proj` | 3584        | 1024         | 24     |

Total LLM `nn.Linear` modules: **168** (7 per layer × 24 layers).

### ViT Encoder — `vit.layers.*` (27 layers)

| Leaf name        | in_features | out_features | Layers |
|------------------|-------------|--------------|--------|
| `q_proj`         | 1152        | 1152         | 27     |
| `k_proj`         | 1152        | 1152         | 27     |
| `v_proj`         | 1152        | 1152         | 27     |
| `o_proj`         | 1152        | 1152         | 27     |
| `dense_h_to_4h`  | 1152        | 4304         | 27     |
| `dense_4h_to_h`  | 4304        | 1152         | 27     |

Total ViT `nn.Linear` modules: **163** (≈ 6 per layer × 27 layers, plus head layers).

### Connector / Other

| Name      | in_features | out_features | Notes                      |
|-----------|-------------|--------------|----------------------------|
| `lm_head` | 1024        | 120818       | Output projection; not targeted |

---

## 2. Parameter Count Table — LLM-Only, Dual LoRA r = 16

For a `DualLoraLinear` wrapping a linear of shape `(in, out)` with `r1 = r2 = r`:
- Magnitude group: A `(r, in)` + B `(out, r)` → `r·in + out·r` params
- Direction group: C `(r, in)` + D `(out, r)` → `r·in + out·r` params
- **Total per layer**: `2 · (r·in + out·r)`

| Layer       | Count | in   | out  | Params/layer | Total params  |
|-------------|-------|------|------|--------------|---------------|
| `q_proj`    | 24    | 1024 | 2048 | 98,304       | 2,359,296     |
| `k_proj`    | 24    | 1024 | 1024 | 65,536       | 1,572,864     |
| `v_proj`    | 24    | 1024 | 1024 | 65,536       | 1,572,864     |
| `o_proj`    | 24    | 2048 | 1024 | 98,304       | 2,359,296     |
| `gate_proj` | 24    | 1024 | 3584 | 147,456      | 3,538,944     |
| `up_proj`   | 24    | 1024 | 3584 | 147,456      | 3,538,944     |
| `down_proj` | 24    | 3584 | 1024 | 147,456      | 3,538,944     |
| **TOTAL**   |       |      |      |              | **18,481,152** |

**Trainable fraction**: 18.48M / 996.2M = **1.86%** — within the 1–5% target range.

For reference, attention-only targeting (q, k, v, o) would yield 7.86M (0.79%) — too low to capture MLP-dependent formatting patterns (color hue, table structure).

---

## 3. Decision: `target_modules` for Primary Config

```python
TARGET_MODULES_LLM = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

**Rationale**:
- All 7 LLM layer types are targeted to give the model full capacity to adapt both attention (formatting span detection) and MLP (color/structure classification) pathways.
- Targeting attention-only (q, k, v, o) gives 0.79% — insufficient for the MLP-heavy color and table features.
- Targeting all 7 gives 1.86% — comfortably within the 1–5% budget without overfitting risk.
- ViT is frozen in the primary config; the ViT encoder already produces adequate visual features for the LLM to detect strikethrough geometry, underlines, and color regions.

---

## 4. ViT Target Layers for Ablation Config (Sub-task 2.4)

In the LLM+ViT ablation, Dual LoRA adapters are additionally applied to ViT attention and MLP layers:

```python
TARGET_MODULES_VIT = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "dense_h_to_4h",
    "dense_4h_to_h",
]
```

Note: `q_proj` / `k_proj` / `v_proj` / `o_proj` exist in both LLM and ViT namespaces. `apply_dual_lora` matches on the leaf module name, so a single call with the combined unique set `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "dense_h_to_4h", "dense_4h_to_h"]` will correctly replace both LLM and ViT layers. Alternatively, two separate `apply_dual_lora` calls (one per sub-model) can be used with their respective lists for clarity.

ViT additional trainable params at r=16 (all 6 types × 27 layers):
- Attention (q,k,v,o): 4 types × 2·(16×1152 + 1152×16) × 27 = ~7.96M
- MLP (dense_h_to_4h, dense_4h_to_h): ~9.43M
- **ViT total ≈ 17.39M** additional trainable params (combined LLM+ViT ≈ 35.87M, 3.60%)
