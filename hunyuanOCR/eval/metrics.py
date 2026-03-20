"""
Evaluation metrics for Phase 3 model evaluation.

Metrics:
- TEDS: table structure score (see teds.py)
- Span F1: for strikethrough / underline text spans
- Color accuracy: color classification + span boundary F1
- CER / WER: general OCR quality
"""

import re
from collections import defaultdict

from hunyuanOCR.eval.teds import teds


# ─── Shared helper ────────────────────────────────────────────────────────────

def _token_set(s: str) -> set[str]:
    return set(s.lower().split())


# ─── Span-level F1 (strikethrough / underline) ────────────────────────────────

def extract_markdown_spans(text: str, marker: str) -> list[str]:
    """Extract spans marked with the given markdown marker, e.g. ~~text~~."""
    pattern = re.escape(marker) + r"(.+?)" + re.escape(marker)
    return re.findall(pattern, text, re.DOTALL)


def extract_html_spans(text: str, tag: str) -> list[str]:
    """Extract text inside <tag>...</tag>."""
    pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
    return re.findall(pattern, text, re.DOTALL | re.IGNORECASE)


def extract_color_spans(text: str) -> list[tuple[str, str]]:
    """Extract (color_name, text) tuples from <span style='background-color:NAME;'>text</span>."""
    pattern = r'<span[^>]+background-color\s*:\s*([a-zA-Z]+)[^>]*>(.*?)</span>'
    return [(m[0].lower(), m[1]) for m in re.findall(pattern, text, re.DOTALL | re.IGNORECASE)]


def token_set_f1(pred_spans: list[str], gold_spans: list[str]) -> float:
    """
    Compute span-level F1 using token sets.
    A predicted span matches if its token set overlaps >= 50% with any gold span.
    """
    if not gold_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    tp = 0
    matched_gold = set()
    for pred in pred_spans:
        pred_toks = _token_set(pred)
        best_overlap = 0.0
        best_j = -1
        for j, gold in enumerate(gold_spans):
            if j in matched_gold:
                continue
            gold_toks = _token_set(gold)
            if not gold_toks:
                continue
            overlap = len(pred_toks & gold_toks) / len(pred_toks | gold_toks)
            if overlap > best_overlap:
                best_overlap = overlap
                best_j = j
        if best_overlap >= 0.5:
            tp += 1
            matched_gold.add(best_j)

    precision = tp / len(pred_spans)
    recall = tp / len(gold_spans)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def span_f1_strikethrough(pred: str, gold: str) -> float:
    return token_set_f1(extract_markdown_spans(pred, "~~"), extract_markdown_spans(gold, "~~"))


def span_f1_underline(pred: str, gold: str) -> float:
    return token_set_f1(extract_html_spans(pred, "u"), extract_html_spans(gold, "u"))



# ─── Color accuracy ────────────────────────────────────────────────────────────

def color_classification_accuracy(pred: str, gold: str) -> float:
    """
    Accuracy of color label assignment for color spans.
    Matches spans by text content (token overlap), compares color names.
    """
    pred_spans = extract_color_spans(pred)
    gold_spans = extract_color_spans(gold)
    if not gold_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    correct = 0
    for gold_color, gold_text in gold_spans:
        gold_toks = _token_set(gold_text)
        for pred_color, pred_text in pred_spans:
            overlap = len(_token_set(pred_text) & gold_toks) / max(len(_token_set(pred_text) | gold_toks), 1)
            if overlap >= 0.5 and pred_color == gold_color:
                correct += 1
                break

    return correct / len(gold_spans)


def color_span_boundary_f1(pred: str, gold: str) -> float:
    """F1 on color span texts (ignoring color label)."""
    pred_texts = [t for _, t in extract_color_spans(pred)]
    gold_texts = [t for _, t in extract_color_spans(gold)]
    return token_set_f1(pred_texts, gold_texts)


# ─── CER / WER ────────────────────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1))
    return dp[n]


def cer(pred: str, gold: str) -> float:
    """Character Error Rate."""
    if not gold:
        return 0.0 if not pred else 1.0
    return _edit_distance(list(pred), list(gold)) / len(gold)


def wer(pred: str, gold: str) -> float:
    """Word Error Rate."""
    gold_words = gold.split()
    if not gold_words:
        return 0.0 if not pred.split() else 1.0
    return _edit_distance(pred.split(), gold_words) / len(gold_words)


# ─── Batch evaluation ─────────────────────────────────────────────────────────

def evaluate_batch(records: list[dict]) -> dict:
    """
    Evaluate a batch of {prediction, ground_truth, feature} records.

    Returns dict with per-feature and overall metrics.
    """
    results: dict[str, list] = defaultdict(list)

    for rec in records:
        pred = rec.get("prediction", "")
        gold = rec.get("ground_truth", "")
        feature = rec.get("feature", "general")

        if feature == "tables":
            results["teds"].append(teds(pred, gold))
        elif feature == "strikethrough":
            results["strike_f1"].append(span_f1_strikethrough(pred, gold))
            results["strike_exact"].append(float(pred.strip() == gold.strip()))
        elif feature == "underline":
            results["underline_f1"].append(span_f1_underline(pred, gold))
            results["underline_exact"].append(float(pred.strip() == gold.strip()))
        elif feature == "color":
            results["color_acc"].append(color_classification_accuracy(pred, gold))
            results["color_boundary_f1"].append(color_span_boundary_f1(pred, gold))

        results["cer"].append(cer(pred, gold))
        results["wer"].append(wer(pred, gold))

    summary = {}
    for k, v in results.items():
        summary[k] = sum(v) / len(v) if v else None
    summary["n"] = len(records)
    return summary
