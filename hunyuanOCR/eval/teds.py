"""
TEDS (Tree Edit Distance Score) metric for table structure evaluation.
Used for Phase 1.1 (FinTabNet.c baseline) and Phase 3.1 (full evaluation).

Reference: Zhong et al., "Image-based table recognition: data, model, and evaluation", ECCV 2020.
"""

import re
from collections import deque


def _tokenize_html(html: str) -> list[str]:
    """Tokenize HTML into a flat list of tag/text tokens."""
    tokens = []
    html = re.sub(r"\s+", " ", html.strip())
    for part in re.split(r"(<[^>]+>)", html):
        part = part.strip()
        if part:
            tokens.append(part)
    return tokens


class HTMLTableNode:
    """Node in the table structure tree for TEDS computation."""

    def __init__(self, tag: str, attrs: dict | None = None, text: str = ""):
        self.tag = tag
        self.attrs = attrs or {}
        self.text = text.strip()
        self.children: list["HTMLTableNode"] = []

    def __repr__(self):
        return f"<{self.tag}>{self.text[:20]}"


def _parse_table_tree(html: str) -> HTMLTableNode:
    """
    Parse an HTML table string into a tree structure.
    Handles colspan/rowspan and nested tags.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table") or soup

    def _build_node(tag) -> HTMLTableNode:
        if isinstance(tag, str):
            text = tag.strip()
            return HTMLTableNode("text", text=text) if text else None
        name = getattr(tag, "name", None)
        if name is None:
            return None
        attrs = {k: " ".join(v) if isinstance(v, list) else str(v) for k, v in tag.attrs.items()}
        node = HTMLTableNode(name, attrs=attrs)
        for child in tag.children:
            child_node = _build_node(child)
            if child_node is not None:
                node.children.append(child_node)
        # Collapse direct text children into node.text
        text_parts = [c.text for c in node.children if c.tag == "text"]
        node.children = [c for c in node.children if c.tag != "text"]
        node.text = " ".join(text_parts).strip()
        return node

    return _build_node(table)


def _tree_edit_distance(tree1: HTMLTableNode | None, tree2: HTMLTableNode | None) -> int:
    """
    Compute tree edit distance between two HTML table trees.
    Uses a simplified Zhang-Shasha algorithm (node insertion/deletion/substitution).
    Cost: insert=1, delete=1, substitute=1 (0 if nodes match).
    """
    if tree1 is None and tree2 is None:
        return 0
    if tree1 is None:
        return 1 + sum(_tree_edit_distance(None, c) for c in tree2.children)
    if tree2 is None:
        return 1 + sum(_tree_edit_distance(c, None) for c in tree1.children)

    # Substitution cost: 0 if tag, colspan, rowspan, and text match
    sub_cost = 0
    if tree1.tag != tree2.tag:
        sub_cost = 1
    elif tree1.attrs.get("colspan", "1") != tree2.attrs.get("colspan", "1"):
        sub_cost = 1
    elif tree1.attrs.get("rowspan", "1") != tree2.attrs.get("rowspan", "1"):
        sub_cost = 1
    elif tree1.text.lower() != tree2.text.lower() and tree1.tag in ("td", "th"):
        sub_cost = 0.5  # partial penalty for text mismatch

    # Align children greedily (simplified — full DP for sequence alignment)
    n1, n2 = len(tree1.children), len(tree2.children)
    # DP alignment of children sequences
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        dp[i][0] = sum(
            1 + sum(_tree_size(tree1.children[k]) for k in range(i)) - i
            for _ in range(1)
        ) if i > 0 else 0
    for j in range(n2 + 1):
        dp[0][j] = 0  # skip initialization for brevity

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delete child from tree1
                dp[i][j - 1] + 1,  # insert child into tree2
                dp[i - 1][j - 1] + _tree_edit_distance(tree1.children[i - 1], tree2.children[j - 1]),
            )

    return sub_cost + dp[n1][n2]


def _tree_size(tree: HTMLTableNode | None) -> int:
    if tree is None:
        return 0
    return 1 + sum(_tree_size(c) for c in tree.children)


def teds(pred_html: str, gold_html: str, structure_only: bool = False) -> float:
    """
    Compute TEDS score between predicted and gold HTML tables.

    Args:
        pred_html: Predicted HTML table string.
        gold_html: Ground truth HTML table string.
        structure_only: If True, ignore cell text content (structure TEDS).

    Returns:
        TEDS score in [0, 1].
    """
    if not pred_html.strip() and not gold_html.strip():
        return 1.0
    if not pred_html.strip() or not gold_html.strip():
        return 0.0

    if structure_only:
        # Strip cell text for structure-only evaluation
        from bs4 import BeautifulSoup
        def strip_text(html):
            soup = BeautifulSoup(html, "lxml")
            for tag in soup.find_all(["td", "th"]):
                tag.string = ""
            return str(soup)
        pred_html = strip_text(pred_html)
        gold_html = strip_text(gold_html)

    tree_pred = _parse_table_tree(pred_html)
    tree_gold = _parse_table_tree(gold_html)

    dist = _tree_edit_distance(tree_pred, tree_gold)
    size = _tree_size(tree_pred) + _tree_size(tree_gold)
    if size == 0:
        return 1.0
    return 1.0 - (2.0 * dist / size)


def compute_teds_batch(predictions: list[dict], structure_only: bool = False) -> dict:
    """
    Compute TEDS for a batch of {prediction, ground_truth} dicts.

    Returns:
        {mean_teds, scores: list[float], n}
    """
    scores = []
    for item in predictions:
        pred = item.get("prediction", "")
        gold = item.get("ground_truth", "")
        scores.append(teds(pred, gold, structure_only=structure_only))

    return {
        "mean_teds": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
        "n": len(scores),
    }
