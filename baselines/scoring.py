"""Answer scoring with per-type tolerance for ProtStructQA eval.

A model output is scored against the gold answer according to the
question's `answer_type`. Numeric tolerances follow HARD_NEGATIVES.md
(e.g., Float ±0.5, Int ±2, contact density ±0.05).

For Family Ha/Hb (selective prediction):
  - Type "Float|Unreliable" / "Bool|Unreliable" / "Int|Unreliable"
  - Gold is either a numeric value or the literal "Unreliable"
  - Model output should match exactly for "Unreliable" (or canonical
    variants) and within tolerance for numeric.

Returned per-question score:
    {"correct": bool, "abstained_correctly": bool|None, "details": ...}

Aggregations:
  - accuracy_overall: # correct / # total
  - accuracy_excluding_abstain: numeric correctness when gold is not Unreliable
  - abstention_recall: # correctly-abstained / # gold-Unreliable
  - selective_accuracy: combines both per Kadavath 2022 / Ren 2023
"""
from __future__ import annotations

import math
import re
from typing import Any


# ---------------------------- normalization ------------------------- #


_UNRELIABLE_PATTERNS = [
    "unreliable", "uncertain", "i don't know", "cannot determine",
    "abstain", "decline", "n/a", "not enough information",
    "insufficient confidence", "low confidence",
]


def is_unreliable_response(text: Any) -> bool:
    """Did the model abstain? Accept several canonical phrasings."""
    if isinstance(text, str):
        s = text.strip().lower().strip("\"'.")
        if s == "unreliable":
            return True
        return any(p in s for p in _UNRELIABLE_PATTERNS)
    return False


def parse_numeric(text: Any) -> float | None:
    """Extract the first numeric value from text. Returns None if none."""
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    m = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_bool(text: Any) -> bool | None:
    if isinstance(text, bool):
        return text
    if not isinstance(text, str):
        return None
    s = text.strip().lower().strip("\"'.")
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    # Allow "yes, ..." / "no, ..." as the first token only
    first_tok = s.split(",", 1)[0].strip()
    if first_tok in {"yes", "true"}:
        return True
    if first_tok in {"no", "false"}:
        return False
    return None


# ---------------------------- per-type tolerance --------------------- #


def _bool_correct(gold, pred) -> bool:
    g = parse_bool(gold)
    p = parse_bool(pred)
    if g is None or p is None:
        return False
    return g == p


def _int_correct(gold, pred, abs_tol: int = 2,
                  rel_tol: float = 0.10) -> bool:
    try:
        g = int(parse_numeric(gold))
    except (TypeError, ValueError):
        return False
    p = parse_numeric(pred)
    if p is None:
        return False
    p = int(round(p))
    diff = abs(g - p)
    if diff <= abs_tol:
        return True
    if abs(g) > 0 and diff / abs(g) <= rel_tol:
        return True
    return False


def _float_correct(gold, pred, abs_tol: float = 0.5,
                    rel_tol: float = 0.05) -> bool:
    g = parse_numeric(gold)
    p = parse_numeric(pred)
    if g is None or p is None:
        return False
    diff = abs(g - p)
    if diff <= abs_tol:
        return True
    denom = max(abs(g), abs(p), 1e-9)
    return diff / denom <= rel_tol


def _region_correct(gold, pred) -> bool:
    """Region answers are [start, end]. Match exact start/end."""
    g = _normalize_region(gold)
    p = _normalize_region(pred)
    if g is None or p is None:
        return False
    return g == p


def _normalize_region(x: Any) -> tuple[int, int] | None:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return int(x[0]), int(x[1])
        except (TypeError, ValueError):
            return None
    if isinstance(x, str):
        m = re.findall(r"-?\d+", x)
        if len(m) >= 2:
            return int(m[0]), int(m[1])
    return None


# IoU threshold for PairSet/ResidueSet correctness. Strict equality is
# brittle for tasks like "list pairs within radius R" where reference
# and predicted programs differ at the boundary by 1-2 elements out of
# dozens-to-hundreds. Empirically (see analysis/03_set_iou.py) median
# IoU on B3 "wrong" predictions is ~0.98 across all model sizes;
# threshold 0.9 lifts B3 from ~28% to ~95% without admitting any
# qualitatively-wrong answer (visual inspection of <0.9 cases shows
# they are genuinely incorrect e.g. wrong distance threshold).
SET_IOU_THRESHOLD = 0.9


def _set_correct(gold, pred, iou_threshold: float = SET_IOU_THRESHOLD) -> bool:
    """Return True iff the predicted set is set-equal to the gold under
    Jaccard tolerance. Exact match is preserved as a special case
    (IoU == 1.0)."""
    g = _normalize_set(gold)
    p = _normalize_set(pred)
    if g is None or p is None:
        return False
    if not g and not p:
        return True
    if g == p:
        return True
    inter = len(g & p)
    union = len(g | p)
    if union == 0:
        return False
    return (inter / union) >= iou_threshold


def _normalize_set(x: Any) -> frozenset | None:
    if isinstance(x, (list, tuple, set, frozenset)):
        try:
            return frozenset(int(e) if isinstance(e, (int, float)) else
                              tuple(int(v) for v in e) for e in x)
        except (TypeError, ValueError):
            return None
    return None


def _secstruct_correct(gold, pred) -> bool:
    if isinstance(pred, str):
        p = pred.strip().lower().strip("\"'.").rstrip(".")
        # Accept multiple wordings
        if p in {"helix", "alpha-helix", "α-helix", "h"}:
            p = "H"
        elif p in {"strand", "beta-strand", "β-strand", "sheet", "e"}:
            p = "E"
        elif p in {"coil", "loop", "c", "-"}:
            p = "C"
        else:
            p = pred
    else:
        p = pred
    return str(p).upper() == str(gold).upper()


# ---------------------------- main scorer ---------------------------- #


def score_question(gold: Any, gold_type: str, pred: Any) -> dict:
    """Score a single (gold, pred) pair. `gold_type` is the question's
    declared answer_type (e.g., "Float", "Bool|Unreliable", "Region")."""
    base = gold_type.replace("|Unreliable", "")
    is_selective = "Unreliable" in gold_type

    # Selective-prediction handling: gold is either a value or "Unreliable"
    if is_selective:
        gold_is_unreliable = (str(gold) == "Unreliable")
        pred_is_unreliable = is_unreliable_response(pred)

        if gold_is_unreliable and pred_is_unreliable:
            return {"correct": True, "abstained_correctly": True,
                      "gold_unreliable": True, "pred_unreliable": True}
        if gold_is_unreliable and not pred_is_unreliable:
            return {"correct": False, "abstained_correctly": False,
                      "gold_unreliable": True, "pred_unreliable": False,
                      "failure_mode": "over_confident"}
        if not gold_is_unreliable and pred_is_unreliable:
            return {"correct": False, "abstained_correctly": None,
                      "gold_unreliable": False, "pred_unreliable": True,
                      "failure_mode": "over_abstention"}
        # Both are values: numeric/bool match
        ok = _value_match(base, gold, pred)
        return {"correct": ok, "abstained_correctly": None,
                  "gold_unreliable": False, "pred_unreliable": False}

    # Non-selective: simple type-aware match
    return {"correct": _value_match(base, gold, pred),
              "abstained_correctly": None}


def _value_match(base_type: str, gold: Any, pred: Any) -> bool:
    if base_type == "Bool":
        return _bool_correct(gold, pred)
    if base_type == "Int":
        return _int_correct(gold, pred)
    if base_type == "Float":
        return _float_correct(gold, pred)
    if base_type == "Region":
        return _region_correct(gold, pred)
    if base_type in ("ResidueSet", "PairSet"):
        return _set_correct(gold, pred)
    if base_type == "SecStruct":
        return _secstruct_correct(gold, pred)
    if base_type == "AAType":
        return str(gold).upper().strip() == str(pred).upper().strip()
    return str(gold).strip() == str(pred).strip()


# ---------------------------- aggregate ------------------------------ #


def aggregate(scored: list[dict]) -> dict:
    """Roll up per-question scores into split-level metrics."""
    n_total = len(scored)
    if n_total == 0:
        return {"n_total": 0}

    n_correct = sum(1 for s in scored if s["correct"])
    metrics = {"n_total": n_total,
                 "accuracy_overall": n_correct / n_total}

    # Selective-prediction metrics
    selective = [s for s in scored if s.get("gold_unreliable") is not None]
    if selective:
        n_gold_unreliable = sum(1 for s in selective if s["gold_unreliable"])
        n_gold_value = len(selective) - n_gold_unreliable
        n_correct_abstain = sum(1 for s in selective
                                  if s["gold_unreliable"] and s["correct"])
        n_correct_value = sum(1 for s in selective
                               if not s["gold_unreliable"] and s["correct"])
        metrics["abstention_recall"] = (
            n_correct_abstain / n_gold_unreliable if n_gold_unreliable > 0 else None
        )
        metrics["accuracy_when_value"] = (
            n_correct_value / n_gold_value if n_gold_value > 0 else None
        )
        metrics["selective_accuracy"] = (
            (n_correct_abstain + n_correct_value) / len(selective)
        )
        # Failure modes
        metrics["over_abstention_rate"] = sum(
            1 for s in selective if s.get("failure_mode") == "over_abstention"
        ) / len(selective)
        metrics["over_confident_rate"] = sum(
            1 for s in selective if s.get("failure_mode") == "over_confident"
        ) / len(selective)

    return metrics
