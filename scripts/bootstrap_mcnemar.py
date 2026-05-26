"""Bootstrap CIs + McNemar's paired test for the 32-cell ablation.

Output:
  1. Per-cell accuracy with 95% bootstrap CI (1000 resamples)
  2. McNemar's exact paired test for all {EV-v1 vs EV+CoT, L0 vs L0+CoT,
     L0+CoT vs EV+CoT} comparisons within each (model, split) pair
  3. Significance verdict (Bonferroni-corrected p < 0.05)

Usage:
  python scripts/bootstrap_mcnemar.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parents[1]

ROOTS = {
    "L0":     "baseline_runs_v3dataset",
    "L0+CoT": "baseline_runs_v3dataset_l0_cot",
    "EV-v1":  "baseline_runs_v3dataset",
    "EV+CoT": "baseline_runs_v3dataset_ev_cot",
}
REGIMES = {"L0": "L0", "L0+CoT": "L0", "EV-v1": "EV", "EV+CoT": "EV"}

SPLITS = ["test_iid", "test_compositional_eval",
          "test_cross_species_eval", "test_hn_eval"]
SPLIT_SHORT = {"test_iid": "iid", "test_compositional_eval": "comp",
                "test_cross_species_eval": "cs", "test_hn_eval": "hn"}
MODELS = ["Qwen3-1.7B", "Qwen3-8B"]
VARIANTS = ["L0", "L0+CoT", "EV-v1", "EV+CoT"]

# Comparisons we care about for the paper (paired McNemar)
PAIRWISE = [
    ("L0",     "L0+CoT"),  # CoT effect on free-form
    ("EV-v1",  "EV+CoT"),  # CoT effect on grammar+k=3
    ("L0+CoT", "EV+CoT"),  # grammar effect on top of CoT
    ("L0",     "EV-v1"),   # grammar effect on free-form
]


def load_per_question(model: str, split: str, variant: str) -> dict[str, bool]:
    """Return {qid: correct_bool}. Empty dict if file missing."""
    root = ROOTS[variant]
    regime = REGIMES[variant]
    path = HERE / "benchmark" / root / model / split / regime / "per_question.jsonl"
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("qid")
            if qid is not None:
                out[qid] = bool(d.get("correct", False))
    return out


def bootstrap_ci(corrects: list[bool], n_resamples: int = 1000,
                  seed: int = 42) -> tuple[float, float, float]:
    """Return (point_acc, ci_lo, ci_hi) at 95% confidence."""
    arr = np.asarray(corrects, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means.append(arr[idx].mean())
    means = np.asarray(means)
    return (float(arr.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def mcnemar_paired(a_correct: dict[str, bool],
                     b_correct: dict[str, bool]) -> tuple[int, int, int, int, float]:
    """Paired comparison on shared qids. Returns:
        (n_shared, n_a_only_correct, n_b_only_correct, n_both_or_neither,
         p_value_exact_binomial)
    """
    shared = set(a_correct) & set(b_correct)
    n_shared = len(shared)
    n_a_only = 0
    n_b_only = 0
    n_both = 0
    n_neither = 0
    for q in shared:
        a = a_correct[q]
        b = b_correct[q]
        if a and not b:
            n_a_only += 1
        elif b and not a:
            n_b_only += 1
        elif a and b:
            n_both += 1
        else:
            n_neither += 1
    # McNemar's exact: under H0 of symmetric disagreement, the count of
    # "A correct & B incorrect" follows Binomial(n_a_only + n_b_only, 0.5).
    n_disagree = n_a_only + n_b_only
    if n_disagree == 0:
        p = 1.0
    else:
        # Two-sided p-value
        k = min(n_a_only, n_b_only)
        p = 2 * stats.binom.cdf(k, n_disagree, 0.5)
        p = min(p, 1.0)
    return n_shared, n_a_only, n_b_only, n_both + n_neither, p


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}"


def fmt_ci(point: float, lo: float, hi: float) -> str:
    return f"{100*point:5.2f} [{100*lo:5.2f}, {100*hi:5.2f}]"


def fmt_p(p: float) -> str:
    if p < 1e-4: return "<0.0001"
    if p < 1e-3: return "<0.001"
    if p < 1e-2: return f"{p:.3f}"
    return f"{p:.3f}"


def stars(p: float, alpha: float = 0.05) -> str:
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < alpha:  return "*"
    return "ns"


def main() -> None:
    n_comparisons_total = len(MODELS) * len(SPLITS) * len(PAIRWISE)
    bonferroni_alpha = 0.05 / n_comparisons_total

    print("=" * 80)
    print("PER-CELL ACCURACY + 95% BOOTSTRAP CI (1000 resamples)")
    print("=" * 80)

    cell_data: dict[tuple[str, str, str], dict[str, bool]] = {}
    for model in MODELS:
        print(f"\n----- {model} -----")
        for split in SPLITS:
            print(f"\n  {SPLIT_SHORT[split]}:")
            for variant in VARIANTS:
                pq = load_per_question(model, split, variant)
                cell_data[(model, split, variant)] = pq
                if not pq:
                    print(f"    {variant:<10} (missing)")
                    continue
                vals = list(pq.values())
                pt, lo, hi = bootstrap_ci(vals)
                print(f"    {variant:<10} n={len(vals):>5}  "
                      f"acc={fmt_ci(pt, lo, hi)}")

    print()
    print("=" * 80)
    print(f"PAIRED McNEMAR'S TESTS  "
          f"(Bonferroni alpha = 0.05 / {n_comparisons_total} "
          f"= {bonferroni_alpha:.5f})")
    print("=" * 80)

    rows = []
    for model in MODELS:
        print(f"\n----- {model} -----")
        for split in SPLITS:
            print(f"\n  {SPLIT_SHORT[split]}:")
            for a_label, b_label in PAIRWISE:
                a = cell_data.get((model, split, a_label), {})
                b = cell_data.get((model, split, b_label), {})
                if not a or not b:
                    print(f"    {a_label:<8} vs {b_label:<8}  (missing data)")
                    continue
                n_shared, a_only, b_only, both_or_neither, p = mcnemar_paired(a, b)
                a_acc = sum(a[q] for q in a) / max(1, len(a))
                b_acc = sum(b[q] for q in b) / max(1, len(b))
                delta = (b_acc - a_acc) * 100
                bonf_sig = "BONF-SIG" if p < bonferroni_alpha else ""
                print(f"    {a_label:<8} vs {b_label:<8}  "
                      f"Δ = {delta:+6.2f}pp  "
                      f"n_shared={n_shared:>5}  "
                      f"flips: A={a_only:>4} B={b_only:>4}  "
                      f"p={fmt_p(p):>9}  {stars(p):<5}  {bonf_sig}")
                rows.append({
                    "model": model,
                    "split": SPLIT_SHORT[split],
                    "a": a_label,
                    "b": b_label,
                    "delta_pp": float(delta),
                    "n_shared": int(n_shared),
                    "a_only": int(a_only),
                    "b_only": int(b_only),
                    "p": float(p),
                    "bonferroni_significant": bool(p < bonferroni_alpha),
                })

    # Save machine-readable for the paper pipeline
    out = HERE / "benchmark" / "stats" / "bootstrap_mcnemar.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump({"alpha_bonferroni": bonferroni_alpha,
                   "rows": rows}, f, indent=2)
    print(f"\nMachine-readable output: {out}")


if __name__ == "__main__":
    main()
