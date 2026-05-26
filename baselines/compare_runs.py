"""Compare two run directories (L0 vs L1, or any two regimes/models)
on aligned per-question outputs. Produces a per-template / per-family
comparison table and emits a LaTeX block ready for paper inclusion.

Usage:
    python -m baselines.compare_runs \
        --a benchmark/baseline_runs/Qwen3-1.7B/test_iid_sample50/L0/ \
        --b benchmark/baseline_runs/Qwen3-1.7B/test_iid_sample50/L1/ \
        --label-a L0 --label-b L1 \
        --out paper/tables/headline.tex

Aligns rows by qid; reports only questions present in both runs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(run_dir: Path, prefer_reeval: bool = True) -> dict:
    """Load per_question rows keyed by qid. Use reeval if available."""
    cands = []
    if prefer_reeval:
        cands.append(run_dir / "per_question_reeval.jsonl")
    cands.append(run_dir / "per_question.jsonl")
    for p in cands:
        if p.exists():
            rows: dict = {}
            with p.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    rows[r["qid"]] = r
            return rows
    raise SystemExit(f"no per_question.jsonl in {run_dir}")


def acc(rows: list[dict]) -> tuple[int, int, float]:
    n = len(rows)
    c = sum(1 for r in rows if r.get("correct"))
    return c, n, (c / n if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--label-a", default="L0")
    ap.add_argument("--label-b", default="L1")
    ap.add_argument("--out", type=Path, default=None,
                      help="optional path for LaTeX table")
    args = ap.parse_args()

    A = load_rows(args.a)
    B = load_rows(args.b)
    common = sorted(set(A) & set(B))
    print(f"# {args.label_a}: {len(A)} rows | {args.label_b}: {len(B)} rows | common: {len(common)}")

    aA = [A[q] for q in common]
    aB = [B[q] for q in common]

    cA, nA, accA = acc(aA)
    cB, nB, accB = acc(aB)
    print(f"\nOverall on {len(common)} aligned questions:")
    print(f"  {args.label_a}: {cA}/{nA} = {100*accA:.2f}%")
    print(f"  {args.label_b}: {cB}/{nB} = {100*accB:.2f}%")
    print(f"  Δ ({args.label_b} - {args.label_a}): {100*(accB-accA):+.2f}pp")

    # Per-family
    fA = defaultdict(list); fB = defaultdict(list)
    for r in aA: fA[r["family"]].append(r)
    for r in aB: fB[r["family"]].append(r)
    print(f"\nPer-family:")
    for f in sorted(set(fA) | set(fB)):
        _, _, a = acc(fA[f]); _, _, b = acc(fB[f])
        print(f"  {f}: {args.label_a}={100*a:.1f}%  {args.label_b}={100*b:.1f}%  Δ={100*(b-a):+.1f}pp")

    # Per-template
    tA = defaultdict(list); tB = defaultdict(list)
    for r in aA: tA[r["template"]].append(r)
    for r in aB: tB[r["template"]].append(r)
    print(f"\nPer-template (sorted by Δ desc):")
    rows = []
    for t in sorted(set(tA) | set(tB)):
        _, nA_t, a = acc(tA[t]); _, nB_t, b = acc(tB[t])
        rows.append((t, a, b, b - a, nA_t))
    for t, a, b, d, n in sorted(rows, key=lambda r: -r[3]):
        print(f"  {t}: {args.label_a}={100*a:.1f}%  {args.label_b}={100*b:.1f}%  Δ={100*d:+.1f}pp (n={n})")

    if args.out:
        # Build LaTeX table. Family-level summary.
        lines = [
            "\\begin{table}[t]",
            "\\centering\\small",
            "\\begin{tabular}{lccr}",
            "\\toprule",
            f"Family & {args.label_a} & {args.label_b} & $\\Delta$ \\\\",
            "\\midrule",
        ]
        for f in sorted(set(fA) | set(fB)):
            _, _, a = acc(fA[f]); _, _, b = acc(fB[f])
            lines.append(f"{f} & {100*a:.1f}\\% & {100*b:.1f}\\% & {100*(b-a):+.1f} \\\\")
        lines.append("\\midrule")
        lines.append(f"\\textbf{{Overall}} & \\textbf{{{100*accA:.2f}\\%}} & "
                       f"\\textbf{{{100*accB:.2f}\\%}} & "
                       f"\\textbf{{{100*(accB-accA):+.2f}}} \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Per-family accuracy of {args.label_a} vs.\\ "
            f"{args.label_b} on the aligned stratified sample "
            f"($n$={len(common)} questions, {len(common)//23} per template).}}",
            "\\label{tab:l0_vs_l1}",
            "\\end{table}",
        ]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote LaTeX table → {args.out}")


if __name__ == "__main__":
    main()
