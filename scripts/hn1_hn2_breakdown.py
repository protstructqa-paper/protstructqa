"""Compute HN1 vs HN2 accuracy breakdown across all available result dirs."""
import json
import glob
from collections import defaultdict
from pathlib import Path

BENCH_ROOT = Path(
    "./benchmark"
)
HN_EVAL = BENCH_ROOT / "splits" / "test_hn_eval.jsonl"


def load_qid_to_hnclass(path: Path) -> dict[str, str]:
    m = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            m[d["qid"]] = d.get("hn_class", "UNK")
    return m


def compute_breakdown(per_q_path: Path, qid_to_hnclass: dict[str, str]) -> dict:
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    with open(per_q_path) as f:
        for line in f:
            d = json.loads(line)
            hn = qid_to_hnclass.get(d["qid"], "UNK")
            by_class[hn]["total"] += 1
            if d.get("correct"):
                by_class[hn]["correct"] += 1
    out = {}
    total_correct = 0
    total_total = 0
    for hn, s in by_class.items():
        out[hn] = {
            "n": s["total"],
            "acc": 100.0 * s["correct"] / s["total"] if s["total"] else float("nan"),
        }
        total_correct += s["correct"]
        total_total += s["total"]
    out["ALL"] = {
        "n": total_total,
        "acc": 100.0 * total_correct / total_total if total_total else float("nan"),
    }
    return out


def main() -> None:
    qid_to_hnclass = load_qid_to_hnclass(HN_EVAL)
    print(f"HN eval split loaded: {len(qid_to_hnclass)} questions")
    counts = defaultdict(int)
    for v in qid_to_hnclass.values():
        counts[v] += 1
    print(f"  hn_class dist: {dict(counts)}")
    print()

    pattern = str(BENCH_ROOT / "baseline_runs_v3dataset*/Qwen3-*/test_hn_eval/*/per_question.jsonl")
    paths = sorted(glob.glob(pattern))
    print(f"Found {len(paths)} per_question.jsonl files for test_hn_eval")
    print()

    rows = []
    for p in paths:
        p = Path(p)
        parts = p.parts
        try:
            run_dir = parts[-5]
            model = parts[-4]
            regime = parts[-2]
        except IndexError:
            continue
        breakdown = compute_breakdown(p, qid_to_hnclass)
        rows.append({
            "run_dir": run_dir,
            "model": model,
            "regime": regime,
            "path": str(p),
            "breakdown": breakdown,
        })

    # Print as table: run_dir / model / regime / N(HN1) / acc(HN1) / N(HN2) / acc(HN2) / N(ALL) / acc(ALL)
    print(
        f"{'run_dir':<48} {'model':<14} {'regime':<10} "
        f"{'HN1_n':>6} {'HN1_acc':>8} {'HN2_n':>6} {'HN2_acc':>8} {'ALL_n':>6} {'ALL_acc':>8}"
    )
    print("-" * 130)
    for r in rows:
        b = r["breakdown"]
        hn1 = b.get("HN1", {"n": 0, "acc": float("nan")})
        hn2 = b.get("HN2", {"n": 0, "acc": float("nan")})
        all_ = b.get("ALL", {"n": 0, "acc": float("nan")})
        print(
            f"{r['run_dir']:<48} {r['model']:<14} {r['regime']:<10} "
            f"{hn1['n']:>6} {hn1['acc']:>8.2f} {hn2['n']:>6} {hn2['acc']:>8.2f} "
            f"{all_['n']:>6} {all_['acc']:>8.2f}"
        )


if __name__ == "__main__":
    main()
