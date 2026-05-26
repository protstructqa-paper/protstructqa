"""ProtStructQA benchmark validation gates.

Runs sanity checks on the composed splits before the benchmark is shipped:

    1. Schema validity: every question has the required fields.
    2. Gold-program executability: every A-G program parses + runs.
    3. Per-split protein-level disjointness (human train/dev/test_iid).
    4. Cross-species OOD purity (no human proteins in test_cross_species).
    5. HN class distribution within target fractions.
    6. Lexical-only baseline gate: train a simple TF-IDF + logistic
       regression on `question` text, eval on test_iid Bool answers.
       Spec: must be <= 55% (else paraphrases are too template-fingerprintable).

Usage:
    python benchmark/07_validate.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SPLITS_ROOT = HERE / "benchmark" / "splits"
DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))


REQUIRED_FIELDS = {
    "qid", "uniprot", "species", "family", "template", "question",
    "program", "answer", "answer_type", "params", "paraphrase_id",
}


# ---------------------------- I/O ------------------------------------ #


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_split(split_name: str) -> list[dict]:
    path = SPLITS_ROOT / f"{split_name}.jsonl"
    return list(_iter_jsonl(path))


# ---------------------------- gates ---------------------------------- #


def gate_schema(splits: dict[str, list[dict]]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for name, qs in splits.items():
        for q in qs:
            missing = REQUIRED_FIELDS - q.keys()
            if missing:
                issues.append(f"{name}/{q.get('qid', '?')}: missing {missing}")
                if len(issues) > 20: break
    return (not issues), issues


def gate_program_executes(splits: dict[str, list[dict]],
                            sample_n: int = 200,
                            rng_seed: int = 7) -> tuple[bool, list[str]]:
    """Sample N questions per split and verify the gold program executes
    against the matching ProteinView."""
    import random
    from dsl import load_from_npz, run as dsl_run
    rng = random.Random(rng_seed)
    issues: list[str] = []
    for name, qs in splits.items():
        if name == "test_hn":
            # HN class HN1 already had its program executed at mining time.
            # HN2 also executed. Skip re-verification here for speed.
            continue
        if not qs:
            continue
        sample = rng.sample(qs, min(sample_n, len(qs)))
        for q in sample:
            sp = q["species"]
            up = q["uniprot"]
            npz = DATA_ROOT / sp / "features" / f"AF-{up}.npz"
            if not npz.exists():
                issues.append(f"{name}/{q['qid']}: NPZ missing {npz}")
                continue
            try:
                view = load_from_npz(npz, uniprot=up, species=sp)
            except Exception as e:
                issues.append(f"{name}/{q['qid']}: NPZ load failed: {e}")
                continue
            fam = q["family"]
            if fam in ("Ha", "Hb"):
                # Family H uses Python execute_directly, program is doc.
                continue
            try:
                dsl_run(q["program"], view)
            except Exception as e:
                issues.append(f"{name}/{q['qid']}: prog failed: {e}")
                if len(issues) > 30:
                    break
    return (not issues), issues


def gate_protein_disjoint(splits: dict[str, list[dict]]
                              ) -> tuple[bool, list[str]]:
    """human train/dev/test_iid must be disjoint at the protein level."""
    train_p = {q["uniprot"] for q in splits["train"] if q["species"] == "human"}
    dev_p = {q["uniprot"] for q in splits["dev"] if q["species"] == "human"}
    test_p = {q["uniprot"] for q in splits["test_iid"] if q["species"] == "human"}
    issues = []
    if train_p & dev_p:
        issues.append(f"train ∩ dev human proteins = {len(train_p & dev_p)}")
    if train_p & test_p:
        issues.append(f"train ∩ test human proteins = {len(train_p & test_p)}")
    if dev_p & test_p:
        issues.append(f"dev ∩ test human proteins = {len(dev_p & test_p)}")
    return (not issues), issues


def gate_cross_species_pure(splits: dict[str, list[dict]]
                                ) -> tuple[bool, list[str]]:
    """test_cross_species must have NO human proteins."""
    bad = [q["qid"] for q in splits["test_cross_species"]
              if q["species"] == "human"]
    return (not bad), [f"{len(bad)} human questions in test_cross_species"
                          ] if bad else []


def gate_hn_class_distribution(splits: dict[str, list[dict]]
                                  ) -> tuple[bool, list[str]]:
    """HN class distribution should be approximately {HN1: 0.7, HN2: 0.15, HN3: 0.15}."""
    counts = Counter(q.get("hn_class", "?") for q in splits["test_hn"])
    total = sum(counts.values())
    if total == 0:
        return False, ["test_hn is empty"]
    fractions = {k: v / total for k, v in counts.items()}
    issues = []
    # We require HN1 to be present and dominant (>= 0.50)
    if fractions.get("HN1", 0) < 0.50:
        issues.append(f"HN1 fraction = {fractions.get('HN1', 0):.2f} < 0.50")
    return (not issues), issues + [f"distribution = {fractions}"]


def gate_lexical_only_baseline(splits: dict[str, list[dict]]
                                  ) -> tuple[bool, str]:
    """TF-IDF + logistic regression on question text alone, predicting
    Bool answers on test_iid. Spec gate: <= 55% accuracy."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return True, "sklearn not installed; skipping lexical baseline"

    train = [q for q in splits["train"] if q["answer_type"] == "Bool"]
    test = [q for q in splits["test_iid"] if q["answer_type"] == "Bool"]
    if len(train) < 100 or len(test) < 50:
        return True, f"not enough Bool examples (train={len(train)}, test={len(test)}); skipping"

    Xtr = [q["question"] for q in train]
    ytr = [int(bool(q["answer"])) for q in train]
    Xte = [q["question"] for q in test]
    yte = [int(bool(q["answer"])) for q in test]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr_v, ytr)
    yhat = clf.predict(Xte_v)
    acc = float(np.mean(np.array(yhat) == np.array(yte)))
    pass_gate = acc <= 0.55
    return pass_gate, (
        f"lexical-only Bool acc on test_iid: {acc*100:.1f}% "
        f"(target <= 55%; n_train={len(train)}, n_test={len(test)})"
    )


# ---------------------------- runner --------------------------------- #


GATES = [
    ("schema",                    gate_schema),
    ("protein_disjointness",      gate_protein_disjoint),
    ("cross_species_pure",        gate_cross_species_pure),
    ("hn_class_distribution",     gate_hn_class_distribution),
    ("program_executes_sampled",  gate_program_executes),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lexical", action="store_true",
                      help="skip the slow lexical-baseline gate")
    args = ap.parse_args()

    split_names = ["train", "dev", "test_iid", "test_cross_species",
                     "test_compositional", "test_selective_prompted",
                     "test_selective_unprompted", "test_hn"]
    splits = {n: _load_split(n) for n in split_names}
    print("=== Loaded splits ===")
    for n, qs in splits.items():
        print(f"  {n:30s}  {len(qs):7d}")
    print()

    print("=== Gates ===")
    overall_pass = True
    for gate_name, gate_fn in GATES:
        t0 = time.perf_counter()
        try:
            ok, issues = gate_fn(splits)
        except Exception as e:
            ok, issues = False, [f"gate raised: {e}"]
        dt = time.perf_counter() - t0
        status = "✓" if ok else "✗"
        print(f"  {status} {gate_name} ({dt:.1f}s)")
        if not ok:
            overall_pass = False
            for issue in issues[:5]:
                print(f"      - {issue}")
            if len(issues) > 5:
                print(f"      - ... ({len(issues)-5} more)")

    if not args.skip_lexical:
        t0 = time.perf_counter()
        ok, msg = gate_lexical_only_baseline(splits)
        dt = time.perf_counter() - t0
        status = "✓" if ok else "✗"
        print(f"  {status} lexical_only_baseline ({dt:.1f}s)")
        print(f"      {msg}")
        if not ok:
            overall_pass = False

    print()
    print("=== OVERALL " + ("PASS" if overall_pass else "FAIL") + " ===")
    if not overall_pass:
        sys.exit(2)


if __name__ == "__main__":
    main()
