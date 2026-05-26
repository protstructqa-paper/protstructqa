"""ProtStructQA split composition.

Reads positives from benchmark/questions/{species}/{template}.jsonl and
hard negatives from benchmark/hard_negatives/{species}/{template}.jsonl,
then composes the train / dev / test splits documented in the spec.

Splits produced (each is a JSONL file):

    train.jsonl                    Family A-F questions on the 80% of human
                                   proteins designated train.
    dev.jsonl                      Family A-F questions on 10% of human (dev).
    test_iid.jsonl                 Family A-F questions on 10% of human (test).
    test_cross_species.jsonl       Family A-F questions on ALL mouse/fly/chicken
                                   proteins (cross-species OOD test).
    test_compositional.jsonl       Family G questions across ALL species
                                   (compositional generalization track).
    test_selective_prompted.jsonl  Family Ha questions across ALL species.
    test_selective_unprompted.jsonl Family Hb questions across ALL species.
    test_hn.jsonl                  All hard negatives across families/species.
    manifest.json                  Counts + protein-list + seed.

Protein-level split (not question-level): deterministic random partition of
the 4000 human canonical UniProts into 80/10/10. All questions for a given
protein land in exactly one split, preserving the OOD property of the held-
out test sets.

Usage:
    python benchmark/06_compose_splits.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent.parent
QUESTIONS_ROOT = HERE / "benchmark" / "questions"
HN_ROOT = HERE / "benchmark" / "hard_negatives"
SPLITS_ROOT = HERE / "benchmark" / "splits"

SPECIES = ["human", "mouse", "fly", "chicken"]
HUMAN_TRAIN_FRAC = 0.80
HUMAN_DEV_FRAC = 0.10   # remaining 0.10 is test_iid


# ----------------------------- I/O helpers ---------------------------- #


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_all_questions(root: Path, species: str) -> list[dict]:
    out: list[dict] = []
    sp_dir = root / species
    if not sp_dir.exists():
        return out
    for f in sorted(sp_dir.glob("*.jsonl")):
        out.extend(_iter_jsonl(f))
    return out


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ----------------------------- protein-level partition --------------- #


def partition_proteins(human_uniprots: list[str], rng: random.Random
                          ) -> tuple[set[str], set[str], set[str]]:
    """Deterministic protein-level 80/10/10 split."""
    shuffled = list(human_uniprots)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * HUMAN_TRAIN_FRAC))
    n_dev = int(round(n * HUMAN_DEV_FRAC))
    train = set(shuffled[:n_train])
    dev = set(shuffled[n_train: n_train + n_dev])
    test = set(shuffled[n_train + n_dev:])
    return train, dev, test


# ----------------------------- main split composer ------------------- #


def compose(seed: int = 42, out_dir: Path = SPLITS_ROOT
              ) -> dict:
    rng = random.Random(seed)

    # Read everything
    positives_by_species: dict[str, list[dict]] = {}
    hns_by_species: dict[str, list[dict]] = {}
    for sp in SPECIES:
        positives_by_species[sp] = _load_all_questions(QUESTIONS_ROOT, sp)
        hns_by_species[sp] = _load_all_questions(HN_ROOT, sp)

    # Protein-level partition for human
    human_uniprots = sorted({q["uniprot"] for q in positives_by_species["human"]})
    train_set, dev_set, test_set = partition_proteins(human_uniprots, rng)

    AF_FAMILIES = {"A", "B", "C", "D", "E", "F"}

    train: list[dict] = []
    dev: list[dict] = []
    test_iid: list[dict] = []
    test_cross_species: list[dict] = []
    test_compositional: list[dict] = []
    test_selective_prompted: list[dict] = []
    test_selective_unprompted: list[dict] = []
    test_hn: list[dict] = []

    # Human positives: split by family + protein assignment
    for q in positives_by_species["human"]:
        fam = q["family"]
        up = q["uniprot"]
        if fam in AF_FAMILIES:
            if up in train_set:
                train.append(q)
            elif up in dev_set:
                dev.append(q)
            elif up in test_set:
                test_iid.append(q)
            else:
                # Defensive: shouldn't happen, but if so put in test
                test_iid.append(q)
        elif fam == "G":
            test_compositional.append(q)
        elif fam == "Ha":
            test_selective_prompted.append(q)
        elif fam == "Hb":
            test_selective_unprompted.append(q)

    # Cross-species (mouse / fly / chicken): all A-F → test_cross_species,
    # G → test_compositional, Ha/Hb → test_selective_*
    for sp in ("mouse", "fly", "chicken"):
        for q in positives_by_species[sp]:
            fam = q["family"]
            if fam in AF_FAMILIES:
                test_cross_species.append(q)
            elif fam == "G":
                test_compositional.append(q)
            elif fam == "Ha":
                test_selective_prompted.append(q)
            elif fam == "Hb":
                test_selective_unprompted.append(q)

    # Hard negatives: all → test_hn (regardless of family/species)
    for sp in SPECIES:
        for q in hns_by_species[sp]:
            test_hn.append(q)

    # Write splits
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train":                       _write_jsonl(out_dir / "train.jsonl", train),
        "dev":                         _write_jsonl(out_dir / "dev.jsonl", dev),
        "test_iid":                    _write_jsonl(out_dir / "test_iid.jsonl", test_iid),
        "test_cross_species":          _write_jsonl(out_dir / "test_cross_species.jsonl", test_cross_species),
        "test_compositional":          _write_jsonl(out_dir / "test_compositional.jsonl", test_compositional),
        "test_selective_prompted":     _write_jsonl(out_dir / "test_selective_prompted.jsonl", test_selective_prompted),
        "test_selective_unprompted":   _write_jsonl(out_dir / "test_selective_unprompted.jsonl", test_selective_unprompted),
        "test_hn":                     _write_jsonl(out_dir / "test_hn.jsonl", test_hn),
    }

    # Per-split family counts (for paper data section)
    family_counts: dict[str, dict[str, int]] = {}
    for split_name, items in [
        ("train", train), ("dev", dev), ("test_iid", test_iid),
        ("test_cross_species", test_cross_species),
        ("test_compositional", test_compositional),
        ("test_selective_prompted", test_selective_prompted),
        ("test_selective_unprompted", test_selective_unprompted),
        ("test_hn", test_hn),
    ]:
        c = Counter(q["family"] for q in items)
        family_counts[split_name] = dict(c)

    manifest = {
        "seed": seed,
        "human_train_fraction": HUMAN_TRAIN_FRAC,
        "human_dev_fraction": HUMAN_DEV_FRAC,
        "n_human_train_proteins": len(train_set),
        "n_human_dev_proteins": len(dev_set),
        "n_human_test_proteins": len(test_set),
        "split_sizes": counts,
        "family_counts_per_split": family_counts,
        "train_protein_ids_sample": sorted(train_set)[:10],
        "test_protein_ids_sample": sorted(test_set)[:10],
        "total_questions": sum(counts.values()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=SPLITS_ROOT)
    args = ap.parse_args()

    manifest = compose(seed=args.seed, out_dir=args.out_dir)
    print("\n=== SPLITS ===")
    for k, v in manifest["split_sizes"].items():
        print(f"  {k:30s}  {v:7d}")
    print(f"\n  TOTAL  {manifest['total_questions']}")
    print(f"\n  human proteins:  train={manifest['n_human_train_proteins']}  "
          f"dev={manifest['n_human_dev_proteins']}  "
          f"test={manifest['n_human_test_proteins']}")
    print(f"\n  manifest at {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
