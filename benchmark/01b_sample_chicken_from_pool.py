"""Sample 2K SBE-disjoint chicken UniProts from the pre-downloaded AFDB pool.

Why this script exists separately from `01_download_proteomes.py`:

  AlphaFold DB ships organism-level proteome tarballs only for select
  species (human, mouse, fly, yeast, ~20 others). Chicken (UP000000539,
  Gallus gallus) is NOT among them: it is only available as per-protein
  files via https://alphafold.ebi.ac.uk/files/. SBE has already paid the
  cost of bulk-downloading 39,625 chicken PDBs to
  ./datasets/chicken_alphafold_full/, so we sample from
  that pool instead of re-downloading.

Pipeline:
    1. Enumerate all `AF-{up}-F1-model_v6.pdb` files in the pool.
    2. Load the SBE exclusion set (anchor_pool.tsv ∪ eval_set.tsv,
       1,405 UniProts) so ProtStructQA stays disjoint from SBE's variant-
       scoring evaluation set (salami-slicing protection,
       memory: feedback_no_method_carryover_to_tts.md).
    3. Stratified-sample TARGET (default 2,000) UniProts by PDB file size
       (a strong proxy for sequence length). 5 size buckets × 400 each.
    4. Symlink chosen PDBs into ./data/chicken/structures/
       to match the layout produced by 01_download_proteomes.py for the
       other species. Symlink (not copy) keeps disk usage flat.
    5. Write ./data/chicken/uniprot_ids.txt so
       02_download_pae.py can fetch the PAE JSONs for the sampled subset.

After this script, run:
    python benchmark/02_download_pae.py --species chicken

to fetch PAE JSONs for the 2,000 sampled UniProts.

Tests: benchmark/tests/test_chicken_sampling.py (synthetic-fixture TDD).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

POOL_DIR = Path(os.environ.get(
    "PROTSTRUCTQA_CHICKEN_POOL",
    "./datasets/chicken_alphafold_full",
))
DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
OUT_DIR = DATA_ROOT / "chicken"

SBE_EXCL_FILES = [
    Path("<project-root>/"
         "SBE/data/splits/chicken_tier_a/anchor_pool.tsv"),
    Path("<project-root>/"
         "SBE/data/splits/chicken_tier_a/eval_set.tsv"),
]
SBE_EXCL_COL = "chicken_uniprot"

DEFAULT_TARGET = 2000


def load_exclusion_set(files: list[Path], col: str) -> set[str]:
    excluded: set[str] = set()
    for path in files:
        p = Path(path)
        if not p.exists():
            print(f"[exclude] WARNING: file not found: {p}", flush=True)
            continue
        with p.open() as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if col in row and row[col]:
                    excluded.add(row[col])
    if excluded:
        print(f"[exclude] loaded {len(excluded)} UniProts to exclude",
              flush=True)
    return excluded


def enumerate_pool(pool_dir: Path) -> dict[str, int]:
    """Return {uniprot: pdb_size_bytes} for every AF-*-F1-model_v6.pdb in pool."""
    avail: dict[str, int] = {}
    for pdb in pool_dir.glob("AF-*-F1-model_v6.pdb"):
        parts = pdb.name.split("-")
        if len(parts) < 2 or parts[0] != "AF":
            continue
        avail[parts[1]] = pdb.stat().st_size
    return avail


def stratified_sample(items: dict[str, int],
                       target: int,
                       n_buckets: int = 5,
                       rng_seed: int = 42) -> list[str]:
    """Stratified random sample by size, with deterministic rng_seed.

    Items are sorted by size, divided into `n_buckets` equal-size buckets
    (by count), and ~target/n_buckets are randomly drawn from each bucket.

    If target >= len(items), returns all items (sorted by size).
    """
    if target >= len(items):
        return sorted(items.keys(), key=lambda k: items[k])

    rng = random.Random(rng_seed)
    sorted_items = sorted(items.items(), key=lambda kv: kv[1])  # by size
    n = len(sorted_items)
    per_bucket = target // n_buckets
    remainder = target - per_bucket * n_buckets

    chosen: list[str] = []
    for b in range(n_buckets):
        lo = (n * b) // n_buckets
        hi = (n * (b + 1)) // n_buckets
        bucket = [up for up, _ in sorted_items[lo:hi]]
        k = per_bucket + (1 if b < remainder else 0)
        if k >= len(bucket):
            chosen.extend(bucket)
        else:
            chosen.extend(rng.sample(bucket, k))
    return chosen[:target]


def link_pdbs(chosen: list[str], pool_dir: Path, out_dir: Path) -> int:
    """Symlink chosen PDBs from pool into out_dir. Returns # newly created links."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for up in chosen:
        src = pool_dir / f"AF-{up}-F1-model_v6.pdb"
        if not src.exists():
            print(f"[warn] missing in pool: {up}", flush=True)
            continue
        dst = out_dir / f"AF-{up}-F1-model_v6.pdb"
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src)
        n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[chicken] pool        = {args.pool_dir}", flush=True)
    print(f"[chicken] out         = {args.out_dir}", flush=True)
    print(f"[chicken] target N    = {args.target}", flush=True)

    excl = load_exclusion_set(SBE_EXCL_FILES, col=SBE_EXCL_COL)
    avail = enumerate_pool(args.pool_dir)
    print(f"[chicken] pool size   = {len(avail)} PDBs", flush=True)

    avail_filt = {u: s for u, s in avail.items() if u not in excl}
    print(f"[chicken] after SBE exclusion = {len(avail_filt)} PDBs "
          f"(removed {len(avail) - len(avail_filt)})", flush=True)

    chosen = stratified_sample(avail_filt, target=args.target,
                                  n_buckets=5, rng_seed=args.rng_seed)
    print(f"[chicken] sampled     = {len(chosen)} UniProts", flush=True)

    structures_dir = args.out_dir / "structures"
    n_new = link_pdbs(chosen, pool_dir=args.pool_dir, out_dir=structures_dir)
    print(f"[chicken] symlinked   = {n_new} new PDBs into {structures_dir}",
          flush=True)

    ids_file = args.out_dir / "uniprot_ids.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(chosen) + "\n")
    print(f"[chicken] wrote       = {ids_file} ({len(chosen)} IDs)",
          flush=True)
    print(f"[chicken] next step   = python benchmark/02_download_pae.py "
          f"--species chicken", flush=True)


if __name__ == "__main__":
    main()
