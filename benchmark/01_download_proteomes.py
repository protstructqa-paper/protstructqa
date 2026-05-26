"""Download AlphaFold proteome subsets for the next-paper benchmark.

Pulls AlphaFold v6 PDB + PAE JSON files for:
  - Human (Homo sapiens)         taxid 9606,  primary benchmark
  - Mouse (Mus musculus)          taxid 10090, OOD-near (Section 6)
  - Drosophila (D. melanogaster) taxid 7227,  OOD-far (Section 6)

These species are disjoint from the variant paper's chicken / cattle /
salmon / pig panel by design (see PLAN.md §6 for rationale).

Source: AlphaFold UniProt release on Google Cloud Public Datasets,
        CC-BY 4.0 license.

Strategy:
  - For each species, fetch the AlphaFold organism-level proteome download
    (a single tar.gz containing all PDB + PAE for that organism).
  - Subsample at most 2K (human), 1.5K (mouse), 1K (Drosophila) proteins
    by stratified sampling on protein length to keep distributions matched.
  - Write the chosen UniProt IDs and structure files to a per-species
    directory.

Usage:
    python text_to_structurequery/benchmark/01_download_proteomes.py \
        [--species human,mouse,fly] [--max-per-species 2000] [--no-pae]

    Default DATA_ROOT is ./data/, NOT /home,
    because the proteome tarballs are several GB each and /home is
    quota-capped (200 GB). Override with $PROTSTRUCTQA_DATA.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import os
import urllib.request

# Storage convention: AF tarballs and extracted PDB+PAE files live on
# /work, NOT /home, because they're multi-GB and /home is quota-capped.
# Per-species directories land at {DATA_ROOT}/{human,mouse,fly}/.
# Override with $PROTSTRUCTQA_DATA if needed.
DATA_ROOT = Path(os.environ.get(
    "PROTSTRUCTQA_DATA", "./data"
))

# Organism-level AlphaFold downloads (proteome bundles).
# These URLs are the public AlphaFold archive; if the URL pattern shifts
# in a future release, update here.
ORGANISMS = {
    "human": {
        "taxid":     9606,
        "label":     "Homo sapiens",
        "max_proteins": 4000,
        "tar_urls": [
            "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar",
        ],
    },
    "mouse": {
        "taxid":     10090,
        "label":     "Mus musculus",
        "max_proteins": 2500,
        "tar_urls": [
            "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v6.tar",
        ],
    },
    "fly": {
        "taxid":     7227,
        "label":     "Drosophila melanogaster",
        "max_proteins": 1500,
        "tar_urls": [
            "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000803_7227_DROME_v6.tar",
        ],
    },
    "chicken": {
        "taxid":     9031,
        "label":     "Gallus gallus",
        "max_proteins": 2000,
        "tar_urls": [
            "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000539_9031_CHICK_v6.tar",
        ],
        # Salami-slicing protection: exclude any UniProt used by the SBE
        # variant-scoring paper, so ProtStructQA chicken proteins are
        # disjoint from SBE's evaluation subset.
        "exclude_uniprots_from": [
            "<external-resource>/data/splits/chicken_tier_a/anchor_pool.tsv",
            "<external-resource>/data/splits/chicken_tier_a/eval_set.tsv",
        ],
        # Column name in the TSVs that holds the UniProt IDs to exclude.
        "exclude_uniprot_column": "chicken_uniprot",
    },
}


def load_exclusion_set(meta: dict) -> set:
    """Load the set of UniProts to exclude (e.g., the 1,405 SBE chicken UniProts).
    Returns empty set if no exclusion configured."""
    excl_files = meta.get("exclude_uniprots_from", [])
    col = meta.get("exclude_uniprot_column", "uniprot")
    excluded: set[str] = set()
    for path in excl_files:
        p = Path(path)
        if not p.exists():
            print(f"[exclude] WARNING: exclusion file not found: {p}", flush=True)
            continue
        import csv as _csv
        with p.open() as fh:
            reader = _csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if col in row and row[col]:
                    excluded.add(row[col])
    if excluded:
        print(f"[exclude] loaded {len(excluded)} UniProts to exclude", flush=True)
    return excluded


def fetch_tar(url: str, dest_path: Path) -> Path:
    """Stream-download a tar archive."""
    print(f"[download] {url}", flush=True)
    t0 = time.perf_counter()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with dest_path.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    dt = time.perf_counter() - t0
    print(f"[download] wrote {dest_path}  ({dest_path.stat().st_size/1e9:.2f} GB, {dt:.0f}s)",
          flush=True)
    return dest_path


def extract_proteins(tar_path: Path, out_dir: Path,
                       max_proteins: int,
                       want_pae: bool = True,
                       exclude_uniprots: set | None = None) -> list[str]:
    """Extract AF-{uniprot}-F1-model_v*.pdb + predicted_aligned_error*.json
    files from the proteome tar archive into out_dir, capped at
    max_proteins, with stratified-by-length sampling.

    If `exclude_uniprots` is provided, those UniProts are filtered out
    BEFORE stratified sampling (used to keep ProtStructQA's chicken subset
    disjoint from SBE's variant-scoring eval subset).

    Returns the list of uniprot IDs kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] reading {tar_path}", flush=True)
    t0 = time.perf_counter()

    # Pass 1: enumerate (uniprot, length-proxy) for stratified sampling.
    # Length-proxy = compressed PDB size (correlates strongly with sequence length).
    members: dict[str, dict] = {}  # uniprot → {pdb_member, pae_member, size}
    with tarfile.open(tar_path, "r") as tar:
        for m in tar:
            name = m.name
            if not name.endswith(".pdb.gz"):
                # also check predicted_aligned_error JSON
                if "predicted_aligned_error" in name and name.endswith(".json.gz"):
                    parts = Path(name).name.split("-")
                    if len(parts) >= 2:
                        up = parts[1]
                        members.setdefault(up, {"size": 0})["pae_member"] = name
                continue
            parts = Path(name).name.split("-")
            if len(parts) < 2 or parts[0] != "AF":
                continue
            up = parts[1]
            members.setdefault(up, {"size": 0})["pdb_member"] = name
            members[up]["size"] = m.size

    have_pdb = {up: d for up, d in members.items() if "pdb_member" in d}

    # Apply exclusion filter (e.g., SBE-disjoint sampling for chicken)
    if exclude_uniprots:
        n_before = len(have_pdb)
        have_pdb = {up: d for up, d in have_pdb.items() if up not in exclude_uniprots}
        print(f"[extract] excluded {n_before - len(have_pdb)} UniProts "
              f"(SBE-overlap protection); {len(have_pdb)} remain",
              flush=True)

    print(f"[extract] {len(have_pdb)} proteins with PDB; "
          f"{sum(1 for d in have_pdb.values() if 'pae_member' in d)} with PAE",
          flush=True)

    # Stratified sample by size (=length proxy): bucket into 5 size deciles
    # and sample evenly. Caps total at max_proteins.
    items = sorted(have_pdb.items(), key=lambda kv: kv[1]["size"])
    n = len(items)
    if n <= max_proteins:
        chosen = list(have_pdb.keys())
    else:
        # 5 buckets, take max_proteins/5 from each
        per_bucket = max_proteins // 5
        chosen: list[str] = []
        for b in range(5):
            lo = (n * b) // 5
            hi = (n * (b + 1)) // 5
            bucket = items[lo:hi]
            stride = max(1, len(bucket) // per_bucket)
            chosen.extend(up for up, _ in bucket[::stride][:per_bucket])
        chosen = chosen[:max_proteins]
    print(f"[extract] chose {len(chosen)} proteins (stratified by length)",
          flush=True)

    # Pass 2: extract chosen members.
    chosen_set = set(chosen)
    n_extracted = 0
    n_pae_extracted = 0
    with tarfile.open(tar_path, "r") as tar:
        for m in tar:
            name = m.name
            parts = Path(name).name.split("-")
            if len(parts) < 2 or parts[0] != "AF":
                continue
            up = parts[1]
            if up not in chosen_set:
                continue
            if name.endswith(".pdb.gz"):
                # decompress + write as .pdb
                with tar.extractfile(m) as src:  # type: ignore[union-attr]
                    raw = gzip.decompress(src.read())
                out_pdb = out_dir / f"AF-{up}-F1-model_v6.pdb"
                out_pdb.write_bytes(raw)
                n_extracted += 1
            elif "predicted_aligned_error" in name and name.endswith(".json.gz"):
                if not want_pae:
                    continue
                with tar.extractfile(m) as src:  # type: ignore[union-attr]
                    raw = gzip.decompress(src.read())
                out_pae = out_dir / f"AF-{up}-F1-predicted_aligned_error_v6.json"
                out_pae.write_bytes(raw)
                n_pae_extracted += 1

    dt = time.perf_counter() - t0
    print(f"[extract] wrote {n_extracted} PDBs + {n_pae_extracted} PAEs "
          f"in {dt:.0f}s", flush=True)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="human,mouse,fly",
                    help="comma-separated subset of {human, mouse, fly}")
    ap.add_argument("--max-per-species", type=int, default=None,
                    help="override max_proteins for all species")
    ap.add_argument("--no-pae", action="store_true",
                    help="skip PAE JSONs (saves ~30 percent space)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep the downloaded tar archive")
    args = ap.parse_args()

    species_keys = [s.strip() for s in args.species.split(",")]
    for sp in species_keys:
        if sp not in ORGANISMS:
            raise SystemExit(f"unknown species: {sp}")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for sp in species_keys:
        meta = ORGANISMS[sp]
        max_proteins = (args.max_per_species
                          if args.max_per_species is not None
                          else meta["max_proteins"])
        print(f"\n{'='*70}\n{sp.upper()}: {meta['label']} "
              f"(taxid {meta['taxid']}, target {max_proteins} proteins)\n{'='*70}",
              flush=True)
        sp_dir = DATA_ROOT / sp
        sp_dir.mkdir(parents=True, exist_ok=True)

        # Try each candidate URL in turn until one succeeds.
        last_exc = None
        tar_path = None
        for url in meta["tar_urls"]:
            try:
                tar_path = fetch_tar(url, sp_dir / "_proteome.tar")
                break
            except Exception as exc:
                print(f"[download] {url} failed: {exc}", flush=True)
                last_exc = exc
        if tar_path is None:
            raise RuntimeError(f"all download URLs failed for {sp}: {last_exc}")

        # Salami-slicing protection: load any UniProt-exclusion set
        # configured for this species (used for chicken to stay disjoint
        # from SBE's variant-scoring eval subset).
        exclude_set = load_exclusion_set(meta)

        chosen = extract_proteins(tar_path, sp_dir / "structures",
                                    max_proteins=max_proteins,
                                    want_pae=not args.no_pae,
                                    exclude_uniprots=exclude_set)
        with (sp_dir / "uniprot_ids.txt").open("w") as fh:
            for up in chosen:
                fh.write(up + "\n")
        summary[sp] = {
            "n_proteins": len(chosen),
            "out_dir":    str(sp_dir / "structures"),
        }
        if not args.keep_tar:
            tar_path.unlink()
            print(f"[cleanup] removed {tar_path}", flush=True)

    summary_path = DATA_ROOT / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}")
    for sp, s in summary.items():
        print(f"  {sp}: {s['n_proteins']:5d} proteins  →  {s['out_dir']}")


if __name__ == "__main__":
    main()
