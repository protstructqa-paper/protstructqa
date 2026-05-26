"""Download per-protein PAE (predicted_aligned_error) JSONs from EBI.

The organism-level v6 tarballs in `01_download_proteomes.py` only contain
PDBs and per-residue confidence (pLDDT). PAE matrices are hosted as
separate per-protein files. This script fetches them for every UniProt
listed in `{DATA_ROOT}/{species}/uniprot_ids.txt` and writes them
alongside the existing PDBs in `{DATA_ROOT}/{species}/structures/`.

Source: https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v6.json
License: CC-BY 4.0 (AlphaFold DB).

Cache-aware: if the destination file already exists with non-zero size, the
download is skipped. Uses 8 parallel HTTP threads with retries.

Usage:
    python text_to_structurequery/benchmark/02_download_pae.py
    python text_to_structurequery/benchmark/02_download_pae.py --species human,mouse
    python text_to_structurequery/benchmark/02_download_pae.py --threads 16
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DATA_ROOT = Path(os.environ.get(
    "PROTSTRUCTQA_DATA", "./data"
))
URL_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/"
    "AF-{uniprot}-F1-predicted_aligned_error_v6.json"
)


def download_one(uniprot: str, dest: Path,
                  retries: int = 3, timeout: int = 60) -> tuple[str, str, int]:
    """Returns (uniprot, status, size_bytes). status in {ok, skipped, fail}."""
    if dest.exists() and dest.stat().st_size > 0:
        return uniprot, "skipped", dest.stat().st_size
    url = URL_TEMPLATE.format(uniprot=uniprot)
    last_err = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return uniprot, "ok", len(data)
        except HTTPError as e:
            if e.code == 404:
                # No PAE for this protein (rare; e.g. fragmented entries).
                return uniprot, "not_found", 0
            last_err = e
        except (URLError, TimeoutError) as e:
            last_err = e
        time.sleep(2 ** attempt)
    print(f"  [fail] {uniprot}: {last_err}", flush=True)
    return uniprot, "fail", 0


def download_species(species: str, threads: int) -> dict:
    sp_dir = DATA_ROOT / species
    ids_file = sp_dir / "uniprot_ids.txt"
    out_dir = sp_dir / "structures"
    if not ids_file.exists():
        raise SystemExit(f"missing {ids_file}; run 01_download_proteomes.py first")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [line.strip() for line in ids_file.open() if line.strip()]
    print(f"\n[{species}] {len(ids)} UniProt IDs → {out_dir}", flush=True)

    counts = {"ok": 0, "skipped": 0, "not_found": 0, "fail": 0}
    bytes_dl = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = []
        for up in ids:
            dest = out_dir / f"AF-{up}-F1-predicted_aligned_error_v6.json"
            futs.append(ex.submit(download_one, up, dest))
        for i, f in enumerate(as_completed(futs), 1):
            up, status, sz = f.result()
            counts[status] += 1
            bytes_dl += sz
            if i % 100 == 0 or i == len(ids):
                dt = time.perf_counter() - t0
                rate = i / dt if dt else 0
                print(f"  [{species}] {i}/{len(ids)}  "
                      f"ok={counts['ok']} skip={counts['skipped']} "
                      f"nf={counts['not_found']} fail={counts['fail']}  "
                      f"{bytes_dl/1e6:.1f} MB  {rate:.1f} req/s",
                      flush=True)

    return {"species": species, **counts,
            "bytes_downloaded": bytes_dl,
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="human,mouse,fly",
                    help="comma-separated species subset")
    ap.add_argument("--threads", type=int, default=8,
                    help="parallel HTTP downloads (EBI tolerates ~10-20)")
    args = ap.parse_args()

    species_keys = [s.strip() for s in args.species.split(",")]
    summary = []
    for sp in species_keys:
        summary.append(download_species(sp, threads=args.threads))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_bytes = 0
    for s in summary:
        print(f"  {s['species']:8s}  ok={s['ok']:5d}  skip={s['skipped']:5d}  "
              f"nf={s['not_found']:3d}  fail={s['fail']:3d}  "
              f"{s['bytes_downloaded']/1e9:.2f} GB  "
              f"{s['elapsed_s']:.0f}s")
        total_bytes += s["bytes_downloaded"]
    print(f"\n  TOTAL downloaded: {total_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
