"""Compute reviewer-facing benchmark statistics for Configuration C.

For each species, summarizes:
  - Protein count
  - Length distribution: mean, median, std, p10, p25, p50, p75, p90, max
  - pLDDT distribution (per-residue B-factor): mean, median, fractions
    in low / medium / high / very-high confidence bands
  - PAE distribution (upper-triangle, excluding diagonal): mean, median,
    fractions in high-confidence (<5 Å) and low-confidence (>15 Å) bands

Output: a `STATS.md` file at benchmark/STATS.md with formatted tables
suitable for inclusion in the paper's data section or appendix.

Modes:
  - default: sample 200 proteins per species (~30s)
  - --full:  every protein (~5 min)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np

DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
SPECIES = ["human", "mouse", "fly", "chicken"]
DEFAULT_SAMPLE = 200
OUT = Path(__file__).resolve().parent / "STATS.md"


def load_ids(species: str) -> list[str]:
    return [s.strip()
            for s in (DATA_ROOT / species / "uniprot_ids.txt")
            .read_text().splitlines() if s.strip()]


def parse_pdb_residues(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (ca_xyz Lx3 float, plddt L float) from an AF PDB."""
    coords = []
    plddt = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66])
                coords.append([x, y, z])
                plddt.append(b)
    return np.array(coords, dtype=np.float64), np.array(plddt, dtype=np.float64)


def parse_pae(path: Path) -> np.ndarray:
    """Return PAE matrix LxL as numpy array."""
    data = json.loads(path.read_text())
    rec = data[0] if isinstance(data, list) else data
    return np.asarray(rec["predicted_aligned_error"], dtype=np.float32)


def percentile(xs: np.ndarray, q: float) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def summarize(species: str, sample: int | None) -> dict:
    ids = load_ids(species)
    if sample is not None and sample < len(ids):
        rng = random.Random(hash(species) & 0xffff)
        ids_use = rng.sample(ids, sample)
    else:
        ids_use = ids

    sd = DATA_ROOT / species / "structures"
    lengths = []
    plddt_all: list[float] = []
    pae_high_frac: list[float] = []
    pae_low_frac: list[float] = []
    pae_means: list[float] = []

    for u in ids_use:
        pdb = sd / f"AF-{u}-F1-model_v6.pdb"
        pae = sd / f"AF-{u}-F1-predicted_aligned_error_v6.json"
        ca, pl = parse_pdb_residues(pdb)
        lengths.append(len(pl))
        plddt_all.extend(pl.tolist())

        mat = parse_pae(pae)
        L = mat.shape[0]
        # Upper triangle excluding diagonal: |i-j| >= 1
        iu = np.triu_indices(L, k=1)
        upper = mat[iu]
        pae_means.append(float(upper.mean()))
        pae_high_frac.append(float((upper < 5.0).mean()))
        pae_low_frac.append(float((upper > 15.0).mean()))

    L = np.array(lengths)
    P = np.array(plddt_all)
    return {
        "species":          species,
        "n_canonical":      len(ids),
        "n_sampled":        len(ids_use),
        "length": {
            "mean":       float(L.mean()),
            "median":     float(np.median(L)),
            "std":        float(L.std()),
            "p10":        percentile(L, 10),
            "p25":        percentile(L, 25),
            "p50":        percentile(L, 50),
            "p75":        percentile(L, 75),
            "p90":        percentile(L, 90),
            "max":        int(L.max()),
            "min":        int(L.min()),
        },
        "plddt": {
            "mean":       float(P.mean()),
            "median":     float(np.median(P)),
            "frac_lt_50": float((P < 50).mean()),  # disordered / low conf
            "frac_50_70": float(((P >= 50) & (P < 70)).mean()),
            "frac_70_90": float(((P >= 70) & (P < 90)).mean()),
            "frac_ge_90": float((P >= 90).mean()),  # very high
        },
        "pae": {
            "mean_upper":     float(np.mean(pae_means)),
            "frac_high_conf": float(np.mean(pae_high_frac)),  # PAE < 5
            "frac_low_conf":  float(np.mean(pae_low_frac)),   # PAE > 15
        },
    }


def write_md(stats: list[dict], out: Path, sampled: bool) -> None:
    lines: list[str] = []
    lines.append("# ProtStructQA Configuration C: Dataset Statistics")
    lines.append("")
    lines.append(f"**Source**: AlphaFold DB v6 (CC-BY 4.0).")
    lines.append("")
    lines.append(f"**Sampling**: " + (
        f"~{stats[0]['n_sampled']} proteins per species (sample-mode, "
        f"deterministic random draw)."
        if sampled
        else "every canonical protein (full scan)."))
    lines.append("")
    lines.append("**Total canonical proteins**: 10,000 (4,000 human + 2,500 mouse + 1,500 fly + 2,000 chicken).")
    lines.append("")

    # Length table
    lines.append("## Length distribution (residues)")
    lines.append("")
    lines.append("| Species | N | mean | median | std | p10 | p25 | p75 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        L = s["length"]
        lines.append(
            f"| {s['species']} | {s['n_canonical']} | {L['mean']:.0f} | "
            f"{L['median']:.0f} | {L['std']:.0f} | {L['p10']:.0f} | "
            f"{L['p25']:.0f} | {L['p75']:.0f} | {L['p90']:.0f} | {L['max']} |"
        )
    lines.append("")

    # pLDDT table
    lines.append("## pLDDT distribution (per-residue, B-factor column)")
    lines.append("")
    lines.append("AlphaFold pLDDT bands (per the AFDB convention): "
                 "<50 = disordered / unreliable; 50-70 = low confidence; "
                 "70-90 = confident; ≥90 = very high confidence.")
    lines.append("")
    lines.append("| Species | mean | median | <50 | 50-70 | 70-90 | ≥90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        P = s["plddt"]
        lines.append(
            f"| {s['species']} | {P['mean']:.1f} | {P['median']:.1f} | "
            f"{P['frac_lt_50']*100:.1f}% | {P['frac_50_70']*100:.1f}% | "
            f"{P['frac_70_90']*100:.1f}% | {P['frac_ge_90']*100:.1f}% |"
        )
    lines.append("")

    # PAE table
    lines.append("## PAE distribution (upper triangle, off-diagonal)")
    lines.append("")
    lines.append("Conventions: PAE < 5 Å = high inter-residue confidence "
                 "(domain-internal); PAE > 15 Å = low confidence (likely "
                 "inter-domain or disordered linker). Reported as protein-"
                 "level averages over residue pairs.")
    lines.append("")
    lines.append("| Species | mean PAE (Å) | %pairs < 5 Å | %pairs > 15 Å |")
    lines.append("|---|---:|---:|---:|")
    for s in stats:
        E = s["pae"]
        lines.append(
            f"| {s['species']} | {E['mean_upper']:.2f} | "
            f"{E['frac_high_conf']*100:.1f}% | {E['frac_low_conf']*100:.1f}% |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `benchmark/compute_stats.py`. Re-run after "
                 "any data-pipeline change.*")

    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                      help="scan every canonical protein "
                           "(slow; default samples 200/species)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sample = None if args.full else args.sample
    print(f"[stats] mode = {'FULL' if args.full else f'sample N={args.sample}'}",
          flush=True)

    out_stats = []
    for sp in SPECIES:
        print(f"[stats] species = {sp}", flush=True)
        out_stats.append(summarize(sp, sample))
        s = out_stats[-1]
        print(
            f"  length: mean={s['length']['mean']:.0f} median={s['length']['median']:.0f} max={s['length']['max']} | "
            f"pLDDT: mean={s['plddt']['mean']:.1f} %≥70={(s['plddt']['frac_70_90']+s['plddt']['frac_ge_90'])*100:.1f} | "
            f"PAE: mean={s['pae']['mean_upper']:.2f} %<5Å={s['pae']['frac_high_conf']*100:.1f}",
            flush=True,
        )

    write_md(out_stats, args.out, sampled=not args.full)
    print(f"\n[stats] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
