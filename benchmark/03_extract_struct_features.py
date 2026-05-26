"""Per-protein structural feature extraction for ProtStructQA.

For every canonical UniProt in Configuration C, produces an NPZ at
{species}/features/AF-{up}.npz containing:

    seq           (str L)               : 1-letter amino-acid sequence
    residue_nums  (uint16 L)             : PDB residue numbers
    ca_xyz        (float32 L×3)          : CA atom coordinates (Å)
    plddt         (float32 L)            : pLDDT (B-factor column)
    ss3           (uint8 L)              : secondary structure: 0=H, 1=E, 2=C
    sasa          (float32 L)            : solvent-accessible surface (Å²)
    pae           (uint8 L×L)            : PAE matrix (integer-quantized)
    backbone_xyz  (float32 L×4×3)        : N, CA, C, O atoms (for SS recompute)

Distance matrices, contact matrices, and neighbor lists are NOT stored;
they are derived on-demand from `ca_xyz` because (a) they double the
storage cost and (b) the question generator caches what it needs.

Toolchain:
    - PDB parsing: BioPython 1.87 + manual fallback
    - PAE parsing: stdlib json
    - Secondary structure: pydssp (pure-Python DSSP, torch backend)
    - SASA: freesasa (C library via Python bindings)

Usage:
    # Smoke test on a small subset
    python benchmark/03_extract_struct_features.py --species human --limit 100

    # Full extraction
    python benchmark/03_extract_struct_features.py
    # → ~17 min sequential, ~5 min with --workers 8

Tests: benchmark/tests/test_extract_features.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# pydssp / freesasa imports are deferred into the functions that need them
# so the import-time cost is isolated when running tests.

DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
SPECIES = ["human", "mouse", "fly", "chicken"]

# 3-letter to 1-letter amino acid mapping. Non-standard residues map to 'X'.
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

SS_INDEX = {"H": 0, "E": 1, "C": 2}  # ss3 encoding


# ============================ pure parsers ============================== #

def parse_pdb(pdb_path: Path) -> dict:
    """Parse an AlphaFold v6 PDB: return per-residue features.

    AlphaFold PDBs are reliably formatted:
      - One chain ('A')
      - Residues numbered contiguously from 1
      - Each residue has standard backbone atoms (N, CA, C, O) + side chain
      - B-factor column holds pLDDT (0-100)

    This parser uses the fixed-column PDB layout for speed (~5ms per
    protein): BioPython's parser is correct but ~50ms.

    Returns a dict with: seq, residue_nums, ca_xyz, plddt, backbone_xyz.
    """
    seq_chars: list[str] = []
    residue_nums: list[int] = []
    ca_xyz: list[list[float]] = []
    plddt: list[float] = []
    # backbone[res] = {'N': xyz, 'CA': xyz, 'C': xyz, 'O': xyz}
    backbone: dict[int, dict[str, list[float]]] = {}
    res_name_by_num: dict[int, str] = {}

    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "O"):
                continue
            res_name = line[17:20].strip()
            res_num = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if res_num not in backbone:
                backbone[res_num] = {}
                res_name_by_num[res_num] = res_name
            backbone[res_num][atom_name] = [x, y, z]

            if atom_name == "CA":
                bf = float(line[60:66])
                ca_xyz.append([x, y, z])
                plddt.append(bf)
                residue_nums.append(res_num)
                seq_chars.append(AA3_TO_1.get(res_name, "X"))

    L = len(residue_nums)
    bb_arr = np.zeros((L, 4, 3), dtype=np.float32)
    for i, rnum in enumerate(residue_nums):
        for j, aname in enumerate(("N", "CA", "C", "O")):
            xyz = backbone[rnum].get(aname)
            if xyz is None:
                # Fallback: copy CA so pydssp doesn't crash on missing O
                xyz = ca_xyz[i]
            bb_arr[i, j] = xyz

    return {
        "seq":          "".join(seq_chars),
        "residue_nums": np.array(residue_nums, dtype=np.uint16),
        "ca_xyz":       np.array(ca_xyz, dtype=np.float32),
        "plddt":        np.array(plddt, dtype=np.float32),
        "backbone_xyz": bb_arr,
    }


def parse_pae(pae_path: Path) -> np.ndarray:
    """AFDB PAE is integer-quantized; cast to uint8 for storage savings."""
    data = json.loads(pae_path.read_text())
    rec = data[0] if isinstance(data, list) else data
    mat = np.asarray(rec["predicted_aligned_error"], dtype=np.float32)
    # AFDB max is ~32; uint8 (0-255) is safe and halves storage.
    if (mat < 0).any() or (mat > 255).any():
        raise ValueError("PAE out of uint8 range")
    return mat.astype(np.uint8)


# ============================ feature computers ========================= #

def compute_ss3(backbone_xyz: np.ndarray) -> np.ndarray:
    """3-state secondary structure via pydssp. Returns uint8 array of
    length L with values 0=H, 1=E, 2=C."""
    import pydssp
    # pydssp expects (L, 4, 3) with N, CA, C, O: exactly our backbone_xyz.
    ss_chars = pydssp.assign(backbone_xyz, out_type="c3")
    # pydssp uses '-' for coil; remap to 'C' then to indices.
    out = np.zeros(len(ss_chars), dtype=np.uint8)
    for i, c in enumerate(ss_chars):
        if c == "H":
            out[i] = 0
        elif c == "E":
            out[i] = 1
        else:  # '-' or anything else
            out[i] = 2
    return out


_FREESASA_QUIET = False


def compute_sasa(pdb_path: Path, n_residues: int) -> np.ndarray:
    """Per-residue SASA via freesasa (Shrake-Rupley). Returns float32 (L,)."""
    import freesasa
    global _FREESASA_QUIET
    if not _FREESASA_QUIET:
        freesasa.setVerbosity(freesasa.silent)
        _FREESASA_QUIET = True

    struct = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(struct)
    residues = result.residueAreas()
    chain = next(iter(residues))  # AlphaFold uses single chain 'A'
    res_areas = residues[chain]
    # Sort residue keys numerically (they come as strings)
    keys = sorted(res_areas.keys(), key=lambda x: int(x))
    out = np.array([res_areas[k].total for k in keys], dtype=np.float32)
    if len(out) != n_residues:
        # Defensive: this should never happen for AFDB single-chain PDBs.
        raise ValueError(
            f"SASA count {len(out)} != PDB residue count {n_residues}"
        )
    return out


# ============================ I/O ====================================== #

def save_features(feats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        seq=np.array(list(feats["seq"]) if isinstance(feats["seq"], str)
                       else feats["seq"]),
        residue_nums=feats["residue_nums"].astype(np.uint16),
        ca_xyz=feats["ca_xyz"].astype(np.float32),
        plddt=feats["plddt"].astype(np.float32),
        ss3=feats["ss3"].astype(np.uint8),
        sasa=feats["sasa"].astype(np.float32),
        pae=feats["pae"].astype(np.uint8),
    )


def load_features(in_path: Path) -> dict:
    z = np.load(in_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ============================ pipeline ================================= #

def extract_one(uniprot: str, struct_dir: Path, out_dir: Path) -> bool:
    """Full pipeline for one UniProt. Returns True on success.

    Idempotent: if {out_dir}/AF-{up}.npz already exists with non-zero
    size, skips computation and returns True.
    """
    out_path = out_dir / f"AF-{uniprot}.npz"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    pdb_path = struct_dir / f"AF-{uniprot}-F1-model_v6.pdb"
    pae_path = struct_dir / f"AF-{uniprot}-F1-predicted_aligned_error_v6.json"

    try:
        pdb = parse_pdb(pdb_path)
        L = len(pdb["seq"])

        pae = parse_pae(pae_path)
        if pae.shape != (L, L):
            raise ValueError(
                f"{uniprot}: PAE shape {pae.shape} != (L,L)=({L},{L})"
            )

        ss3 = compute_ss3(pdb["backbone_xyz"])
        sasa = compute_sasa(pdb_path, n_residues=L)

        feats = {
            "seq":          pdb["seq"],
            "residue_nums": pdb["residue_nums"],
            "ca_xyz":       pdb["ca_xyz"],
            "plddt":        pdb["plddt"],
            "ss3":          ss3,
            "sasa":         sasa,
            "pae":          pae,
        }
        save_features(feats, out_path)
        return True
    except Exception as e:
        print(f"[fail] {uniprot}: {type(e).__name__}: {e}", flush=True)
        return False


def _worker(args):
    """Top-level so it pickles cleanly for ProcessPoolExecutor."""
    uniprot, struct_dir, out_dir = args
    return uniprot, extract_one(uniprot, Path(struct_dir), Path(out_dir))


def extract_species(species: str, limit: int | None = None,
                       workers: int = 1) -> dict:
    sp_dir = DATA_ROOT / species
    struct_dir = sp_dir / "structures"
    out_dir = sp_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [s.strip() for s in (sp_dir / "uniprot_ids.txt").read_text().splitlines()
            if s.strip()]
    if limit is not None:
        ids = ids[:limit]

    print(f"\n=== {species}: extracting features for {len(ids)} proteins "
          f"(workers={workers}) ===", flush=True)

    counts = {"ok": 0, "skipped": 0, "fail": 0}
    t0 = time.perf_counter()

    if workers == 1:
        for i, up in enumerate(ids, 1):
            ok = extract_one(up, struct_dir, out_dir)
            counts["ok" if ok else "fail"] += 1
            if i % 100 == 0:
                dt = time.perf_counter() - t0
                rate = i / dt
                eta = (len(ids) - i) / rate
                print(f"  [{species}] {i}/{len(ids)}  "
                      f"ok={counts['ok']} fail={counts['fail']}  "
                      f"{rate:.1f} proteins/s  eta={eta:.0f}s", flush=True)
    else:
        # Use ProcessPoolExecutor for CPU-bound work
        tasks = [(up, str(struct_dir), str(out_dir)) for up in ids]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            done = 0
            for f in as_completed(futures):
                up, ok = f.result()
                counts["ok" if ok else "fail"] += 1
                done += 1
                if done % 100 == 0:
                    dt = time.perf_counter() - t0
                    rate = done / dt
                    eta = (len(ids) - done) / rate
                    print(f"  [{species}] {done}/{len(ids)}  "
                          f"ok={counts['ok']} fail={counts['fail']}  "
                          f"{rate:.1f} proteins/s  eta={eta:.0f}s",
                          flush=True)

    dt = time.perf_counter() - t0
    print(f"  [{species}] DONE  ok={counts['ok']} fail={counts['fail']}  "
          f"{dt:.0f}s  ({len(ids)/dt:.1f} proteins/s)", flush=True)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None,
                      help="comma-separated subset (default: all 4)")
    ap.add_argument("--limit", type=int, default=None,
                      help="cap proteins per species (smoke test)")
    ap.add_argument("--workers", type=int, default=1,
                      help="number of parallel processes")
    args = ap.parse_args()

    species_keys = (args.species.split(",") if args.species
                       else SPECIES)
    species_keys = [s.strip() for s in species_keys]
    bad = [s for s in species_keys if s not in SPECIES]
    if bad:
        raise SystemExit(f"unknown species: {bad}")

    overall = {"ok": 0, "fail": 0}
    for sp in species_keys:
        c = extract_species(sp, limit=args.limit, workers=args.workers)
        overall["ok"] += c["ok"]
        overall["fail"] += c["fail"]

    print(f"\n=== TOTAL  ok={overall['ok']}  fail={overall['fail']} ===",
          flush=True)
    if overall["fail"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
