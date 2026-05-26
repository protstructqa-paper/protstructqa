"""ProteinView: read-only wrapper around an AlphaFold prediction.

Loads from either:
  - a precomputed feature parquet row + a PAE matrix file (fast path)
  - or a raw .pdb + .pae JSON (slower fallback)

Used by `dsl.executor` to answer all primitive queries.

All residue indices are 1-indexed (matching ClinVar / UniProt convention)
externally; internally we store 0-indexed numpy arrays for efficiency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Side-chain physico-chemistry tables (kept private; primitives operate
# on them through the public methods below).
_AA_VOLUME = {
    "G":  60.1, "A":  88.6, "V": 140.0, "L": 166.7, "I": 166.7, "P": 112.7,
    "F": 189.9, "W": 227.8, "M": 162.9, "C": 108.5, "S":  89.0, "T": 116.1,
    "Y": 193.6, "N": 114.1, "Q": 143.8, "D": 111.1, "E": 138.4, "K": 168.5,
    "R": 173.4, "H": 153.2,
}


# Module-level cache for per-protein KDTree-derived neighbor lists at 8Å.
# Avoids recomputing the O(n²) pairwise distance matrix every time the DSL
# evaluates contact_density / long_range_contacts.
_NBR8_CACHE: dict[tuple[str, str], list[list[int]]] = {}


def _neighbor_list_8A(view: "ProteinView") -> list[list[int]]:
    """Lazy KDTree neighbor list: nbrs[i] = sorted residue-0-idx neighbors
    of i within 8 Å Cα distance (excludes i itself). Cached per protein."""
    key = (view.species, view.uniprot)
    cached = _NBR8_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.spatial import cKDTree
    tree = cKDTree(view.ca_xyz)
    pairs = tree.query_pairs(r=8.0, output_type="set")
    n = view.n_residues
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i, j in pairs:
        nbrs[i].append(j)
        nbrs[j].append(i)
    for lst in nbrs:
        lst.sort()
    _NBR8_CACHE[key] = nbrs
    return nbrs


@dataclass(frozen=True)
class ProteinView:
    """Immutable view over a single AlphaFold prediction.

    Residue indices are passed in/out as 1-indexed. Internal numpy arrays
    are 0-indexed.
    """
    uniprot:     str
    species:    str
    n_residues: int
    ref_aa:     np.ndarray         # shape (N,), dtype <U1
    plddt:      np.ndarray         # shape (N,), float32
    sasa:       np.ndarray         # shape (N,), float32 (absolute Å²)
    rel_sasa:   np.ndarray         # shape (N,), float32 ([0, ~1])
    n_neigh:    np.ndarray         # shape (N,), int32 (8 Å)
    ss_3:       np.ndarray         # shape (N,), dtype <U1, "H"/"E"/"C"
    ca_xyz:     np.ndarray         # shape (N, 3), float32: Cα coords (Å)
    pae:        np.ndarray | None  # shape (N, N), float32, or None if absent

    # ----- residue-index helpers (1-indexed external, 0-indexed internal) ----

    def _idx(self, residue_1based: int) -> int:
        if not (1 <= residue_1based <= self.n_residues):
            raise IndexError(
                f"residue {residue_1based} out of range "
                f"[1, {self.n_residues}] for {self.uniprot}"
            )
        return residue_1based - 1

    def _idx_unsafe(self, residue_1based: int) -> int:
        """Like _idx but returns -1 for out-of-range instead of raising."""
        if 1 <= residue_1based <= self.n_residues:
            return residue_1based - 1
        return -1

    def all_residues(self) -> Iterator[int]:
        return iter(range(1, self.n_residues + 1))

    # ----- per-residue primitives -----

    def plddt_at(self, r: int) -> float:
        return float(self.plddt[self._idx(r)])

    def ref_aa_at(self, r: int) -> str:
        return str(self.ref_aa[self._idx(r)])

    def ss_at(self, r: int) -> str:
        return str(self.ss_3[self._idx(r)])

    def sasa_at(self, r: int) -> float:
        return float(self.sasa[self._idx(r)])

    def rel_sasa_at(self, r: int) -> float:
        return float(self.rel_sasa[self._idx(r)])

    def n_neighbors_at(self, r: int, radius: float = 8.0) -> int:
        if abs(radius - 8.0) < 1e-6:
            return int(self.n_neigh[self._idx(r)])
        # General-radius case: recompute from coords.
        i = self._idx(r)
        d = np.linalg.norm(self.ca_xyz - self.ca_xyz[i], axis=1)
        # exclude self; count residues within radius
        return int(np.sum((d > 0) & (d <= radius)))

    def ca_xyz_at(self, r: int) -> tuple[float, float, float]:
        i = self._idx(r)
        return tuple(map(float, self.ca_xyz[i]))

    # ----- per-pair primitives -----

    def distance(self, r1: int, r2: int) -> float:
        i, j = self._idx(r1), self._idx(r2)
        return float(np.linalg.norm(self.ca_xyz[i] - self.ca_xyz[j]))

    def seq_separation(self, r1: int, r2: int) -> int:
        return abs(r1 - r2)

    def pae_at(self, r1: int, r2: int) -> float:
        if self.pae is None:
            raise ValueError(f"PAE not loaded for {self.uniprot}")
        i, j = self._idx(r1), self._idx(r2)
        return float(self.pae[i, j])

    # ----- per-region primitives -----

    def _region_slice(self, start: int, end: int) -> slice:
        if start < 1 or end > self.n_residues or start > end:
            raise IndexError(
                f"region [{start}, {end}] out of range for {self.uniprot} "
                f"(n_residues={self.n_residues})"
            )
        return slice(start - 1, end)

    def mean_plddt(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.plddt[s].mean())

    def min_plddt(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.plddt[s].min())

    def max_plddt(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.plddt[s].max())

    def std_plddt(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.plddt[s].std())

    def mean_pae(self, a_start: int, a_end: int,
                  b_start: int, b_end: int) -> float:
        if self.pae is None:
            raise ValueError(f"PAE not loaded for {self.uniprot}")
        sa = self._region_slice(a_start, a_end)
        sb = self._region_slice(b_start, b_end)
        return float(self.pae[sa, sb].mean())

    def max_pae(self, a_start: int, a_end: int,
                  b_start: int, b_end: int) -> float:
        """Maximum PAE in the inter-region block [a_start..a_end] x
        [b_start..b_end]. Parallels mean_pae; useful as a worst-case
        confidence indicator (e.g., template C3)."""
        if self.pae is None:
            raise ValueError(f"PAE not loaded for {self.uniprot}")
        sa = self._region_slice(a_start, a_end)
        sb = self._region_slice(b_start, b_end)
        return float(self.pae[sa, sb].max())

    def count_high_pae(self, a_start: int, a_end: int,
                          b_start: int, b_end: int,
                          threshold: float) -> int:
        """Count PAE entries strictly above `threshold` in the
        inter-region block [a_start..a_end] x [b_start..b_end]. Useful
        for template C4 (uncertainty volume across an inter-domain
        block)."""
        if self.pae is None:
            raise ValueError(f"PAE not loaded for {self.uniprot}")
        sa = self._region_slice(a_start, a_end)
        sb = self._region_slice(b_start, b_end)
        return int((self.pae[sa, sb] > float(threshold)).sum())

    def contact_density(self, start: int, end: int,
                         radius: float = 8.0) -> float:
        s = self._region_slice(start, end)
        n = s.stop - s.start
        if n < 2:
            return 0.0
        n_pairs = n * (n - 1) // 2
        if abs(radius - 8.0) < 1e-6:
            # Fast path: use cached KDTree neighbor list at 8 Å.
            nbrs = _neighbor_list_8A(self)
            lo, hi = s.start, s.stop  # 0-idx half-open
            n_contacts = 0
            for i in range(lo, hi):
                # Each pair (i, j) counted once when j > i.
                for j in nbrs[i]:
                    if j > i and j < hi:
                        n_contacts += 1
            return n_contacts / n_pairs
        # General-radius fallback: vectorized upper-triangle distances.
        coords = self.ca_xyz[s]
        ii, jj = np.triu_indices(n, k=1)
        d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
        n_contacts = int(np.sum(d <= radius))
        return n_contacts / n_pairs

    def long_range_contacts(self, start: int, end: int,
                              sep: int = 12, radius: float = 8.0) -> int:
        s = self._region_slice(start, end)
        n = s.stop - s.start
        if n < 2:
            return 0
        if abs(radius - 8.0) < 1e-6:
            # Fast path via cached neighbor list. `sep` is residue
            # separation in the region's own frame, so we threshold on
            # (j - i) directly within the local index window.
            nbrs = _neighbor_list_8A(self)
            lo, hi = s.start, s.stop
            n_contacts = 0
            for i in range(lo, hi):
                for j in nbrs[i]:
                    if j > i and j < hi and (j - i) > sep:
                        n_contacts += 1
            return n_contacts
        # General-radius fallback: upper-triangle distances + sep mask.
        coords = self.ca_xyz[s]
        ii, jj = np.triu_indices(n, k=1)
        d = np.linalg.norm(coords[ii] - coords[jj], axis=1)
        seps = jj - ii
        return int(np.sum((d <= radius) & (seps > sep)))

    def radius_of_gyration(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        coords = self.ca_xyz[s]
        if coords.shape[0] == 0:
            raise ValueError("empty region for radius_of_gyration")
        center = coords.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))

    # ----- protein-level primitives -----

    def length(self) -> int:
        return self.n_residues

    def n_helices(self) -> int:
        # Count contiguous runs of "H".
        return self._n_runs("H")

    def n_strands(self) -> int:
        return self._n_runs("E")

    def _n_runs(self, label: str) -> int:
        runs = 0
        prev = None
        for c in self.ss_3:
            if c == label and prev != label:
                runs += 1
            prev = c
        return runs

    def mean_protein_plddt(self) -> float:
        return float(self.plddt.mean())

    # ----- additional region primitives -----

    def mean_rel_sasa(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.rel_sasa[s].mean())

    def mean_n_neighbors(self, start: int, end: int) -> float:
        s = self._region_slice(start, end)
        return float(self.n_neigh[s].mean())

    def ss_runs(self, label: str) -> list[tuple[int, int]]:
        """Return contiguous runs of `label` ('H' / 'E' / 'C') as
        1-indexed (start, end) inclusive tuples."""
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for i, c in enumerate(self.ss_3):
            if c == label:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    runs.append((run_start + 1, i))
                    run_start = None
        if run_start is not None:
            runs.append((run_start + 1, len(self.ss_3)))
        return runs


# -----------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------

def load_from_feature_parquet(uniprot: str, species: str,
                               feature_parquet_path: str | Path,
                               pae_dir: str | Path | None = None) -> ProteinView:
    """Build a ProteinView for `uniprot` from a precomputed feature parquet
    (one row per residue) plus optional PAE matrix loaded from a JSON.

    The parquet schema must contain at least:
        uniprot, residue, plddt, ss_3state, sasa, rel_sasa, ref_aa,
        n_neighbors_8A, ca_x, ca_y, ca_z

    If `ca_x/y/z` are absent we fall back to loading the .pdb structure.
    """
    import pandas as pd

    df = pd.read_parquet(feature_parquet_path,
                          filters=[("uniprot", "=", uniprot)])
    if df.empty:
        raise ValueError(f"No rows for uniprot={uniprot} in {feature_parquet_path}")
    df = df.sort_values("residue").reset_index(drop=True)
    n = int(df["residue"].max())

    plddt    = np.zeros(n, dtype=np.float32)
    sasa     = np.zeros(n, dtype=np.float32)
    rel_sasa = np.zeros(n, dtype=np.float32)
    n_neigh  = np.zeros(n, dtype=np.int32)
    ss_3     = np.full(n, "C", dtype="<U1")
    ref_aa   = np.full(n, "X", dtype="<U1")
    ca_xyz   = np.zeros((n, 3), dtype=np.float32)

    has_xyz = all(c in df.columns for c in ("ca_x", "ca_y", "ca_z"))

    for _, r in df.iterrows():
        i = int(r["residue"]) - 1
        if not (0 <= i < n):
            continue
        plddt[i]    = float(r["plddt"])
        sasa[i]     = float(r["sasa"])
        rel_sasa[i] = float(r["rel_sasa"])
        n_neigh[i]  = int(r["n_neighbors_8A"])
        ss_3[i]     = str(r["ss_3state"])[:1]
        ref_aa[i]   = str(r["ref_aa"])[:1]
        if has_xyz:
            ca_xyz[i] = (float(r["ca_x"]), float(r["ca_y"]), float(r["ca_z"]))

    if not has_xyz:
        raise ValueError(
            f"feature parquet at {feature_parquet_path} lacks ca_x/y/z; "
            "PDB-fallback loading not implemented yet"
        )

    pae = None
    if pae_dir is not None:
        pae_path = Path(pae_dir) / f"AF-{uniprot}-F1-predicted_aligned_error_v6.json"
        if pae_path.exists():
            with pae_path.open() as fh:
                d = json.load(fh)
            if isinstance(d, list):
                d = d[0]
            pae = np.asarray(d["predicted_aligned_error"], dtype=np.float32)
            # Fold to (n, n) if not already
            if pae.shape != (n, n):
                # resize / pad if mismatched
                m = min(pae.shape[0], n)
                tmp = np.zeros((n, n), dtype=np.float32)
                tmp[:m, :m] = pae[:m, :m]
                pae = tmp

    return ProteinView(
        uniprot=uniprot, species=species, n_residues=n,
        ref_aa=ref_aa, plddt=plddt, sasa=sasa, rel_sasa=rel_sasa,
        n_neigh=n_neigh, ss_3=ss_3, ca_xyz=ca_xyz, pae=pae,
    )


def load_from_npz(npz_path: str | Path, uniprot: str, species: str) -> ProteinView:
    """Build a ProteinView from a Phase-1 feature NPZ produced by
    `benchmark/03_extract_struct_features.py`. Expected keys:
        seq (str/array of <U1, length L)
        residue_nums (uint16, length L)
        ca_xyz (float32, L×3)
        plddt (float32, L)
        ss3 (uint8, L; 0=H, 1=E, 2=C)
        sasa (float32, L)
        pae (uint8, L×L)

    Computes rel_sasa from absolute SASA via per-residue Tien et al. 2013
    maximum-SASA values, and n_neigh from ca_xyz at 8 Å.
    """
    z = np.load(npz_path, allow_pickle=False)
    seq_arr = z["seq"]
    seq = "".join(seq_arr.astype(str).tolist())
    L = len(seq)

    plddt = z["plddt"].astype(np.float32)
    sasa = z["sasa"].astype(np.float32)
    ss3_idx = z["ss3"].astype(np.uint8)
    ca_xyz = z["ca_xyz"].astype(np.float32)
    pae = z["pae"].astype(np.float32) if z["pae"].size else None

    # ss3: 0=H, 1=E, 2=C
    ss_3 = np.full(L, "C", dtype="<U1")
    ss_3[ss3_idx == 0] = "H"
    ss_3[ss3_idx == 1] = "E"
    # 2 stays as "C" (default)

    # rel_sasa = sasa / max_sasa(aa) per Tien et al. 2013 theoretical max
    # (numerically clamped to [0, 1.5] to absorb rare overestimates).
    _MAX_SASA_TIEN = {
        "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
        "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
        "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
        "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
        "X": 200.0,
    }
    max_per_res = np.array(
        [_MAX_SASA_TIEN.get(aa, 200.0) for aa in seq],
        dtype=np.float32,
    )
    rel_sasa = np.clip(sasa / np.maximum(max_per_res, 1e-3), 0.0, 1.5).astype(np.float32)

    # n_neigh at 8 Å, computed once.
    diff = ca_xyz[:, None, :] - ca_xyz[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    n_neigh = (d <= 8.0).sum(axis=1).astype(np.int32)

    ref_aa = np.array(list(seq), dtype="<U1")

    return ProteinView(
        uniprot=uniprot, species=species, n_residues=L,
        ref_aa=ref_aa, plddt=plddt, sasa=sasa, rel_sasa=rel_sasa,
        n_neigh=n_neigh, ss_3=ss_3, ca_xyz=ca_xyz, pae=pae,
    )


def load_from_pdb(pdb_path: str | Path,
                  pae_path: str | Path | None = None,
                  uniprot: str = "?", species: str = "?") -> ProteinView:
    """Slow path: parse a .pdb file directly. Used for unit tests on small
    structures where we don't have a precomputed parquet yet."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(uniprot, str(pdb_path))
    model = next(structure.get_models())
    chain = next(model.get_chains())

    residues = []
    plddt_list = []
    ref_aa_list = []
    ca_xyz_list = []

    AA_3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    for res in chain:
        if res.id[0] != " ":  # heteroatoms
            continue
        rname = res.get_resname()
        if rname not in AA_3TO1:
            continue
        if "CA" not in res:
            continue
        ca = res["CA"]
        residues.append(res.id[1])
        plddt_list.append(float(ca.bfactor))
        ref_aa_list.append(AA_3TO1[rname])
        ca_xyz_list.append([float(c) for c in ca.coord])

    if not residues:
        raise ValueError(f"no resolved residues in {pdb_path}")

    n = max(residues)
    plddt    = np.zeros(n, dtype=np.float32)
    ref_aa   = np.full(n, "X", dtype="<U1")
    ca_xyz   = np.zeros((n, 3), dtype=np.float32)
    sasa     = np.zeros(n, dtype=np.float32)
    rel_sasa = np.zeros(n, dtype=np.float32)
    n_neigh  = np.zeros(n, dtype=np.int32)
    ss_3     = np.full(n, "C", dtype="<U1")

    for r, p, aa, xyz in zip(residues, plddt_list, ref_aa_list, ca_xyz_list):
        i = r - 1
        if 0 <= i < n:
            plddt[i] = p
            ref_aa[i] = aa
            ca_xyz[i] = xyz

    # Quick neighbor count (within 8 Å of Cα).
    diff = ca_xyz[:, None, :] - ca_xyz[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    n_neigh = (d <= 8.0).sum(axis=1).astype(np.int32)

    pae = None
    if pae_path is not None:
        with Path(pae_path).open() as fh:
            d_json = json.load(fh)
        if isinstance(d_json, list):
            d_json = d_json[0]
        pae = np.asarray(d_json["predicted_aligned_error"], dtype=np.float32)

    return ProteinView(
        uniprot=uniprot, species=species, n_residues=n,
        ref_aa=ref_aa, plddt=plddt, sasa=sasa, rel_sasa=rel_sasa,
        n_neigh=n_neigh, ss_3=ss_3, ca_xyz=ca_xyz, pae=pae,
    )
