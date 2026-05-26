"""Verify KDTree-based neighbor-list fast path for contact_density and
long_range_contacts is numerically equivalent to the original O(n²)
matrix path. Also confirms the cache is keyed correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from dsl.protein_view import (
    ProteinView,
    _NBR8_CACHE,
    _neighbor_list_8A,
)


def _make_synthetic_view(n: int, seed: int = 0,
                          uniprot: str = "TEST",
                          species: str = "synthetic") -> ProteinView:
    """Build a synthetic ProteinView with random Cα coords inside a
    moderately dense box, so plenty of pairs fall within 8 Å."""
    rng = np.random.default_rng(seed)
    # Place residues in a 25 Å cube → expect ~5–15 neighbors per residue
    # within 8 Å for moderate n. Tune box size to hit that density.
    box = 25.0 * (n / 50.0) ** (1 / 3)
    ca = rng.uniform(0.0, box, size=(n, 3)).astype(np.float32)
    return ProteinView(
        uniprot=uniprot,
        species=species,
        n_residues=n,
        ref_aa=np.full(n, "A", dtype="<U1"),
        plddt=np.full(n, 90.0, dtype=np.float32),
        sasa=np.zeros(n, dtype=np.float32),
        rel_sasa=np.zeros(n, dtype=np.float32),
        n_neigh=np.zeros(n, dtype=np.int32),
        ss_3=np.full(n, "C", dtype="<U1"),
        ca_xyz=ca,
        pae=None,
    )


def _reference_contact_density(view: ProteinView, start: int, end: int,
                                  radius: float) -> float:
    """O(n²) implementation: used as ground truth for comparison."""
    s = view._region_slice(start, end)
    coords = view.ca_xyz[s]
    n = coords.shape[0]
    if n < 2:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    n_pairs = n * (n - 1) // 2
    n_contacts = int(np.sum(d <= radius)) // 2
    return n_contacts / n_pairs


def _reference_long_range(view: ProteinView, start: int, end: int,
                            sep: int, radius: float) -> int:
    s = view._region_slice(start, end)
    coords = view.ca_xyz[s]
    n = coords.shape[0]
    if n < 2:
        return 0
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    mask = (d <= radius) & (np.abs(ii - jj) > sep)
    return int(np.sum(mask)) // 2


@pytest.fixture(autouse=True)
def _clear_cache_each_test():
    _NBR8_CACHE.clear()
    yield
    _NBR8_CACHE.clear()


def test_contact_density_8A_matches_reference():
    view = _make_synthetic_view(n=120, seed=42)
    for (s, e) in [(1, 30), (10, 80), (1, 120), (50, 60)]:
        fast = view.contact_density(s, e, radius=8.0)
        ref = _reference_contact_density(view, s, e, radius=8.0)
        assert abs(fast - ref) < 1e-9, f"region [{s},{e}]: fast={fast} ref={ref}"


def test_contact_density_general_radius_matches_reference():
    """Non-8 Å radii take the general-radius fallback (upper-triangle
    pdist). Verify it still matches the O(n²) reference."""
    view = _make_synthetic_view(n=80, seed=11)
    for r in (5.0, 10.0, 12.0):
        fast = view.contact_density(1, 80, radius=r)
        ref = _reference_contact_density(view, 1, 80, radius=r)
        assert abs(fast - ref) < 1e-9, f"radius={r}: fast={fast} ref={ref}"


def test_long_range_contacts_8A_matches_reference():
    view = _make_synthetic_view(n=120, seed=7)
    for (s, e, sep) in [(1, 120, 12), (1, 60, 6), (30, 90, 20)]:
        fast = view.long_range_contacts(s, e, sep=sep, radius=8.0)
        ref = _reference_long_range(view, s, e, sep=sep, radius=8.0)
        assert fast == ref, (
            f"region [{s},{e}] sep={sep}: fast={fast} ref={ref}"
        )


def test_long_range_contacts_general_radius_matches_reference():
    view = _make_synthetic_view(n=70, seed=99)
    for r in (6.0, 9.0, 11.0):
        fast = view.long_range_contacts(1, 70, sep=8, radius=r)
        ref = _reference_long_range(view, 1, 70, sep=8, radius=r)
        assert fast == ref, f"radius={r}: fast={fast} ref={ref}"


def test_neighbor_list_cache_reuse_same_protein():
    """Repeated calls on the same protein should hit the cache."""
    view = _make_synthetic_view(n=100, seed=0,
                                  uniprot="P_CACHE", species="syn")
    first = _neighbor_list_8A(view)
    second = _neighbor_list_8A(view)
    assert first is second, "cache miss on repeated call for same protein"
    # Cache key must be (species, uniprot)
    assert ("syn", "P_CACHE") in _NBR8_CACHE


def test_neighbor_list_cache_separate_proteins():
    """Different (species, uniprot) tuples must not collide."""
    a = _make_synthetic_view(n=50, seed=1, uniprot="A", species="syn")
    b = _make_synthetic_view(n=50, seed=2, uniprot="B", species="syn")
    na = _neighbor_list_8A(a)
    nb = _neighbor_list_8A(b)
    assert na is not nb
    assert len(_NBR8_CACHE) == 2


def test_contact_density_zero_neighbors_far_apart():
    """Sanity: residues placed >> 8 Å apart should have 0 contact density."""
    n = 20
    ca = np.zeros((n, 3), dtype=np.float32)
    ca[:, 0] = np.arange(n, dtype=np.float32) * 100.0  # 100 Å spacing
    view = ProteinView(
        uniprot="FAR", species="syn", n_residues=n,
        ref_aa=np.full(n, "A", dtype="<U1"),
        plddt=np.full(n, 90.0, dtype=np.float32),
        sasa=np.zeros(n, dtype=np.float32),
        rel_sasa=np.zeros(n, dtype=np.float32),
        n_neigh=np.zeros(n, dtype=np.int32),
        ss_3=np.full(n, "C", dtype="<U1"),
        ca_xyz=ca, pae=None,
    )
    assert view.contact_density(1, n, radius=8.0) == 0.0
    assert view.long_range_contacts(1, n, sep=4, radius=8.0) == 0


def test_contact_density_empty_or_singleton_region():
    view = _make_synthetic_view(n=50, seed=3)
    # n=1: empty pairs → 0.0
    assert view.contact_density(5, 5, radius=8.0) == 0.0
    assert view.long_range_contacts(5, 5, sep=2, radius=8.0) == 0
