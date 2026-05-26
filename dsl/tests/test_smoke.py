"""Smoke tests for ProtStructQA parser + executor on real AlphaFold
structures. Covers all 7 question template families (Confidence, Distance,
PAE, Solvent, SS, Topology, Compositional) plus pair-tuple binding.

Run with:
    python -m pytest text_to_structurequery/dsl/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project package importable as `dsl.*` regardless of where
# pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # …/text_to_structurequery
sys.path.insert(0, str(_PROJECT_ROOT))

from dsl import parse, run                                # noqa: E402
from dsl.executor import Region, Pair, ProtStructQAError      # noqa: E402
from dsl.protein_view import load_from_pdb                # noqa: E402


# ----------------------------------------------------------------------
# Fixtures: load real AlphaFold structures from any available panel
# ----------------------------------------------------------------------

# Candidate AF panels in priority order. Tests skip if none exist.
_AF_PANEL_CANDIDATES = [
    # Future canonical location once we download human AFDB:
    Path("./data/human/structures"),
    # Existing on-disk pig panel (1,842 PDB+PAE+conf triplets): fallback
    # for now; species-agnostic since the DSL never inspects species ID.
    Path("<project-root>/probcot_bio/"
         "data/aiv/processed/expanded_pig_structures"),
]


def _resolve_panel() -> Path:
    for d in _AF_PANEL_CANDIDATES:
        if d.is_dir() and any(d.glob("AF-*-F1-model_v6.pdb")):
            return d
    pytest.skip(
        "no AlphaFold panel available; download human/mouse/fly via "
        "benchmark/01_download_proteomes.py or point at any AF dir"
    )


def _load(min_residues: int = 0, want_pae: bool = False):
    """Load any AF structure that meets the constraints."""
    panel = _resolve_panel()
    candidates = sorted(panel.glob("AF-*-F1-model_v6.pdb"))
    assert candidates, f"no PDBs in {panel}"
    for p in candidates:
        pae_path = p.with_name(p.name.replace(
            "model_v6.pdb", "predicted_aligned_error_v6.json"))
        if want_pae and not pae_path.exists():
            continue
        # Check residue count quickly via file size heuristic.
        if p.stat().st_size < min_residues * 80:
            continue
        uniprot = p.stem.split("-")[1]
        return load_from_pdb(
            p, pae_path=pae_path if pae_path.exists() else None,
            uniprot=uniprot, species=panel.parent.name or "?",
        )
    raise RuntimeError("no suitable PDB found")


@pytest.fixture(scope="module")
def view_small():
    """Smallest available; for fast tests."""
    return _load(min_residues=50)


@pytest.fixture(scope="module")
def view_medium():
    """At least 100 residues; for compositional tests."""
    return _load(min_residues=100)


@pytest.fixture(scope="module")
def view_pae():
    """Has PAE file; for PAE-related tests."""
    return _load(min_residues=100, want_pae=True)


# ======================================================================
# Parser-only tests
# ======================================================================

def test_parse_literals():
    assert parse("true").expression.__class__.__name__ == "BoolLit"
    assert parse("3.14").expression.value == 3.14
    assert parse("42").expression.value == 42

def test_parse_arithmetic():
    p = parse("1 + 2 * 3")
    assert p.expression is not None

def test_parse_comparison():
    p = parse("plddt(residue(10)) > 80.0")
    assert p.expression.__class__.__name__ == "Compare"

def test_parse_comprehension_single_var():
    p = parse("count r in all_residues where rel_sasa(r) < 0.10")
    assert p.expression.__class__.__name__ == "Count"
    assert p.expression.vars == ["r"]

def test_parse_comprehension_pair_binder():
    p = parse(
        "filter (i, j) in all_pairs(min_sep=50) where distance(i, j) < 8.0"
    )
    assert p.expression.__class__.__name__ == "Filter"
    assert p.expression.vars == ["i", "j"]

def test_parse_argmax_with_where():
    p = parse(
        "argmax reg in sliding_window(60) "
        "by long_range_contacts(reg) "
        "where mean_plddt(reg) > 80.0"
    )
    assert p.expression.__class__.__name__ == "ArgMax"
    assert p.expression.where is not None


# ======================================================================
# Family A: Confidence (pLDDT)
# ======================================================================

def test_A1_region_mean_plddt(view_medium):
    out = run("mean_plddt(range(1, 30))", view_medium)
    assert 0 <= out <= 100

def test_A2_n_vs_c_terminal(view_medium):
    out = run("mean_plddt(last(20)) < mean_plddt(first(20))", view_medium)
    assert isinstance(out, bool)

def test_A3_lowest_confidence_window(view_medium):
    n = view_medium.length()
    if n < 30:
        pytest.skip()
    reg = run("argmin reg in sliding_window(30) by mean_plddt(reg)", view_medium)
    assert isinstance(reg, Region)
    assert reg.end - reg.start + 1 == 30

def test_A4_count_above_threshold(view_medium):
    out = run("count r in all_residues where plddt(r) > 90.0", view_medium)
    assert 0 <= out <= view_medium.length()

def test_A5_high_conf_region_exists(view_medium):
    out = run(
        "exists reg in sliding_window(20) where mean_plddt(reg) > 80.0",
        view_medium,
    )
    assert isinstance(out, bool)


# ======================================================================
# Family B: Distance
# ======================================================================

def test_B1_pairwise_distance(view_small):
    out = run("distance(residue(1), residue(2))", view_small)
    assert out > 0  # consecutive Cα are typically ~3.8 Å apart

def test_B2_proximity_bool(view_small):
    out = run("distance(residue(1), residue(2)) < 8.0", view_small)
    assert out is True   # consecutive residues are within 8 Å

def test_B3_long_range_contacts_pairset(view_medium):
    n = view_medium.length()
    if n < 60:
        pytest.skip()
    out = run(
        "filter (i, j) in all_pairs(min_sep=20) where distance(i, j) < 8.0",
        view_medium,
    )
    assert isinstance(out, frozenset)

def test_B4_long_range_contact_count(view_medium):
    n = view_medium.length()
    if n < 60:
        pytest.skip()
    out = run(
        "size(filter (i, j) in all_pairs(min_sep=20) where distance(i, j) < 8.0)",
        view_medium,
    )
    assert isinstance(out, int) and out >= 0


# ======================================================================
# Family C: PAE
# ======================================================================

def test_C1_region_pair_pae(view_pae):
    n = view_pae.length()
    if n < 150:
        pytest.skip()
    out = run("mean_pae(range(1, 50), range(80, 130))", view_pae)
    assert isinstance(out, float) and out >= 0

def test_C2_orientation_bool(view_pae):
    n = view_pae.length()
    if n < 150:
        pytest.skip()
    out = run(
        "mean_pae(range(1, 50), range(80, 130)) < 5.0",
        view_pae,
    )
    assert isinstance(out, bool)


# ======================================================================
# Family D: Solvent / packing
# ======================================================================

def test_D1_buried_bool(view_small):
    out = run("rel_sasa(residue(1)) < 0.10", view_small)
    assert isinstance(out, bool)

def test_D2_most_exposed_window(view_medium):
    n = view_medium.length()
    if n < 40:
        pytest.skip()
    reg = run("argmax reg in sliding_window(20) by mean_rel_sasa(reg)", view_medium)
    assert isinstance(reg, Region)

def test_D3_buried_count(view_medium):
    out = run("count r in all_residues where rel_sasa(r) < 0.10", view_medium)
    assert 0 <= out <= view_medium.length()

def test_D4_neighbor_count(view_small):
    out = run("n_neighbors(residue(1))", view_small)
    assert isinstance(out, int) and out >= 0

def test_D5_densely_packed(view_small):
    out = run("n_neighbors(residue(1)) > 12", view_small)
    assert isinstance(out, bool)


# ======================================================================
# Family E: Secondary structure (incl. SS runs)
# ======================================================================

def test_E1_per_residue_ss(view_small):
    out = run("ss(residue(1))", view_small)
    assert out in {"H", "E", "C"}

def test_E2_helix_check(view_small):
    out = run('ss(residue(1)) == "H"', view_small)
    assert isinstance(out, bool)

def test_E3_helix_count(view_medium):
    out = run('count r in all_residues where ss(r) == "H"', view_medium)
    assert 0 <= out <= view_medium.length()

def test_E4_longest_helix(view_medium):
    """Use the new `longest_run` helper for SS regions."""
    n = view_medium.length()
    if n < 40:
        pytest.skip()
    try:
        reg = run('longest_run("H")', view_medium)
        assert isinstance(reg, Region)
        assert reg.end >= reg.start
    except ProtStructQAError:
        # Some small proteins may have no helices; acceptable.
        pass

def test_E5_runs_helper(view_medium):
    """`runs("H")` returns a tuple of Region objects."""
    runs_h = run('runs("H")', view_medium)
    assert isinstance(runs_h, tuple)
    if runs_h:
        assert isinstance(runs_h[0], Region)


# ======================================================================
# Family F: Topology / contacts
# ======================================================================

def test_F1_long_range_contact_density(view_medium):
    out = run("long_range_contacts(range(1, 50), sep=12)", view_medium)
    assert isinstance(out, int) and out >= 0

def test_F2_compact_core_detection(view_medium):
    n = view_medium.length()
    if n < 60:
        pytest.skip()
    out = run(
        'exists reg in sliding_window(40) where '
        'mean_plddt(reg) > 80.0 and contact_density(reg) > 0.15',
        view_medium,
    )
    assert isinstance(out, bool)

def test_F3_radius_of_gyration(view_medium):
    out = run("radius_of_gyration(range(1, 30))", view_medium)
    assert out > 0

def test_F4_most_compact_window(view_medium):
    n = view_medium.length()
    if n < 60:
        pytest.skip()
    reg = run(
        "argmin reg in sliding_window(30) by radius_of_gyration(reg)",
        view_medium,
    )
    assert isinstance(reg, Region)


# ======================================================================
# Family G: Compositional (held-out at training time)
# ======================================================================

def test_G1_buried_low_plddt(view_medium):
    out = run(
        "filter r in all_residues where rel_sasa(r) < 0.10 and plddt(r) < 60.0",
        view_medium,
    )
    assert isinstance(out, frozenset)

def test_G2_high_conf_contact_rich(view_medium):
    n = view_medium.length()
    if n < 80:
        pytest.skip()
    try:
        reg = run(
            "argmax reg in sliding_window(40) "
            "by long_range_contacts(reg) "
            "where mean_plddt(reg) > 80.0 and contact_density(reg) > 0.15",
            view_medium,
        )
        assert isinstance(reg, Region)
    except ProtStructQAError:
        pass  # may have no qualifying region

def test_G3_helix_strand_interface(view_medium):
    out = run(
        'exists r in all_residues where ss(r) == "H" and '
        'exists s in all_residues where ss(s) == "E" and distance(r, s) < 8.0',
        view_medium,
    )
    assert isinstance(out, bool)

def test_G4_compositional_pair(view_medium):
    """Pair binder + structural compositional predicate."""
    n = view_medium.length()
    if n < 80:
        pytest.skip()
    out = run(
        "filter (i, j) in all_pairs(min_sep=30) where "
        "distance(i, j) < 8.0 and plddt(i) > 80.0 and plddt(j) > 80.0",
        view_medium,
    )
    assert isinstance(out, frozenset)


# ======================================================================
# Error semantics
# ======================================================================

def test_out_of_range_residue_errors(view_small):
    with pytest.raises(ProtStructQAError):
        run("plddt(residue(999999))", view_small)

def test_division_by_zero_errors(view_small):
    with pytest.raises(ProtStructQAError):
        run("1.0 / 0", view_small)

def test_argmin_empty_errors(view_small):
    with pytest.raises(ProtStructQAError):
        run(
            "argmin r in all_residues by plddt(r) where plddt(r) > 100.0",
            view_small,
        )


# ======================================================================
# Symmetry / determinism
# ======================================================================

def test_distance_symmetric(view_small):
    a = run("distance(residue(1), residue(2))", view_small)
    b = run("distance(residue(2), residue(1))", view_small)
    assert abs(a - b) < 1e-6

def test_self_distance_zero(view_small):
    out = run("distance(residue(1), residue(1))", view_small)
    assert out == 0.0

def test_idempotent_eval(view_small):
    """Running the same program twice gives the same answer."""
    p = "mean_plddt(first(20))"
    a = run(p, view_small)
    b = run(p, view_small)
    assert a == b
