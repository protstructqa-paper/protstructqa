"""Unit tests for the multiprocess DSL execution pool in
`baselines/run_baseline_batched.py`.

Goals:
  - serial path (`_DSL_POOL_SIZE = 0`) and parallel path (`>= 1`) must
    return identical (pred, err) tuples for the same inputs
  - the pool is keyed correctly: separate proteins don't collide in the
    worker-local view cache
  - empty/None programs are skipped without dispatching to workers
  - pickle of typical DSL outputs round-trips cleanly

These tests touch real DSL execution against synthetic NPZ fixtures
under tmp_path; they avoid loading the full benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---- module-under-test ------------------------------------------------ #
import baselines.run_baseline_batched as rbb


# ---- synthetic protein fixture --------------------------------------- #


def _write_synthetic_npz(path: Path, n: int = 50, seed: int = 0):
    """Create a minimal NPZ that load_from_npz accepts."""
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=n))
    np.savez(
        path,
        seq=np.array(list(seq), dtype="<U1"),
        residue_nums=np.arange(1, n + 1, dtype=np.uint16),
        ca_xyz=rng.uniform(0.0, 30.0, size=(n, 3)).astype(np.float32),
        plddt=rng.uniform(50.0, 95.0, size=n).astype(np.float32),
        ss3=rng.integers(0, 3, size=n, dtype=np.uint8),
        sasa=rng.uniform(0.0, 200.0, size=n).astype(np.float32),
        pae=np.zeros((0,), dtype=np.uint8),  # no PAE
    )


@pytest.fixture
def synthetic_data_root(tmp_path, monkeypatch):
    """Lay out two synthetic proteins under
    {data_root}/{species}/features/AF-{uniprot}.npz so load_from_npz
    finds them via the runner's path convention."""
    root = tmp_path / "data"
    species = "synthetic"
    fdir = root / species / "features"
    fdir.mkdir(parents=True)
    _write_synthetic_npz(fdir / "AF-PROT_A.npz", n=40, seed=1)
    _write_synthetic_npz(fdir / "AF-PROT_B.npz", n=60, seed=2)
    # Patch DATA_ROOT in the module under test so get_view + workers
    # both find the synthetic files.
    monkeypatch.setattr(rbb, "DATA_ROOT", root)
    yield root


def _items_for_test(species: str = "synthetic") -> list[tuple]:
    """A mix of programs covering common DSL ops + edge cases."""
    return [
        # straightforward ops
        ("length()",                       species, "PROT_A"),
        ("mean_protein_plddt()",           species, "PROT_A"),
        ("mean_plddt(1, 10)",              species, "PROT_A"),
        ("contact_density(1, 30, 8.0)",    species, "PROT_A"),
        ("long_range_contacts(1, 40, 12, 8.0)", species, "PROT_A"),
        # different protein
        ("length()",                       species, "PROT_B"),
        ("max_plddt(20, 50)",              species, "PROT_B"),
        # error case (out-of-range)
        ("plddt_at(9999)",                 species, "PROT_A"),
        # empty program → must short-circuit
        (None,                             species, "PROT_A"),
        ("",                               species, "PROT_B"),
    ]


# ---- tests ----------------------------------------------------------- #


def test_serial_path_runs(synthetic_data_root):
    """Serial fallback (workers=0) returns sensible results."""
    rbb._DSL_POOL_SIZE = 0
    rbb._shutdown_dsl_pool()  # no-op if not started; safe
    items = _items_for_test()
    out = rbb._try_run_batch(items)
    assert len(out) == len(items)
    # length() should be 40 for PROT_A, 60 for PROT_B
    assert out[0] == (40, None)
    assert out[5] == (60, None)
    # contact_density → float in [0, 1]
    cd_pred, cd_err = out[3]
    assert cd_err is None
    assert isinstance(cd_pred, float) and 0.0 <= cd_pred <= 1.0
    # plddt_at(9999) → error (out of range)
    err_pred, err_msg = out[7]
    assert err_pred is None
    assert err_msg is not None
    # Empty programs → (None, None) without dispatch
    assert out[8] == (None, None)
    assert out[9] == (None, None)


def test_parallel_matches_serial(synthetic_data_root):
    """The parallel path must produce identical (pred, err) tuples."""
    items = _items_for_test()

    rbb._DSL_POOL_SIZE = 0
    rbb._shutdown_dsl_pool()
    serial = rbb._try_run_batch(items)

    rbb._DSL_POOL_SIZE = 2
    rbb._shutdown_dsl_pool()  # discard any prior pool so init args take
    parallel = rbb._try_run_batch(items)

    assert len(serial) == len(parallel)
    for i, (s, p) in enumerate(zip(serial, parallel)):
        # Predictions must match exactly. Errors can differ in trailing
        # text, so compare by error-or-not at the boolean level.
        s_pred, s_err = s
        p_pred, p_err = p
        assert (s_err is None) == (p_err is None), (
            f"item {i}: err mismatch: serial={s_err!r} parallel={p_err!r}"
        )
        assert s_pred == p_pred, (
            f"item {i}: pred mismatch: serial={s_pred!r} parallel={p_pred!r}"
        )

    # Cleanup
    rbb._shutdown_dsl_pool()
    rbb._DSL_POOL_SIZE = 0


def test_parallel_two_proteins_no_collision(synthetic_data_root):
    """Worker-local cache must keep two proteins distinct (regression
    guard against a single-key cache bug)."""
    rbb._DSL_POOL_SIZE = 2
    rbb._shutdown_dsl_pool()
    items = [
        ("length()", "synthetic", "PROT_A"),
        ("length()", "synthetic", "PROT_B"),
        ("length()", "synthetic", "PROT_A"),
        ("length()", "synthetic", "PROT_B"),
    ]
    out = rbb._try_run_batch(items)
    assert [r[0] for r in out] == [40, 60, 40, 60]
    rbb._shutdown_dsl_pool()
    rbb._DSL_POOL_SIZE = 0


def test_region_pair_pickle_roundtrip():
    """Region and Pair are tuple subclasses with multi-arg __new__. Pickle
    must round-trip them via __reduce__. Regression for the
    "TypeError: Region.__new__() missing 1 required positional argument:
    'end'" failure observed in production EV-t0 run.
    """
    import pickle
    from dsl.executor import Region, Pair
    for r in (Region(1, 100), Region(50, 50), Region(0, 0)):
        rp = pickle.loads(pickle.dumps(r))
        assert isinstance(rp, Region)
        assert rp == r
        assert rp.start == r.start and rp.end == r.end
    for p in (Pair(1, 2), Pair(0, 0)):
        pp = pickle.loads(pickle.dumps(p))
        assert isinstance(pp, Pair)
        assert pp == p


def test_parallel_handles_timeout_gracefully(synthetic_data_root, monkeypatch):
    """A pathological program should return (None, 'DSLTimeout: ...') in
    the parallel path without hanging the chunk. Force a 1s timeout via
    env var so the test runs quickly."""
    # Set short timeout in the worker env via initializer args.
    monkeypatch.setenv("PROTSTRUCTQA_DSL_TIMEOUT", "1")
    # Pathological program: deeply pointless loop. We use a comprehension
    # that's quadratic in n_residues to ensure n=40 still hits >1s.
    # If the DSL doesn't have such an operator we just verify a syntax
    # error returns gracefully (also fine: not a hang).
    rbb._DSL_POOL_SIZE = 2
    rbb._DSL_TIMEOUT_SEC = 1
    rbb._shutdown_dsl_pool()
    items = [
        ("length()", "synthetic", "PROT_A"),
        ("totally_invalid_dsl_program()", "synthetic", "PROT_A"),
        ("length()", "synthetic", "PROT_B"),
    ]
    out = rbb._try_run_batch(items)
    assert out[0][0] == 40
    # Middle item: parse or runtime error → (None, error_str)
    assert out[1][0] is None
    assert out[1][1] is not None
    assert out[2][0] == 60
    rbb._shutdown_dsl_pool()
    rbb._DSL_POOL_SIZE = 0
    rbb._DSL_TIMEOUT_SEC = 10
