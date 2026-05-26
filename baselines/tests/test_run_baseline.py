"""End-to-end smoke test for run_baseline.

Stubs the model adapter and uses synthetic split JSONLs in tmp_path.
Verifies the full pipeline:
  load split → load views → build prompts → call (stub) model →
  parse → execute → score → write metrics.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0,
    "."
)


# --------------------------- module loader --------------------- #


def _load_run_baseline():
    here = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_bl", here / "run_baseline.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_run_bl"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def rb():
    return _load_run_baseline()


@pytest.fixture(scope="module")
def real_view_uniprot():
    npz = Path("./data/human/features/AF-A0A024RBG1.npz")
    if not npz.exists():
        pytest.skip("real human NPZ not present")
    return "A0A024RBG1"


# --------------------------- stub adapter -------------------- #


class StubAdapter:
    """Returns canned outputs for testing."""
    def __init__(self, response_text: str = "mean_plddt(range(1, 50))"):
        self.response_text = response_text
        self.calls = []

    def generate(self, prompt, max_tokens=512, temperature=0.0, stop=None):
        self.calls.append(prompt)
        return {
            "text": self.response_text,
            "n_input_tokens": len(prompt) // 4,
            "n_output_tokens": len(self.response_text) // 4,
            "stop_reason": "stop",
            "elapsed_s": 0.01,
        }


# --------------------------- helpers ------------------------- #


def _write_split(splits_dir: Path, name: str, rows: list[dict]):
    splits_dir.mkdir(parents=True, exist_ok=True)
    with (splits_dir / f"{name}.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_q(uniprot: str, family: str, template: str,
              question: str, program: str, answer, answer_type: str,
              params: dict) -> dict:
    return {
        "qid": f"human/{uniprot}/{template}/0",
        "uniprot": uniprot, "species": "human",
        "family": family, "template": template,
        "question": question, "program": program,
        "answer": answer, "answer_type": answer_type,
        "params": params, "paraphrase_id": 0,
    }


# --------------------------- pick_exemplars ------------------ #


def test_pick_exemplars_balanced(rb):
    train = []
    for fam in ["A", "B", "C", "D", "E", "F"]:
        for i in range(5):
            train.append(_make_q(f"U{fam}{i}", fam, f"{fam}1",
                                       f"{fam} q{i}", f"prog_{fam}_{i}", 1.0,
                                       "Float", {}))
    target = _make_q("UT", "A", "A2", "Q?", "prog_t", 1.0, "Float", {})
    chosen = rb.pick_exemplars(train, target, n=6, seed=0)
    assert len(chosen) == 6
    # Should not include same template
    assert all(c["template"] != "A2" for c in chosen)
    # Should span multiple families (with 6 picks across 6 families,
    # we expect each family represented)
    fams = {c["family"] for c in chosen}
    assert len(fams) >= 4   # high diversity


def test_pick_exemplars_deterministic(rb):
    train = [_make_q(f"U{i}", "A", "A1", f"Q{i}", "prog", 1.0, "Float", {})
                for i in range(20)]
    target = _make_q("UT", "B", "B2", "?", "x", 1.0, "Float", {})
    a = rb.pick_exemplars(train, target, n=4, seed=0)
    b = rb.pick_exemplars(train, target, n=4, seed=0)
    assert [x["qid"] for x in a] == [x["qid"] for x in b]


# --------------------------- evaluate_one (uses real view) -- #


def test_evaluate_one_program_path(rb, real_view_uniprot):
    """Stub returns a valid ProtStructQA program; pipeline should execute it
    against the real protein view and score correctly."""
    from dsl import load_from_npz, run as dsl_run

    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    # Construct a question whose gold matches what the stub program produces
    program = "mean_plddt(range(1, 50))"
    gold = dsl_run(program, view)
    q = _make_q(real_view_uniprot, "A", "A1",
                  "What is the mean pLDDT of residues 1 to 50?",
                  program, gold, "Float",
                  {"start": 1, "end": 50})

    adapter = StubAdapter(response_text=f"```\n{program}\n```")
    row = rb.evaluate_one(adapter, q, view, exemplars=[], regime="L0")
    assert row["used_program"] is True
    assert row["extracted_program"] == program
    assert row["correct"] is True
    assert row["pred_answer"] == pytest.approx(gold)


def test_evaluate_one_scalar_path(rb, real_view_uniprot):
    """Stub returns a numeric scalar; runner should fall back to scalar
    parsing and score against gold (possibly with tolerance)."""
    from dsl import load_from_npz

    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    q = _make_q(real_view_uniprot, "A", "A1",
                  "What is the mean pLDDT of residues 1 to 50?",
                  "mean_plddt(range(1, 50))", 90.5, "Float",
                  {"start": 1, "end": 50})
    adapter = StubAdapter(response_text="The answer is approximately 90.7.")
    row = rb.evaluate_one(adapter, q, view, exemplars=[], regime="L0")
    assert row["used_program"] is False
    assert row["extracted_scalar"] == 90.7
    # Within Float tolerance (±0.5)
    assert row["correct"] is True


def test_l1_grammar_constrained_first_attempt_succeeds(rb, real_view_uniprot):
    """L1 stub returns a valid program on the FIRST attempt → no retry."""
    from dsl import load_from_npz, run as dsl_run
    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    program = "mean_plddt(range(1, 30))"
    gold = dsl_run(program, view)
    q = _make_q(real_view_uniprot, "A", "A1",
                  "Mean pLDDT 1-30?", program, gold, "Float",
                  {"start": 1, "end": 30})

    class L1Stub:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, max_tokens=512, temperature=0.0,
                       stop=None, guided_grammar=None):
            self.calls += 1
            return {"text": program, "n_input_tokens": 100,
                      "n_output_tokens": 10, "stop_reason": "stop",
                      "elapsed_s": 0.01}

    stub = L1Stub()
    row = rb.evaluate_one(stub, q, view, exemplars=[], regime="L1")
    assert stub.calls == 1
    assert row["correct"] is True
    assert row["n_attempts"] == 1


def test_l1_retry_on_execution_error(rb, real_view_uniprot):
    """L1 stub returns a BAD program first (out-of-range residue) then a
    GOOD one. The retry loop should converge."""
    from dsl import load_from_npz, run as dsl_run
    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    bad = "distance(residue(99999), residue(1))"   # out-of-range
    good = "distance(residue(1), residue(50))"
    gold = dsl_run(good, view)

    q = _make_q(real_view_uniprot, "B", "B1",
                  "Distance 1-50?", good, gold, "Float", {"i": 1, "j": 50})

    class FlakyStub:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, max_tokens=512, temperature=0.0,
                       stop=None, guided_grammar=None):
            self.calls += 1
            text = bad if self.calls == 1 else good
            return {"text": text, "n_input_tokens": 100,
                      "n_output_tokens": 10, "stop_reason": "stop",
                      "elapsed_s": 0.01}

    stub = FlakyStub()
    row = rb.evaluate_one(stub, q, view, exemplars=[], regime="L1")
    assert stub.calls == 2     # one bad, one good
    assert row["correct"] is True
    assert row["n_attempts"] == 2


def test_evaluate_one_abstention_path(rb, real_view_uniprot):
    """Family Hb question with stub returning 'unreliable'."""
    from dsl import load_from_npz

    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    q = _make_q(real_view_uniprot, "Hb", "Hb1",
                  "Distance? Unreliable if uncertain.",
                  "if plddt(...) ... else distance(...)",
                  "Unreliable", "Float|Unreliable",
                  {"i": 1, "j": 2, "plddt_thr": 50})

    adapter = StubAdapter(response_text="unreliable")
    row = rb.evaluate_one(adapter, q, view, exemplars=[], regime="L0")
    assert row["abstained"] is True
    assert row["correct"] is True


# --------------------------- run() smoke (full pipeline) ----- #


def test_run_full_pipeline_smoke(tmp_path, rb, real_view_uniprot, monkeypatch):
    """End-to-end: small synthetic split + stub adapter → run() writes
    per_question.jsonl + metrics.json without crashing."""
    splits = tmp_path / "splits"
    out = tmp_path / "baseline_runs"

    from dsl import load_from_npz, run as dsl_run
    npz = Path(f"./data/human/features/AF-{real_view_uniprot}.npz")
    view = load_from_npz(npz, uniprot=real_view_uniprot, species="human")

    # Build 3 train + 3 test questions, all answerable by the same program
    program = "mean_plddt(range(1, 50))"
    gold = dsl_run(program, view)
    train_rows = [_make_q(real_view_uniprot, fam, f"{fam}1", f"Q-{fam}",
                              program, gold, "Float", {})
                    for fam in ["A", "B", "C"]]
    test_rows = [_make_q(real_view_uniprot, "A", "A1",
                             f"Test Q {i}", program, gold, "Float", {})
                   for i in range(3)]
    _write_split(splits, "train", train_rows)
    _write_split(splits, "test_iid", test_rows)

    # Patch the runner's roots so it uses our tmp dirs
    monkeypatch.setattr(rb, "SPLITS_ROOT", splits)

    # Patch the model adapter factory to return our stub
    stub = StubAdapter(response_text=f"```{program}```")
    monkeypatch.setattr(rb.model_adapter, "VLLMHTTPAdapter",
                          lambda **kwargs: stub)

    metrics = rb.run(
        split_name="test_iid", regime="L0",
        model_backend="vllm-http", model_name="stub-model",
        vllm_url=None, max_examples=None, n_few_shot=2,
        out_root=out,
    )

    assert metrics["n_total"] == 3
    assert metrics["accuracy_overall"] == 1.0
    # Outputs written
    out_dir = out / "stub-model" / "test_iid" / "L0"
    assert (out_dir / "per_question.jsonl").exists()
    assert (out_dir / "metrics.json").exists()
    # Per-template + per-family breakdowns present
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert "by_template" in saved and "by_family" in saved
