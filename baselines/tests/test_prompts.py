"""TDD tests for baselines/prompts.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0,
    "."
)

from baselines import prompts


@pytest.fixture(scope="module")
def real_view():
    npz = Path("./data/human/features/AF-A0A024RBG1.npz")
    if not npz.exists():
        pytest.skip("real human NPZ not present")
    from dsl import load_from_npz
    return load_from_npz(npz, uniprot="A0A024RBG1", species="human")


# --------------------------- protein_summary ----------------------- #


def test_protein_summary_basic(real_view):
    s = prompts.protein_summary(real_view)
    assert "A0A024RBG1" in s
    assert "Length: 181" in s
    assert "Mean pLDDT" in s
    assert "pLDDT band" in s
    assert "SS band" in s
    assert "Helix runs" in s


def test_protein_summary_band_chars(real_view):
    s = prompts.protein_summary(real_view, max_band_len=20)
    # The band should be 20 characters using the {#, +, ., ?} alphabet
    band_line = [l for l in s.split("\n") if "pLDDT band" in l][0]
    band = band_line.split(":", 2)[-1].strip()
    # Allow either band first (before legend): extract last segment
    band = band.split()[-1] if band.split() else ""
    # Some chars from band alphabet should be present
    assert any(c in band for c in "#+.?")


# --------------------------- few-shot block --------------------- #


def test_build_few_shot_block_renders_n_examples():
    exemplars = [
        {"question": f"Q{i}?", "program": f"prog_{i}", "answer": i * 1.5}
        for i in range(10)
    ]
    block = prompts.build_few_shot_block(exemplars, n=4)
    assert "Examples:" in block
    # 4 numbered "Example N:" plus the top-level "Examples:" header
    assert block.count("Example") == 5
    assert "Q0?" in block
    assert "Q3?" in block
    assert "Q4?" not in block


# --------------------------- L0 prompt --------------------------- #


def test_l0_prompt_contains_pieces(real_view):
    q = {"question": "What is the mean pLDDT of residues 30 to 50?",
          "answer": 88.5, "answer_type": "Float"}
    prompt = prompts.build_l0_prompt(q, real_view)
    assert "ProtStructQA" in prompt
    assert "plddt(r)" in prompt
    assert "A0A024RBG1" in prompt
    assert q["question"] in prompt
    # Without exemplars, no "Examples:" block
    assert "Examples:" not in prompt


def test_l0_prompt_with_exemplars(real_view):
    q = {"question": "Test question?", "answer_type": "Float"}
    exemplars = [
        {"question": "Sample Q1?", "program": "mean_plddt(range(1, 50))",
          "answer": 75.0},
        {"question": "Sample Q2?", "program": "n_helices()", "answer": 5},
    ]
    prompt = prompts.build_l0_prompt(q, real_view, exemplars=exemplars,
                                          n_shots=2)
    assert "Examples:" in prompt
    assert "Sample Q1?" in prompt
    assert "Sample Q2?" in prompt
    assert "75.000" in prompt    # float formatted with 3 decimals


# --------------------------- L1 prompt --------------------------- #


def test_l1_prompt_no_prose_instruction(real_view):
    q = {"question": "Q?", "answer_type": "Float"}
    prompt = prompts.build_l1_prompt(q, real_view)
    assert "ProtStructQA program" in prompt
    assert "Do NOT include any prose" in prompt
    assert "Program:" in prompt   # ends with the program: cue
