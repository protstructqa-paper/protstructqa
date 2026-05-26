"""TDD tests for baselines/output_parser.py."""
from __future__ import annotations

import sys
import pytest

sys.path.insert(0,
    "."
)

from baselines import output_parser as op


# --------------------------- extract_program ----------------------- #


def test_extract_fenced_code_block():
    text = """Here is the answer:
```
mean_plddt(range(50, 100))
```
"""
    assert op.extract_program(text) == "mean_plddt(range(50, 100))"


def test_extract_fenced_with_language_tag():
    text = "```python\ndistance(residue(45), residue(160))\n```"
    assert op.extract_program(text) == "distance(residue(45), residue(160))"


def test_extract_inline_backtick():
    text = "The query is `n_helices()` for this protein."
    assert op.extract_program(text) == "n_helices()"


def test_extract_program_prefix():
    text = "Program: mean_plddt(range(1, 50))"
    assert op.extract_program(text) == "mean_plddt(range(1, 50))"


def test_extract_raw_line():
    text = "argmin reg in sliding_window(30) by mean_plddt(reg)"
    assert op.extract_program(text) == "argmin reg in sliding_window(30) by mean_plddt(reg)"


def test_extract_strips_thinking():
    text = "<think>I need to compute the mean</think>\nmean_plddt(range(1, 50))"
    assert op.extract_program(text) == "mean_plddt(range(1, 50))"


def test_no_program_returns_none():
    text = "I think the answer is 42."   # no DSL tokens
    assert op.extract_program(text) is None


# --------------------------- parse_llm_output: program -------------- #


def test_parse_llm_output_program(real_view=None):
    out = op.parse_llm_output(
        "```mean_plddt(range(1, 50))```", expected_type="Float"
    )
    assert out["program"] == "mean_plddt(range(1, 50))"
    assert out["scalar"] is None
    assert not out["abstained"]


# --------------------------- parse_llm_output: scalar fallback ----- #


def test_parse_llm_output_scalar_float():
    out = op.parse_llm_output("8.5 Å", expected_type="Float")
    assert out["program"] is None
    assert out["scalar"] == 8.5


def test_parse_llm_output_scalar_int():
    out = op.parse_llm_output("there are 42 long-range contacts",
                                  expected_type="Int")
    assert out["program"] is None
    assert out["scalar"] == 42.0


def test_parse_llm_output_scalar_bool_yes():
    out = op.parse_llm_output("Yes", expected_type="Bool")
    assert out["scalar"] is True


def test_parse_llm_output_scalar_bool_no():
    out = op.parse_llm_output("No", expected_type="Bool")
    assert out["scalar"] is False


def test_parse_llm_output_scalar_secstruct():
    out = op.parse_llm_output("helix", expected_type="SecStruct")
    assert out["scalar"] == "H"


def test_parse_llm_output_scalar_region():
    out = op.parse_llm_output("residues 100 to 130",
                                  expected_type="Region")
    assert out["scalar"] == [100, 130]


# --------------------------- selective abstention ----------------- #


def test_parse_llm_output_unreliable():
    out = op.parse_llm_output("unreliable", expected_type="Float|Unreliable")
    assert out["abstained"]
    assert out["scalar"] == "Unreliable"


def test_parse_llm_output_unreliable_phrase():
    out = op.parse_llm_output("I cannot determine this confidently.",
                                  expected_type="Bool|Unreliable")
    assert out["abstained"]


def test_parse_llm_output_value_when_selective():
    """Even with a selective answer_type, returning a value is valid."""
    out = op.parse_llm_output("8.3 Å", expected_type="Float|Unreliable")
    assert not out["abstained"]
    assert out["scalar"] == 8.3


# --------------------------- robustness ------------------------- #


def test_parse_llm_output_handles_none():
    out = op.parse_llm_output(None, expected_type="Float")
    assert out["program"] is None
    assert out["scalar"] is None
    assert not out["abstained"]


def test_parse_llm_output_empty_string():
    out = op.parse_llm_output("", expected_type="Bool")
    assert out["program"] is None
    assert out["scalar"] is None
