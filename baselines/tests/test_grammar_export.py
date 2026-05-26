"""TDD tests for baselines/grammar_export.py."""
from __future__ import annotations

import sys

sys.path.insert(0,
    "."
)

from baselines import grammar_export


def test_export_gbnf_returns_string():
    g = grammar_export.export_gbnf()
    assert isinstance(g, str)
    assert len(g) > 100


def test_grammar_contains_root_rule():
    g = grammar_export.export_gbnf()
    assert "root ::=" in g


def test_primitives_alternation_substituted():
    g = grammar_export.export_gbnf()
    # Placeholder must be replaced
    assert "__PRIMITIVE_ALTS__" not in g
    # All canonical primitives must appear as quoted alternatives
    for prim in grammar_export.PRIMITIVES:
        assert f'"{prim}"' in g, f"missing primitive in grammar: {prim}"


def test_grammar_has_expected_structure():
    g = grammar_export.export_gbnf()
    for required in ("expression", "or_expr", "and_expr", "comparison",
                       "add_expr", "mul_expr", "pow_expr", "unary",
                       "atom", "comprehension", "call", "args", "literal",
                       "string", "number", "name"):
        assert f"{required} ::=" in g, f"missing rule: {required}"


def test_includes_comprehension_keywords():
    g = grammar_export.export_gbnf()
    for kw in ("exists", "forall", "count", "filter", "argmin", "argmax",
                 "if", "then", "else"):
        assert f'"{kw}"' in g, f"missing comprehension keyword: {kw}"


def test_size_reasonable_for_vllm():
    """vLLM has to parse the grammar on every request; keep it lean."""
    g = grammar_export.export_gbnf()
    assert len(g) < 5000, f"grammar got too big: {len(g)} chars"
    assert len(g.splitlines()) < 100
