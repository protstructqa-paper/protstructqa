"""Export ProtStructQA grammar as GBNF (the format vLLM/xgrammar accepts
via the `guided_grammar` request parameter).

The full Lark grammar lives in `dsl/grammar.py`. GBNF is similar to EBNF
but with subtle differences (no inline `?` rules, no Lark transformations,
explicit whitespace handling). Rather than auto-translate, we hand-write
a compact GBNF that captures the same surface language, so the LLM is
constrained to emit only parseable ProtStructQA during sampling.

Reference for GBNF: https://github.com/ggerganov/llama.cpp/tree/master/grammars
xgrammar (vLLM's backend) supports a superset of GBNF.

The compactness goal: total grammar < 200 lines so vLLM parses it quickly
on every request. We omit a few rare DSL primitives (`coordinates`,
`runs`, `longest_run`) since they aren't used by any of our 33 question
templates: re-add to the grammar if a future template needs them.
"""
from __future__ import annotations


# All primitive function names the executor knows. Kept in sync with
# dsl/executor.py and dsl/protein_view.py.
PRIMITIVES = [
    # Per-residue
    "plddt", "ref_aa", "ss", "sasa", "rel_sasa", "n_neighbors",
    # Per-pair
    "distance", "ca_distance", "cb_distance", "seq_separation", "pae",
    # Per-region
    "mean_plddt", "min_plddt", "max_plddt", "std_plddt",
    "mean_pae", "max_pae", "count_high_pae",
    "contact_density", "long_range_contacts", "radius_of_gyration",
    "length", "mean_rel_sasa", "mean_n_neighbors",
    # Region constructors
    "residue", "range", "first", "last", "window",
    # Iterator forms
    "sliding_window", "all_pairs",
    # Protein-level
    "protein_length", "n_helices", "n_strands", "mean_protein_plddt",
    # Set ops
    "union", "intersection", "difference", "contains", "size",
    "to_set", "in_region",
    # Math helpers
    "abs", "min", "max", "round", "floor", "ceil",
]


def _alts(items: list[str]) -> str:
    """Render `items` as GBNF alternation: '"a" | "b" | "c"'."""
    return " | ".join(f'"{w}"' for w in items)


GBNF_GRAMMAR = r'''# ProtStructQA GBNF for xgrammar / vLLM guided_grammar
# Hand-written to match dsl/grammar.py (Lark): keep these in sync.

root ::= ws expression ws

expression ::= or_expr

or_expr ::= and_expr (ws "or" ws and_expr)*
and_expr ::= not_expr (ws "and" ws not_expr)*
not_expr ::= ("not" ws not_expr) | comparison
comparison ::= add_expr (ws cmpop ws add_expr)? | "between" ws "(" ws add_expr ws "," ws add_expr ws "," ws add_expr ws ")"

cmpop ::= "<=" | ">=" | "==" | "!=" | "<" | ">"

add_expr ::= mul_expr (ws ("+" | "-") ws mul_expr)*
mul_expr ::= pow_expr (ws ("*" | "/") ws pow_expr)*
pow_expr ::= unary (ws "**" ws unary)*
unary ::= ("-" ws unary) | atom

atom ::= literal | comprehension | call | "(" ws expression ws ")" | name

comprehension ::= ("exists" | "forall" | "count" | "filter") ws binder ws "in" ws expression ws "where" ws expression | "argmin" ws binder ws "in" ws expression ws "by" ws expression (ws "where" ws expression)? | "argmax" ws binder ws "in" ws expression ws "by" ws expression (ws "where" ws expression)? | "if" ws expression ws "then" ws expression ws "else" ws expression

binder ::= name | "(" ws name ws "," ws name ws ")"

call ::= primitive ws "(" ws args? ws ")"
args ::= arg (ws "," ws arg)*
arg ::= name ws "=" ws expression | expression

primitive ::= __PRIMITIVE_ALTS__

literal ::= number | string | "true" | "false" | "all_residues"

string ::= "\"" string_chars "\""
string_chars ::= [^"\\]*

number ::= "-"? ([0-9]+) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?

name ::= [a-zA-Z_] [a-zA-Z0-9_]*

ws ::= [ \t\n]*
'''


def export_gbnf() -> str:
    """Return the ProtStructQA GBNF grammar string, with PRIMITIVES embedded
    as an alternation. Suitable for vLLM's `guided_grammar` payload."""
    return GBNF_GRAMMAR.replace("__PRIMITIVE_ALTS__", _alts(PRIMITIVES))


def export_for_xgrammar() -> str:
    """xgrammar-specific dialect (currently the same as GBNF; placeholder
    for future tweaks if vLLM diverges)."""
    return export_gbnf()


if __name__ == "__main__":
    grammar = export_gbnf()
    print(grammar)
    print(f"\n# Grammar size: {len(grammar)} chars, "
          f"{len(grammar.splitlines())} lines, "
          f"{len(PRIMITIVES)} primitives")
