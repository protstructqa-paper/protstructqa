"""ProtStructQA: a small, executable, domain-specific language for natural-
language questions about AlphaFold predicted protein structures.

See `../specs/DSL_SPEC.md` for the v0.1 specification.

Top-level entry points:
    parse(source: str) -> AST
    execute(ast: AST, view: ProteinView) -> Any
    run(source: str, view: ProteinView) -> Any   # parse + execute
"""

from .protein_view import (
    ProteinView,
    load_from_npz,
    load_from_pdb,
    load_from_feature_parquet,
)
from .ast_nodes import Program
from .grammar import parse, parse_grammar
from .executor import execute, run

__all__ = ["ProteinView", "Program", "parse", "parse_grammar",
           "execute", "run", "load_from_npz", "load_from_pdb",
           "load_from_feature_parquet"]
