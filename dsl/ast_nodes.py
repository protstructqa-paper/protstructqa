"""Abstract syntax tree (AST) for ProtStructQA.

A program is a single Expression. The Expression tree is built by the
parser in `grammar.py` and traversed by the executor in `executor.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------- base ----------------

@dataclass
class Node:
    """All AST nodes inherit from this for type hinting."""
    pass


# ---------------- literals ----------------

@dataclass
class BoolLit(Node):     value: bool
@dataclass
class IntLit(Node):      value: int
@dataclass
class FloatLit(Node):    value: float
@dataclass
class StringLit(Node):   value: str  # used for "H"/"E"/"C", "A".."Y"


# ---------------- regions / residues ----------------

@dataclass
class ResidueExpr(Node):  # residue(i)
    index: "Expression"
@dataclass
class RangeExpr(Node):    # range(start, end)
    start: "Expression"
    end:   "Expression"
@dataclass
class FirstExpr(Node):    # first(n)
    n: "Expression"
@dataclass
class LastExpr(Node):     # last(n)
    n: "Expression"
@dataclass
class WindowExpr(Node):   # window(start, length)
    start:  "Expression"
    length: "Expression"
@dataclass
class AllResidues(Node): pass


# ---------------- per-residue / per-pair / per-region primitives ----------------

@dataclass
class FuncCall(Node):
    name: str
    args: list["Expression"] = field(default_factory=list)
    kwargs: dict[str, "Expression"] = field(default_factory=dict)


# ---------------- comparison / logical / arithmetic ----------------

@dataclass
class Compare(Node):
    op: str      # "<", "<=", ">", ">=", "==", "!="
    lhs: "Expression"
    rhs: "Expression"

@dataclass
class BinOp(Node):
    op: str      # "+", "-", "*", "/", "**"
    lhs: "Expression"
    rhs: "Expression"

@dataclass
class UnaryOp(Node):
    op: str      # "-", "not"
    operand: "Expression"

@dataclass
class LogicalOp(Node):
    op: str      # "and", "or"
    lhs: "Expression"
    rhs: "Expression"

@dataclass
class Between(Node):
    value: "Expression"
    lo:    "Expression"
    hi:    "Expression"


# ---------------- iterator forms ----------------

@dataclass
class SlidingWindow(Node):
    size: "Expression"
    step: "Expression" = None  # default 1

@dataclass
class AllPairs(Node):
    min_sep: "Expression" = None  # default 1


# ---------------- comprehensions / quantifiers ----------------

@dataclass
class Exists(Node):
    vars: list[str]       # 1 name for residues/regions, 2 names for pairs
    domain: "Expression"  # AllResidues / SlidingWindow / AllPairs / etc.
    predicate: "Expression"

@dataclass
class ForAll(Node):
    vars: list[str]
    domain: "Expression"
    predicate: "Expression"

@dataclass
class Count(Node):
    vars: list[str]
    domain: "Expression"
    predicate: "Expression"

@dataclass
class Filter(Node):
    vars: list[str]
    domain: "Expression"
    predicate: "Expression"

@dataclass
class ArgMin(Node):
    vars: list[str]
    domain: "Expression"
    by: "Expression"
    where: "Expression" = None

@dataclass
class ArgMax(Node):
    vars: list[str]
    domain: "Expression"
    by: "Expression"
    where: "Expression" = None


# ---------------- variable reference ----------------

@dataclass
class Var(Node):
    name: str


# ---------------- conditional ----------------

@dataclass
class IfThenElse(Node):
    cond:  "Expression"
    then:  "Expression"
    else_: "Expression"


# ---------------- pair / tuple ----------------

@dataclass
class PairLit(Node):
    first:  "Expression"
    second: "Expression"


# ---------------- top-level ----------------

@dataclass
class Program(Node):
    expression: "Expression"


# ---------------- expression union ----------------

Expression = Any  # for type-hinting; in practice one of the Node subclasses
