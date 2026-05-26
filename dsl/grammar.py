"""ProtStructQA grammar (Lark) and parser.

Grammar in EBNF-like form (see specs/DSL_SPEC.md for full semantics):

    program     ::= expression
    expression  ::= or_expr
    or_expr     ::= and_expr ("or" and_expr)*
    and_expr    ::= not_expr ("and" not_expr)*
    not_expr    ::= "not" not_expr | comparison
    comparison  ::= add_expr (CMPOP add_expr)?
                  | "between" "(" add_expr "," add_expr "," add_expr ")"
    add_expr    ::= mul_expr (("+" | "-") mul_expr)*
    mul_expr    ::= pow_expr (("*" | "/") pow_expr)*
    pow_expr    ::= unary ("**" unary)*
    unary       ::= "-" unary | atom
    atom        ::= literal | call | comprehension | "(" expression ")" | NAME
    call        ::= NAME "(" [args] ")"
    args        ::= arg ("," arg)*
    arg         ::= expression | NAME "=" expression
    comprehension ::= QUANT NAME "in" expression "where" expression
                    | "argmin" NAME "in" expression "by" expression
                          ["where" expression]
                    | "argmax" NAME "in" expression "by" expression
                          ["where" expression]
                    | "if" expression "then" expression "else" expression

    QUANT       ::= "exists" | "forall" | "count" | "filter"
    CMPOP       ::= "<" | "<=" | ">" | ">=" | "==" | "!="
"""

from __future__ import annotations

from lark import Lark, Transformer, v_args

from .ast_nodes import (
    AllPairs, AllResidues, ArgMax, ArgMin, Between, BinOp, BoolLit, Compare,
    Count, Exists, Filter, FirstExpr, FloatLit, ForAll, FuncCall, IfThenElse,
    IntLit, LastExpr, LogicalOp, PairLit, Program, RangeExpr, ResidueExpr,
    SlidingWindow, StringLit, UnaryOp, Var, WindowExpr,
)


_GRAMMAR = r"""
start: expression

?expression: or_expr

?or_expr: and_expr
        | or_expr "or" and_expr  -> or_op

?and_expr: not_expr
         | and_expr "and" not_expr  -> and_op

?not_expr: "not" not_expr  -> not_op
        | comparison

?comparison: add_expr
           | add_expr CMPOP add_expr  -> compare
           | "between" "(" add_expr "," add_expr "," add_expr ")" -> between

?add_expr: mul_expr
        | add_expr "+" mul_expr  -> add
        | add_expr "-" mul_expr  -> sub

?mul_expr: pow_expr
        | mul_expr "*" pow_expr  -> mul
        | mul_expr "/" pow_expr  -> div

?pow_expr: unary
        | unary "**" pow_expr  -> pow

?unary: "-" unary             -> neg
      | atom

?atom: literal
     | callable
     | "(" expression ")"
     | NAME                    -> var

?literal: SIGNED_INT           -> int_lit
        | SIGNED_FLOAT         -> float_lit
        | "true"               -> true_lit
        | "false"              -> false_lit
        | ESCAPED_STRING       -> string_lit
        | "all_residues"       -> all_residues_lit

callable: comprehension | call_expr

call_expr: NAME "(" [args] ")"

args: arg ("," arg)*
arg: NAME "=" expression  -> kwarg
   | expression           -> posarg

comprehension: "exists" binder "in" expression "where" expression  -> exists
             | "forall" binder "in" expression "where" expression  -> forall
             | "count" binder "in" expression "where" expression   -> count
             | "filter" binder "in" expression "where" expression  -> filter
             | "argmin" binder "in" expression "by" expression "where" expression -> argmin
             | "argmin" binder "in" expression "by" expression                    -> argmin_nowhere
             | "argmax" binder "in" expression "by" expression "where" expression -> argmax
             | "argmax" binder "in" expression "by" expression                    -> argmax_nowhere
             | "if" expression "then" expression "else" expression                -> if_then_else

binder: NAME                          -> single_binder
      | "(" NAME "," NAME ")"          -> pair_binder

CMPOP: "<=" | ">=" | "==" | "!=" | "<" | ">"
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.SIGNED_INT
%import common.SIGNED_FLOAT
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


# Names of built-in callables (vs user comprehensions).
_BUILTIN_NAMES = {
    # Region / residue constructors
    "residue", "range", "first", "last", "window",
    # Iterator forms
    "sliding_window", "all_pairs",
    # Per-residue
    "plddt", "ref_aa", "ss", "sasa", "rel_sasa", "n_neighbors", "coordinates",
    # Per-pair
    "distance", "ca_distance", "cb_distance", "seq_separation", "pae",
    # Per-region
    "mean_plddt", "min_plddt", "max_plddt", "std_plddt",
    "mean_pae", "contact_density", "long_range_contacts",
    "radius_of_gyration", "length",
    "mean_rel_sasa", "mean_n_neighbors",
    # Secondary-structure runs
    "runs", "longest_run",
    # Protein-level
    "protein_length", "n_helices", "n_strands", "mean_protein_plddt",
    # Set ops
    "union", "intersection", "difference", "contains", "size",
    "to_set", "in_region",
    # Helpers
    "abs", "min", "max", "round", "floor", "ceil",
    # Pair literals
    "pair",
}


class _AstBuilder(Transformer):
    """Lark Transformer → AST nodes."""

    # ----- literals -----
    @v_args(inline=True)
    def int_lit(self, tok):    return IntLit(int(tok))
    @v_args(inline=True)
    def float_lit(self, tok):  return FloatLit(float(tok))
    def true_lit(self, _):     return BoolLit(True)
    def false_lit(self, _):    return BoolLit(False)
    @v_args(inline=True)
    def string_lit(self, tok): return StringLit(tok[1:-1])  # strip quotes
    def all_residues_lit(self, _): return AllResidues()

    # ----- vars -----
    @v_args(inline=True)
    def var(self, tok):        return Var(str(tok))

    # ----- arithmetic / logical / comparison -----
    @v_args(inline=True)
    def add(self, lhs, rhs):  return BinOp("+", lhs, rhs)
    @v_args(inline=True)
    def sub(self, lhs, rhs):  return BinOp("-", lhs, rhs)
    @v_args(inline=True)
    def mul(self, lhs, rhs):  return BinOp("*", lhs, rhs)
    @v_args(inline=True)
    def div(self, lhs, rhs):  return BinOp("/", lhs, rhs)
    @v_args(inline=True)
    def pow(self, lhs, rhs):  return BinOp("**", lhs, rhs)
    @v_args(inline=True)
    def neg(self, x):         return UnaryOp("-", x)
    @v_args(inline=True)
    def or_op(self, lhs, rhs):  return LogicalOp("or",  lhs, rhs)
    @v_args(inline=True)
    def and_op(self, lhs, rhs): return LogicalOp("and", lhs, rhs)
    @v_args(inline=True)
    def not_op(self, x):        return UnaryOp("not", x)
    @v_args(inline=True)
    def compare(self, lhs, op, rhs): return Compare(str(op), lhs, rhs)
    @v_args(inline=True)
    def between(self, v, lo, hi):    return Between(v, lo, hi)

    # ----- args / kwargs -----
    def args(self, items):
        # returns (positional_list, kwarg_dict)
        positional, kw = [], {}
        for kind, *payload in items:
            if kind == "pos":
                positional.append(payload[0])
            else:
                k, v = payload
                kw[k] = v
        return positional, kw

    def posarg(self, items):
        return ("pos", items[0])

    def kwarg(self, items):
        return ("kw", str(items[0]), items[1])

    # ----- callables -----
    def call_expr(self, items):
        # items: [NAME, args] or [NAME] if no args
        name = str(items[0])
        positional, kw = ([], {})
        if len(items) > 1 and items[1] is not None:
            positional, kw = items[1]
        # Some names are first-class constructors → AST nodes; others go via FuncCall.
        if name == "residue":
            return ResidueExpr(positional[0])
        if name == "range":
            return RangeExpr(positional[0], positional[1])
        if name == "first":
            return FirstExpr(positional[0])
        if name == "last":
            return LastExpr(positional[0])
        if name == "window":
            return WindowExpr(positional[0], positional[1])
        if name == "sliding_window":
            step = kw.get("step")
            if step is None and len(positional) > 1:
                step = positional[1]
            return SlidingWindow(size=positional[0], step=step)
        if name == "all_pairs":
            min_sep = kw.get("min_sep")
            if min_sep is None and positional:
                min_sep = positional[0]
            return AllPairs(min_sep=min_sep)
        if name == "pair":
            return PairLit(positional[0], positional[1])
        if name in _BUILTIN_NAMES:
            return FuncCall(name=name, args=positional, kwargs=kw)
        # Anything else: treat as a generic FuncCall (user is on their own).
        return FuncCall(name=name, args=positional, kwargs=kw)

    # ----- binders -----
    def single_binder(self, items):
        return [str(items[0])]
    def pair_binder(self, items):
        return [str(items[0]), str(items[1])]

    # ----- comprehensions -----
    def exists(self, items):
        return Exists(vars=items[0], domain=items[1], predicate=items[2])
    def forall(self, items):
        return ForAll(vars=items[0], domain=items[1], predicate=items[2])
    def count(self, items):
        return Count(vars=items[0], domain=items[1], predicate=items[2])
    def filter(self, items):
        return Filter(vars=items[0], domain=items[1], predicate=items[2])
    def argmin(self, items):
        return ArgMin(vars=items[0], domain=items[1], by=items[2], where=items[3])
    def argmin_nowhere(self, items):
        return ArgMin(vars=items[0], domain=items[1], by=items[2], where=None)
    def argmax(self, items):
        return ArgMax(vars=items[0], domain=items[1], by=items[2], where=items[3])
    def argmax_nowhere(self, items):
        return ArgMax(vars=items[0], domain=items[1], by=items[2], where=None)
    def if_then_else(self, items):
        return IfThenElse(cond=items[0], then=items[1], else_=items[2])

    def callable(self, items): return items[0]

    # ----- top-level -----
    def start(self, items): return Program(expression=items[0])


_PARSER = Lark(_GRAMMAR, start="start", parser="lalr",
                maybe_placeholders=False)
_BUILDER = _AstBuilder()


def parse(source: str) -> Program:
    """Parse a ProtStructQA source string into a Program AST."""
    tree = _PARSER.parse(source)
    return _BUILDER.transform(tree)


def parse_grammar() -> str:
    """Return the underlying Lark grammar string (useful for the paper appendix)."""
    return _GRAMMAR
