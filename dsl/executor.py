"""ProtStructQA executor: interprets an AST against a ProteinView.

Implements all primitives listed in `specs/DSL_SPEC.md` v0.1.

Domain values (what an Expression can evaluate to):
    bool / int / float / str
    Residue           int (1-indexed)
    Region            (start, end) tuple of ints (1-indexed, inclusive)
    Pair              (Residue, Residue) tuple
    ResidueSet        frozenset of ints
    PairSet           frozenset of (i, j) tuples
    RegionSet         tuple of (start, end) tuples

The executor is purely deterministic given a ProteinView, by design.
"""

from __future__ import annotations

import math
from typing import Any

from .ast_nodes import (
    AllPairs, AllResidues, ArgMax, ArgMin, Between, BinOp, BoolLit, Compare,
    Count, Exists, Filter, FirstExpr, FloatLit, ForAll, FuncCall, IfThenElse,
    IntLit, LastExpr, LogicalOp, PairLit, Program, RangeExpr, ResidueExpr,
    SlidingWindow, StringLit, UnaryOp, Var, WindowExpr,
)
from .grammar import parse
from .protein_view import ProteinView


class ProtStructQAError(RuntimeError):
    """Raised on any execution-time error (out-of-range, type mismatch, etc.)."""


# ----------------------------------------------------------------------
# Type tags (so we can distinguish Region from Pair from Tuple<...>)
# ----------------------------------------------------------------------

class Region(tuple):
    """A region (start, end), 1-indexed inclusive."""
    __slots__ = ()
    def __new__(cls, start: int, end: int):
        return tuple.__new__(cls, (int(start), int(end)))
    @property
    def start(self): return self[0]
    @property
    def end(self):   return self[1]
    def __reduce__(self):
        # Pickle support: tuple subclasses with multi-arg __new__ require an
        # explicit reconstructor. Without this, unpickling calls
        # __new__(cls, state_tuple) and fails with
        # "missing 1 required positional argument: 'end'".
        return (self.__class__, (self[0], self[1]))


class Pair(tuple):
    """A pair of residues."""
    __slots__ = ()
    def __new__(cls, a: int, b: int):
        return tuple.__new__(cls, (int(a), int(b)))
    def __reduce__(self):
        return (self.__class__, (self[0], self[1]))


# ----------------------------------------------------------------------
# Top-level entry points
# ----------------------------------------------------------------------

def execute(ast: Program, view: ProteinView) -> Any:
    return _Eval(view).eval(ast.expression, env={})


def run(source: str, view: ProteinView) -> Any:
    return execute(parse(source), view)


# ----------------------------------------------------------------------
# The evaluator
# ----------------------------------------------------------------------

class _Eval:
    def __init__(self, view: ProteinView):
        self.view = view

    def eval(self, node, env: dict) -> Any:
        if isinstance(node, BoolLit):    return node.value
        if isinstance(node, IntLit):     return node.value
        if isinstance(node, FloatLit):   return node.value
        if isinstance(node, StringLit):  return node.value

        if isinstance(node, Var):
            if node.name in env:
                return env[node.name]
            raise ProtStructQAError(f"unbound variable: {node.name}")

        if isinstance(node, AllResidues):
            return frozenset(range(1, self.view.length() + 1))

        if isinstance(node, ResidueExpr):
            return int(self.eval(node.index, env))
        if isinstance(node, RangeExpr):
            return Region(int(self.eval(node.start, env)),
                          int(self.eval(node.end, env)))
        if isinstance(node, FirstExpr):
            n = int(self.eval(node.n, env))
            return Region(1, n)
        if isinstance(node, LastExpr):
            n = int(self.eval(node.n, env))
            L = self.view.length()
            return Region(L - n + 1, L)
        if isinstance(node, WindowExpr):
            s = int(self.eval(node.start, env))
            l = int(self.eval(node.length, env))
            return Region(s, s + l - 1)
        if isinstance(node, PairLit):
            return Pair(int(self.eval(node.first, env)),
                        int(self.eval(node.second, env)))

        if isinstance(node, SlidingWindow):
            size = int(self.eval(node.size, env))
            step = int(self.eval(node.step, env)) if node.step is not None else 1
            L = self.view.length()
            return tuple(Region(i, i + size - 1) for i in range(1, L - size + 2, step))

        if isinstance(node, AllPairs):
            min_sep = int(self.eval(node.min_sep, env)) if node.min_sep is not None else 1
            L = self.view.length()
            return tuple(Pair(i, j)
                          for i in range(1, L + 1)
                          for j in range(i + min_sep, L + 1))

        if isinstance(node, FuncCall):
            return self._call(node.name, node.args, node.kwargs, env)

        if isinstance(node, BinOp):
            a = self.eval(node.lhs, env); b = self.eval(node.rhs, env)
            if node.op == "+": return a + b
            if node.op == "-": return a - b
            if node.op == "*": return a * b
            if node.op == "/":
                if b == 0:
                    raise ProtStructQAError("division by zero")
                return a / b
            if node.op == "**": return a ** b
            raise ProtStructQAError(f"unknown binop: {node.op}")

        if isinstance(node, UnaryOp):
            if node.op == "-":   return -self.eval(node.operand, env)
            if node.op == "not": return not self.eval(node.operand, env)
            raise ProtStructQAError(f"unknown unary op: {node.op}")

        if isinstance(node, LogicalOp):
            a = self.eval(node.lhs, env)
            if node.op == "and": return a and self.eval(node.rhs, env)
            if node.op == "or":  return a or  self.eval(node.rhs, env)
            raise ProtStructQAError(f"unknown logical op: {node.op}")

        if isinstance(node, Compare):
            a = self.eval(node.lhs, env); b = self.eval(node.rhs, env)
            if node.op == "<":  return a <  b
            if node.op == "<=": return a <= b
            if node.op == ">":  return a >  b
            if node.op == ">=": return a >= b
            if node.op == "==": return a == b
            if node.op == "!=": return a != b
            raise ProtStructQAError(f"unknown cmp op: {node.op}")

        if isinstance(node, Between):
            v = self.eval(node.value, env)
            lo = self.eval(node.lo, env)
            hi = self.eval(node.hi, env)
            return lo <= v <= hi

        if isinstance(node, IfThenElse):
            return self.eval(node.then if self.eval(node.cond, env)
                             else node.else_, env)

        if isinstance(node, Exists):
            domain = self.eval(node.domain, env)
            for x in self._iter(domain):
                if self.eval(node.predicate, self._bind(env, node.vars, x)):
                    return True
            return False

        if isinstance(node, ForAll):
            domain = self.eval(node.domain, env)
            for x in self._iter(domain):
                if not self.eval(node.predicate, self._bind(env, node.vars, x)):
                    return False
            return True

        if isinstance(node, Count):
            domain = self.eval(node.domain, env)
            n = 0
            for x in self._iter(domain):
                if self.eval(node.predicate, self._bind(env, node.vars, x)):
                    n += 1
            return n

        if isinstance(node, Filter):
            domain = self.eval(node.domain, env)
            keep = []
            for x in self._iter(domain):
                if self.eval(node.predicate, self._bind(env, node.vars, x)):
                    keep.append(x)
            if keep and isinstance(keep[0], Pair):
                return frozenset(keep)
            if keep and isinstance(keep[0], Region):
                return tuple(keep)
            return frozenset(keep)

        if isinstance(node, ArgMin):
            return self._argext(node, env, mode="min")
        if isinstance(node, ArgMax):
            return self._argext(node, env, mode="max")

        raise ProtStructQAError(f"unknown AST node: {type(node).__name__}")

    # ------------------------------------------------------------
    # Built-in callables
    # ------------------------------------------------------------

    def _call(self, name: str, args, kwargs, env) -> Any:
        a = [self.eval(x, env) for x in args]
        k = {kk: self.eval(v, env) for kk, v in kwargs.items()}
        v = self.view
        try:
            return self._call_inner(name, a, k, v)
        except (IndexError, ValueError, KeyError, TypeError) as exc:
            raise ProtStructQAError(f"{name}({a}, {k}): {exc}") from exc

    def _call_inner(self, name: str, a, k, v) -> Any:

        # ----- per-residue -----
        if name == "plddt":           return v.plddt_at(int(a[0]))
        if name == "ref_aa":          return v.ref_aa_at(int(a[0]))
        if name == "ss":              return v.ss_at(int(a[0]))
        if name == "sasa":            return v.sasa_at(int(a[0]))
        if name == "rel_sasa":        return v.rel_sasa_at(int(a[0]))
        if name == "n_neighbors":
            radius = float(k.get("radius", a[1] if len(a) > 1 else 8.0))
            return v.n_neighbors_at(int(a[0]), radius=radius)
        if name == "coordinates":     return v.ca_xyz_at(int(a[0]))

        # ----- per-pair -----
        if name in ("distance", "ca_distance"):
            return v.distance(int(a[0]), int(a[1]))
        if name == "cb_distance":
            return v.distance(int(a[0]), int(a[1]))     # Cα fallback in v0.1
        if name == "seq_separation":
            return v.seq_separation(int(a[0]), int(a[1]))
        if name == "pae":
            return v.pae_at(int(a[0]), int(a[1]))

        # ----- per-region -----
        if name == "mean_plddt":     return v.mean_plddt(*self._region_args(a))
        if name == "min_plddt":      return v.min_plddt(*self._region_args(a))
        if name == "max_plddt":      return v.max_plddt(*self._region_args(a))
        if name == "std_plddt":      return v.std_plddt(*self._region_args(a))
        if name == "mean_pae":
            r1 = self._region_args(a[:1] if isinstance(a[0], Region) else a[:2])
            r2 = self._region_args(a[1:2] if isinstance(a[0], Region) else a[2:4])
            return v.mean_pae(r1[0], r1[1], r2[0], r2[1])
        if name == "max_pae":
            r1 = self._region_args(a[:1] if isinstance(a[0], Region) else a[:2])
            r2 = self._region_args(a[1:2] if isinstance(a[0], Region) else a[2:4])
            return v.max_pae(r1[0], r1[1], r2[0], r2[1])
        if name == "count_high_pae":
            # Args: (region_a, region_b, threshold) or
            #       (a_start, a_end, b_start, b_end, threshold)
            if isinstance(a[0], Region):
                r1 = self._region_args(a[:1])
                r2 = self._region_args(a[1:2])
                thr = float(a[2])
            else:
                r1 = self._region_args(a[:2])
                r2 = self._region_args(a[2:4])
                thr = float(a[4])
            return v.count_high_pae(r1[0], r1[1], r2[0], r2[1], thr)
        if name == "contact_density":
            radius = float(k.get("radius", 8.0))
            return v.contact_density(*self._region_args(a), radius=radius)
        if name == "long_range_contacts":
            sep = int(k.get("sep", 12))
            radius = float(k.get("radius", 8.0))
            return v.long_range_contacts(*self._region_args(a),
                                            sep=sep, radius=radius)
        if name == "radius_of_gyration":
            return v.radius_of_gyration(*self._region_args(a))
        if name == "length":
            if len(a) == 1 and isinstance(a[0], Region):
                return a[0].end - a[0].start + 1
            return v.length()
        if name == "mean_rel_sasa":
            return v.mean_rel_sasa(*self._region_args(a))
        if name == "mean_n_neighbors":
            return v.mean_n_neighbors(*self._region_args(a))

        # ----- SS run primitives -----
        if name == "runs":
            label = str(a[0])
            return tuple(Region(s, e) for s, e in v.ss_runs(label))
        if name == "longest_run":
            label = str(a[0])
            rs = v.ss_runs(label)
            if not rs:
                raise ProtStructQAError(f"no runs of '{label}' found")
            s, e = max(rs, key=lambda se: se[1] - se[0] + 1)
            return Region(s, e)

        # ----- protein-level -----
        if name == "protein_length":     return v.length()
        if name == "n_helices":          return v.n_helices()
        if name == "n_strands":          return v.n_strands()
        if name == "mean_protein_plddt": return v.mean_protein_plddt()

        # ----- set ops -----
        if name == "union":          return frozenset(a[0]) | frozenset(a[1])
        if name == "intersection":   return frozenset(a[0]) & frozenset(a[1])
        if name == "difference":     return frozenset(a[0]) - frozenset(a[1])
        if name == "contains":       return a[1] in a[0]
        if name == "size":           return len(a[0])
        if name == "to_set":
            r = a[0]
            return frozenset(range(r.start, r.end + 1))
        if name == "in_region":
            i, r = int(a[0]), a[1]
            return r.start <= i <= r.end

        # ----- math helpers -----
        if name == "abs":   return abs(a[0])
        if name == "min":   return min(a)
        if name == "max":   return max(a)
        if name == "round": return round(a[0], int(a[1]) if len(a) > 1 else 0)
        if name == "floor": return math.floor(a[0])
        if name == "ceil":  return math.ceil(a[0])

        raise ProtStructQAError(f"unknown function: {name}")

    @staticmethod
    def _region_args(values) -> tuple[int, int]:
        """Accept either Region(start, end) or (start, end) ints; return (start, end)."""
        if len(values) == 1 and isinstance(values[0], Region):
            return values[0].start, values[0].end
        if len(values) >= 2 and not isinstance(values[0], Region):
            return int(values[0]), int(values[1])
        if len(values) == 2 and isinstance(values[0], Region):
            return values[0].start, values[0].end
        raise ProtStructQAError(f"could not interpret region args: {values}")

    @staticmethod
    def _iter(domain):
        if isinstance(domain, frozenset):
            return iter(sorted(domain))
        if isinstance(domain, (tuple, list)):
            return iter(domain)
        if isinstance(domain, Region):
            return iter(range(domain.start, domain.end + 1))
        raise ProtStructQAError(f"cannot iterate over {type(domain).__name__}")

    @staticmethod
    def _bind(env: dict, names: list[str], value) -> dict:
        """Bind one or two names from env. Single-name binds the whole value;
        two-name binders unpack a Pair / 2-tuple."""
        if len(names) == 1:
            return {**env, names[0]: value}
        if len(names) == 2:
            if not (isinstance(value, tuple) and len(value) == 2):
                raise ProtStructQAError(
                    f"cannot unpack {type(value).__name__} into "
                    f"({names[0]}, {names[1]})")
            return {**env, names[0]: value[0], names[1]: value[1]}
        raise ProtStructQAError(f"unsupported binder arity {len(names)}")

    def _argext(self, node, env, *, mode: str):
        domain = self.eval(node.domain, env)
        best, best_score = None, None
        for x in self._iter(domain):
            local = self._bind(env, node.vars, x)
            if node.where is not None and not self.eval(node.where, local):
                continue
            score = self.eval(node.by, local)
            if best is None or (
                (mode == "min" and score < best_score) or
                (mode == "max" and score > best_score)
            ):
                best, best_score = x, score
        if best is None:
            raise ProtStructQAError(f"argmin/argmax over empty domain")
        return best
