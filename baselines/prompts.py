"""Prompt construction for ProtStructQA baselines.

Three prompting modes share most of this scaffolding:
  - L0 zero-shot:   protein context + DSL primer + question → "answer or program"
  - L1 grammar-constrained:  same as L0 plus xgrammar-enforced output schema
  - L2 ReAct:       multi-turn with tool calls (defer to runner)

Few-shot exemplars: 4-8 (question, program, answer) triples drawn from the
TRAIN split, balanced across families (avoid leaking template-specific
hints). For Family Hb (non-prompted abstention), exemplars must include
both confidently-answerable AND unreliable cases so the model learns the
pLDDT threshold from data rather than from the question text.

Protein context: provided as a compact text summary derived from the
ProteinView (length, mean pLDDT, n_helices, n_strands, plus a per-residue
pLDDT band sequence to give the model location-of-low-confidence
information).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dsl import ProteinView


# Env-gated hint: when PROTSTRUCTQA_TYPE_HINT=1, inject a type-aware primitive
# block into L0/L1 prompts. The block is derived directly from the DSL's
# return-type signatures (mechanically, not hand-engineered against
# observed failure modes). Default OFF preserves prior baselines.
#
# CLEANER VERSION: only positive type-derived guidance; no "Do NOT use X"
# anti-pattern bans (which were hand-engineered against observed
# primitive-selection failures and conflated principled type-info with
# benchmark-specific hardcoding).
def _type_hint_for_prompt(answer_type: str | None) -> str:
    if os.environ.get("PROTSTRUCTQA_TYPE_HINT", "0") != "1":
        return ""
    base = (answer_type or "").split("|", 1)[0].strip()
    # Each entry lists ONLY the DSL primitives whose return type matches
    # the expected answer_type. This information is mechanical (read off
    # the DSL spec's typing rules); no anti-patterns are encoded.
    primitive_map = {
        "Bool":       "primitives that return Bool include `exists ... where ...`, "
                      "`forall ... where ...`, and comparison expressions "
                      "of the form `f(...) op c` where f returns a numeric "
                      "value and op is one of <, <=, >, >=, ==, !=.",
        "Int":        "primitives that return Int include `count(... where ...)` "
                      "and `length(...)`.",
        "Float":      "primitives that return Float include `mean_plddt(...)`, "
                      "`distance(...)`, `mean_pae(...)`, `contact_density(...)`, "
                      "and `rel_sasa(...)`.",
        "Region":     "primitives that return Region include `argmax ... by ...`, "
                      "`argmin ... by ...`, and `longest_run(...)`.",
        "ResidueSet": "primitives that return ResidueSet are of the form "
                      "`filter r in ... where ...`.",
        "PairSet":    "primitives that return PairSet are of the form "
                      "`filter (r1, r2) in all_pairs(...) where ...`.",
        "SecStruct":  "primitives that return SecStruct include `ss(residue(N))`.",
    }
    info = primitive_map.get(base, "")
    if not info:
        return ""
    return (f"# Expected answer type: {base}\n"
            f"# Type information: {info}\n")


# --------------------------- DSL primer ------------------------------ #


DSL_PRIMER = r"""
ProtStructQA is a small executable language for questions about a single
AlphaFold predicted protein structure. The available primitives are:

  Per-residue:
    plddt(r)            -- pLDDT in [0, 100]
    ss(r)               -- "H" / "E" / "C" (helix / strand / coil)
    rel_sasa(r)         -- relative solvent-accessible surface in [0, ~1]
    sasa(r)             -- absolute SASA in Å²
    n_neighbors(r, radius=8.0)
    coordinates(r)      -- (x, y, z)

  Per-pair:
    distance(r1, r2)    -- CA-CA Å
    pae(r1, r2)         -- predicted aligned error in Å
    seq_separation(r1, r2)

  Per-region (range(start, end)):
    mean_plddt(reg), min_plddt(reg), max_plddt(reg), std_plddt(reg)
    mean_pae(reg_a, reg_b), contact_density(reg, radius=8.0)
    long_range_contacts(reg, sep=12), radius_of_gyration(reg)
    mean_rel_sasa(reg), length(reg)

  Protein-level:
    protein_length(), n_helices(), n_strands(), mean_protein_plddt()

  Comparison/logic:  < <= > >= == != and or not, between(x, lo, hi)
  Aggregation:       count r in S where P(r), exists r in S where P(r),
                     forall r in S where P(r), filter r in S where P(r),
                     argmin/argmax r in S by f(r)
  Region quantifier: argmin/argmax/exists/filter reg in sliding_window(size) ...
  Pair quantifier:   filter (i, j) in all_pairs(min_sep=K) where ...
  Conditional:       if cond then e1 else e2

Residue indices are 1-based. Region notation: range(50, 100) = residues 50 to 100 inclusive.
"""


# --------------------------- protein summary ------------------------- #


def protein_summary(view: ProteinView, max_band_len: int = 80) -> str:
    """Compact text summary of a ProteinView for the LLM context.

    Includes: UniProt + species + length + mean pLDDT + n_helices/strands +
    per-residue pLDDT band (downsampled to max_band_len characters).
    """
    L = view.length()
    mean_pl = view.mean_protein_plddt()
    n_h = view.n_helices()
    n_s = view.n_strands()

    # Downsample pLDDT to a band sequence
    plddt = view.plddt
    if L > max_band_len:
        idx = np.linspace(0, L - 1, max_band_len).astype(int)
        plddt_ds = plddt[idx]
    else:
        plddt_ds = plddt
    band = "".join(_plddt_char(p) for p in plddt_ds)

    # Downsample SS analogously
    ss = view.ss_3
    if L > max_band_len:
        ss_ds = ss[idx]
    else:
        ss_ds = ss
    ss_band = "".join(ss_ds.tolist())

    return (
        f"Protein: {view.uniprot} ({view.species})\n"
        f"  Length: {L} residues\n"
        f"  Mean pLDDT: {mean_pl:.1f}\n"
        f"  Helix runs: {n_h}, strand runs: {n_s}\n"
        f"  pLDDT band ({max_band_len} bins, range {min(plddt_ds):.0f}-{max(plddt_ds):.0f}): {band}\n"
        f"  SS band:   {ss_band}\n"
        f"  Legend for pLDDT band: '#' >=90, '+' 70-89, '.' 50-69, '?' <50.\n"
    )


def _plddt_char(p: float) -> str:
    if p >= 90:
        return "#"
    if p >= 70:
        return "+"
    if p >= 50:
        return "."
    return "?"


# --------------------------- few-shot ------------------------------ #


def build_few_shot_block(exemplars: list[dict], n: int = 6) -> str:
    """Render a list of (question, program, answer) triples as a few-shot
    block. Caller is responsible for picking exemplars (balanced across
    families, ideally from TRAIN split)."""
    lines = ["Examples:"]
    for i, ex in enumerate(exemplars[:n], 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Program: {ex['program']}")
        ans = ex["answer"]
        if isinstance(ans, float):
            ans_str = f"{ans:.3f}"
        else:
            ans_str = str(ans)
        lines.append(f"Answer: {ans_str}")
    return "\n".join(lines)


# PAL-style (Gao+23, Chen+22) reasoning derivation: from the exemplar's
# gold program + answer_type, derive 2-3 # comments that the model can
# emulate. This is the canonical "reasoning as inline comments" pattern
# used by PAL and Program-of-Thoughts.
_PRIMITIVE_DESCRIPTIONS = {
    "mean_plddt":        "aggregate per-residue pLDDT over a region",
    "mean_pae":          "aggregate pairwise alignment error over a region pair",
    "distance":          "compute CA-CA Euclidean distance between two residues",
    "pae":               "look up pairwise alignment error for a residue pair",
    "plddt":             "look up per-residue pLDDT confidence",
    "ss":                "look up DSSP three-state secondary structure (H/E/C)",
    "rel_sasa":          "compute relative solvent-accessible surface area",
    "n_neighbors":       "count residues within a radius of a target",
    "n_helices":         "count alpha-helix segments",
    "n_strands":         "count beta-strand segments",
    "contact_density":   "compute long-range contact density",
    "radius_of_gyration":"compute compactness via radius of gyration",
    "seq_separation":    "compute sequence-distance between two residues",
    "exists":            "existential quantifier over a domain",
    "forall":            "universal quantifier over a domain",
    "filter":            "filter elements by a predicate",
    "count":             "count elements satisfying a predicate",
    "argmin":            "find the element minimizing a value",
    "argmax":            "find the element maximizing a value",
    "sliding_window":    "iterate over sliding residue windows",
    "all_pairs":         "iterate over residue pairs (with min_sep)",
    "all_residues":      "iterate over all residues",
    "range":             "form a contiguous residue region [a, b]",
    "residue":           "reference a single residue by index",
    "length":            "size of a set / region",
}

import re as _re
_TOPCALL = _re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*\(")


def _pal_reasoning(ex: dict) -> list[str]:
    """Derive 2-3 PAL-style reasoning comments from an exemplar's gold program.
    Comments describe: (a) the answer type, (b) the top-level primitive's
    role, (c) any nested primitives that carry semantic load."""
    prog = (ex.get("program") or "").strip()
    ans_type = (ex.get("answer_type") or "").split("|", 1)[0].strip()
    lines = []
    if ans_type:
        lines.append(f"# The answer type is {ans_type}.")
    # Top-level primitive
    m = _TOPCALL.match(prog)
    if m:
        top = m.group(1)
        desc = _PRIMITIVE_DESCRIPTIONS.get(top)
        if desc:
            lines.append(f"# Use `{top}` to {desc}.")
    # Detect predicate / comparison structure
    if any(op in prog for op in ["<", ">", "<=", ">=", "==", "!="]):
        if not any("predicate" in L or "compari" in L for L in lines):
            lines.append("# The expression compares a computed value against a numeric threshold.")
    # Detect composition (filter / exists / forall over sliding_window or all_pairs)
    if "sliding_window" in prog and any(q in prog for q in ["exists", "forall", "filter"]):
        lines.append("# Slide over fixed-size residue windows and test a predicate per window.")
    elif "all_pairs" in prog and any(q in prog for q in ["filter", "count", "exists"]):
        lines.append("# Iterate over residue pairs and apply a predicate.")
    # Fallback: at least one comment if neither primitive nor predicate matched
    if not lines:
        lines.append("# Identify the primitive and parameters from the question.")
    return lines[:3]


def build_few_shot_block_pal(exemplars: list[dict], n: int = 4) -> str:
    """Render exemplars in PAL-style (Gao+23, Chen+22): reasoning embedded
    as `#` comments preceding the program expression. No separate
    "Reasoning:" preamble: comments are interleaved with code as in PAL."""
    lines = ["Examples (reasoning shown as `#` comments before each program):"]
    for i, ex in enumerate(exemplars[:n], 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Question: {ex['question']}")
        for c in _pal_reasoning(ex):
            lines.append(c)
        lines.append(f"Program: {ex['program']}")
        ans = ex["answer"]
        ans_str = f"{ans:.3f}" if isinstance(ans, float) else str(ans)
        lines.append(f"Answer: {ans_str}")
    return "\n".join(lines)


# --------------------------- prompt assembly ----------------------- #


def build_l0_prompt(question: dict, view: ProteinView,
                       exemplars: list[dict] | None = None,
                       n_shots: int = 6,
                       cot_prefix: bool = False,
                       **kwargs) -> str:
    """Compose a zero-shot (or few-shot if exemplars provided) L0 prompt.

    Output should be a single ProtStructQA program OR a direct answer (model's
    choice). Output parser handles both.

    `cot_prefix=True` prepends the same step-by-step reasoning checklist
    used in `build_l1_prompt(cot_prefix=True)`. This isolates the
    reasoning-prefix effect (no grammar constraint, no k-sample consensus)
    as an ablation against EV+CoT.
    """
    parts = [
        DSL_PRIMER.strip(),
        protein_summary(view),
    ]
    cot_variant = kwargs.get('cot_variant', 0) if cot_prefix else 0
    if exemplars:
        # v3 (PAL-style): exemplars carry inline-comment reasoning;
        # other variants use the plain Q/Program/Answer format.
        if cot_variant == 3:
            parts.append(build_few_shot_block_pal(exemplars, n_shots))
        else:
            parts.append(build_few_shot_block(exemplars, n_shots))
    type_hint = _type_hint_for_prompt(question.get("answer_type"))
    if cot_prefix:
        if cot_variant == 1:
            # Variant 1: generic "think step by step" (minimal CoT)
            parts.append(
                "Think step by step before answering.\n"
                f"{type_hint}"
                "Now answer the question. Output a ProtStructQA program or "
                "direct answer.\n\n"
                f"Question: {question['question']}"
            )
        elif cot_variant == 2:
            # Variant 2: 2-step brief checklist
            parts.append(
                "Before answering, identify: "
                "(1) the DSL primitive(s) needed; "
                "(2) any explicit numeric thresholds to preserve.\n"
                f"{type_hint}"
                "Now answer the question. Output a ProtStructQA program or "
                "direct answer.\n\n"
                f"Question: {question['question']}"
            )
        elif cot_variant == 3:
            # Variant 3: PAL-style (Gao+23, Chen+22): reasoning carried
            # by exemplar comments (rendered via build_few_shot_block_pal
            # above), no explicit checklist or trigger. Mirror the
            # exemplar pattern: emit comments before the program.
            parts.append(
                "Following the exemplar pattern above, emit your reasoning "
                "as `#` comment lines, then a single `Program:` line with "
                "the ProtStructQA expression.\n"
                f"{type_hint}\n"
                f"Question: {question['question']}"
            )
        else:
            # Variant 0 (default): original 4-step checklist
            parts.append(
                "Approach this carefully: for each question, mentally check:\n"
                "  (1) What is the question asking for? (a Region? a Bool? a count?)\n"
                "  (2) Which DSL primitive(s) does it need? (e.g., for "
                "compositional questions, you may need to combine `filter`, "
                "`exists`, `argmax`, with conditions joined by `and`/`or`.)\n"
                "  (3) What's the right scope? (`all_residues`, "
                "`sliding_window(N)`, a specific `range(a,b)`?)\n"
                "  (4) Are the thresholds in the question explicit "
                "(e.g., 'pLDDT > 70')? Use those exact values.\n"
                f"{type_hint}"
                "Now answer the question. You may output a single ProtStructQA "
                "program (preferred: we will execute it) or the answer "
                "directly.\n\n"
                f"Question: {question['question']}"
            )
    else:
        parts.append(
            f"{type_hint}"
            "Now answer the following question. "
            "You may either output a single ProtStructQA program (preferred: "
            "we will execute it), or output the answer directly.\n\n"
            f"Question: {question['question']}"
        )
    return "\n\n".join(parts)


def build_l1_prompt(question: dict, view: ProteinView,
                       exemplars: list[dict] | None = None,
                       n_shots: int = 6,
                       cot_prefix: bool = False) -> str:
    """L1 grammar-constrained variant: the prompt asks for ONLY a
    program (no prose). The actual grammar enforcement happens at the
    sampler level via xgrammar.

    `cot_prefix=True` adds an in-prompt step-by-step reasoning checklist
    (no output format change: grammar still forces program-only output).
    Targets compositional G family where multi-step reasoning helps.
    """
    parts = [
        DSL_PRIMER.strip(),
        protein_summary(view),
    ]
    if exemplars:
        parts.append(build_few_shot_block(exemplars, n_shots))
    type_hint = _type_hint_for_prompt(question.get("answer_type"))
    if cot_prefix:
        parts.append(
            "Approach this carefully: for each question, mentally check:\n"
            "  (1) What is the question asking for? (a Region? a Bool? a count?)\n"
            "  (2) Which DSL primitive(s) does it need? (e.g., for "
            "compositional questions, you may need to combine `filter`, "
            "`exists`, `argmax`, with conditions joined by `and`/`or`.)\n"
            "  (3) What's the right scope? (`all_residues`, "
            "`sliding_window(N)`, a specific `range(a,b)`?)\n"
            "  (4) Are the thresholds in the question explicit "
            "(e.g., 'pLDDT > 70')? Use those exact values.\n"
            f"{type_hint}"
            "Now output ONE valid ProtStructQA program (program only, no prose).\n\n"
            f"Question: {question['question']}\n\n"
            "Program:"
        )
    else:
        parts.append(
            f"{type_hint}"
            "Output a single ProtStructQA program that answers the question. "
            "Do NOT include any prose, explanation, or formatting: just the program.\n\n"
            f"Question: {question['question']}\n\n"
            "Program:"
        )
    return "\n\n".join(parts)


# --------------------------- L2 ReAct ------------------------------ #


L2_V3_TOOL_PRIMER = r"""
You answer a question about an AlphaFold-predicted protein structure.

DEFAULT BEHAVIOR: answer directly. Most questions can be answered
from the protein summary already provided. In that case emit only:
    <think>brief reasoning</think>
    <answer>final answer</answer>
DO NOT invoke tools when you can answer from context. Tool use is
costly and frequently wrong; it is reserved for genuine aggregation.

WHEN TO USE TOOLS: only when the question requires an aggregate
over many residues (argmin / argmax / count / scan over a sliding
window or all_residues), the protein summary does not contain the
needed value (radius_of_gyration, long_range_contacts), or you are
genuinely uncertain after thinking. In that case emit one
    <act>tool_name(args)</act>
and STOP. The system will return <obs>...</obs> and you may continue.

You have TWO tools (kept minimal to reduce decision noise):

  summarize_region(start, end)
      Returns mean_plddt, min_plddt, max_plddt, std_plddt,
      mean_rel_sasa, n_helix_residues, n_strand_residues,
      n_coil_residues, length, radius_of_gyration for residues
      [start..end].

  run_dsl("<ProtStructQA program>")
      Executes any ProtStructQA program against the protein and returns
      its result. Use for argmin / argmax / count / exists / forall /
      filter / contact_density / long_range_contacts / etc. Pass the
      program as a STRING, with positional args:
          run_dsl("argmin reg in sliding_window(30) by mean_plddt(reg)")
          run_dsl("count r in all_residues where plddt(r) > 90")
          run_dsl("long_range_contacts(range(1, 200))")

Format every turn EXACTLY as one of:
    <think>brief reasoning</think><act>tool_name(args)</act>
    <think>brief reasoning</think><answer>final answer</answer>

Final-answer formats (output ONLY the bare value, no prose, no units):
  - if the answer is yes/no       --> True   or  False
  - if it is a count or measure   --> a single bare number, e.g. 3 or 18.7
  - if it is a residue range      --> [start, end], e.g. [120, 149]
  - if it is one residue's SS     --> H  or  E  or  C   (single capital letter)
  - if it is a set of residues    --> a JSON list, e.g. [1, 14, 22]

Always provide your best concrete answer. Do not refuse, hedge, or
abstain unless the question itself explicitly tells you to do so.

Three short worked examples follow. Note examples 1 and 2 use NO tool.

EXAMPLE 1 (point lookup -- direct answer, no tool):
Question: What secondary structure is at residue 137?
<think>The protein summary's pLDDT/SS sequence shows H at position 137.</think><answer>H</answer>

EXAMPLE 2 (Bool from summary -- direct answer, no tool):
Question: Does the protein contain at least one alpha-helix?
<think>The protein summary lists n_helices=4, so yes.</think><answer>True</answer>

EXAMPLE 3 (compositional aggregation -- run_dsl):
Question: Which 30-residue window has the lowest mean pLDDT?
<think>Argmin over sliding windows; the summary has aggregate stats but not per-window. Use run_dsl.</think><act>run_dsl("argmin reg in sliding_window(30) by mean_plddt(reg)")</act>
<obs>result = [180, 209]</obs><answer>[180, 209]</answer>
"""


L2_TOOL_PRIMER = r"""
You are an agent investigating an AlphaFold-predicted protein
structure. You answer the question by calling tools in a ReAct loop:
emit one <act>tool_name(args)</act> tag and STOP. The system returns
<obs>...</obs>. Plan the next step from the observed value, call
another tool, or commit a final <answer>.

=== TOOLS (agent-side; names start with `tool_*`) ===

  tool_inspect(i)
      Returns pLDDT, SS (H/E/C), rel_sasa, aa for residue i (1-based).

  tool_distance(i, j)
      CA-CA Angstroms between residues i and j.

  tool_pae_mean(a_start, a_end, b_start, b_end)
      Mean PAE between residue range [a_start..a_end] and [b_start..b_end].

  tool_region_stats(start, end)
      mean_plddt, min_plddt, max_plddt, std_plddt, mean_rel_sasa,
      n_helix_residues / n_strand_residues / n_coil_residues, length,
      radius_of_gyration for residues [start..end].

  tool_dsl("<ProtStructQA program>")
      Executes an arbitrary ProtStructQA DSL program (see grammar below).
      Use for aggregates: argmin/argmax/exists/forall/count over
      sliding_window(K) / all_residues / all_pairs.

=== PROTSTRUCTQA DSL QUICK-REFERENCE (use ONLY these inside tool_dsl) ===

The DSL is a SEPARATE namespace. `tool_*` names are AGENT actions and
are INVALID inside a tool_dsl program. Inside tool_dsl, use these:

  Per-residue:    plddt(r), ss(r), rel_sasa(r), n_neighbors(r, radius)
  Per-pair:       distance(r1, r2), pae(r1, r2), seq_separation(r1, r2)
  Region:         mean_plddt(reg), min_plddt(reg), max_plddt(reg),
                  contact_density(reg), radius_of_gyration(reg),
                  rel_sasa_mean(reg), mean_pae(reg1, reg2)
  Whole-protein:  long_range_contacts(), n_helices(), n_strands()
  Constructors:   residue(i), range(s, e), sliding_window(K),
                  all_residues, all_pairs(min_sep=K)
  Comprehensions: filter r in COLL where COND, count r in COLL where COND,
                  exists r in COLL where COND, forall r in COLL where COND,
                  argmin/argmax expr in COLL by EXPR

WRONG (will error):
    tool_dsl("count r in all_residues where tool_distance(r, 50) < 8")
                                            ^^^^^^^^^^^^^: tool_* invalid here
    tool_dsl("count r1, r2 where compute_distance(r1, r2) < 10")
                                  ^^^^^^^^^^^^^^^^: not a DSL primitive

RIGHT:
    tool_dsl("count (i, j) in all_pairs(min_sep=20) where distance(i, j) < 8")

=== FORMAT (per turn, EXACTLY one of) ===
    <think>brief reasoning</think>
    <act>tool_name(args)</act>
or, when ready to commit:
    <think>brief reasoning</think>
    <answer>final answer</answer>

If a tool returns "tool_dsl error: ..." or unexpected output, REVISE the
program syntax or try a different tool. Do NOT keep retrying the same
malformed expression. Commit a <answer> if you have enough information.

=== FINAL-ANSWER FORMATS ===
  - Bool:        "True" or "False"
  - Int / Float: a single number, no units
  - Region:      [start, end]   (e.g., [120, 149])
  - ResidueSet:  list of residue indices, e.g., [12, 47, 89]
  - SecStruct:   "H", "E", or "C"

=== WORKED EXAMPLES ===

EXAMPLE 1 (aggregate, single-shot):
Question: Which 30-residue window has the lowest mean pLDDT?
<think>Argmin over sliding windows; use tool_dsl directly.</think>
<act>tool_dsl("argmin reg in sliding_window(30) by mean_plddt(reg)")</act>
<obs>result = [180, 209]</obs>
<answer>[180, 209]</answer>

EXAMPLE 2 (point query, use agent tool):
Question: What secondary structure is at residue 137?
<think>One residue lookup; use tool_inspect.</think>
<act>tool_inspect(137)</act>
<obs>residue 137: pLDDT=84.10, SS=H, rel_sasa=0.31, aa=L</obs>
<answer>H</answer>

EXAMPLE 3 (compositional existential, single-shot tool_dsl):
Question: Is at least one 40-residue stretch high-confidence (mean pLDDT > 70) AND contact-rich (contact_density > 0.3)?
<think>Conjunction over windows. One tool_dsl with an existential expression handles it.</think>
<act>tool_dsl("exists reg in sliding_window(40) where mean_plddt(reg) > 70 and contact_density(reg) > 0.3")</act>
<obs>result = False</obs>
<answer>False</answer>

EXAMPLE 4 (multi-turn DISCOVER-then-INSPECT):
Question: What is the secondary structure of the residue with the highest pLDDT in the first 200 residues?
<think>Find argmax residue first, then inspect for SS.</think>
<act>tool_dsl("argmax r in range(1, 200) by plddt(r)")</act>
<obs>result = 137</obs>
<think>Inspect residue 137 for SS.</think>
<act>tool_inspect(137)</act>
<obs>residue 137: pLDDT=89.42, SS=H, rel_sasa=0.21, aa=L</obs>
<answer>H</answer>
"""


def build_l2_prompt(question: dict, view: ProteinView) -> str:
    """L2 ReAct initial user prompt. The runner loops, appending
    <obs>...</obs> after each <act>, until <answer> or max_turns.
    Uses the v2 strong-tool primer (5 tools, ReAct exemplars). The v3
    lazy-prompt variant was tested and shown to regress: L2_V3_TOOL_PRIMER
    is kept in this module for ablations but is NOT the default."""
    parts = [
        L2_TOOL_PRIMER.strip(),
        protein_summary(view),
        (f"Question: {question['question']}\n\n"
         "Now begin. Output exactly one <think>...</think> block "
         "followed by either one <act>tool(args)</act> or one "
         "<answer>...</answer>. STOP after the closing tag --- do "
         "NOT invent your own <obs>; the system provides it."),
    ]
    return "\n\n".join(parts)
