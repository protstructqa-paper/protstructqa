"""LLM output → ProtStructQA program / scalar answer extractor.

LLMs respond to ProtStructQA questions in many surface forms:
  1. Just the program:     mean_plddt(range(50, 100))
  2. Wrapped in fences:    ```mean_plddt(range(50, 100))```
  3. With prefix:          The program is: mean_plddt(range(50, 100))
  4. Numeric only:         86.27
  5. Bool with prose:      "Yes, residue 50 is buried."
  6. Selective abstain:    "Unreliable" or "I cannot answer this confidently."
  7. Reasoning + answer:   "<think>…</think> answer: 86.27"

`parse_llm_output(text, expected_type)` returns:
    {
      "program":   str | None     : the ProtStructQA program if extracted
      "scalar":    Any             : direct scalar (used when program is absent)
      "abstained": bool            : True if the model declined
      "raw":       str             : the original text
    }

For L0 zero-shot eval the caller will:
  - if `program` parses: execute it against the ProteinView, compare to gold
  - elif `scalar` not None: compare directly to gold (with type tolerance)
  - elif `abstained`: handle per Family Ha/Hb selective-prediction rules
  - else: count as wrong
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

from . import scoring


def _try_unwrap_json(s: str) -> tuple[str, Any]:
    """If `s` is a JSON object containing a known key for program/answer,
    return (extracted_string, extracted_scalar). Otherwise (s, None).

    Handles frontier-model outputs like::
        ```json
        {"program": "mean_plddt(range(28, 92))"}
        ```
    or bare::
        {"answer": 96.71}
    """
    s = s.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return s, None
    try:
        obj = _json.loads(s)
    except (_json.JSONDecodeError, ValueError):
        return s, None
    if not isinstance(obj, dict):
        return s, None
    # Look for a program-like field first
    for key in ("program", "protstructqa", "query", "code", "dsl"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), None
    # Then a scalar-like field
    for key in ("answer", "result", "value", "output"):
        if key in obj:
            return s, obj[key]
    return s, None


# ProtStructQA identifier regex: matches our DSL surface tokens.
_PROTSTRUCTQA_TOKENS = re.compile(
    r"\b(mean_plddt|min_plddt|max_plddt|std_plddt|distance|ca_distance|"
    r"cb_distance|seq_separation|mean_pae|pae|n_neighbors|coordinates|"
    r"plddt|sasa|rel_sasa|ss|ref_aa|residue|range|first|last|window|"
    r"all_residues|sliding_window|all_pairs|filter|exists|forall|count|"
    r"argmin|argmax|contact_density|long_range_contacts|radius_of_gyration|"
    r"mean_rel_sasa|mean_n_neighbors|protein_length|n_helices|n_strands|"
    r"mean_protein_plddt|union|intersection|difference|contains|size|"
    r"to_set|in_region|abs|min|max|round|floor|ceil|between|runs|"
    r"longest_run)\b"
)


_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z]*\n)?(.*?)```", re.DOTALL
)
_INLINE_RE = re.compile(r"`([^`\n]+)`")
_PROGRAM_PREFIX_RE = re.compile(
    r"(?:program|answer|query|query|ProtStructQA\s+program)\s*[:=]\s*"
    r"(.+?)$",
    re.IGNORECASE | re.MULTILINE,
)


def _looks_like_protstructqa(s: str) -> bool:
    """Heuristic: does this string contain ProtStructQA surface tokens?"""
    if "(" not in s or ")" not in s:
        return False
    return bool(_PROTSTRUCTQA_TOKENS.search(s))


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> and similar reasoning blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    return text


def extract_program(text: str) -> str | None:
    """Best-effort extraction of a ProtStructQA program from LLM text.
    Returns None if no plausible program is found.

    Handles frontier-model JSON wrapping (e.g., ```json{"program":...}```)
    by attempting JSON unwrap inside the fence loop and on bare text."""
    text = _strip_thinking(text)

    # 1. Try fenced code blocks first (highest priority).
    for m in _FENCE_RE.findall(text):
        candidate = m.strip().strip(";")
        # NEW: if the fence content is JSON-wrapped, unwrap first
        unwrapped, _scalar = _try_unwrap_json(candidate)
        if _looks_like_protstructqa(unwrapped):
            return unwrapped
        if _looks_like_protstructqa(candidate):
            return candidate

    # 2. Try inline backticks.
    for m in _INLINE_RE.findall(text):
        candidate = m.strip().strip(";")
        if _looks_like_protstructqa(candidate):
            return candidate

    # 3. NEW: Try unwrapping the whole text as JSON (bare JSON response)
    unwrapped, _scalar = _try_unwrap_json(text)
    if unwrapped != text and _looks_like_protstructqa(unwrapped):
        return unwrapped

    # 4. Try "program:"/"answer:" / "ProtStructQA:" prefix lines.
    for m in _PROGRAM_PREFIX_RE.finditer(text):
        candidate = _trim_outer(m.group(1))
        if _looks_like_protstructqa(candidate):
            return candidate

    # 5. Try the raw text: first line that looks like a program.
    for line in text.splitlines():
        line = _trim_outer(line)
        if _looks_like_protstructqa(line):
            return line

    return None


def _trim_outer(s: str) -> str:
    """Trim whitespace, trailing semicolons, and ONLY symmetric outer
    quotes from a candidate program. Critically does not strip quotes
    that appear inside the program (e.g. `if c then \"Yes\" else \"No\"`)."""
    s = s.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_llm_output(text: str, expected_type: str | None = None) -> dict:
    """Parse an LLM response into one of: program / scalar / abstained.
    `expected_type` is the question's declared answer_type (used to bias
    the scalar parser)."""
    if text is None:
        return {"program": None, "scalar": None, "abstained": False,
                  "raw": ""}
    raw = text
    # If a <answer>...</answer> tag is present (L2 ReAct), restrict the
    # downstream parser to its content. This prevents tool-call traffic
    # earlier in the conversation from polluting program/scalar extraction.
    m_ans = _ANSWER_TAG_RE.search(text)
    if m_ans:
        text = m_ans.group(1)
    text = _strip_thinking(text).strip()

    # Selective-prediction abstention always wins
    if expected_type and "Unreliable" in expected_type:
        if scoring.is_unreliable_response(text):
            return {"program": None, "scalar": "Unreliable",
                      "abstained": True, "raw": raw}

    program = extract_program(text)
    if program is not None:
        return {"program": program, "scalar": None, "abstained": False,
                  "raw": raw}

    # No program: try to parse a scalar based on expected_type.
    # We aggressively normalise prose-wrapped answers (L2 ReAct often
    # emits things like "Yes, the protein has a helix.") so that the
    # downstream scorer doesn't reject correct answers due to formatting.
    base = (expected_type or "").replace("|Unreliable", "")
    scalar: Any = None
    if base == "Bool":
        scalar = scoring.parse_bool(text)
        if scalar is None:
            # Look for a yes/no/true/false anywhere in the text
            tl = text.lower()
            # Strip leading "the answer is", "yes,", etc.
            m_y = re.search(r"\b(yes|true|correct|present|contains|does)\b",
                              tl)
            m_n = re.search(r"\b(no|false|incorrect|absent|does not|not present)\b",
                              tl)
            if m_y and (not m_n or m_y.start() < m_n.start()):
                scalar = True
            elif m_n:
                scalar = False
    elif base in ("Int", "Float"):
        scalar = scoring.parse_numeric(text)
    elif base == "SecStruct":
        s = text.strip().lower().strip("\"'.")
        if s in {"helix", "alpha-helix", "α-helix", "h"}:
            scalar = "H"
        elif s in {"strand", "beta-strand", "β-strand", "sheet", "e"}:
            scalar = "E"
        elif s in {"coil", "loop", "c", "-"}:
            scalar = "C"
        else:
            # Prose-wrapped: search for the canonical letter or keyword
            tl = text.lower()
            if re.search(r"\bhelix\b|\balpha-helix\b", tl) or \
                  re.search(r"\bH\b", text):
                scalar = "H"
            elif re.search(r"\bstrand\b|\bbeta-strand\b|\bsheet\b", tl) or \
                  re.search(r"\bE\b", text):
                scalar = "E"
            elif re.search(r"\bcoil\b|\bloop\b", tl) or \
                  re.search(r"\bC\b", text):
                scalar = "C"
    elif base == "Region":
        # Prefer [start, end] bracketed form first.
        m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        if m:
            scalar = [int(m.group(1)), int(m.group(2))]
        else:
            # Fall back to "residues 120 to 149" / "120-149" / first two ints
            m = re.search(r"(-?\d+)\s*(?:-|to|through)\s*(-?\d+)", text)
            if m:
                scalar = [int(m.group(1)), int(m.group(2))]
            else:
                nums = re.findall(r"-?\d+", text)
                if len(nums) >= 2:
                    scalar = [int(nums[0]), int(nums[1])]
    elif base in ("ResidueSet",):
        # Prefer [n1, n2, ...] bracketed list
        m = re.search(r"\[([^\]]+)\]", text)
        nums = re.findall(r"-?\d+", m.group(1) if m else text)
        if nums:
            scalar = [int(n) for n in nums]
    return {"program": None, "scalar": scalar, "abstained": False,
              "raw": raw}
