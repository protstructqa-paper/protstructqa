"""ProtStructQA question generation: produce (question, program, gold_answer)
triples from a canonical UniProt set + extracted features.

Output schema (one JSONL line per question):
    {
      "qid":          "human/A0A024RBG1/A1/0",
      "uniprot":      "A0A024RBG1",
      "species":      "human",
      "family":       "A",            // top-level letter
      "template":     "A1",           // specific template
      "question":     "What is the mean pLDDT of residues 50 to 100?",
      "program":      "mean_plddt(range(50, 100))",
      "answer":       75.32,
      "answer_type":  "Float",
      "params":       {"start": 50, "end": 100},
      "paraphrase_id": 0
    }

Per-protein recipe:
    - Sample 30 questions from families A-F using weights from spec
    - Sample 3 Family Ha questions (prompted-rule abstention)
    - Sample 3 Family Hb questions (non-prompted abstention)
    - Family G held out, generated for compositional split only

Each emitted question is verified by RE-RUNNING its `program` through the
DSL parser+executor against the same ProteinView. If the program-derived
answer != template-derived answer, the question is dropped (logged as
'verify_failed').

Outputs:
    <project-root>/
        protstructqa/benchmark/questions/{species}/{family}.jsonl

Usage:
    # Smoke test on 100 human proteins
    python benchmark/04_generate_questions.py --species human --limit 100

    # Full generation across all 4 species (~330K questions)
    python benchmark/04_generate_questions.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Make the dsl package importable when run as a script
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dsl import load_from_npz, run as dsl_run, ProteinView

DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
OUT_ROOT = HERE / "benchmark" / "questions"
SPECIES = ["human", "mouse", "fly", "chicken"]

# Family weights from QUESTION_TEMPLATES.md "Generation parameters" section.
# A-F sum to 0.93; Family G = 0.07 (held out for compositional split).
# Family H is sampled separately, NOT mixed in.
FAMILY_WEIGHTS_AF = {
    "A": 0.18, "B": 0.18, "C": 0.13, "D": 0.18, "E": 0.13, "F": 0.13, "G": 0.07,
}
N_QUESTIONS_PER_PROTEIN = 30
N_FAMILY_G_PER_PROTEIN = 3   # held out for compositional generalization split
N_FAMILY_H_PER_PROTEIN = 3   # split between Ha (prompted) and Hb (non-prompted)


# ============================ template framework ====================== #


@dataclass
class Question:
    qid: str
    uniprot: str
    species: str
    family: str
    template: str
    question: str
    program: str
    answer: Any
    answer_type: str
    params: dict
    paraphrase_id: int

    def to_jsonl(self) -> str:
        return json.dumps({
            "qid":           self.qid,
            "uniprot":       self.uniprot,
            "species":       self.species,
            "family":        self.family,
            "template":      self.template,
            "question":      self.question,
            "program":       self.program,
            "answer":        _serialize_answer(self.answer),
            "answer_type":   self.answer_type,
            "params":        self.params,
            "paraphrase_id": self.paraphrase_id,
        }, ensure_ascii=False)


def _serialize_answer(x: Any) -> Any:
    """JSON-friendly answer encoding."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, frozenset):
        return sorted(list(x))
    if isinstance(x, tuple):
        # DSL Region/Pair are tuples
        return list(x)
    if isinstance(x, bool):
        return bool(x)
    return x


class Template:
    """Base class for a (family-letter, template-name, paraphrase-list, sampler).

    Subclasses set class-level attributes:
        family      : "A" / "B" / ... / "G" / "Ha" / "Hb"
        name        : "A1" / "A2" / ...
        answer_type : "Float" / "Bool" / "Int" / "Region" / "ResidueSet" / ...
        paraphrase_list: list[str] of NL templates with {placeholder}s

    And override:
        sample_params(view, rng) -> dict | None
        gold_program(params) -> str
    """
    family: str = ""
    name: str = ""
    answer_type: str = ""
    paraphrase_list: list[str] = []

    def sample_params(self, view: "ProteinView", rng: random.Random) -> dict | None:
        raise NotImplementedError

    def gold_program(self, params: dict) -> str:
        raise NotImplementedError

    def render_question(self, params: dict, paraphrase_id: int) -> str:
        if not self.paraphrase_list:
            raise ValueError(f"{self.name}: no paraphrases defined")
        idx = paraphrase_id % len(self.paraphrase_list)
        return self.paraphrase_list[idx].format(**params)

    def n_paraphrases(self) -> int:
        return len(self.paraphrase_list)


# ============================ Family A: pLDDT ========================= #


class A1_RegionMeanPLDDT(Template):
    family = "A"; name = "A1"; answer_type = "Float"
    paraphrase_list = [
        "What is the mean pLDDT of residues {start} to {end}?",
        "How confident is AlphaFold about residues {start}-{end} on average?",
        "Average pLDDT for residues {start} through {end}?",
        "What is the mean per-residue confidence in the {start}-{end} window?",
        "For residues {start}-{end}, what is the average pLDDT score?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 30:
            return None
        # 20-80 residue windows; ensure end <= L
        size = rng.randint(20, min(80, L - 1))
        start = rng.randint(1, L - size)
        end = start + size - 1
        return {"start": start, "end": end}

    def gold_program(self, params):
        return f"mean_plddt(range({params['start']}, {params['end']}))"


class A2_NCConfidenceComparison(Template):
    family = "A"; name = "A2"; answer_type = "Bool"
    paraphrase_list = [
        "Is the {term1}-terminal {window} residues less reliable than the {term2}-terminal {window}?",
        "Does the {term1}-terminus have lower confidence than the {term2}-terminus over a {window}-residue window?",
        "Compare confidence between the first and last {window} residues - is the {term1}-end worse?",
        "Across the first and last {window} residues, is the {term1}-terminal pLDDT lower?",
        "Is the mean pLDDT of the {term1}-terminal {window} residues below that of the {term2}-terminal {window}?",
        "For terminal {window}-residue windows, does the {term1}-terminus have weaker prediction confidence than the {term2}-terminus?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 60:
            return None
        window = rng.choice([15, 20, 25, 30])
        if 2 * window > L:
            return None
        # Decide which terminus we're querying about
        ask_n = rng.random() < 0.5
        return {
            "term1": "N" if ask_n else "C",
            "term2": "C" if ask_n else "N",
            "window": window,
        }

    def gold_program(self, params):
        w = params["window"]
        if params["term1"] == "N":
            # "Is N-terminal {w} less reliable than C-terminal {w}?"
            return f"mean_plddt(first({w})) < mean_plddt(last({w}))"
        else:
            return f"mean_plddt(last({w})) < mean_plddt(first({w}))"


class A3_LowestConfidenceWindow(Template):
    family = "A"; name = "A3"; answer_type = "Region"
    paraphrase_list = [
        "Which {window}-residue window has the lowest mean pLDDT?",
        "Where is the least confident {window}-residue stretch?",
        "Find the {window}-residue window with the worst pLDDT.",
        "Locate the {window}-residue region of minimum mean pLDDT.",
        "Identify the {window}-consecutive-residue span where AlphaFold is least confident.",
        "Among all sliding {window}-residue windows, which has the lowest average pLDDT score?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([20, 30, 40, 50])
        if window >= L:
            return None
        return {"window": window}

    def gold_program(self, params):
        return f"argmin reg in sliding_window({params['window']}) by mean_plddt(reg)"


class A4_ConfidenceThresholdedCount(Template):
    family = "A"; name = "A4"; answer_type = "Int"
    paraphrase_list = [
        "How many residues have pLDDT > {threshold}?",
        "How many residues are predicted with pLDDT above {threshold}?",
        "Count the residues with pLDDT > {threshold}.",
        "What is the number of residues whose pLDDT exceeds {threshold}?",
        "Determine the count of high-confidence residues (pLDDT > {threshold}).",
        "Across the full sequence, how many residues meet pLDDT > {threshold}?",
    ]

    def sample_params(self, view, rng):
        threshold = rng.choice([50, 70, 80, 90])
        return {"threshold": threshold}

    def gold_program(self, params):
        return f"count r in all_residues where plddt(r) > {params['threshold']}"


class A5_ConfidentRegionDetection(Template):
    family = "A"; name = "A5"; answer_type = "Bool"
    paraphrase_list = [
        "Does this protein have a {window}-residue region with mean pLDDT above {threshold}?",
        "Is there a {window}-residue stretch where mean pLDDT exceeds {threshold}?",
        "Does any sliding {window}-residue window have a mean pLDDT > {threshold}?",
        "Can a contiguous {window}-residue region of pLDDT > {threshold} be found in this protein?",
        "Is at least one {window}-residue span confidently predicted (mean pLDDT > {threshold})?",
        "Are there {window} consecutive residues with average pLDDT above {threshold}?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([20, 30, 50])
        if window >= L:
            return None
        threshold = rng.choice([70, 80, 90])
        return {"window": window, "threshold": threshold}

    def gold_program(self, params):
        return (f"exists reg in sliding_window({params['window']}) "
                f"where mean_plddt(reg) > {params['threshold']}")


# ============================ Family B: Distance ====================== #


def _sample_residue_pair(view, rng, min_sep: int = 1) -> tuple[int, int] | None:
    L = view.length()
    if L < min_sep + 2:
        return None
    for _ in range(20):
        i = rng.randint(1, L)
        j = rng.randint(1, L)
        if abs(i - j) >= min_sep:
            return i, j
    return None


class B1_PairwiseDistance(Template):
    family = "B"; name = "B1"; answer_type = "Float"
    paraphrase_list = [
        "What is the C-alpha distance between residues {i} and {j}?",
        "How far apart are residues {i} and {j} in 3D (CA-CA, in Angstroms)?",
        "What is the spatial distance between residue {i} and residue {j}?",
        "Compute the CA-CA Euclidean distance for residues {i} and {j}.",
        "Report the 3D distance between the alpha-carbons of residues {i} and {j}.",
        "How many Angstroms separate the CA atoms of residues {i} and {j}?",
    ]

    def sample_params(self, view, rng):
        p = _sample_residue_pair(view, rng, min_sep=2)
        if p is None: return None
        return {"i": p[0], "j": p[1]}

    def gold_program(self, params):
        return f"distance(residue({params['i']}), residue({params['j']}))"


class B2_SpatialProximity(Template):
    family = "B"; name = "B2"; answer_type = "Bool"
    paraphrase_list = [
        "Are residues {i} and {j} spatially close (within {threshold} Angstroms)?",
        "Do residues {i} and {j} make contact at a {threshold}-Angstrom threshold?",
        "Is residue {i} within {threshold} angstroms of residue {j}?",
        "Is the CA-CA distance between residues {i} and {j} below {threshold} A?",
        "Are residues {i} and {j} in contact (CA-CA < {threshold} A)?",
        "Does the alpha-carbon of residue {i} sit within {threshold} A of residue {j}'s alpha-carbon?",
    ]

    def sample_params(self, view, rng):
        p = _sample_residue_pair(view, rng, min_sep=4)
        if p is None: return None
        threshold = rng.choice([6, 8, 10, 12])
        return {"i": p[0], "j": p[1], "threshold": threshold}

    def gold_program(self, params):
        return (f"distance(residue({params['i']}), residue({params['j']}))"
                f" < {params['threshold']}")


class B3_LongRangeContactPairs(Template):
    family = "B"; name = "B3"; answer_type = "PairSet"
    paraphrase_list = [
        "Find residue pairs separated by more than {sep} positions in sequence but within {threshold} Angstroms in space.",
        "Which pairs of residues are far apart in sequence (>{sep}) but spatially close (<{threshold} A)?",
        "List residue pairs with sequence separation > {sep} and 3D distance < {threshold} A.",
        "Identify all (i, j) pairs where |i-j| > {sep} and CA-CA distance < {threshold} A.",
        "Enumerate contacts that span more than {sep} residues in sequence at < {threshold} A in space.",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 50: return None
        sep = rng.choice([20, 30, 50])
        if sep >= L - 5: return None
        threshold = rng.choice([6, 8, 10])
        return {"sep": sep, "threshold": threshold}

    def gold_program(self, params):
        return (f"filter (i, j) in all_pairs(min_sep={params['sep']})"
                f" where distance(i, j) < {params['threshold']}")


class B4_LongRangeContactCount(Template):
    family = "B"; name = "B4"; answer_type = "Int"
    paraphrase_list = [
        "How many long-range contacts (sequence separation > {sep}, distance < {threshold} Angstroms) are there?",
        "Count residue pairs separated by > {sep} positions in sequence and within {threshold} A in space.",
        "What is the number of long-range contacts in this protein at sequence sep > {sep} and distance < {threshold} A?",
        "Tally pairs (i, j) with |i-j| > {sep} and CA-CA distance < {threshold} A.",
        "How many contacts span more than {sep} residues with CA-CA < {threshold} A?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 50: return None
        sep = rng.choice([20, 30, 50])
        if sep >= L - 5: return None
        threshold = rng.choice([6, 8, 10])
        return {"sep": sep, "threshold": threshold}

    def gold_program(self, params):
        return (f"size(filter (i, j) in all_pairs(min_sep={params['sep']})"
                f" where distance(i, j) < {params['threshold']})")


# ============================ Family C: PAE =========================== #


def _sample_two_disjoint_regions(view, rng, min_size=15, max_size=60,
                                    min_gap=10) -> dict | None:
    L = view.length()
    if L < 2 * min_size + min_gap: return None
    a_size_hi = min(max_size, max(min_size, L // 3))
    if a_size_hi < min_size:
        return None
    for _ in range(30):
        a_size = rng.randint(min_size, a_size_hi)
        a_start_hi = L - 2 * a_size - min_gap
        if a_start_hi < 1:
            continue
        a_start = rng.randint(1, a_start_hi)
        a_end = a_start + a_size - 1
        b_start = a_end + min_gap + 1
        b_size_max = L - b_start + 1
        if b_size_max < min_size:
            continue
        b_size_hi = min(max_size, b_size_max)
        if b_size_hi < min_size:
            continue
        b_size = rng.randint(min_size, b_size_hi)
        b_end = b_start + b_size - 1
        if b_end > L:
            continue
        return {"a_start": a_start, "a_end": a_end,
                "b_start": b_start, "b_end": b_end}
    return None


class C1_RegionPairPAE(Template):
    family = "C"; name = "C1"; answer_type = "Float"
    paraphrase_list = [
        "What is the mean PAE between residues {a_start}-{a_end} and {b_start}-{b_end}?",
        "Average predicted aligned error between the {a_start}-{a_end} region and the {b_start}-{b_end} region?",
        "Compute the mean PAE for the residue block ({a_start}-{a_end}) x ({b_start}-{b_end}).",
        "What is the inter-region predicted aligned error between residues {a_start}-{a_end} and {b_start}-{b_end} on average?",
        "Report the mean PAE across the ({a_start}-{a_end}, {b_start}-{b_end}) region pair.",
    ]

    def sample_params(self, view, rng):
        if view.pae is None: return None
        return _sample_two_disjoint_regions(view, rng)

    def gold_program(self, params):
        return (f"mean_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']}))")


class C2_DomainOrientationReliability(Template):
    family = "C"; name = "C2"; answer_type = "Bool"
    paraphrase_list = [
        "Do residues {a_start}-{a_end} and {b_start}-{b_end} have a confident relative orientation (mean PAE < {threshold})?",
        "Is the orientation between the {a_start}-{a_end} and {b_start}-{b_end} regions reliably predicted (mean PAE < {threshold})?",
        "Are the {a_start}-{a_end} and {b_start}-{b_end} segments confidently positioned relative to each other (mean PAE < {threshold} A)?",
        "Does the inter-region PAE for ({a_start}-{a_end}, {b_start}-{b_end}) fall below {threshold} A on average?",
        "Is the relative geometry between residues {a_start}-{a_end} and {b_start}-{b_end} reliable (mean PAE < {threshold})?",
    ]

    def sample_params(self, view, rng):
        if view.pae is None: return None
        params = _sample_two_disjoint_regions(view, rng)
        if params is None: return None
        params["threshold"] = rng.choice([5, 8, 12])
        return params

    def gold_program(self, params):
        return (f"mean_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']}))"
                f" < {params['threshold']}")


# ============================ Family D: SASA / packing ================ #


class D1_BuriedExposed(Template):
    family = "D"; name = "D1"; answer_type = "Bool"
    paraphrase_list = [
        "Is residue {i} buried in the protein core (relative SASA < {threshold})?",
        "Is residue {i} more buried than the threshold {threshold}?",
        "Does residue {i} satisfy rel_sasa < {threshold} (i.e., is it buried)?",
        "Is residue {i} a core residue at the {threshold} relative-SASA cutoff?",
        "At a relative-SASA threshold of {threshold}, would residue {i} be classified as buried?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        i = rng.randint(1, L)
        threshold = rng.choice([0.10, 0.15, 0.20])
        return {"i": i, "threshold": threshold}

    def gold_program(self, params):
        return f"rel_sasa(residue({params['i']})) < {params['threshold']}"


class D2_MostExposedWindow(Template):
    family = "D"; name = "D2"; answer_type = "Region"
    paraphrase_list = [
        "Which {window}-residue window has the highest average solvent accessibility?",
        "Find the {window}-residue stretch with the most exposed residues (max mean rel SASA).",
        "Where in the protein is the {window}-residue window with the largest mean relative SASA?",
        "Identify the most solvent-exposed {window}-consecutive-residue region.",
        "Among all sliding {window}-residue windows, which has the highest mean rel_sasa?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([15, 20, 30])
        if window >= L: return None
        return {"window": window}

    def gold_program(self, params):
        return f"argmax reg in sliding_window({params['window']}) by mean_rel_sasa(reg)"


class D3_BuriedResidueCount(Template):
    family = "D"; name = "D3"; answer_type = "Int"
    paraphrase_list = [
        "How many residues have relative SASA < {threshold}?",
        "Count buried residues (rel SASA < {threshold}).",
        "What is the number of residues classified as buried at rel_sasa threshold {threshold}?",
        "How many residues lie below {threshold} relative SASA?",
        "Tally residues whose relative SASA falls under {threshold}.",
    ]

    def sample_params(self, view, rng):
        threshold = rng.choice([0.10, 0.15, 0.20])
        return {"threshold": threshold}

    def gold_program(self, params):
        return f"count r in all_residues where rel_sasa(r) < {params['threshold']}"


class D4_NeighborCount(Template):
    family = "D"; name = "D4"; answer_type = "Int"
    paraphrase_list = [
        "How many heavy-atom neighbors does residue {i} have within 8 Angstroms?",
        "Count the residues within 8 A of residue {i}.",
        "What is the 8-A neighbor count at residue {i}?",
        "How many other residues lie within 8 A of residue {i}'s alpha-carbon?",
        "Determine the number of contacts (CA-CA <= 8 A) for residue {i}.",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        return {"i": rng.randint(1, L)}

    def gold_program(self, params):
        return f"n_neighbors(residue({params['i']}))"


class D5_DenselyPackedRegion(Template):
    family = "D"; name = "D5"; answer_type = "Bool"
    paraphrase_list = [
        "Is residue {i} in a densely-packed region (more than {threshold} neighbors within 8 Angstroms)?",
        "Does residue {i} have more than {threshold} contacts within 8 A?",
        "Is residue {i}'s 8-A neighbor count above {threshold}?",
        "Does residue {i} sit in a tightly-packed environment (>{threshold} CA neighbors at 8 A)?",
        "At residue {i}, are there more than {threshold} alpha-carbon neighbors within 8 A?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        return {"i": rng.randint(1, L), "threshold": rng.choice([8, 12, 16])}

    def gold_program(self, params):
        return f"n_neighbors(residue({params['i']})) > {params['threshold']}"


# ============================ Family E: Secondary structure ============ #


class E1_PerResidueSS(Template):
    family = "E"; name = "E1"; answer_type = "SecStruct"
    paraphrase_list = [
        "What is the secondary structure at residue {i}?",
        "Is residue {i} in a helix, strand, or coil?",
        "Report the 3-state secondary structure assignment at residue {i}.",
        "Which secondary-structure class (H/E/C) does residue {i} belong to?",
        "What SS state does residue {i} occupy?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        return {"i": rng.randint(1, L)}

    def gold_program(self, params):
        return f"ss(residue({params['i']}))"


class E2_SSCheck(Template):
    family = "E"; name = "E2"; answer_type = "Bool"
    paraphrase_list = [
        "Is residue {i} part of an alpha-helix?",
        "Does residue {i} belong to a helix?",
        "Is residue {i} in a helical conformation?",
        "Is the secondary structure at residue {i} a helix (H)?",
        "At residue {i}, is the 3-state SS assignment 'H'?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        return {"i": rng.randint(1, L)}

    def gold_program(self, params):
        return f'ss(residue({params["i"]})) == "H"'


class E3_SSTaggedCount(Template):
    family = "E"; name = "E3"; answer_type = "Int"
    paraphrase_list = [
        "How many residues are in alpha-helices?",
        "Count the residues whose secondary structure is helix.",
        "What is the total number of helix residues in this protein?",
        "Tally residues with SS == 'H'.",
        "How many residues are assigned to the helix state?",
    ]

    def sample_params(self, view, rng):
        return {}

    def gold_program(self, params):
        return 'count r in all_residues where ss(r) == "H"'


# ============================ Family F: Topology/contacts ============== #


class F1_LongRangeContactDensity(Template):
    family = "F"; name = "F1"; answer_type = "Float"
    paraphrase_list = [
        "What fraction of pairs in residues {start}-{end} are in contact (within 8 A)?",
        "What is the contact density of residues {start}-{end}?",
        "Compute the fraction of pairs within 8 A in the {start}-{end} region.",
        "Report the contact density (pairs <= 8 A / total pairs) for residues {start}-{end}.",
        "How tightly packed is the {start}-{end} region (8-A pair fraction)?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 30: return None
        size = rng.randint(20, min(80, L - 1))
        start = rng.randint(1, L - size)
        return {"start": start, "end": start + size - 1}

    def gold_program(self, params):
        return f"contact_density(range({params['start']}, {params['end']}))"


class F2_CompactCoreDetection(Template):
    family = "F"; name = "F2"; answer_type = "Bool"
    paraphrase_list = [
        "Is there a compact, high-confidence core ({window} residues with mean pLDDT > 80 and contact density > {cd_thr}) in this protein?",
        "Does this protein contain a {window}-residue region that is both confident (mean pLDDT > 80) and tightly packed (contact density > {cd_thr})?",
        "Can a {window}-residue stretch satisfying mean_plddt > 80 AND contact_density > {cd_thr} be found?",
        "Is at least one {window}-residue region simultaneously high-pLDDT (> 80) and high-contact-density (> {cd_thr})?",
        "Does any sliding {window}-window meet both confidence (mean pLDDT > 80) and packing (contact density > {cd_thr}) criteria?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([20, 30, 40])
        if window >= L: return None
        cd_thr = rng.choice([0.20, 0.30, 0.40])
        return {"window": window, "cd_thr": cd_thr}

    def gold_program(self, params):
        return (f"exists reg in sliding_window({params['window']})"
                f" where mean_plddt(reg) > 80 and contact_density(reg) > {params['cd_thr']}")


class F3_RadiusOfGyration(Template):
    family = "F"; name = "F3"; answer_type = "Float"
    paraphrase_list = [
        "What is the radius of gyration of residues {start}-{end}?",
        "Compute Rg over the {start}-{end} region.",
        "What is Rg (in Angstroms) for the {start}-{end} segment?",
        "Report the radius of gyration for the residue range {start}-{end}.",
        "Compute the spatial spread (Rg) of residues {start} through {end}.",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        if L < 30: return None
        size = rng.randint(20, min(80, L - 1))
        start = rng.randint(1, L - size)
        return {"start": start, "end": start + size - 1}

    def gold_program(self, params):
        return f"radius_of_gyration(range({params['start']}, {params['end']}))"


class F4_MostCompactWindow(Template):
    family = "F"; name = "F4"; answer_type = "Region"
    paraphrase_list = [
        "Which {window}-residue window has the smallest radius of gyration (most compact)?",
        "Find the most compact {window}-residue window in this protein.",
        "Where in the protein is the {window}-residue stretch with minimum Rg?",
        "Identify the most spatially-compact {window}-residue segment.",
        "Among sliding {window}-residue windows, which has the lowest radius of gyration?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([20, 30, 40])
        if window >= L: return None
        return {"window": window}

    def gold_program(self, params):
        return f"argmin reg in sliding_window({params['window']}) by radius_of_gyration(reg)"


# ============================ Family G: Compositional (held-out) ====== #
#
# Family G is held out for the compositional-generalization test split.
# It is generated for ALL proteins but never mixed into the main 30
# questions per protein: it lives on a separate track for the
# compositional-generalization eval. The compositional structure is
# layered: G templates compose primitives from A-F via conjunctive
# predicates, multi-step quantifiers, and cross-feature filters.


class G1_BuriedLowPLDDT(Template):
    family = "G"; name = "G1"; answer_type = "ResidueSet"
    paraphrase_list = [
        "Find residues that are both buried (relative SASA < {sasa_thr}) and low-confidence (pLDDT < {plddt_thr}).",
        "Which residues are simultaneously buried (rel SASA < {sasa_thr}) AND poorly predicted (pLDDT < {plddt_thr})?",
        "List residues that satisfy both rel_sasa(r) < {sasa_thr} and plddt(r) < {plddt_thr}.",
        "Identify all residues in the buried-and-uncertain set: rel_sasa < {sasa_thr} AND pLDDT < {plddt_thr}.",
        "Return the residue set with rel_sasa(r) < {sasa_thr} and plddt(r) < {plddt_thr}.",
        "Enumerate residues that are both core (rel_sasa < {sasa_thr}) and weakly predicted (pLDDT < {plddt_thr}).",
    ]

    def sample_params(self, view, rng):
        return {
            "sasa_thr": rng.choice([0.10, 0.15, 0.20]),
            "plddt_thr": rng.choice([60, 70, 80]),
        }

    def gold_program(self, params):
        return (f"filter r in all_residues where "
                f"rel_sasa(r) < {params['sasa_thr']} "
                f"and plddt(r) < {params['plddt_thr']}")


class G2_HighConfContactRichRegion(Template):
    family = "G"; name = "G2"; answer_type = "Bool"
    paraphrase_list = [
        "Is there a {window}-residue region with mean pLDDT > {plddt_thr} AND contact density > {cd_thr}?",
        "Does this protein have a {window}-residue stretch that is both confident (pLDDT > {plddt_thr}) and tightly packed (contact density > {cd_thr})?",
        "Can a {window}-residue window with mean_plddt > {plddt_thr} AND contact_density > {cd_thr} be found?",
        "Is at least one {window}-residue stretch high-confidence (mean pLDDT > {plddt_thr}) AND contact-rich (contact density > {cd_thr})?",
        "Does any sliding {window}-residue window satisfy mean pLDDT > {plddt_thr} and contact density > {cd_thr}?",
    ]

    def sample_params(self, view, rng):
        L = view.length()
        window = rng.choice([20, 30, 40])
        if window >= L: return None
        return {
            "window": window,
            "plddt_thr": rng.choice([70, 80]),
            "cd_thr": rng.choice([0.20, 0.30]),
        }

    def gold_program(self, params):
        return (f"exists reg in sliding_window({params['window']}) "
                f"where mean_plddt(reg) > {params['plddt_thr']} "
                f"and contact_density(reg) > {params['cd_thr']}")


class G3_HelixStrandInterface(Template):
    family = "G"; name = "G3"; answer_type = "Bool"
    paraphrase_list = [
        "Are there residues in alpha-helices that are spatially close (< {threshold} A) to residues in beta-strands?",
        "Does any helix residue come within {threshold} A of a strand residue?",
        "Is there a helix-strand contact (CA-CA distance < {threshold} A)?",
        "Do any helical residues sit within {threshold} A of any beta-strand residues?",
        "Does this protein have any (helix, strand) residue pair with CA-CA < {threshold} A?",
        "Are alpha-helix and beta-strand secondary structures in contact (< {threshold} A) anywhere in this protein?",
    ]

    def sample_params(self, view, rng):
        return {"threshold": rng.choice([6, 8, 10])}

    def gold_program(self, params):
        return (f"exists r in all_residues where ss(r) == \"H\" "
                f"and exists s in all_residues where ss(s) == \"E\" "
                f"and distance(r, s) < {params['threshold']}")


# ============================ Family H: Abstention-Capable ============= #
#
# Family H templates differ from A-F in two ways:
#  1. The gold answer is either the computed value OR the literal string
#     "Unreliable", depending on whether AlphaFold's own confidence
#     signals (pLDDT, PAE) at the relevant residues meet a reliability
#     threshold.
#  2. Each H template has TWO variants:
#     - Ha = "prompted": the abstention rule is stated in the question
#       text (sanity baseline; reduces to conditional execution)
#     - Hb = "non-prompted": the rule is hidden; the model must learn
#       pLDDT/PAE reliability semantics from few-shot or background
#       knowledge (the genuine selective-prediction angle, mapping to
#       Kadavath 2022 / Yin 2023 / Ren 2023 / Cole 2023)
#
# Both Ha and Hb use the SAME gold program / gold answer. Only the
# question wording differs.
#
# The DSL does not natively type "Unreliable"; gold is computed in
# Python via execute_directly(). The emitted `program` string uses
# `if-then-else` syntax for documentation but is NOT re-executed by the
# DSL. This is a deliberate asymmetry because Family H tests
# meta-reasoning, not pure structural lookup.


class HTemplate(Template):
    """Family H base. Subclasses override execute_directly() and
    gold_program(); paraphrase_list is split by Ha vs Hb in subclasses.
    is_family_h marks instances so the main loop bypasses DSL verification."""

    is_family_h = True

    def execute_directly(self, view: ProteinView, params: dict) -> Any:
        raise NotImplementedError


# ----- H1: Distance with abstention on low pLDDT ----- #


class _H1_Base(HTemplate):
    answer_type = "Float|Unreliable"

    def sample_params(self, view, rng):
        L = view.length()
        if L < 20: return None
        # Try to balance answerable vs unreliable. Aim for 50/50 across
        # protein samples by random thresholding.
        plddt_thr = rng.choice([50, 60, 70])
        for _ in range(20):
            i = rng.randint(1, L)
            j = rng.randint(1, L)
            if abs(i - j) < 4:
                continue
            return {"i": i, "j": j, "plddt_thr": plddt_thr}
        return None

    def gold_program(self, params):
        return (f"if plddt(residue({params['i']})) < {params['plddt_thr']} "
                f"or plddt(residue({params['j']})) < {params['plddt_thr']} "
                f"then \"Unreliable\" "
                f"else distance(residue({params['i']}), residue({params['j']}))")

    def execute_directly(self, view, params):
        if (view.plddt_at(params["i"]) < params["plddt_thr"] or
            view.plddt_at(params["j"]) < params["plddt_thr"]):
            return "Unreliable"
        return view.distance(params["i"], params["j"])


class Ha1_DistanceAbstention(_H1_Base):
    family = "Ha"; name = "Ha1"
    paraphrase_list = [
        "What is the C-alpha distance between residues {i} and {j}? "
        "Answer 'unreliable' if either residue has pLDDT below {plddt_thr}.",
        "Distance between residues {i} and {j}? "
        "If either residue's pLDDT is below {plddt_thr}, respond 'unreliable'.",
        "Compute the CA-CA distance between residues {i} and {j}. "
        "Treat the answer as unreliable when pLDDT < {plddt_thr} at either position.",
        "Report the CA-CA distance for residues {i} and {j}, "
        "but answer 'unreliable' when min(plddt({i}), plddt({j})) < {plddt_thr}.",
        "What is the inter-residue distance for ({i}, {j})? "
        "If pLDDT at residue {i} or residue {j} is below {plddt_thr}, return 'unreliable'.",
    ]


class Hb1_DistanceAbstention(_H1_Base):
    family = "Hb"; name = "Hb1"
    # Audit-tightened wording (2026-05-03): the Hb question must NOT
    # state any pLDDT/PAE threshold. The model must learn from few-shot
    # exemplars (provided at eval time) when pLDDT signals make a residue
    # unreliable to reason about. Phrasing emphasizes that abstention is
    # tied to STRUCTURAL UNCERTAINTY at the named residues, not generic
    # answer-uncertainty.
    paraphrase_list = [
        "What is the C-alpha distance between residues {i} and {j}? "
        "Respond 'unreliable' only if AlphaFold's prediction at residues {i} and/or {j} is too uncertain to support a confident distance.",
        "Compute the CA-CA distance between residues {i} and {j}. "
        "If the predicted backbone position at either residue is not confidently determined, return 'unreliable' instead of a number.",
        "Distance between residues {i} and {j}? "
        "Say 'unreliable' if the structural prediction at residues {i} or {j} lacks the confidence needed to commit to a numeric distance.",
        "How far apart are the alpha-carbons of residues {i} and {j}? "
        "Reply 'unreliable' when AlphaFold's prediction at residue {i} or residue {j} is too uncertain to commit to a number.",
        "What is the inter-residue distance for {i}-{j}? "
        "Return 'unreliable' iff AlphaFold's local confidence at residue {i} or residue {j} is insufficient.",
    ]


# ----- H2: Region-pair PAE with abstention on absent/unreliable PAE ----- #


class _H2_Base(HTemplate):
    answer_type = "Float|Unreliable"

    def sample_params(self, view, rng):
        if view.pae is None:
            return None
        params = _sample_two_disjoint_regions(view, rng)
        if params is None: return None
        params["pae_max"] = rng.choice([15, 20, 25])
        return params

    def gold_program(self, params):
        return (f"if mean_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']})) > {params['pae_max']} "
                f"then \"Unreliable\" "
                f"else mean_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']}))")

    def execute_directly(self, view, params):
        m = view.mean_pae(params["a_start"], params["a_end"],
                            params["b_start"], params["b_end"])
        if m > params["pae_max"]:
            return "Unreliable"
        return m


class Ha2_PAERegionAbstention(_H2_Base):
    family = "Ha"; name = "Ha2"
    paraphrase_list = [
        "Compute the mean PAE between residues {a_start}-{a_end} and {b_start}-{b_end}. "
        "If mean PAE exceeds {pae_max}, return 'unreliable'.",
        "Mean PAE for the {a_start}-{a_end} vs {b_start}-{b_end} region pair? "
        "Treat values above {pae_max} as 'unreliable'.",
        "Report mean_pae(({a_start}-{a_end}), ({b_start}-{b_end})). "
        "If the mean exceeds {pae_max}, answer 'unreliable' instead.",
        "What is the mean PAE between the {a_start}-{a_end} region and the {b_start}-{b_end} region? "
        "Return 'unreliable' when the value would be > {pae_max}.",
        "Compute the inter-region predicted aligned error for ({a_start}-{a_end}) x ({b_start}-{b_end}). "
        "If mean PAE > {pae_max}, the answer is 'unreliable'.",
    ]


class Hb2_PAERegionAbstention(_H2_Base):
    family = "Hb"; name = "Hb2"
    paraphrase_list = [
        "Compute the mean PAE between residues {a_start}-{a_end} and {b_start}-{b_end}. "
        "If the predicted relative orientation between these two regions is too uncertain to commit to a number, respond 'unreliable'.",
        "Mean PAE for the {a_start}-{a_end} vs {b_start}-{b_end} region pair? "
        "Return 'unreliable' if the inter-domain alignment confidence is too low to support a meaningful answer.",
        "What is the average inter-region predicted aligned error for ({a_start}-{a_end}, {b_start}-{b_end})? "
        "Answer 'unreliable' if AlphaFold does not confidently place these regions relative to each other.",
        "Report the mean PAE block({a_start}-{a_end}, {b_start}-{b_end}). "
        "If the relative geometry between the two segments is poorly determined, say 'unreliable' instead.",
        "Compute mean_pae(({a_start}-{a_end}), ({b_start}-{b_end})): but only when the inter-region orientation is confidently predicted; otherwise answer 'unreliable'.",
    ]


# ----- H3: Buried/exposed with abstention on disordered context ----- #


class _H3_Base(HTemplate):
    answer_type = "Bool|Unreliable"

    def sample_params(self, view, rng):
        L = view.length()
        if L < 12: return None
        # Sample residue at least 5 from each end so the ±5 window fits
        i = rng.randint(6, L - 5)
        sasa_thr = rng.choice([0.10, 0.15, 0.20])
        plddt_thr = rng.choice([50, 60, 70])
        return {"i": i, "sasa_thr": sasa_thr, "plddt_thr": plddt_thr}

    def gold_program(self, params):
        return (f"if mean_plddt(range({params['i']-5}, {params['i']+5})) < {params['plddt_thr']} "
                f"then \"Unreliable\" "
                f"else (rel_sasa(residue({params['i']})) < {params['sasa_thr']})")

    def execute_directly(self, view, params):
        i = params["i"]
        if view.mean_plddt(max(1, i - 5), min(view.length(), i + 5)) < params["plddt_thr"]:
            return "Unreliable"
        return view.rel_sasa_at(i) < params["sasa_thr"]


class Ha3_BuriedAbstentionDisorder(_H3_Base):
    family = "Ha"; name = "Ha3"
    paraphrase_list = [
        "Is residue {i} buried in the protein core (relative SASA < {sasa_thr})? "
        "Treat as 'unreliable' if the local pLDDT (residues {i}+/-5) is below {plddt_thr} (likely disordered).",
        "Is residue {i} buried (rel SASA < {sasa_thr})? "
        "If the surrounding {i}+/-5 mean pLDDT is < {plddt_thr}, answer 'unreliable'.",
        "Determine whether residue {i} is buried (rel_sasa < {sasa_thr}). "
        "If mean pLDDT in window {i}+/-5 < {plddt_thr}, the answer is 'unreliable'.",
        "Is rel_sasa({i}) < {sasa_thr}? "
        "Return 'unreliable' when mean_plddt(range({i}-5, {i}+5)) < {plddt_thr}.",
        "Classify residue {i} as buried or not at the {sasa_thr} threshold. "
        "Answer 'unreliable' if the local mean pLDDT (window {i}+/-5) falls below {plddt_thr}.",
    ]


class Hb3_BuriedAbstentionDisorder(_H3_Base):
    family = "Hb"; name = "Hb3"
    paraphrase_list = [
        "Is residue {i} buried (rel SASA < {sasa_thr})? "
        "If the local backbone around residue {i} is poorly resolved (likely disordered region), answer 'unreliable'.",
        "Is residue {i} buried in the protein core (relative SASA < {sasa_thr})? "
        "Return 'unreliable' if the structural prediction in the immediate neighborhood of residue {i} is too uncertain to fix burial.",
        "Determine burial of residue {i} (rel_sasa < {sasa_thr}). "
        "Respond 'unreliable' when the local pLDDT around residue {i} indicates an unreliable backbone: burial cannot be assessed in disordered regions.",
        "Is residue {i} a core residue at the {sasa_thr} threshold? "
        "Answer 'unreliable' iff residue {i}'s local environment is not confidently predicted enough to determine burial.",
        "At residue {i}, is rel_sasa < {sasa_thr}? "
        "Say 'unreliable' if the local structural prediction surrounding residue {i} is too low-confidence to commit to a Boolean.",
    ]


# ----- H4: Long-range contact count with abstention on truncation risk ----- #


class _H4_Base(HTemplate):
    answer_type = "Int|Unreliable"

    def sample_params(self, view, rng):
        L = view.length()
        if L < 50: return None
        sep = rng.choice([20, 30, 50])
        if sep >= L - 5: return None
        threshold = rng.choice([6, 8, 10])
        plddt_thr = rng.choice([50, 60])
        coverage_thr = rng.choice([20, 30, 40])  # percent
        return {"sep": sep, "threshold": threshold,
                "plddt_thr": plddt_thr, "coverage_thr": coverage_thr}

    def gold_program(self, params):
        return (f"if (count r in all_residues where plddt(r) < {params['plddt_thr']}) "
                f"* 100 / protein_length() > {params['coverage_thr']} "
                f"then \"Unreliable\" "
                f"else size(filter (i, j) in all_pairs(min_sep={params['sep']})"
                f" where distance(i, j) < {params['threshold']})")

    def execute_directly(self, view, params):
        L = view.length()
        low_count = int((view.plddt < params["plddt_thr"]).sum())
        if low_count * 100 / L > params["coverage_thr"]:
            return "Unreliable"
        # Compute long-range contacts within the whole protein
        # (sep > params['sep'], dist < threshold)
        coords = view.ca_xyz
        n = coords.shape[0]
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mask = (d <= params["threshold"]) & (np.abs(ii - jj) > params["sep"])
        return int(mask.sum() // 2)


class Ha4_LRContactsAbstention(_H4_Base):
    family = "Ha"; name = "Ha4"
    paraphrase_list = [
        "How many long-range contacts (sequence separation > {sep}, distance < {threshold} A) does this protein have? "
        "Return 'unreliable' if more than {coverage_thr}% of residues have pLDDT below {plddt_thr}.",
        "Count long-range contacts (>{sep} apart in sequence, <{threshold} A in space). "
        "If >{coverage_thr}% of the protein has pLDDT < {plddt_thr}, answer 'unreliable'.",
        "Tally pairs (i, j) with |i-j| > {sep} and CA-CA distance < {threshold} A. "
        "Answer 'unreliable' when the fraction of residues with pLDDT < {plddt_thr} exceeds {coverage_thr}%.",
        "What is the number of long-range contacts? "
        "Return 'unreliable' if more than {coverage_thr}% of residues are below pLDDT {plddt_thr}.",
        "Count contacts where |i-j| > {sep} AND CA-CA < {threshold} A. "
        "If the protein-wide low-pLDDT fraction (pLDDT < {plddt_thr}) exceeds {coverage_thr}%, return 'unreliable'.",
    ]


class Hb4_LRContactsAbstention(_H4_Base):
    family = "Hb"; name = "Hb4"
    paraphrase_list = [
        "How many long-range contacts (sequence separation > {sep}, distance < {threshold} A) does this protein have? "
        "If the protein-wide structural prediction is not confidently determined, respond 'unreliable': the contact map cannot be trusted.",
        "Count long-range contacts (>{sep} apart in sequence, <{threshold} A in space). "
        "Answer 'unreliable' iff the protein-wide prediction confidence is too low to make the contact tally meaningful.",
        "Tally pairs (i, j) with |i-j| > {sep} and CA-CA < {threshold} A. "
        "Say 'unreliable' when the global structural prediction has too many low-confidence regions to support a contact count.",
        "What is the long-range contact count for sep > {sep} and distance < {threshold} A? "
        "Return 'unreliable' if a substantial fraction of the protein is poorly predicted (the contact map would inherit that uncertainty).",
        "Compute the number of long-range contacts in this protein. "
        "If AlphaFold's confidence is insufficient across the protein to commit to a count, answer 'unreliable'.",
    ]


# ----- H5: Helix-strand interface with abstention on uncertain SS ----- #


class _H5_Base(HTemplate):
    answer_type = "Bool|Unreliable"

    def sample_params(self, view, rng):
        L = view.length()
        if L < 50: return None
        threshold = rng.choice([6, 8, 10])
        plddt_thr = rng.choice([50, 60])
        coverage_thr = rng.choice([20, 30, 40])
        return {"threshold": threshold, "plddt_thr": plddt_thr,
                "coverage_thr": coverage_thr}

    def gold_program(self, params):
        return (f"if (count r in all_residues where plddt(r) < {params['plddt_thr']}) "
                f"* 100 / protein_length() > {params['coverage_thr']} "
                f"then \"Unreliable\" "
                f"else (exists r in all_residues where ss(r) == \"H\" and "
                f"exists s in all_residues where ss(s) == \"E\" "
                f"and distance(r, s) < {params['threshold']})")

    def execute_directly(self, view, params):
        L = view.length()
        low_count = int((view.plddt < params["plddt_thr"]).sum())
        if low_count * 100 / L > params["coverage_thr"]:
            return "Unreliable"
        # Find helix and strand residues; check any pair < threshold A
        helix_idx = np.where(view.ss_3 == "H")[0]
        strand_idx = np.where(view.ss_3 == "E")[0]
        if len(helix_idx) == 0 or len(strand_idx) == 0:
            return False
        h_xyz = view.ca_xyz[helix_idx]
        e_xyz = view.ca_xyz[strand_idx]
        diff = h_xyz[:, None, :] - e_xyz[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        return bool((d < params["threshold"]).any())


class Ha5_HelixStrandAbstention(_H5_Base):
    family = "Ha"; name = "Ha5"
    paraphrase_list = [
        "Are there residues in alpha-helices that are spatially close (< {threshold} A) to residues in beta-strands? "
        "Treat as 'unreliable' if more than {coverage_thr}% of residues have pLDDT below {plddt_thr}.",
        "Is there any helix-strand contact (CA-CA < {threshold} A)? "
        "If >{coverage_thr}% of the protein has pLDDT < {plddt_thr}, answer 'unreliable'.",
        "Does this protein contain any (helix, strand) residue pair with CA-CA < {threshold} A? "
        "Return 'unreliable' when the fraction of residues with pLDDT < {plddt_thr} exceeds {coverage_thr}%.",
        "Are alpha-helix and beta-strand residues in contact (< {threshold} A) anywhere? "
        "If more than {coverage_thr}% of residues are below pLDDT {plddt_thr}, answer 'unreliable'.",
        "At a {threshold}-A cutoff, does at least one helix-strand pair exist? "
        "If protein-wide (low-pLDDT < {plddt_thr}) fraction exceeds {coverage_thr}%, the answer is 'unreliable'.",
    ]


class Hb5_HelixStrandAbstention(_H5_Base):
    family = "Hb"; name = "Hb5"
    paraphrase_list = [
        "Are there residues in alpha-helices that are spatially close (< {threshold} A) to residues in beta-strands? "
        "If the secondary-structure assignment cannot be trusted because too much of the protein is poorly predicted, respond 'unreliable'.",
        "Is there any helix-strand contact (CA-CA < {threshold} A)? "
        "Answer 'unreliable' if AlphaFold's prediction is too uncertain over enough of the protein that the SS labels themselves become unreliable.",
        "Does this protein contain any (helix, strand) residue pair with CA-CA distance < {threshold} A? "
        "Say 'unreliable' when the protein-wide structural confidence is too low to commit to SS-derived contacts.",
        "Are alpha-helix and beta-strand residues in contact (< {threshold} A) anywhere in this protein? "
        "Return 'unreliable' if a meaningful fraction of the residues are poorly predicted (their SS labels can't be trusted).",
        "At a {threshold}-A cutoff, is there at least one (helix, strand) pair? "
        "Respond 'unreliable' if the global pLDDT distribution indicates too many residues are unreliable for SS-conditional reasoning.",
    ]


# ============================ NEW templates (v3) ====================== #
# Coverage-gap fill (see analysis/06_coverage_audit.py):
#   - C family had only 2 templates → add C3 (max PAE), C4 (count high-PAE).
#   - E family had only 3 templates, all helix-only → add E4 (strand count),
#     E5 (longest helix length), E6 (# distinct helix segments).


class C3_MaxRegionPairPAE(Template):
    family = "C"; name = "C3"; answer_type = "Float"
    paraphrase_list = []   # filled in below from PARAPHRASE_POOLS_V3

    def sample_params(self, view, rng):
        if view.pae is None: return None
        return _sample_two_disjoint_regions(view, rng)

    def gold_program(self, params):
        return (f"max_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']}))")


class C4_HighPAEPairCount(Template):
    family = "C"; name = "C4"; answer_type = "Int"
    paraphrase_list = []

    def sample_params(self, view, rng):
        if view.pae is None: return None
        params = _sample_two_disjoint_regions(view, rng)
        if params is None: return None
        params["threshold"] = rng.choice([5, 8, 12, 15])
        return params

    def gold_program(self, params):
        return (f"count_high_pae(range({params['a_start']}, {params['a_end']}),"
                f" range({params['b_start']}, {params['b_end']}),"
                f" {params['threshold']})")


class E4_StrandTaggedCount(Template):
    family = "E"; name = "E4"; answer_type = "Int"
    paraphrase_list = []

    def sample_params(self, view, rng):
        return {}

    def gold_program(self, params):
        return 'count r in all_residues where ss(r) == "E"'


class E5_LongestHelixLength(Template):
    family = "E"; name = "E5"; answer_type = "Int"
    paraphrase_list = []

    def sample_params(self, view, rng):
        # Only valid when the protein has at least one helix segment.
        if view.n_helices() == 0:
            return None
        return {}

    def gold_program(self, params):
        return 'length(longest_run("H"))'


class E6_NumHelixSegments(Template):
    family = "E"; name = "E6"; answer_type = "Int"
    paraphrase_list = []

    def sample_params(self, view, rng):
        return {}

    def gold_program(self, params):
        return 'n_helices()'


# ============================ paraphrase pool v3 patch =============== #
# Replaces every Template subclass's `paraphrase_list` with the cleaned,
# leak-free, expanded v3 pool. Also fills the new template classes above.
# Import is local-scope so the rest of this module remains importable
# even if `paraphrase_pools.py` is missing (tests).
try:
    from benchmark.paraphrase_pools import PARAPHRASE_POOLS_V3
except ImportError:
    # When run as a script from benchmark/ directory.
    sys.path.insert(0, str(HERE / "benchmark"))
    from paraphrase_pools import PARAPHRASE_POOLS_V3   # type: ignore

_V3_TEMPLATE_BINDINGS = {
    "A1": A1_RegionMeanPLDDT, "A2": A2_NCConfidenceComparison,
    "A3": A3_LowestConfidenceWindow, "A4": A4_ConfidenceThresholdedCount,
    "A5": A5_ConfidentRegionDetection,
    "B1": B1_PairwiseDistance, "B2": B2_SpatialProximity,
    "B3": B3_LongRangeContactPairs, "B4": B4_LongRangeContactCount,
    "C1": C1_RegionPairPAE, "C2": C2_DomainOrientationReliability,
    "C3": C3_MaxRegionPairPAE, "C4": C4_HighPAEPairCount,
    "D1": D1_BuriedExposed, "D2": D2_MostExposedWindow,
    "D3": D3_BuriedResidueCount, "D4": D4_NeighborCount,
    "D5": D5_DenselyPackedRegion,
    "E1": E1_PerResidueSS, "E2": E2_SSCheck, "E3": E3_SSTaggedCount,
    "E4": E4_StrandTaggedCount, "E5": E5_LongestHelixLength,
    "E6": E6_NumHelixSegments,
    "F1": F1_LongRangeContactDensity, "F2": F2_CompactCoreDetection,
    "F3": F3_RadiusOfGyration, "F4": F4_MostCompactWindow,
}
for _tname, _cls in _V3_TEMPLATE_BINDINGS.items():
    if _tname in PARAPHRASE_POOLS_V3:
        _cls.paraphrase_list = list(PARAPHRASE_POOLS_V3[_tname])


# ============================ template registry ======================= #


TEMPLATES_BY_FAMILY: dict[str, list[Template]] = {
    "A": [A1_RegionMeanPLDDT(), A2_NCConfidenceComparison(),
            A3_LowestConfidenceWindow(), A4_ConfidenceThresholdedCount(),
            A5_ConfidentRegionDetection()],
    "B": [B1_PairwiseDistance(), B2_SpatialProximity(),
            B3_LongRangeContactPairs(), B4_LongRangeContactCount()],
    "C": [C1_RegionPairPAE(), C2_DomainOrientationReliability(),
            C3_MaxRegionPairPAE(), C4_HighPAEPairCount()],
    "D": [D1_BuriedExposed(), D2_MostExposedWindow(),
            D3_BuriedResidueCount(), D4_NeighborCount(), D5_DenselyPackedRegion()],
    "E": [E1_PerResidueSS(), E2_SSCheck(), E3_SSTaggedCount(),
            E4_StrandTaggedCount(), E5_LongestHelixLength(),
            E6_NumHelixSegments()],
    "F": [F1_LongRangeContactDensity(), F2_CompactCoreDetection(),
            F3_RadiusOfGyration(), F4_MostCompactWindow()],
    "G": [G1_BuriedLowPLDDT(), G2_HighConfContactRichRegion(),
            G3_HelixStrandInterface()],
    "Ha": [Ha1_DistanceAbstention(), Ha2_PAERegionAbstention(),
              Ha3_BuriedAbstentionDisorder(), Ha4_LRContactsAbstention(),
              Ha5_HelixStrandAbstention()],
    "Hb": [Hb1_DistanceAbstention(), Hb2_PAERegionAbstention(),
              Hb3_BuriedAbstentionDisorder(), Hb4_LRContactsAbstention(),
              Hb5_HelixStrandAbstention()],
}


# ============================ generator pipeline ======================= #


def _params_key(params: dict) -> str:
    """Canonical string key for dedup (stable JSON, sorted keys)."""
    return json.dumps(params, sort_keys=True, default=str)


def generate_one_question(
    view: ProteinView, family_letter: str, rng: random.Random,
    qid_idx: int,
    seen: set[str] | None = None,
    bool_balance: dict | None = None,
) -> Question | None:
    """Pick a random template within `family_letter`, sample params, build
    the question + program + verified gold answer.

    Improvements:
      - Per-protein dedup: skip if (template, params) already emitted for
        this protein (caller threads `seen` through).
      - Bool answer balance: for Bool-typed templates, do up to 4
        retries to find params whose answer is the rarer of the two
        observed so far for this template on this protein (caller
        threads `bool_balance`).
    """
    family_templates = TEMPLATES_BY_FAMILY.get(family_letter, [])
    if not family_templates:
        return None
    tpl = rng.choice(family_templates)
    is_h = getattr(tpl, "is_family_h", False)
    is_bool = tpl.answer_type.startswith("Bool")

    # Decide whether to push for the under-represented Bool answer.
    # Only kick in when the bias is severe (>=4 imbalance) to avoid the
    # 5x DSL-execution overhead per Bool question that aggressive
    # balance entails. With >=4 threshold, balance-retry runs maybe once
    # every 8-10 Bool questions on average.
    target_bool = None
    if is_bool and bool_balance is not None:
        counts = bool_balance.get(tpl.name, {True: 0, False: 0})
        if abs(counts[True] - counts[False]) >= 4:
            target_bool = counts[True] > counts[False]

    params = None; program = None; answer = None
    # Always try up to 3 times to absorb dedup hits and sampling
    # failures. Bool-balance retry uses the same budget when triggered.
    for attempt in range(3):
        params_try = tpl.sample_params(view, rng)
        if params_try is None:
            continue
        # Dedup check
        if seen is not None:
            k = f"{tpl.name}::{_params_key(params_try)}"
            if k in seen:
                continue
        program_try = tpl.gold_program(params_try)
        try:
            if is_h:
                answer_try = tpl.execute_directly(view, params_try)
            else:
                answer_try = dsl_run(program_try, view)
        except Exception:
            continue
        # Bool-balance check
        if target_bool is not None and bool(answer_try) != target_bool:
            params, program, answer = params_try, program_try, answer_try
            continue   # keep trying for the target
        params, program, answer = params_try, program_try, answer_try
        break

    if params is None:
        return None

    paraphrase_id = rng.randint(0, max(0, tpl.n_paraphrases() - 1))
    question_text = tpl.render_question(params, paraphrase_id)
    qid = f"{view.species}/{view.uniprot}/{tpl.name}/{qid_idx}"

    # Update dedup / balance bookkeeping
    if seen is not None:
        seen.add(f"{tpl.name}::{_params_key(params)}")
    if is_bool and bool_balance is not None:
        b = bool_balance.setdefault(tpl.name, {True: 0, False: 0})
        try:
            b[bool(answer)] += 1
        except Exception:
            pass

    return Question(
        qid=qid, uniprot=view.uniprot, species=view.species,
        family=tpl.family, template=tpl.name,
        question=question_text, program=program, answer=answer,
        answer_type=tpl.answer_type, params=params,
        paraphrase_id=paraphrase_id,
    )


def sample_family_letter(rng: random.Random) -> str:
    """Sample a family letter A-G from FAMILY_WEIGHTS_AF (G held out)."""
    keys = ["A", "B", "C", "D", "E", "F"]   # G held out for compositional split
    weights = [FAMILY_WEIGHTS_AF[k] for k in keys]
    total = sum(weights)
    weights = [w / total for w in weights]
    return rng.choices(keys, weights=weights, k=1)[0]


def generate_for_protein(
    view: ProteinView, n_questions: int, rng: random.Random,
    n_family_g: int = N_FAMILY_G_PER_PROTEIN,
    n_family_h: int = N_FAMILY_H_PER_PROTEIN,
) -> list[Question]:
    """Emit:
        - `n_questions` from Families A-F (the main track)
        - `n_family_g` Family G questions (compositional held-out track)
        - `n_family_h` Ha/Hb questions (selective-prediction held-out track)

    Per-protein dedup: same (template, params) tuple never fires twice
    for the same protein. Bool-answer balance: for Bool-typed templates,
    the generator nudges sampling toward whichever answer is currently
    under-represented for that template on this protein (up to 4
    resampling tries before giving up)."""
    out: list[Question] = []
    qid_idx = 0
    seen: set[str] = set()
    bool_balance: dict = {}

    # ----- Main A-F track -----
    attempts = 0
    while len(out) < n_questions and attempts < n_questions * 12:
        attempts += 1
        family = sample_family_letter(rng)
        q = generate_one_question(view, family, rng, qid_idx,
                                     seen=seen, bool_balance=bool_balance)
        if q is not None:
            out.append(q)
            qid_idx += 1

    # ----- Family G track (compositional held-out) -----
    g_attempts = 0
    g_emitted = 0
    while g_emitted < n_family_g and g_attempts < n_family_g * 12:
        g_attempts += 1
        q = generate_one_question(view, "G", rng, qid_idx,
                                     seen=seen, bool_balance=bool_balance)
        if q is not None:
            out.append(q)
            qid_idx += 1
            g_emitted += 1

    # ----- Family H track (selective-prediction held-out) -----
    h_attempts = 0
    h_emitted = 0
    while h_emitted < n_family_h and h_attempts < n_family_h * 12:
        h_attempts += 1
        family = "Ha" if rng.random() < 0.5 else "Hb"
        q = generate_one_question(view, family, rng, qid_idx,
                                     seen=seen, bool_balance=bool_balance)
        if q is not None:
            out.append(q)
            qid_idx += 1
            h_emitted += 1

    return out


def write_questions_jsonl(
    questions: list[Question], out_dir: Path, species: str,
) -> dict[str, int]:
    """Write per-template JSONL files. Returns counts per template."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = Counter()
    handles: dict[str, Any] = {}
    try:
        for q in questions:
            key = q.template
            if key not in handles:
                p = out_dir / f"{key}.jsonl"
                handles[key] = p.open("a", encoding="utf-8")
            handles[key].write(q.to_jsonl() + "\n")
            counts[key] += 1
    finally:
        for h in handles.values():
            h.close()
    return dict(counts)


def run_species(species: str, limit: int | None, rng: random.Random,
                  out_root: Path) -> dict[str, Any]:
    sp_dir = DATA_ROOT / species
    feat_dir = sp_dir / "features"
    ids = [s.strip() for s in (sp_dir / "uniprot_ids.txt").read_text().splitlines()
            if s.strip()]
    if limit is not None:
        ids = ids[:limit]
    out_dir = out_root / species
    # Clear existing JSONLs to avoid append-doubling on re-runs
    if out_dir.exists():
        for p in out_dir.glob("*.jsonl"):
            p.unlink()

    print(f"\n=== {species}: generating ~{N_QUESTIONS_PER_PROTEIN}/protein "
          f"× {len(ids)} = ~{N_QUESTIONS_PER_PROTEIN*len(ids)} questions ===",
          flush=True)
    counts: Counter[str] = Counter()
    n_failed = 0
    t0 = time.perf_counter()
    for i, up in enumerate(ids, 1):
        npz = feat_dir / f"AF-{up}.npz"
        if not npz.exists():
            n_failed += 1
            continue
        try:
            view = load_from_npz(npz, uniprot=up, species=species)
        except Exception as e:
            print(f"  load fail {up}: {e}", flush=True)
            n_failed += 1
            continue
        qs = generate_for_protein(view, N_QUESTIONS_PER_PROTEIN, rng)
        per_template = write_questions_jsonl(qs, out_dir, species)
        counts.update(per_template)
        if i % 100 == 0:
            dt = time.perf_counter() - t0
            rate = i / dt
            eta = (len(ids) - i) / rate
            print(f"  [{species}] {i}/{len(ids)}  {rate:.1f} proteins/s  "
                  f"eta={eta:.0f}s  total_q={sum(counts.values())}",
                  flush=True)

    dt = time.perf_counter() - t0
    print(f"  [{species}] DONE  {len(ids)} proteins  "
          f"{sum(counts.values())} questions  failed={n_failed}  {dt:.0f}s",
          flush=True)
    print(f"  per-template counts:", flush=True)
    for k in sorted(counts.keys()):
        print(f"    {k}: {counts[k]}", flush=True)
    return {"counts": dict(counts), "n_failed": n_failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None,
                      help="comma-separated subset (default: all 4)")
    ap.add_argument("--limit", type=int, default=None,
                      help="cap proteins per species (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    species_keys = (args.species.split(",") if args.species else SPECIES)
    species_keys = [s.strip() for s in species_keys]
    bad = [s for s in species_keys if s not in SPECIES]
    if bad:
        raise SystemExit(f"unknown species: {bad}")

    rng = random.Random(args.seed)
    summary: dict[str, Any] = {}
    for sp in species_keys:
        summary[sp] = run_species(sp, args.limit, rng, args.out_root)

    print(f"\n=== SUMMARY ===", flush=True)
    grand_total = 0
    for sp, info in summary.items():
        n = sum(info["counts"].values())
        grand_total += n
        print(f"  {sp:8s} {n:6d} questions  ({info['n_failed']} load failures)",
              flush=True)
    print(f"  TOTAL  {grand_total} questions", flush=True)


if __name__ == "__main__":
    main()
