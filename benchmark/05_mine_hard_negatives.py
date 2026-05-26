"""ProtStructQA hard-negative miner.

For every positive question in benchmark/questions/{species}/{template}.jsonl,
emit one hard negative that is:
  - structurally well-formed (same template, valid params)
  - factually different from the positive (answer flipped or distinct
    beyond a per-type tolerance)
  - lexically natural (same template paraphrase pool, no negation tells)

Three classes per HARD_NEGATIVES.md:
  - HN1 "structural counterfactual" (~70%): same template, resample params
    on the SAME protein until the answer differs.
  - HN2 "threshold flip" (~15%): keep residues/regions fixed, flip the
    numeric threshold.
  - HN3 "cross-protein" (~15%): same template+params, swap to a protein
    where the answer is opposite.

This v1 implements HN1 and HN2 (HN3 deferred (needs a per-template
"opposite-protein" index).

Outputs JSONL at:
    <project-root>/
        protstructqa/benchmark/hard_negatives/{species}/{template}.jsonl

with the same schema as positives plus:
    is_hard_negative: True
    hn_class:         "HN1" | "HN2"
    hn_source_qid:    qid of the positive this was derived from

Usage:
    python benchmark/05_mine_hard_negatives.py --species human --limit 100
    python benchmark/05_mine_hard_negatives.py             # full
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dsl import load_from_npz, run as dsl_run, ProteinView


# Import the question-generator module by file path (filename starts with
# a digit, so a normal `import` won't work).
_GEN_PATH = Path(__file__).resolve().parent / "04_generate_questions.py"
_spec = importlib.util.spec_from_file_location("_gen_questions", _GEN_PATH)
_gen = importlib.util.module_from_spec(_spec)
sys.modules["_gen_questions"] = _gen
_spec.loader.exec_module(_gen)


DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
QUESTIONS_ROOT = HERE / "benchmark" / "questions"
HN_ROOT = HERE / "benchmark" / "hard_negatives"
SPECIES = ["human", "mouse", "fly", "chicken"]

# Class probabilities per HARD_NEGATIVES.md §5
CLASS_WEIGHTS = {
    "A":  [0.70, 0.15, 0.15],   # HN1, HN2, HN3
    "B":  [0.70, 0.15, 0.15],
    "C":  [0.70, 0.15, 0.15],
    "D":  [0.70, 0.15, 0.15],
    "E":  [0.85, 0.05, 0.10],
    "F":  [0.65, 0.15, 0.20],
    "G":  [1.00, 0.00, 0.00],   # always HN1 for compositional
    "Ha": [1.00, 0.00, 0.00],
    "Hb": [1.00, 0.00, 0.00],
}

# HN ratio: 1 HN per ~10 positives -> emit HN for 10% of positives.
DEFAULT_HN_RATIO = 0.10


# --------------------------- answer comparison ------------------------- #


def answers_differ(a: Any, b: Any, answer_type: str,
                    numeric_rel_tol: float = 0.07,
                    numeric_abs_tol: float = 0.6) -> bool:
    """Return True iff `a` is meaningfully different from `b`, judged by
    a tolerance just slightly stricter than the scorer's. The HN must
    give an answer that the scorer would NOT accept as the same as the
    positive, otherwise the HN isn't actually wrong vs the gold.

    Scorer tolerances: Float abs<=0.5 OR rel<=0.05; Int abs<=2 OR rel<=0.10.
    HN must exceed those: Float abs>0.6 OR rel>0.07; Int abs>=3 OR rel>0.12.
    """
    base = answer_type.replace("|Unreliable", "")
    if a == "Unreliable" or b == "Unreliable":
        return a != b
    if base == "Bool":
        return bool(a) != bool(b)
    if base == "Float":
        try:
            af, bf = float(a), float(b)
        except (TypeError, ValueError):
            return a != b
        diff = abs(af - bf)
        if diff < 1e-9:
            return False
        denom = max(abs(af), abs(bf), 1e-9)
        return (diff / denom > numeric_rel_tol) or (diff > numeric_abs_tol)
    if base == "Int":
        try:
            ai, bi = int(a), int(b)
        except (TypeError, ValueError):
            return a != b
        diff = abs(ai - bi)
        denom = max(abs(ai), abs(bi), 1)
        return (diff / denom > 0.12) or (diff >= 3)
    if base == "Region":
        return tuple(a) != tuple(b)
    if base == "ResidueSet":
        return set(a) != set(b)
    if base == "PairSet":
        return set(map(tuple, a)) != set(map(tuple, b))
    if base in ("SecStruct", "AAType"):
        return str(a) != str(b)
    return a != b


# --------------------------- template lookup --------------------------- #


def _all_templates() -> dict[str, "_gen.Template"]:
    """Build a flat name → template instance mapping."""
    out: dict[str, Any] = {}
    for fam, tpls in _gen.TEMPLATES_BY_FAMILY.items():
        for t in tpls:
            out[t.name] = t
    return out


_TEMPLATES = _all_templates()


def _is_h(name: str) -> bool:
    return name.startswith("Ha") or name.startswith("Hb")


def _execute(program: str, params: dict, tpl, view: ProteinView) -> Any:
    """Compute the gold answer for `program` either via DSL execution
    (Families A-G) or via execute_directly() (Families Ha/Hb)."""
    if getattr(tpl, "is_family_h", False):
        return tpl.execute_directly(view, params)
    return dsl_run(program, view)


# --------------------------- HN1: structural counterfactual ----------- #


def generate_hn1(positive: dict, view: ProteinView, rng: random.Random,
                  max_attempts: int = 30) -> dict | None:
    """Resample template params on the same protein until the answer
    differs from the positive's gold."""
    name = positive["template"]
    tpl = _TEMPLATES.get(name)
    if tpl is None:
        return None
    pos_answer = positive["answer"]
    pos_type = positive["answer_type"]

    for _ in range(max_attempts):
        params = tpl.sample_params(view, rng)
        if params is None or params == positive["params"]:
            continue
        program = tpl.gold_program(params)
        try:
            answer = _execute(program, params, tpl, view)
        except Exception:
            continue
        if not answers_differ(answer, pos_answer, pos_type):
            continue
        # Found a flipped-answer instance.
        paraphrase_id = rng.randint(0, max(0, tpl.n_paraphrases() - 1))
        question = tpl.render_question(params, paraphrase_id)
        # JSON-friendly answer
        ans_clean = _gen._serialize_answer(answer)
        return {
            "qid": positive["qid"] + "/HN1",
            "uniprot": positive["uniprot"],
            "species": positive["species"],
            "family": positive["family"],
            "template": positive["template"],
            "question": question,
            "program": program,
            "answer": ans_clean,
            "answer_type": positive["answer_type"],
            "params": params,
            "paraphrase_id": paraphrase_id,
            "is_hard_negative": True,
            "hn_class": "HN1",
            "hn_source_qid": positive["qid"],
        }
    return None


# --------------------------- HN2: threshold flip ---------------------- #
# Strategy: re-use the positive's params, but mutate any numeric threshold
# to a value whose answer flips. Works for Bool / Int templates that have
# a clear numeric threshold parameter ("threshold", "plddt_thr", "sasa_thr",
# "cd_thr", "pae_max", "neighbor_threshold"). Skip otherwise.


_THRESHOLD_KEYS = {
    "threshold", "plddt_thr", "sasa_thr", "cd_thr", "pae_max",
    "coverage_thr", "pae_thr", "neighbor_threshold",
}


def generate_hn2(positive: dict, view: ProteinView, rng: random.Random,
                  max_attempts: int = 30) -> dict | None:
    name = positive["template"]
    tpl = _TEMPLATES.get(name)
    if tpl is None:
        return None
    keys_in_params = [k for k in positive["params"]
                        if k in _THRESHOLD_KEYS]
    if not keys_in_params:
        return None
    pos_answer = positive["answer"]
    pos_type = positive["answer_type"]
    # Try mutating threshold values
    base_params = dict(positive["params"])
    chosen_key = rng.choice(keys_in_params)
    candidates = _threshold_candidates(chosen_key, base_params[chosen_key])

    for cand in rng.sample(candidates, min(len(candidates), max_attempts)):
        params = dict(base_params)
        params[chosen_key] = cand
        if params == positive["params"]:
            continue
        program = tpl.gold_program(params)
        try:
            answer = _execute(program, params, tpl, view)
        except Exception:
            continue
        if not answers_differ(answer, pos_answer, pos_type):
            continue
        paraphrase_id = rng.randint(0, max(0, tpl.n_paraphrases() - 1))
        question = tpl.render_question(params, paraphrase_id)
        return {
            "qid": positive["qid"] + "/HN2",
            "uniprot": positive["uniprot"],
            "species": positive["species"],
            "family": positive["family"],
            "template": positive["template"],
            "question": question,
            "program": program,
            "answer": _gen._serialize_answer(answer),
            "answer_type": positive["answer_type"],
            "params": params,
            "paraphrase_id": paraphrase_id,
            "is_hard_negative": True,
            "hn_class": "HN2",
            "hn_source_qid": positive["qid"],
        }
    return None


def _threshold_candidates(key: str, current: float | int) -> list:
    if key == "threshold":            # distance threshold (Å)
        return [4, 5, 6, 8, 10, 12, 15, 20]
    if key == "plddt_thr":
        return [40, 50, 60, 70, 80, 90]
    if key == "sasa_thr":
        return [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    if key == "cd_thr":
        return [0.10, 0.20, 0.30, 0.40, 0.50]
    if key == "pae_max":
        return [5, 10, 15, 20, 25, 30]
    if key == "coverage_thr":
        return [10, 20, 30, 40, 50, 60]
    if key == "pae_thr":
        return [5, 10, 15, 20, 25]
    if key == "neighbor_threshold":
        return [4, 8, 12, 16, 20]
    return []


# --------------------------- driver ------------------------------------ #


def mine_one_question(positive: dict, view: ProteinView,
                        rng: random.Random) -> tuple[dict | None, str]:
    """Pick an HN class per the family weights, generate, return (hn, class).

    HN3 is deferred; if drawn we fall back to HN1.
    """
    fam = positive["family"]
    base_fam = "Ha" if fam == "Ha" else "Hb" if fam == "Hb" else fam
    weights = CLASS_WEIGHTS.get(base_fam, CLASS_WEIGHTS["A"])
    classes = ["HN1", "HN2", "HN3"]
    cls = rng.choices(classes, weights=weights, k=1)[0]
    if cls == "HN1":
        hn = generate_hn1(positive, view, rng)
        return hn, "HN1"
    if cls == "HN2":
        hn = generate_hn2(positive, view, rng)
        if hn is not None:
            return hn, "HN2"
        # Fallback to HN1 if the threshold mutation found nothing
        hn = generate_hn1(positive, view, rng)
        return hn, "HN1"
    if cls == "HN3":
        # Deferred; use HN1 as fallback
        hn = generate_hn1(positive, view, rng)
        return hn, "HN1"
    return None, ""


def _load_positives(species: str, limit: int | None
                       ) -> dict[str, list[dict]]:
    """Return a dict {uniprot: [positive_question_dict, ...]}."""
    sp_dir = QUESTIONS_ROOT / species
    if not sp_dir.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for jf in sorted(sp_dir.glob("*.jsonl")):
        with jf.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                up = d["uniprot"]
                out.setdefault(up, []).append(d)
    if limit is not None:
        return {k: v for k, v in list(out.items())[:limit]}
    return out


def _existing_source_qids(out_dir: Path) -> set[str]:
    """Read existing HN JSONLs and return the set of source-question qids
    that have already been mined (so resume can skip those positives)."""
    if not out_dir.exists():
        return set()
    seen: set[str] = set()
    for jf in out_dir.glob("*.jsonl"):
        with jf.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = d.get("hn_source_qid")
                if src:
                    seen.add(src)
    return seen


def run_species(species: str, limit: int | None, hn_ratio: float,
                  rng: random.Random, resume: bool = True,
                  force: bool = False) -> dict[str, Any]:
    sp_dir_data = DATA_ROOT / species
    feat_dir = sp_dir_data / "features"
    out_dir = HN_ROOT / species
    if force and out_dir.exists():
        for p in out_dir.glob("*.jsonl"):
            p.unlink()
        print(f"[{species}] --force: wiped existing HNs", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_source_qids = _existing_source_qids(out_dir) if resume else set()
    if done_source_qids:
        print(f"[{species}] RESUME: {len(done_source_qids)} source positives "
              f"already mined; will skip them", flush=True)

    positives = _load_positives(species, limit)
    if not positives:
        print(f"[{species}] no positives at {QUESTIONS_ROOT/species}; "
              f"run 04_generate_questions.py first", flush=True)
        return {"counts": {}, "n_failed": 0}

    print(f"\n=== {species}: mining HNs over {len(positives)} proteins, "
          f"target ratio {hn_ratio*100:.0f}% ===", flush=True)
    counts_class: Counter[str] = Counter()
    counts_template: Counter[str] = Counter()
    n_failed = 0
    n_total = 0
    t0 = time.perf_counter()
    handles: dict[str, Any] = {}
    try:
        for i, (up, qs) in enumerate(positives.items(), 1):
            n_total += len(qs)
            npz = feat_dir / f"AF-{up}.npz"
            if not npz.exists():
                n_failed += len(qs)
                continue
            try:
                view = load_from_npz(npz, uniprot=up, species=species)
            except Exception:
                n_failed += len(qs)
                continue
            # Pick a hn_ratio fraction of positives to mine HNs from
            n_pick = max(1, int(round(len(qs) * hn_ratio)))
            # Resume: filter out positives that we already have HNs for
            qs_pool = [q for q in qs if q["qid"] not in done_source_qids]
            if not qs_pool:
                continue
            picked = rng.sample(qs_pool, min(n_pick, len(qs_pool)))
            for q in picked:
                hn, cls = mine_one_question(q, view, rng)
                if hn is None:
                    n_failed += 1
                    continue
                counts_class[cls] += 1
                counts_template[hn["template"]] += 1
                key = hn["template"]
                if key not in handles:
                    # buffering=1 = line-buffered; flushes on each newline.
                    # Avoids the "B3-only on disk while others buffer up"
                    # confusion that fooled the v1 progress check.
                    handles[key] = (out_dir / f"{key}.jsonl").open(
                        "a", encoding="utf-8", buffering=1,
                    )
                handles[key].write(json.dumps(hn, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                dt = time.perf_counter() - t0
                rate = i / dt
                eta = (len(positives) - i) / rate
                print(f"  [{species}] {i}/{len(positives)}  "
                      f"{rate:.1f} prot/s  eta={eta:.0f}s  "
                      f"hn_total={sum(counts_class.values())}", flush=True)
    finally:
        for h in handles.values():
            h.close()

    dt = time.perf_counter() - t0
    print(f"  [{species}] DONE  positives={n_total}  "
          f"hns={sum(counts_class.values())}  failed={n_failed}  {dt:.0f}s",
          flush=True)
    print(f"  by class: {dict(counts_class)}", flush=True)
    return {"counts_class": dict(counts_class),
              "counts_template": dict(counts_template),
              "n_total_positives": n_total, "n_failed": n_failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=None,
                      help="comma-separated subset (default: all 4)")
    ap.add_argument("--limit", type=int, default=None,
                      help="cap proteins per species (smoke test)")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--hn-ratio", type=float, default=DEFAULT_HN_RATIO)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                      help="ignore existing HN files, restart from scratch (default: resume)")
    ap.add_argument("--force", action="store_true",
                      help="wipe existing HN files before starting (implies --no-resume)")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    species_keys = (args.species.split(",") if args.species else SPECIES)
    species_keys = [s.strip() for s in species_keys]
    bad = [s for s in species_keys if s not in SPECIES]
    if bad:
        raise SystemExit(f"unknown species: {bad}")

    rng = random.Random(args.seed)
    summary: dict[str, Any] = {}
    for sp in species_keys:
        summary[sp] = run_species(sp, args.limit, args.hn_ratio, rng,
                                       resume=args.resume, force=args.force)

    print(f"\n=== SUMMARY ===", flush=True)
    grand = 0
    for sp, info in summary.items():
        n = sum(info.get("counts_class", {}).values())
        grand += n
        print(f"  {sp:8s}  hns={n:6d}  positives_seen={info.get('n_total_positives', 0)}",
              flush=True)
    print(f"  TOTAL  hns={grand}", flush=True)


if __name__ == "__main__":
    main()
