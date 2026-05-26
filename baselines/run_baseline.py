"""L0/L1 baseline runner for ProtStructQA.

Drives a single model through a single split, scores outputs, writes
per-question results + aggregated metrics.

Usage:
    python -m baselines.run_baseline \
        --split test_iid \
        --model-name Qwen3-8B \
        --vllm-url http://localhost:8000 \
        --regime L0 \
        --max-examples 200 \
        --few-shot 6

Output:
    <repo_root>/benchmark/baseline_runs/{model}/{split}/{regime}/
        per_question.jsonl   one row per question with model output + score
        metrics.json          aggregated metrics across the split
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dsl import load_from_npz, run as dsl_run
from baselines import scoring, output_parser, prompts, model_adapter, grammar_export

DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
SPLITS_ROOT = HERE / "benchmark" / "splits"
OUT_ROOT = HERE / "benchmark" / "baseline_runs"


# --------------------------- few-shot exemplar pool --------------- #


def load_split(name: str) -> list[dict]:
    p = SPLITS_ROOT / f"{name}.jsonl"
    if not p.exists():
        return []
    with p.open() as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pick_exemplars(train: list[dict], target_question: dict,
                      n: int = 6, seed: int = 0) -> list[dict]:
    """Pick n exemplars from train, balanced across families and avoiding
    the same template as `target_question`. Deterministic by seed."""
    rng = random.Random(seed + hash(target_question["qid"]))
    candidates = [q for q in train
                    if q["template"] != target_question["template"]]
    by_family = defaultdict(list)
    for q in candidates:
        by_family[q["family"]].append(q)
    fams = sorted(by_family.keys())
    chosen: list[dict] = []
    while len(chosen) < n and fams:
        for f in list(fams):
            if not by_family[f]:
                fams.remove(f); continue
            chosen.append(by_family[f].pop(rng.randrange(len(by_family[f]))))
            if len(chosen) >= n:
                break
    return chosen


# --------------------------- per-question evaluation -------------- #


L1_MAX_RETRIES = 3


def _l1_generate_with_retry(adapter, question, view, exemplars,
                                max_tokens: int = 256,
                                temperature: float = 0.0):
    """L1: generate a grammar-constrained program; if it fails to execute,
    re-prompt with the error message and try again. Returns
    (list_of_generations, last_parsed_output)."""
    grammar = grammar_export.export_gbnf()
    base_prompt = prompts.build_l1_prompt(
        question, view, exemplars=exemplars,
        n_shots=len(exemplars) if exemplars else 0,
    )
    gens: list[dict] = []
    parsed = None
    feedback = ""
    for attempt in range(L1_MAX_RETRIES):
        prompt = (base_prompt + feedback) if feedback else base_prompt
        # Pass the GBNF grammar so vLLM can constrain sampling. If the
        # adapter doesn't support it (HF local), the kwarg is ignored.
        try:
            gen = adapter.generate(prompt, max_tokens=max_tokens,
                                      temperature=temperature,
                                      guided_grammar=grammar)
        except TypeError:
            # Adapter doesn't accept guided_grammar: fall back to plain
            gen = adapter.generate(prompt, max_tokens=max_tokens,
                                      temperature=temperature)
        gens.append(gen)
        parsed = output_parser.parse_llm_output(
            gen.get("text", ""), expected_type=question["answer_type"]
        )
        # If we got a program AND it executes cleanly, accept and return.
        if parsed["program"]:
            try:
                _ = dsl_run(parsed["program"], view)
                return gens, parsed   # success
            except Exception as e:
                # Build feedback for the next retry
                feedback = (f"\n\nThe previous program failed at execution "
                            f"with: {type(e).__name__}: {e}. Please revise.")
                continue
        # No program extracted: treat as failure and retry
        feedback = ("\n\nNo valid ProtStructQA program was extracted. "
                    "Output a single program only.")
    return gens, parsed


def evaluate_one(adapter, question: dict, view, exemplars: list[dict],
                   regime: str, max_tokens: int = 256,
                   temperature: float = 0.0) -> dict:
    """Build prompt → call model → parse → score → return result row.

    L0: single inference, no retries. L1: grammar-constrained sampling +
    execution-feedback retry loop (up to L1_MAX_RETRIES attempts; on
    each program-execution error, re-prompt with the error message)."""
    if regime == "L0":
        prompt = prompts.build_l0_prompt(question, view, exemplars=exemplars,
                                              n_shots=len(exemplars))
        gens = []
        gen = adapter.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature)
        gens.append(gen)
        parsed = output_parser.parse_llm_output(
            gen.get("text", ""), expected_type=question["answer_type"]
        )
    elif regime == "L1":
        gens, parsed = _l1_generate_with_retry(
            adapter, question, view, exemplars,
            max_tokens=max_tokens, temperature=temperature,
        )
    else:
        raise ValueError(f"unknown regime: {regime}")

    # Use the LAST generation as the canonical reply
    gen = gens[-1] if gens else {"text": "", "n_input_tokens": 0,
                                       "n_output_tokens": 0,
                                       "stop_reason": "error",
                                       "elapsed_s": 0.0}

    # Resolve to a final predicted answer
    pred_answer: Any
    used_program = False
    if parsed["program"]:
        try:
            pred_answer = dsl_run(parsed["program"], view)
            used_program = True
        except Exception:
            pred_answer = None
    elif parsed["scalar"] is not None:
        pred_answer = parsed["scalar"]
    elif parsed["abstained"]:
        pred_answer = "Unreliable"
    else:
        pred_answer = None

    # Score
    if pred_answer is None:
        score = {"correct": False, "abstained_correctly": None,
                   "no_parseable_answer": True}
    else:
        score = scoring.score_question(
            question["answer"], question["answer_type"], pred_answer,
        )

    return {
        "qid": question["qid"],
        "uniprot": question["uniprot"],
        "species": question["species"],
        "family": question["family"],
        "template": question["template"],
        "gold_answer": question["answer"],
        "answer_type": question["answer_type"],
        "regime": regime,
        "n_attempts": len(gens) if regime == "L1" else 1,
        "model_text": gen.get("text", ""),
        "extracted_program": parsed["program"],
        "extracted_scalar": parsed["scalar"],
        "abstained": parsed["abstained"],
        "pred_answer": _serialize(pred_answer),
        "used_program": used_program,
        "n_input_tokens": sum(g.get("n_input_tokens", 0) for g in (gens or [gen])),
        "n_output_tokens": sum(g.get("n_output_tokens", 0) for g in (gens or [gen])),
        "elapsed_s": sum(g.get("elapsed_s", 0.0) for g in (gens or [gen])),
        "score": score,
        "correct": score["correct"],
    }


def _serialize(x):
    if isinstance(x, (list, tuple, set, frozenset)):
        return list(x) if not isinstance(x, frozenset) else sorted(list(x))
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


# --------------------------- driver ----------------------------- #


def run(split_name: str, regime: str,
         model_backend: str, model_name: str, vllm_url: str | None,
         max_examples: int | None, n_few_shot: int,
         out_root: Path = OUT_ROOT) -> dict:
    test = load_split(split_name)
    train = load_split("train")
    if not test:
        raise SystemExit(f"split {split_name} is empty or missing.")
    if max_examples:
        test = test[:max_examples]

    if model_backend == "vllm-http":
        adapter = model_adapter.VLLMHTTPAdapter(
            url=vllm_url or "http://localhost:8000", model=model_name
        )
    else:
        adapter = model_adapter.HFTransformersAdapter(model_path=model_name)

    out_dir = out_root / model_name.split("/")[-1] / split_name / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / "per_question.jsonl"

    print(f"[run_baseline] split={split_name} regime={regime} "
          f"model={model_name} n={len(test)}", flush=True)

    # Load each protein view once
    view_cache: dict[str, Any] = {}
    def _view(species: str, uniprot: str):
        key = f"{species}/{uniprot}"
        if key not in view_cache:
            npz = DATA_ROOT / species / "features" / f"AF-{uniprot}.npz"
            view_cache[key] = load_from_npz(npz, uniprot=uniprot,
                                                species=species)
        return view_cache[key]

    results: list[dict] = []
    t0 = time.perf_counter()
    with per_q_path.open("w", encoding="utf-8") as out_fh:
        for i, q in enumerate(test, 1):
            try:
                view = _view(q["species"], q["uniprot"])
            except Exception as e:
                print(f"  ✗ {q['qid']}: view load failed: {e}", flush=True)
                continue
            exemplars = pick_exemplars(train, q, n=n_few_shot)
            row = evaluate_one(adapter, q, view, exemplars, regime)
            results.append(row)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                dt = time.perf_counter() - t0
                rate = i / dt
                acc = sum(1 for r in results if r["correct"]) / len(results)
                print(f"  [{regime}] {i}/{len(test)}  acc={acc*100:.1f}%  "
                      f"{rate:.2f} q/s  eta={(len(test)-i)/rate:.0f}s",
                      flush=True)

    metrics = scoring.aggregate([r["score"] for r in results])
    metrics["regime"] = regime
    metrics["model"] = model_name
    metrics["split"] = split_name
    metrics["n_few_shot"] = n_few_shot
    metrics["wall_clock_s"] = time.perf_counter() - t0

    # Per-template + per-family breakdowns
    by_template = defaultdict(list)
    by_family = defaultdict(list)
    for r in results:
        by_template[r["template"]].append(r["score"])
        by_family[r["family"]].append(r["score"])
    metrics["by_template"] = {k: scoring.aggregate(v)
                                  for k, v in by_template.items()}
    metrics["by_family"] = {k: scoring.aggregate(v)
                                for k, v in by_family.items()}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                          default=str))
    print(f"[run_baseline] DONE  acc_overall={metrics['accuracy_overall']*100:.1f}%  "
          f"out={out_dir}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                      help="e.g., test_iid / test_cross_species / test_compositional")
    ap.add_argument("--regime", default="L0", choices=["L0", "L1"])
    ap.add_argument("--model-backend", default="vllm-http",
                      choices=["vllm-http", "hf-transformers"])
    ap.add_argument("--model-name", required=True,
                      help="vLLM-registered name OR HF path")
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--few-shot", type=int, default=6)
    args = ap.parse_args()

    run(split_name=args.split, regime=args.regime,
         model_backend=args.model_backend, model_name=args.model_name,
         vllm_url=args.vllm_url, max_examples=args.max_examples,
         n_few_shot=args.few_shot)


if __name__ == "__main__":
    main()
