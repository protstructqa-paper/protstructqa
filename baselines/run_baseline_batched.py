"""Batched in-process vLLM L0/L1 baseline runner for ProtStructQA.

Per VLLM_BATCHING.md: replaces per-request HTTP with a single batched
`llm.generate(prompts, sp)` call. ~20x faster than the HTTP path for
bulk eval. Use `run_baseline.py` (HTTP adapter) for streaming/online
scenarios; use this script for offline split evaluation.

Usage:
    python -m baselines.run_baseline_batched \
        --questions benchmark/questions/human/A1.jsonl \
        --model-path ./models/Qwen3-1.7B \
        --regime L0 \
        --max-examples 50 \
        --batch-size 32 \
        --few-shot 4

Output (mirrors run_baseline.py):
    benchmark/baseline_runs/{model_name}/{tag}/{regime}/
        per_question.jsonl   model output + score per question
        metrics.json          aggregated metrics
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dsl import load_from_npz, run as dsl_run
from baselines import scoring, output_parser, prompts, grammar_export

DATA_ROOT = Path(os.environ.get("PROTSTRUCTQA_DATA", "./data"))
SPLITS_ROOT = HERE / "benchmark" / "splits"
OUT_ROOT = HERE / "benchmark" / "baseline_runs"


# --------------------------- I/O helpers --------------------------- #


def load_questions(path: Path, max_examples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def load_train_for_few_shot() -> list[dict]:
    p = SPLITS_ROOT / "train.jsonl"
    if not p.exists():
        # Fallback: pull from human positives if splits aren't composed yet
        questions_dir = HERE / "benchmark" / "questions" / "human"
        if not questions_dir.exists():
            return []
        out = []
        for jf in sorted(questions_dir.glob("*.jsonl")):
            with jf.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    out.append(json.loads(line))
                    if len(out) >= 5000:
                        return out
        return out
    out = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
            if len(out) >= 5000:
                break
    return out


def pick_exemplars(train: list[dict], target: dict, n: int = 4,
                      seed: int = 0,
                      template_match: bool = True) -> list[dict]:
    """Pick few-shot exemplars from `train` for an in-context demo.

    `template_match=True` (default, Method B) prefers same-template
    exemplars on different proteins. Empirically the dominant 8B failure
    mode is "wrong_program_logic" --- valid DSL with wrong semantics ---
    which template-matched exemplars directly attack by showing the
    reference program for that exact template. Falls back to same-family
    exemplars, then any family, if too few same-template exemplars are
    available.

    `template_match=False` reproduces the original cross-family sampling
    (kept for ablation in Section~\\ref{sec:results:ablations})."""
    rng = random.Random(seed + hash(target["qid"]))
    if template_match:
        same_t = [q for q in train if q["template"] == target["template"]
                    and q["uniprot"] != target["uniprot"]]
        chosen: list[dict] = []
        while len(chosen) < n and same_t:
            chosen.append(same_t.pop(rng.randrange(len(same_t))))
        if len(chosen) < n:
            same_f = [q for q in train
                        if q["family"] == target["family"]
                        and q["template"] != target["template"]]
            while len(chosen) < n and same_f:
                chosen.append(same_f.pop(rng.randrange(len(same_f))))
        if len(chosen) < n:
            other = [q for q in train if q["family"] != target["family"]]
            while len(chosen) < n and other:
                chosen.append(other.pop(rng.randrange(len(other))))
        return chosen
    # Original cross-family sampler (Method-B-off ablation).
    candidates = [q for q in train if q["template"] != target["template"]]
    by_family = defaultdict(list)
    for q in candidates:
        by_family[q["family"]].append(q)
    fams = sorted(by_family.keys())
    chosen = []
    while len(chosen) < n and any(by_family.values()):
        for f in list(fams):
            if not by_family[f]:
                continue
            chosen.append(by_family[f].pop(rng.randrange(len(by_family[f]))))
            if len(chosen) >= n:
                break
    return chosen


# --------------------------- view caching ------------------------- #


_VIEW_CACHE: dict[str, Any] = {}


def get_view(species: str, uniprot: str):
    key = f"{species}/{uniprot}"
    if key not in _VIEW_CACHE:
        npz = DATA_ROOT / species / "features" / f"AF-{uniprot}.npz"
        _VIEW_CACHE[key] = load_from_npz(npz, uniprot=uniprot, species=species)
    return _VIEW_CACHE[key]


# --------------------------- prompt rendering -------------------- #


SYSTEM_PROMPT = ("You are a precise structural-biology assistant "
                  "that answers questions about AlphaFold protein "
                  "predictions using the ProtStructQA DSL when possible.")


def _apply_chat(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )


def render_one_prompt(q: dict, train: list[dict], n_few_shot: int,
                          regime: str, tokenizer,
                          cot_prefix: bool = False,
                          cot_variant: int = 0) -> str:
    view = get_view(q["species"], q["uniprot"])
    exemplars = pick_exemplars(train, q, n=n_few_shot) if train else []
    if regime == "L0":
        user = prompts.build_l0_prompt(q, view, exemplars=exemplars,
                                             n_shots=len(exemplars),
                                             cot_prefix=cot_prefix,
                                             cot_variant=cot_variant)
    elif regime in ("L1", "EV", "IEV"):
        # EV/IEV use L1's program-only prompt (model emits a DSL program;
        # the EV regime samples k of them and does executor consensus.
        # IEV adds a reflection step over disagreement.)
        user = prompts.build_l1_prompt(q, view, exemplars=exemplars,
                                             n_shots=len(exemplars),
                                             cot_prefix=cot_prefix)
    elif regime == "L2":
        user = prompts.build_l2_prompt(q, view)
    else:
        raise ValueError(f"unknown regime: {regime}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return _apply_chat(tokenizer, messages)


def render_retry_prompt(q: dict, train: list[dict], n_few_shot: int,
                            regime: str, tokenizer,
                            prev_text: str, err_msg: str) -> str:
    """Build a retry prompt for L1 execution-feedback. Re-renders the chat
    with the prior assistant attempt + a user follow-up that pastes the
    error from DSL execution."""
    view = get_view(q["species"], q["uniprot"])
    exemplars = pick_exemplars(train, q, n=n_few_shot) if train else []
    if regime == "L1":
        user = prompts.build_l1_prompt(q, view, exemplars=exemplars,
                                             n_shots=len(exemplars))
    else:
        user = prompts.build_l0_prompt(q, view, exemplars=exemplars,
                                             n_shots=len(exemplars))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant",
          "content": prev_text.strip()[:600] or "(empty)"},
        {"role": "user",
          "content": (
              "Your previous program failed:\n"
              f"{err_msg}\n\n"
              "Provide a corrected ProtStructQA program that respects the "
              f"protein's actual length (n_residues={view.n_residues}) "
              "and the DSL grammar. Output the program only."
          )},
    ]
    return _apply_chat(tokenizer, messages)


def render_prompts(questions: list[dict], train: list[dict], n_few_shot: int,
                      regime: str, tokenizer,
                      cot_prefix: bool = False,
                      cot_variant: int = 0) -> list[str]:
    """Render initial (round-0) prompts for every question."""
    return [render_one_prompt(q, train, n_few_shot, regime, tokenizer,
                                  cot_prefix=cot_prefix,
                                  cot_variant=cot_variant)
              for q in questions]


_DSL_TIMEOUT_SEC = int(os.environ.get("PROTSTRUCTQA_DSL_TIMEOUT", "10"))


class _DSLTimeout(Exception):
    pass


def _is_valid_type(pred, answer_type: str) -> bool:
    """Check whether a Python value is the right type for the question's
    declared answer_type. Used by EV/IEV to reject samples whose executed
    program returns the wrong type (e.g., a Region for a Bool question).
    Permissive for unknown types."""
    base = (answer_type or "").strip()
    if base == "Bool":
        return isinstance(pred, bool)
    if base == "Int":
        return isinstance(pred, int) and not isinstance(pred, bool)
    if base == "Float":
        return isinstance(pred, (int, float)) and not isinstance(pred, bool)
    if base == "Region":
        return (isinstance(pred, (list, tuple)) and len(pred) == 2
                  and all(isinstance(x, int) for x in pred))
    if base in ("ResidueSet", "PairSet"):
        return isinstance(pred, (frozenset, set, list, tuple))
    if base == "SecStruct":
        return isinstance(pred, str) and pred in {"H", "E", "C"}
    return True   # unknown type: permissive


def _dsl_alarm_handler(signum, frame):
    raise _DSLTimeout("DSL program execution exceeded "
                          f"{_DSL_TIMEOUT_SEC}s timeout")


def _try_run(prog: str, view) -> tuple[Any, str | None]:
    """Run a DSL program against a view; return (pred, err_msg). On
    success err_msg is None. On failure pred is None and err_msg is the
    truncated exception text.

    Uses SIGALRM to enforce a hard timeout per-program. Some pathological
    LLM-generated programs (deeply nested comprehensions, runaway sliding
    windows) trigger an effectively-infinite Python loop that ties up the
    whole driver. A 10s ceiling per program lets the chain continue.
    """
    import signal
    prev_handler = signal.signal(signal.SIGALRM, _dsl_alarm_handler)
    signal.alarm(_DSL_TIMEOUT_SEC)
    try:
        pred = dsl_run(prog, view)
        signal.alarm(0)
        return pred, None
    except _DSLTimeout as e:
        signal.alarm(0)
        return None, f"DSLTimeout: {e}"
    except Exception as e:
        signal.alarm(0)
        return None, f"{type(e).__name__}: {str(e)[:240]}"
    finally:
        signal.signal(signal.SIGALRM, prev_handler)


# ============= Multiprocess DSL exec =============================== #
# Slow chunks (95–139 s observed) come from a handful of pathological
# programs each hitting the 10 s SIGALRM timeout serially. With N workers
# these timeouts run concurrently, so chunk wall time drops from
# sum(slow_times) → max(slow_times). Workers maintain their own
# `_WORKER_VIEW_CACHE` so each protein is loaded from disk at most once
# per worker, after which subsequent programs on the same protein hit
# the in-process numpy cache (incl. the KDTree neighbor list cache in
# `dsl/protein_view.py`).

_DSL_POOL = None
_DSL_POOL_SIZE = 0  # 0 = serial fallback

_WORKER_VIEW_CACHE: dict[str, Any] = {}
_WORKER_DATA_ROOT: Path | None = None

import atexit as _atexit  # noqa: E402


def _dsl_worker_init(data_root_str: str, timeout_sec: int):
    """Worker initializer: pre-import dsl + cache data root."""
    global _WORKER_DATA_ROOT
    _WORKER_DATA_ROOT = Path(data_root_str)
    # Touch imports so first task doesn't pay the import cost.
    from dsl import load_from_npz as _l, run as _r  # noqa: F401
    # Make _DSL_TIMEOUT_SEC consistent in the worker.
    import os as _os
    _os.environ["PROTSTRUCTQA_DSL_TIMEOUT"] = str(timeout_sec)


def _worker_get_view(species: str, uniprot: str):
    """Worker-local protein cache. Independent of parent's _VIEW_CACHE."""
    key = f"{species}/{uniprot}"
    cached = _WORKER_VIEW_CACHE.get(key)
    if cached is not None:
        return cached
    from dsl import load_from_npz
    npz = _WORKER_DATA_ROOT / species / "features" / f"AF-{uniprot}.npz"
    view = load_from_npz(npz, uniprot=uniprot, species=species)
    _WORKER_VIEW_CACHE[key] = view
    return view


def _worker_run_program(payload: tuple) -> tuple:
    """Worker entrypoint: (program, species, uniprot) → (pred, err).
    `pred` is `None` if `program` is None/empty (skipped, no error).
    Result must be picklable; DSL outputs are int/float/bool/str/tuple/
    list/set/frozenset of the same: all fine.
    """
    program, species, uniprot = payload
    if not program:
        return (None, None)
    import signal
    from dsl import run as dsl_run

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout("timeout")

    timeout_sec = int(os.environ.get("PROTSTRUCTQA_DSL_TIMEOUT", "10"))
    prev = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_sec)
    try:
        view = _worker_get_view(species, uniprot)
        pred = dsl_run(program, view)
        signal.alarm(0)
        # Convert sets to sorted lists for stable pickle (set hashing on
        # CPython varies; consensus voting downstream is order-insensitive
        # but we want determinism across workers).
        if isinstance(pred, (set, frozenset)):
            pred = sorted(pred)
        return (pred, None)
    except _Timeout:
        signal.alarm(0)
        return (None, f"DSLTimeout: exceeded {timeout_sec}s")
    except Exception as e:
        signal.alarm(0)
        return (None, f"{type(e).__name__}: {str(e)[:240]}")
    finally:
        signal.signal(signal.SIGALRM, prev)


def _ensure_dsl_pool():
    """Lazy-init the pool. Returns the executor or None if serial mode."""
    global _DSL_POOL
    if _DSL_POOL_SIZE <= 0:
        return None
    if _DSL_POOL is not None:
        return _DSL_POOL
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    ctx = _mp.get_context("fork")  # inherit imports/state, fast
    _DSL_POOL = ProcessPoolExecutor(
        max_workers=_DSL_POOL_SIZE,
        mp_context=ctx,
        initializer=_dsl_worker_init,
        initargs=(str(DATA_ROOT), _DSL_TIMEOUT_SEC),
    )
    _atexit.register(_shutdown_dsl_pool)
    return _DSL_POOL


def _shutdown_dsl_pool():
    global _DSL_POOL
    if _DSL_POOL is not None:
        _DSL_POOL.shutdown(wait=False, cancel_futures=True)
        _DSL_POOL = None


def _try_run_batch(items: list[tuple[str | None, str, str]]
                       ) -> list[tuple[Any, str | None]]:
    """Run many DSL programs in parallel.

    Args:
        items: list of (program_or_None, species, uniprot). When program
            is falsy, the slot returns (None, None) immediately without
            being dispatched.

    Returns: list of (pred, err_msg) parallel to `items`.
    """
    pool = _ensure_dsl_pool()
    n = len(items)
    out: list[tuple[Any, str | None]] = [(None, None)] * n
    if pool is None:
        # Serial fallback: load view in-process (uses parent's _VIEW_CACHE).
        for i, (prog, species, uniprot) in enumerate(items):
            if not prog:
                continue
            view = get_view(species, uniprot)
            out[i] = _try_run(prog, view)
        return out
    # Parallel dispatch. Skip empty programs to avoid pool overhead.
    real_idx = [i for i, t in enumerate(items) if t[0]]
    real_payloads = [items[i] for i in real_idx]
    if not real_payloads:
        return out
    # chunksize=1 is fine here: programs vary wildly in cost, and we
    # want timeouts to overlap rather than serialize within a worker.
    results = list(pool.map(_worker_run_program, real_payloads, chunksize=1))
    for slot, res in zip(real_idx, results):
        out[slot] = res
    return out


# --------------------------- vLLM run ----------------------------- #


def build_runner(model_path: str | Path, *,
                  dtype: str = "float16",
                  gpu_memory_utilization: float = 0.85,
                  max_model_len: int = 4096,
                  enforce_eager: bool = False,
                  guidance_backend: str = "guidance"):
    """One-time setup. Loads weights + compiles CUDA graphs.

    `guidance_backend` selects the structured-outputs engine. 'guidance'
    uses LLGuidance (CPU-light, ~3-5x faster mask construction than
    xgrammar). 'xgrammar' is the older default. 'auto' lets vLLM pick.
    """
    import vllm
    kw = dict(
        model=str(model_path),
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    try:
        from vllm.config import StructuredOutputsConfig
        kw["structured_outputs_config"] = StructuredOutputsConfig(
            backend=guidance_backend
        )
    except ImportError:
        # older vLLM: backend is auto-selected, ignore the request
        pass
    return vllm.LLM(**kw)


def make_sampling(max_tokens: int = 512, temperature: float = 0.0,
                    regime: str = "L0",
                    stop: list[str] | None = None):
    """Build SamplingParams. For L1, attach the ProtStructQA GBNF grammar so
    vLLM constrains sampling to grammatical programs."""
    from vllm import SamplingParams
    kwargs: dict = {"max_tokens": max_tokens, "temperature": temperature}
    # vLLM RNG seed (for multi-seed evaluation reproducibility).
    # Only emit when explicitly requested (non-None); otherwise let vLLM
    # use its internal non-deterministic state for back-compat with
    # existing single-seed runs.
    _vseed = make_sampling._vllm_seed
    if _vseed is not None:
        kwargs["seed"] = int(_vseed)
    # L2 must stop after each <act> or <answer> so the runner can inject
    # the real tool result on the next turn rather than letting the model
    # hallucinate its own <obs>.
    if regime == "L2" and not stop:
        stop = ["</act>", "</answer>"]
    if stop:
        kwargs["stop"] = stop
    if regime in ("L1", "EV", "IEV"):
        # vLLM 0.16+ structured outputs API
        try:
            from vllm.sampling_params import StructuredOutputsParams
            grammar = grammar_export.export_gbnf()
            kwargs["structured_outputs"] = StructuredOutputsParams(
                grammar=grammar
            )
        except (ImportError, TypeError):
            pass   # falls back to free sampling if vLLM lacks support
    return SamplingParams(**kwargs)
# Initialise the module-level vllm-seed sentinel. The CLI updates this.
make_sampling._vllm_seed = None


# --------------------------- L2 ReAct tools -------------------------- #


import re as _re


_ACT_RE = _re.compile(r"<act>\s*(.+?)\s*</act>", _re.DOTALL)
_ANSWER_RE = _re.compile(r"<answer>\s*(.+?)\s*</answer>", _re.DOTALL)
_TOOL_CALL_RE = _re.compile(r"^\s*([a-z_][a-z_0-9]*)\s*\((.*)\)\s*$",
                              _re.IGNORECASE | _re.DOTALL)


def _split_dsl_call(tool_call: str) -> tuple[str, str] | None:
    """Find the outermost balanced (...) of a tool call: tool_name(payload).
    Returns (name, raw_payload_string) or None on parse failure. Used for
    run_dsl where the payload may itself contain commas/parens that the
    naive comma-split _parse_args would mangle."""
    m = _re.match(r"^\s*([a-z_][a-z_0-9]*)\s*\(", tool_call,
                     _re.IGNORECASE)
    if not m:
        return None
    name = m.group(1)
    start = m.end() - 1   # position of the opening '('
    depth = 0
    for i in range(start, len(tool_call)):
        c = tool_call[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return name, tool_call[start + 1: i]
    return None


def _parse_args(arg_str: str) -> list:
    """Parse comma-separated numeric/string args, ignoring kwargs."""
    arg_str = arg_str.strip()
    if not arg_str:
        return []
    out = []
    for part in arg_str.split(","):
        p = part.strip().strip("'\"")
        # ignore kwargs like a_start=10
        if "=" in p and not _re.match(r"^-?\d", p.split("=",1)[1].strip()):
            try:
                p = p.split("=",1)[1].strip().strip("'\"")
            except Exception:
                continue
        elif "=" in p:
            p = p.split("=",1)[1].strip().strip("'\"")
        try:
            if "." in p or "e" in p.lower():
                out.append(float(p))
            else:
                out.append(int(p))
        except ValueError:
            out.append(p)
    return out


def _execute_tool(view, name: str, args: list) -> str:
    """Execute one L2 tool and return a short observation string. Five
    canonical multi-tool ReAct tools, all namespace-prefixed with `tool_`
    to disambiguate from DSL primitives (e.g. tool_distance is the
    agent's distance-lookup TOOL; `distance` is the DSL primitive used
    inside tool_dsl programs)."""
    try:
        # Accept both new (tool_*) names and legacy names for backward compat
        if name in ("tool_inspect", "inspect_residue"):
            i = int(args[0])
            return (f"residue {i}: pLDDT={view.plddt_at(i):.2f}, "
                    f"SS={view.ss_at(i)}, "
                    f"rel_sasa={view.rel_sasa_at(i):.3f}, "
                    f"aa={view.ref_aa[view._idx(i)]}")
        if name in ("tool_distance", "compute_distance"):
            i, j = int(args[0]), int(args[1])
            return f"CA-CA distance between residues {i} and {j} = {view.distance(i, j):.2f} A"
        if name in ("tool_pae_mean", "get_pae_block"):
            if len(args) >= 4:
                a0, a1, b0, b1 = (int(args[0]), int(args[1]),
                                       int(args[2]), int(args[3]))
            else:
                a0, a1, b0, b1 = (int(args[0]), int(args[0]),
                                       int(args[1]), int(args[1]))
            mp = view.mean_pae(a0, a1, b0, b1) if hasattr(view, "mean_pae") \
                  else float(view.pae[a0-1:a1, b0-1:b1].mean())
            return f"mean PAE([{a0},{a1}] vs [{b0},{b1}]) = {mp:.2f}"
        if name in ("tool_region_stats", "summarize_region"):
            s, e = int(args[0]), int(args[1])
            mp = view.mean_plddt(s, e)
            mn = view.min_plddt(s, e)
            mx = view.max_plddt(s, e)
            sd = view.std_plddt(s, e) if hasattr(view, "std_plddt") else 0.0
            mrs = view.mean_rel_sasa(s, e) if hasattr(view, "mean_rel_sasa") else 0.0
            try:
                rg = view.radius_of_gyration(s, e)
            except Exception:
                rg = None
            ss = "".join(view.ss_3[s-1:e].tolist())
            n_h = ss.count("H")
            n_e = ss.count("E")
            n_c = ss.count("C")
            rg_str = f", radius_of_gyration={rg:.2f}" if rg is not None else ""
            return (f"region [{s},{e}]: mean_plddt={mp:.2f}, "
                    f"min_plddt={mn:.2f}, max_plddt={mx:.2f}, "
                    f"std_plddt={sd:.2f}, mean_rel_sasa={mrs:.3f}, "
                    f"n_helix_residues={n_h}, n_strand_residues={n_e}, "
                    f"n_coil_residues={n_c}, length={e-s+1}{rg_str}")
        if name in ("tool_dsl", "run_dsl"):
            # Execute an arbitrary ProtStructQA DSL program.
            if isinstance(args, list) and args and isinstance(args[0], str):
                prog = ", ".join(str(a) for a in args)
            else:
                prog = str(args)
            prog = prog.strip().strip("'\"")
            res, err = _try_run(prog, view)
            if err:
                return f"tool_dsl error: {err}"
            try:
                if isinstance(res, float):
                    s = f"{res:.4f}"
                elif isinstance(res, (list, tuple)):
                    s = "[" + ", ".join(str(x) for x in res) + "]"
                elif hasattr(res, "tolist"):
                    s = str(res.tolist())
                else:
                    s = str(res)
            except Exception:
                s = repr(res)
            return f"result = {s[:240]}"
        return f"unknown tool: {name}. Valid tools: tool_inspect, tool_distance, tool_pae_mean, tool_region_stats, tool_dsl"
    except Exception as e:
        return f"tool error: {type(e).__name__}: {str(e)[:200]}"


def render_l2_initial(q: dict, tokenizer) -> str:
    """Render the first-turn L2 prompt as a chat-templated string."""
    view = get_view(q["species"], q["uniprot"])
    user = prompts.build_l2_prompt(q, view)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return _apply_chat(tokenizer, messages)


def _build_l2_followup(q: dict, tokenizer, history: list[dict]) -> str:
    return _apply_chat(tokenizer, history)


def _generate_l2(llm, sampling, qs_chunk: list[dict],
                    initial_prompts: list[str], tokenizer,
                    max_turns: int = 5,
                    ) -> tuple[list[dict], list[int]]:
    """Run L2 ReAct multi-turn batched. At each turn we generate for all
    still-active questions, parse the assistant message for <act> or
    <answer>, execute tools, append history, and loop. Stops when each
    question hits <answer> or max_turns. Returns final outputs and
    n_turns per question."""
    n = len(qs_chunk)
    final = [None] * n
    n_turns = [0] * n
    histories: list[list[dict]] = []
    for q in qs_chunk:
        view = get_view(q["species"], q["uniprot"])
        user = prompts.build_l2_prompt(q, view)
        histories.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])

    pending = list(range(n))
    cur_prompts = list(initial_prompts)
    for turn in range(max_turns):
        if not pending:
            break
        outs = llm.generate(cur_prompts, sampling, use_tqdm=False)
        next_pending = []
        next_prompts = []
        for j, i in enumerate(pending):
            n_turns[i] = turn + 1
            text = outs[j].outputs[0].text if outs[j].outputs else ""
            # vLLM strips the stop sequence; re-append it so downstream
            # regex matches and the appended history is well-formed.
            stop_reason = getattr(outs[j].outputs[0], "stop_reason", None) if outs[j].outputs else None
            if isinstance(stop_reason, str):
                if stop_reason in ("</act>", "</answer>"):
                    text = text + stop_reason
            else:
                # Heuristic: if text ends mid-tag, close it.
                if "<answer>" in text and "</answer>" not in text:
                    text = text + "</answer>"
                elif "<act>" in text and "</act>" not in text:
                    text = text + "</act>"
            n_in = len(outs[j].prompt_token_ids) if hasattr(outs[j], "prompt_token_ids") else 0
            n_out = len(outs[j].outputs[0].token_ids) if outs[j].outputs else 0
            ans_match = _ANSWER_RE.search(text)
            if ans_match or turn == max_turns - 1:
                # Final
                final[i] = {"text": text, "n_in": n_in, "n_out": n_out}
                continue
            act_match = _ACT_RE.search(text)
            if not act_match:
                final[i] = {"text": text, "n_in": n_in, "n_out": n_out}
                continue
            tool_call = act_match.group(1).strip()
            split = _split_dsl_call(tool_call)
            if not split:
                obs = f"could not parse tool call: {tool_call[:120]}"
            else:
                tname, raw_payload = split
                view = get_view(qs_chunk[i]["species"], qs_chunk[i]["uniprot"])
                if tname == "run_dsl":
                    # Pass the entire payload as the DSL program string
                    # without splitting on commas (programs contain commas).
                    obs = _execute_tool(view, "run_dsl", [raw_payload])
                else:
                    targs = _parse_args(raw_payload)
                    obs = _execute_tool(view, tname, targs)
            histories[i].append({"role": "assistant", "content": text})
            histories[i].append({"role": "user", "content": f"<obs>{obs}</obs>"})
            next_prompts.append(_apply_chat(tokenizer, histories[i]))
            next_pending.append(i)
        pending = next_pending
        cur_prompts = next_prompts
    # Anyone still pending: no <answer> emitted; use last text
    for i in pending:
        if final[i] is None:
            # we didn't write final because we exited via the for-else above
            final[i] = {"text": "", "n_in": 0, "n_out": 0}
    return final, n_turns


# --------------------------- score+output ------------------------- #


def _classify_text(q: dict, text: str) -> dict:
    """Parse LLM text → execute against protein view → return a dict with
    parsed/pred/err. Does NOT score against gold."""
    parsed = output_parser.parse_llm_output(text, expected_type=q["answer_type"])
    view = get_view(q["species"], q["uniprot"])
    used_program = False
    err_msg: str | None = None
    if parsed["program"]:
        pred, err_msg = _try_run(parsed["program"], view)
        used_program = pred is not None
    elif parsed["scalar"] is not None:
        pred = parsed["scalar"]
    else:
        pred = None
        err_msg = "no parseable program or scalar"
    return {"parsed": parsed, "pred": pred, "err_msg": err_msg,
              "used_program": used_program}


def evaluate_outputs(
    questions: list[dict], gen_outputs, regime: str,
    n_attempts: list[int] | None = None,
) -> list[dict]:
    """Per-question: parse model text, optionally execute program against
    the protein view, score against gold.

    `gen_outputs` may be either vLLM RequestOutput objects (for round-0
    direct calls) or dicts of the form {'text': str, 'n_in': int,
    'n_out': int} produced by the retry loop.
    `n_attempts` is an optional per-question count of generation rounds
    used (1 if no retry, up to k_retry+1 otherwise).
    """
    rows: list[dict] = []
    for i, (q, out) in enumerate(zip(questions, gen_outputs)):
        if isinstance(out, dict):
            text = out.get("text", "")
            n_in = out.get("n_in", 0)
            n_out = out.get("n_out", 0)
        else:
            text = out.outputs[0].text if out.outputs else ""
            n_in = len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else 0
            n_out = len(out.outputs[0].token_ids) if out.outputs else 0
        c = _classify_text(q, text)
        parsed = c["parsed"]; pred = c["pred"]
        if pred is None:
            score = {"correct": False, "no_parseable_answer": True}
        else:
            score = scoring.score_question(q["answer"], q["answer_type"], pred)
        rows.append({
            "qid": q["qid"], "uniprot": q["uniprot"], "species": q["species"],
            "family": q["family"], "template": q["template"],
            "gold_answer": q["answer"],
            "answer_type": q["answer_type"], "regime": regime,
            "model_text": text,
            "extracted_program": parsed["program"],
            "extracted_scalar": parsed["scalar"],
            "pred_answer": _serialize(pred),
            "used_program": c["used_program"],
            "n_input_tokens": n_in, "n_output_tokens": n_out,
            "n_attempts": (n_attempts[i] if n_attempts else 1),
            "score": score, "correct": score["correct"],
        })
    return rows


def _serialize(x):
    if isinstance(x, (list, tuple, set, frozenset)):
        return list(x) if not isinstance(x, frozenset) else sorted(list(x))
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def evaluate_ev_outputs(
    questions: list[dict], ev_outs: list[dict],
    n_samples: list[int], agreements: list[float],
    regime: str = "EV",
) -> list[dict]:
    """Score EXEC-VOTE outputs. The consensus value (already executed)
    is taken as the prediction; we run scoring against gold.

    Shared between EV and IEV. The `regime` argument tags rows; for IEV the
    iev_* metadata fields are also propagated from `out` so downstream
    analysis can quantify the reflection mechanism.
    """
    rows: list[dict] = []
    for q, out, k, agree in zip(questions, ev_outs, n_samples, agreements):
        consensus = out.get("ev_consensus")
        text = out.get("text", "")
        n_in = out.get("n_in", 0)
        n_out = out.get("n_out", 0)
        if consensus is None:
            score = {"correct": False, "no_parseable_answer": True}
            pred = None
        else:
            score = scoring.score_question(q["answer"], q["answer_type"],
                                                 consensus)
            pred = consensus
        rows.append({
            "qid": q["qid"], "uniprot": q["uniprot"], "species": q["species"],
            "family": q["family"], "template": q["template"],
            "gold_answer": q["answer"],
            "answer_type": q["answer_type"], "regime": regime,
            "model_text": text,
            "extracted_program": None,
            "extracted_scalar": None,
            "pred_answer": _serialize(pred),
            "used_program": pred is not None,
            "n_input_tokens": n_in, "n_output_tokens": n_out,
            "n_attempts": k,
            "ev_agreement": agree,
            "ev_results": out.get("ev_results", []),
            # IEV-specific metadata. None on plain EV (downstream uses .get).
            "iev_reflected":      out.get("iev_reflected"),
            "iev_skipped_reason": out.get("iev_skipped_reason"),
            "iev_program":        out.get("iev_program"),
            "iev_text":           out.get("iev_text"),
            "score": score, "correct": score["correct"],
        })
    return rows


# --------------------------- driver ----------------------------- #


def _recompute_metrics(per_q_path: Path, out_dir: Path, regime: str,
                          model_name: str, questions_path: Path,
                          dt_gen: float = 0.0) -> dict:
    """Re-aggregate metrics over the full per_question.jsonl. Used after a
    resume run so the metrics.json reflects the entire dataset, not just
    the latest batch."""
    all_rows = []
    with per_q_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    metrics = scoring.aggregate([r["score"] for r in all_rows])
    metrics["regime"] = regime
    metrics["model"] = model_name
    metrics["questions_path"] = str(questions_path)
    metrics["wall_clock_s_this_run"] = dt_gen
    by_t = defaultdict(list); by_f = defaultdict(list)
    for r in all_rows:
        by_t[r["template"]].append(r["score"])
        by_f[r["family"]].append(r["score"])
    metrics["by_template"] = {k: scoring.aggregate(v) for k, v in by_t.items()}
    metrics["by_family"] = {k: scoring.aggregate(v) for k, v in by_f.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                          default=str))
    print(f"[batched] DONE  out_dir={out_dir}  total_rows={len(all_rows)}  "
          f"acc={metrics['accuracy_overall']*100:.1f}%", flush=True)
    return metrics


def _existing_qids(per_q_path: Path) -> set[str]:
    """Read an existing per_question.jsonl and return the set of qids
    already evaluated. Tolerates partial / mid-write files (skips the
    last partial line if JSON-decode fails)."""
    if not per_q_path.exists():
        return set()
    qids: set[str] = set()
    with per_q_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "qid" in d:
                qids.add(d["qid"])
    return qids


def _generate_with_l1_feedback(
    llm, sampling, qs_chunk: list[dict], chunk_prompts: list[str],
    train: list[dict], n_few_shot: int, tokenizer, k_retry: int,
) -> tuple[list[dict], list[int]]:
    """L1 with execution-feedback retry. Returns (final_outputs, n_attempts)
    where final_outputs is a list of dicts with keys text/n_in/n_out, one
    per question in qs_chunk.

    Round 0: generate from initial prompts; for each output that fails to
    parse-and-execute, build a retry prompt that includes the prior
    assistant attempt + the executor error. Repeat up to k_retry rounds.
    A question is considered failed if pred is None (no parseable program)."""
    n = len(qs_chunk)
    final = [None] * n        # type: list[dict | None]
    n_attempts = [0] * n
    pending = list(range(n))
    cur_prompts = list(chunk_prompts)
    for attempt in range(k_retry + 1):
        if not pending:
            break
        outs = llm.generate(cur_prompts, sampling, use_tqdm=False)
        next_pending = []
        next_prompts = []
        for j, i in enumerate(pending):
            n_attempts[i] = attempt + 1
            text = outs[j].outputs[0].text if outs[j].outputs else ""
            n_in = len(outs[j].prompt_token_ids) if hasattr(outs[j], "prompt_token_ids") else 0
            n_out = len(outs[j].outputs[0].token_ids) if outs[j].outputs else 0
            c = _classify_text(qs_chunk[i], text)
            should_retry = (
                attempt < k_retry
                and c["pred"] is None
            )
            if should_retry:
                err = c["err_msg"] or "no parseable / executable program"
                rp = render_retry_prompt(qs_chunk[i], train, n_few_shot,
                                            "L1", tokenizer, text, err)
                next_pending.append(i)
                next_prompts.append(rp)
            else:
                final[i] = {"text": text, "n_in": n_in, "n_out": n_out}
        pending = next_pending
        cur_prompts = next_prompts
    return final, n_attempts


# ============= EXEC-VOTE: agentic self-consistency over executable DSL ==
# Sample k programs at temperature 0.7 under the ProtStructQA grammar,
# execute each against the protein view, then aggregate results by
# answer-type-aware consensus. Compared to L1 (single program + retry on
# error) and L2 (free-form ReAct), EXEC-VOTE combines:
#   (1) grammar-constrained sampling (every sample executes)
#   (2) deterministic environment execution (DSL interpreter)
#   (3) type-aware consensus aggregation:
#         Bool / SecStruct / Int  -> majority vote
#         Float                   -> median
#         Region                  -> IoU-medoid (highest mean IoU vs others)
#         PairSet / ResidueSet    -> IoU-medoid set


def _ev_aggregate_bool(results):
    bs = [bool(r) for r in results if isinstance(r, bool)]
    if not bs:
        return None, 0.0
    n_true = sum(bs)
    n = len(bs)
    if n_true * 2 == n:
        return bool(n_true >= n - n_true), n_true / n
    pred = (n_true * 2 > n)
    agree = max(n_true, n - n_true) / n
    return pred, agree


def _ev_aggregate_int(results):
    ints = []
    for r in results:
        try:
            ints.append(int(r))
        except (TypeError, ValueError):
            pass
    if not ints:
        return None, 0.0
    from collections import Counter
    c = Counter(ints)
    val, cnt = c.most_common(1)[0]
    return val, cnt / len(ints)


def _ev_aggregate_float(results):
    fs = []
    for r in results:
        try:
            fs.append(float(r))
        except (TypeError, ValueError):
            pass
    if not fs:
        return None, 0.0
    fs_sorted = sorted(fs)
    median = fs_sorted[len(fs_sorted) // 2]
    # agreement = fraction within 5% relative tol of median
    near = [f for f in fs if abs(f - median) <= max(0.05 * abs(median), 1e-3)]
    return median, len(near) / len(fs)


def _ev_aggregate_secstruct(results):
    ss = [str(r).strip().upper() for r in results
            if isinstance(r, str) and r.strip().upper() in ("H", "E", "C")]
    if not ss:
        return None, 0.0
    from collections import Counter
    c = Counter(ss)
    val, cnt = c.most_common(1)[0]
    return val, cnt / len(ss)


def _ev_aggregate_region(results):
    """Region = (start, end). Return the IoU-medoid: the candidate
    region whose mean IoU to the others is largest."""
    regions = []
    for r in results:
        if isinstance(r, (list, tuple)) and len(r) == 2:
            try:
                regions.append((int(r[0]), int(r[1])))
            except (TypeError, ValueError):
                pass
    if not regions:
        return None, 0.0

    def iou(a, b):
        s1, e1 = a; s2, e2 = b
        inter_lo = max(s1, s2); inter_hi = min(e1, e2)
        inter = max(0, inter_hi - inter_lo + 1)
        union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
        return inter / union if union > 0 else 0.0

    best_score = -1.0; best_idx = 0
    for i, ri in enumerate(regions):
        score = sum(iou(ri, rj) for j, rj in enumerate(regions) if j != i)
        score = score / max(1, len(regions) - 1)
        if score > best_score:
            best_score = score; best_idx = i
    return list(regions[best_idx]), best_score


def _ev_aggregate_set(results):
    """Set / PairSet aggregation by IoU-medoid."""
    sets = []
    for r in results:
        if not isinstance(r, (list, tuple, set, frozenset)):
            continue
        try:
            members = []
            for e in r:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    members.append((int(e[0]), int(e[1])))
                else:
                    members.append(int(e))
            sets.append(frozenset(members))
        except (TypeError, ValueError):
            pass
    if not sets:
        return None, 0.0

    def iou(a, b):
        if not a and not b:
            return 1.0
        u = len(a | b)
        return len(a & b) / u if u else 0.0

    best_score = -1.0; best_idx = 0
    for i, si in enumerate(sets):
        score = sum(iou(si, sj) for j, sj in enumerate(sets) if j != i)
        score = score / max(1, len(sets) - 1)
        if score > best_score:
            best_score = score; best_idx = i
    return [list(p) if isinstance(p, tuple) else p for p in sets[best_idx]], best_score


def _ev_consensus(answer_type: str, results: list) -> tuple:
    """Return (consensus_value, agreement_score in [0,1])."""
    base = (answer_type or "").strip()
    valid = [r for r in results if r is not None]
    if not valid:
        return None, 0.0
    if base == "Bool":
        return _ev_aggregate_bool(valid)
    if base == "Int":
        return _ev_aggregate_int(valid)
    if base == "Float":
        return _ev_aggregate_float(valid)
    if base == "SecStruct":
        return _ev_aggregate_secstruct(valid)
    if base == "Region":
        return _ev_aggregate_region(valid)
    if base in ("ResidueSet", "PairSet"):
        return _ev_aggregate_set(valid)
    # Unknown type: try majority vote on string repr
    from collections import Counter
    c = Counter(str(r) for r in valid)
    val, cnt = c.most_common(1)[0]
    return val, cnt / len(valid)


def _generate_exec_vote(
    llm, sampling, qs_chunk: list[dict], chunk_prompts: list[str],
    k_samples: int = 5,
    ev_temperature: float = 0.7,
    ev_max_tokens: int | None = None,
    ev_type_gate: bool = True,
):
    """EXEC-VOTE generator. Samples k programs per question (under
    grammar), executes each, aggregates by type-aware consensus.

    Args:
        ev_temperature: sampling temperature for k=k_samples (lower → more
            concentrated, less noisy). 0.7 for diversity; 0.3 for tighter
            sampling on hard questions like compositional G family.
        ev_max_tokens: per-sample token budget. 384 covers multi-step
            compositional programs (formerly 192 truncated G programs).
        ev_type_gate: when True, samples whose executed result has the
            wrong Python type for q['answer_type'] are treated as None
            (e.g., a Region result for a Bool question is rejected).

    Returns: (final_outputs, n_samples_per_q, agreement_per_q,
              all_results_per_q, all_programs_per_q).
    The last element is parallel to all_results: list[list[str|None]] of
    parsed programs (or None when unparseable). Used by IEV for reflection.
    """
    n = len(qs_chunk)
    final = [None] * n
    n_samples = [k_samples] * n
    agreements = [0.0] * n
    all_results: list[list] = [[] for _ in range(n)]
    all_programs: list[list] = [[] for _ in range(n)]

    # vLLM SamplingParams supports `n=` for multi-sample per prompt
    from vllm import SamplingParams as _SP
    base_sp = sampling
    # If ev_max_tokens not explicitly provided, fall back to the runner's
    # max_tokens (preserving prior behavior for cells that don't opt in).
    effective_max_tokens = (ev_max_tokens if ev_max_tokens is not None
                                else getattr(base_sp, "max_tokens", 192))
    sp_kwargs = {
        "max_tokens": effective_max_tokens,
        "temperature": ev_temperature,
        "n": k_samples,
        "top_p": 0.95,
    }
    if hasattr(base_sp, "structured_outputs") and base_sp.structured_outputs:
        sp_kwargs["structured_outputs"] = base_sp.structured_outputs
    if hasattr(base_sp, "stop") and base_sp.stop:
        sp_kwargs["stop"] = base_sp.stop

    multi_sp = _SP(**sp_kwargs)
    outs = llm.generate(chunk_prompts, multi_sp, use_tqdm=False)

    # First pass: parse all samples + collect (program, species, uniprot)
    # tuples for batched DSL execution. Scalars + None fall through.
    parsed_per_q: list[list[dict]] = [[] for _ in range(n)]
    sample_texts_per_q: list[list[str]] = [[] for _ in range(n)]
    exec_items: list[tuple[str | None, str, str]] = []
    exec_back_refs: list[tuple[int, int]] = []  # (q_idx, sample_idx) per item

    for i, out in enumerate(outs):
        q = qs_chunk[i]
        for s in out.outputs:
            text = s.text or ""
            sample_texts_per_q[i].append(text)
            parsed = output_parser.parse_llm_output(
                text, expected_type=q["answer_type"]
            )
            parsed_per_q[i].append(parsed)
            prog = parsed.get("program") or None
            exec_items.append((prog, q["species"], q["uniprot"]))
            exec_back_refs.append((i, len(parsed_per_q[i]) - 1))

    # Batched DSL execution (parallel across workers when --dsl-workers>0).
    exec_results = _try_run_batch(exec_items)

    # Precompute offsets into exec_results for each question (avoids
    # O(n²) re-summing per sample).
    q_offsets: list[int] = [0]
    for k in range(n):
        q_offsets.append(q_offsets[-1] + len(parsed_per_q[k]))

    # Second pass: type-gate + consensus. Scalars handled inline.
    for i, out in enumerate(outs):
        q = qs_chunk[i]
        sample_results: list = []
        sample_programs: list = []
        for s_idx, parsed in enumerate(parsed_per_q[i]):
            sample_programs.append(parsed.get("program"))
            if parsed["program"]:
                pred, _err = exec_results[q_offsets[i] + s_idx]
            elif parsed["scalar"] is not None:
                pred = parsed["scalar"]
            else:
                pred = None
            # Type-validity gate: reject samples whose executed result has
            # the wrong type for the question (e.g., Region pred on Bool
            # question). Diagnoses ~52% of EV's compositional failures.
            if (ev_type_gate and pred is not None
                    and not _is_valid_type(pred, q["answer_type"])):
                pred = None
            sample_results.append(pred)
        all_results[i] = sample_results
        all_programs[i] = sample_programs
        consensus, agree = _ev_consensus(q["answer_type"], sample_results)
        agreements[i] = agree
        final[i] = {
            "text": "\n---\n".join(sample_texts_per_q[i]),
            "n_in": len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else 0,
            "n_out": sum(len(s.token_ids) for s in out.outputs),
            "ev_consensus": consensus,
            "ev_agreement": agree,
            "ev_results": [_serialize(r) for r in sample_results],
        }
    return final, n_samples, agreements, all_results, all_programs


# ============= IEV: Iterative-EV with reflection over ensemble disagreement
# Run EV first; for any question whose k samples disagree (agree < 1.0),
# build a reflection prompt that shows the model the k (program, result)
# pairs and asks it to identify the correct program. Re-execute the
# reflected program; commit if parseable, else fall back to EV consensus.
#
# Novelty vs prior work:
#   - Self-Consistency (Wang+ 2023): sample k, vote (no reflection)
#   - Universal SC (Chen+ 2023):     LLM-as-judge over k samples (no execution)
#   - PoT (Chen+ 2022):              1 program, execute (no ensemble)
#   - Reflexion (Shinn+ 2023):       reflect on past failures (no ensemble,
#                                    no execution)
# IEV = parallel sampling + executable grounding + sequential reflection
# over ensemble disagreement. The combination is novel; the agent
# introspects on its own ensemble's disagreement structure before
# committing.


IEV_REFLECT_SYSTEM = (
    "You are a precise structural-biology assistant that answers questions "
    "about AlphaFold protein predictions using the ProtStructQA DSL."
)


_TYPE_AWARE_HINTS = {
    "Bool": (
        "The question expects a boolean answer (true/false). Identify which "
        "candidate's program correctly evaluates the predicate against the "
        "protein structure. The correct program should:\n"
        "  - Use the right primitive (mean_plddt, distance, ss, rel_sasa, etc.)\n"
        "  - Apply the comparison threshold stated in the question\n"
        "  - Reference the exact residue/region indices the question specifies"
    ),
    "Int": (
        "The question expects an integer count. Identify which candidate's "
        "program correctly counts the requested entity. The correct program "
        "should:\n"
        "  - Use 'count' over the right collection (helices, strands, "
        "all_residues, etc.)\n"
        "  - Filter by the exact predicate stated in the question\n"
        "  - Not double-count or undercount via wrong scoping"
    ),
    "Float": (
        "The question expects a numeric value (distance, mean PAE, "
        "radius of gyration, etc.). Identify which candidate's program "
        "correctly computes that value. The correct program should:\n"
        "  - Use the right primitive (distance, mean_pae, "
        "radius_of_gyration, max_pae, mean_plddt)\n"
        "  - Reference the exact residue/region indices the question states\n"
        "  - The result should be in a plausible range for the structural "
        "feature being asked about"
    ),
    "Region": (
        "The question expects a single contiguous region [start, end]. "
        "Identify which candidate's program correctly returns the requested "
        "region. The correct program should:\n"
        "  - Use 'argmax' or 'argmin' over sliding_window or helices/strands\n"
        "  - Use the right scoring primitive (mean_plddt, "
        "radius_of_gyration, contact_density)\n"
        "  - The result should be a single (start, end) tuple, not multiple "
        "regions or a single residue"
    ),
    "PairSet": (
        "The question expects a set of residue pairs (e.g., contacts). "
        "Identify which candidate's program correctly returns the right "
        "set of pairs. The correct program should:\n"
        "  - Use 'filter' over a pair collection\n"
        "  - Apply the exact distance / sequence-separation thresholds "
        "stated in the question"
    ),
    "ResidueSet": (
        "The question expects a set of residues. Identify which candidate's "
        "program correctly returns the right set. The correct program "
        "should:\n"
        "  - Use 'filter' over all_residues or a sub-region\n"
        "  - Apply exactly the predicate(s) the question states "
        "(plddt threshold, ss type, rel_sasa range, etc.)"
    ),
    "SecStruct": (
        "The question expects a secondary-structure label ('H', 'E', or "
        "'C'). Identify which candidate correctly identifies the structure "
        "at the requested residue. The correct program should call ss(...) "
        "with the correct residue index."
    ),
}


def _type_hint(answer_type: str) -> str:
    """Pick the type-aware hint for a given answer_type."""
    base = (answer_type or "").strip()
    return _TYPE_AWARE_HINTS.get(base, "")


def render_iev_reflection_prompt(q: dict, prog_result_pairs: list,
                                    tokenizer) -> str:
    """Build a type-aware reflection prompt showing k candidate
    (program, result) pairs.

    The model is asked to identify the correct candidate and output a single
    ProtStructQA program. The output is grammar-constrained at sampling time.
    Lever 2: includes a type-specific hint about what a "correct" program
    looks like for the question's answer_type, so the reflection can use
    structural reasoning rather than guessing among similar-looking programs.
    """
    candidates = []
    for idx, (prog, result) in enumerate(prog_result_pairs, 1):
        prog_str = prog if prog is not None else "<unparseable>"
        result_str = _serialize(result)
        candidates.append(
            f"Candidate {idx}:\n  program: {prog_str}\n  result: {result_str}"
        )
    candidates_str = "\n\n".join(candidates)

    type_hint = _type_hint(q.get("answer_type", ""))
    type_block = f"\n\n{type_hint}\n" if type_hint else ""

    user_msg = (
        f"Question: {q.get('question', '')}\n\n"
        f"Three candidate ProtStructQA programs were generated and executed "
        f"against the protein structure. They disagree on the answer:\n\n"
        f"{candidates_str}"
        f"{type_block}\n"
        f"Identify which candidate correctly answers the question, then "
        f"output ONLY the correct ProtStructQA program (no commentary, no "
        f"explanation, no markdown)."
    )
    messages = [
        {"role": "system", "content": IEV_REFLECT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    return _apply_chat(tokenizer, messages)


def _truncate_for_context(prompt: str, tokenizer,
                                max_input_tokens: int = 3800) -> str | None:
    """Tokenize prompt; return as-is if within budget, else None to signal
    'too long, skip reflection for this question'. We leave headroom below
    max_model_len for the output tokens.
    """
    try:
        toks = tokenizer.encode(prompt, add_special_tokens=False)
        if len(toks) <= max_input_tokens:
            return prompt
        return None
    except Exception:
        # If tokenization fails, conservatively skip reflection
        return None


def _result_in_samples(reflected_pred, sample_results) -> bool:
    """Check if a reflected program's result matches any of the k original
    sample results. Used by 'conservative' IEV mode (Lever 4) to reject
    reflections that produce novel (likely hallucinated) results.
    Equality is structural: dicts/sets compared by content; lists by tuple
    coercion to handle nested numerics consistently.
    """
    def _key(x):
        if isinstance(x, list):
            return tuple(_key(e) for e in x)
        if isinstance(x, set):
            return frozenset(_key(e) for e in x)
        if isinstance(x, dict):
            return tuple(sorted((k, _key(v)) for k, v in x.items()))
        return x
    try:
        rk = _key(reflected_pred)
        return any(rk == _key(s) for s in sample_results if s is not None)
    except Exception:
        return False


def _generate_iev(
    llm, sampling, qs_chunk: list[dict], chunk_prompts: list[str],
    tokenizer, k_samples: int = 3,
    reflect_max_tokens: int = 192,
    max_input_tokens: int = 3800,
    iev_reflect_threshold: float = 1.0,
    iev_conservative: bool = False,
    ev_temperature: float = 0.7,
    ev_max_tokens: int | None = None,
    ev_type_gate: bool = True,
):
    """IEV = EV + disagreement-driven reflection.

    Pipeline:
      1. Run EV (k samples, type-aware consensus)
      2. For questions with agreement < `iev_reflect_threshold`, build
         a reflection prompt and (subject to context budget) run a single
         grammar-constrained reflected sample
      3. If reflected program is parseable+executes, overwrite EV consensus

    The `iev_reflect_threshold` controls reflection trigger:
      - 1.0: reflect on any non-unanimous question (incl. 2-1 majority)
      - 0.66: reflect only when there's no 2/3 majority (truly contested)
      - 0.5: reflect only on bare-minimum-majority cases
    Tightening the threshold avoids reflection on cases where EV's majority
    was already correct: empirically these were a regression source.

    Robustness:
      - Pre-tokenize every reflection prompt; drop those over the
        max_input_tokens budget (vLLM rejects > max_model_len). The
        question keeps the EV consensus answer when reflection is skipped.
      - Per-program SIGALRM timeout in `_try_run` keeps a single bad
        reflected program from hanging the chunk.

    Returns: (final_outputs, n_samples_per_q, agreement_per_q, all_results,
              all_programs, reflected_mask).
    """
    # Step 1: run EV
    final, n_samples, agreements, all_results, all_programs = (
        _generate_exec_vote(
            llm, sampling, qs_chunk, chunk_prompts,
            k_samples=k_samples,
            ev_temperature=ev_temperature,
            ev_max_tokens=ev_max_tokens,
            ev_type_gate=ev_type_gate,
        )
    )

    # Step 2: identify disagreeing questions; build prompts; filter
    # over-length ones (vLLM rejects prompts > max_model_len).
    pending: list[int] = []
    pending_prompts: list[str] = []
    skipped_too_long = 0
    for i, agree in enumerate(agreements):
        if agree < iev_reflect_threshold:   # below the reflect-trigger
            q = qs_chunk[i]
            pairs = list(zip(all_programs[i], all_results[i]))
            rp = render_iev_reflection_prompt(q, pairs, tokenizer)
            rp_safe = _truncate_for_context(rp, tokenizer,
                                                  max_input_tokens=max_input_tokens)
            if rp_safe is None:
                skipped_too_long += 1
                final[i]["iev_reflected"] = False
                final[i]["iev_skipped_reason"] = "context_too_long"
                continue
            pending.append(i)
            pending_prompts.append(rp_safe)

    # Initialize reflected mask
    reflected = [False] * len(qs_chunk)

    # Mark above-threshold questions as not-reflected with an explicit
    # reason. Questions whose reflection prompt was over-length already
    # had their metadata set in step 2.
    for i in range(len(qs_chunk)):
        if "iev_reflected" not in final[i]:
            if agreements[i] >= iev_reflect_threshold:
                final[i]["iev_reflected"] = False
                final[i]["iev_skipped_reason"] = (
                    "unanimous" if iev_reflect_threshold >= 1.0
                    else f"agreement>={iev_reflect_threshold}"
                )
                final[i]["iev_program"] = None
                final[i]["iev_text"] = None

    if not pending:
        return final, n_samples, agreements, all_results, all_programs, \
                 reflected

    # Step 3: batched grammar-constrained reflection generation.
    refl_sp = make_sampling(max_tokens=reflect_max_tokens, temperature=0.0,
                                regime="L1")
    try:
        refl_outs = llm.generate(pending_prompts, refl_sp, use_tqdm=False)
    except Exception as e:
        # If the batch fails for any reason (token-length validation, etc.)
        # fall back to EV consensus on every disagreeing question.
        print(f"[IEV] reflection batch failed: {type(e).__name__}: "
              f"{str(e)[:200]}; falling back to EV consensus", flush=True)
        for i in pending:
            final[i]["iev_reflected"] = False
            final[i]["iev_skipped_reason"] = "reflection_batch_error"
            final[i]["iev_program"] = None
            final[i]["iev_text"] = None
        return final, n_samples, agreements, all_results, all_programs, \
                 reflected

    # Step 4: parse + execute reflected programs; replace EV consensus on success
    refl_parsed: list[dict] = []
    refl_texts: list[str] = []
    refl_exec_items: list[tuple[str | None, str, str]] = []
    for j, i in enumerate(pending):
        q = qs_chunk[i]
        text = refl_outs[j].outputs[0].text if refl_outs[j].outputs else ""
        refl_texts.append(text)
        parsed = output_parser.parse_llm_output(text,
                                                  expected_type=q["answer_type"])
        refl_parsed.append(parsed)
        refl_exec_items.append((parsed.get("program") or None,
                                  q["species"], q["uniprot"]))
    refl_exec = _try_run_batch(refl_exec_items)

    for j, i in enumerate(pending):
        q = qs_chunk[i]
        parsed = refl_parsed[j]
        text = refl_texts[j]
        if parsed.get("program"):
            pred, _err = refl_exec[j]
            if pred is not None:
                # Conservative-IEV gate: reject reflections whose result is
                # not among the k original sample results: likely a
                # hallucinated novel program/answer rather than a corrected
                # selection among the existing candidates.
                if iev_conservative and not _result_in_samples(
                        pred, all_results[i]):
                    final[i]["iev_reflected"] = False
                    final[i]["iev_skipped_reason"] = "conservative_rejected_novel_pred"
                    final[i]["iev_program"] = parsed["program"]
                    final[i]["iev_text"] = text
                    continue
                final[i]["ev_consensus"] = pred  # overwrite consensus
                final[i]["iev_reflected"] = True
                final[i]["iev_text"] = text
                final[i]["iev_program"] = parsed["program"]
                final[i]["iev_skipped_reason"] = None
                reflected[i] = True
                continue
        # Fallback: keep EV consensus, mark that reflection was attempted
        # but didn't produce a parseable+executable program.
        final[i]["iev_reflected"] = False
        final[i]["iev_skipped_reason"] = "reflection_unparseable"
        final[i]["iev_program"] = None
        final[i]["iev_text"] = text

    if skipped_too_long > 0:
        print(f"[IEV] skipped {skipped_too_long} reflection prompts "
              f"(over {max_input_tokens}-token budget)", flush=True)
    return final, n_samples, agreements, all_results, all_programs, reflected


def run(questions_path: Path, model_path: Path, regime: str,
         max_examples: int | None, batch_size: int, n_few_shot: int,
         max_tokens: int, dtype: str, gpu_memory_utilization: float,
         max_model_len: int, out_root: Path,
         resume: bool = True, force: bool = False,
         k_retry: int = 0, max_turns: int = 5,
         k_samples: int = 5,
         guidance_backend: str = "guidance",
         iev_reflect_threshold: float = 1.0,
         iev_conservative: bool = False,
         ev_temperature: float = 0.7,
         ev_max_tokens: int | None = None,
         ev_type_gate: bool = True,
         cot_prefix: bool = False,
         cot_variant: int = 0,
         enforce_eager: bool = False) -> dict:

    print(f"[batched] questions = {questions_path}", flush=True)
    print(f"[batched] model     = {model_path}", flush=True)
    print(f"[batched] regime    = {regime}", flush=True)
    if _DSL_POOL_SIZE > 0:
        print(f"[batched] dsl_pool  = {_DSL_POOL_SIZE} workers (parallel)",
              flush=True)
    else:
        print(f"[batched] dsl_pool  = serial fallback", flush=True)

    qs_all = load_questions(questions_path, max_examples=max_examples)
    if not qs_all:
        raise SystemExit(f"no questions found in {questions_path}")

    # ----- Resume support: filter to questions not yet evaluated ----- #
    model_name = Path(model_path).name
    tag = questions_path.stem
    out_dir = out_root / model_name / tag / regime
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / "per_question.jsonl"

    if force and per_q_path.exists():
        per_q_path.unlink()
        print(f"[batched] --force: removed existing {per_q_path}", flush=True)

    done_qids: set[str] = _existing_qids(per_q_path) if resume else set()
    if done_qids:
        qs = [q for q in qs_all if q["qid"] not in done_qids]
        print(f"[batched] RESUME: {len(done_qids)} already evaluated, "
              f"{len(qs)} remaining (of {len(qs_all)} total)", flush=True)
        if not qs:
            print(f"[batched] all questions already evaluated; "
                  f"recomputing metrics from existing per_question.jsonl",
                  flush=True)
            return _recompute_metrics(per_q_path, out_dir, regime,
                                          model_name, questions_path)
    else:
        qs = qs_all
        print(f"[batched] loaded {len(qs)} questions", flush=True)

    train = load_train_for_few_shot()
    print(f"[batched] {len(train)} few-shot pool questions available", flush=True)

    print(f"[batched] loading vLLM ({dtype}, {max_model_len} max tokens, "
          f"util={gpu_memory_utilization}, "
          f"guidance_backend={guidance_backend})...", flush=True)
    t_load = time.perf_counter()
    llm = build_runner(model_path, dtype=dtype,
                          gpu_memory_utilization=gpu_memory_utilization,
                          max_model_len=max_model_len,
                          enforce_eager=enforce_eager,
                          guidance_backend=guidance_backend)
    print(f"[batched] vLLM loaded in {time.perf_counter()-t_load:.1f}s",
          flush=True)

    tokenizer = llm.get_tokenizer()
    print(f"[batched] rendering {len(qs)} prompts...", flush=True)
    t = time.perf_counter()
    rendered = render_prompts(qs, train, n_few_shot, regime, tokenizer,
                                  cot_prefix=cot_prefix,
                                  cot_variant=cot_variant)
    print(f"[batched] rendered in {time.perf_counter()-t:.1f}s. "
          f"Mean prompt len ≈ {sum(len(r) for r in rendered)/len(rendered):.0f} chars",
          flush=True)

    sampling = make_sampling(max_tokens=max_tokens, temperature=0.0,
                                regime=regime)
    n_chunks = (len(rendered) + batch_size - 1) // batch_size
    print(f"[batched] generating (batch_size={batch_size}, "
          f"n_chunks={n_chunks})...", flush=True)
    t_gen = time.perf_counter()
    all_rows: list[dict] = []
    n_done = 0
    # Per-chunk: generate -> evaluate -> append-write so a kill mid-run
    # preserves all completed chunks. Resume picks up at the next qid.
    with per_q_path.open("a", encoding="utf-8") as fh:
        for ci, start in enumerate(range(0, len(rendered), batch_size)):
            chunk = rendered[start: start + batch_size]
            qs_chunk = qs[start: start + batch_size]
            t_chunk = time.perf_counter()
            if regime == "L1" and k_retry > 0:
                final_outs, n_attempts = _generate_with_l1_feedback(
                    llm, sampling, qs_chunk, chunk, train, n_few_shot,
                    tokenizer, k_retry,
                )
                chunk_rows = evaluate_outputs(qs_chunk, final_outs, regime,
                                                  n_attempts=n_attempts)
            elif regime == "L2":
                final_outs, n_turns_list = _generate_l2(
                    llm, sampling, qs_chunk, chunk, tokenizer,
                    max_turns=max_turns,
                )
                chunk_rows = evaluate_outputs(qs_chunk, final_outs, regime,
                                                  n_attempts=n_turns_list)
            elif regime == "EV":
                (final_outs, n_samples_list, agreements,
                 _all_results, _all_programs) = _generate_exec_vote(
                    llm, sampling, qs_chunk, chunk,
                    k_samples=k_samples,
                    ev_temperature=ev_temperature,
                    ev_max_tokens=ev_max_tokens,
                    ev_type_gate=ev_type_gate,
                )
                chunk_rows = evaluate_ev_outputs(
                    qs_chunk, final_outs, n_samples_list, agreements,
                    regime="EV",
                )
            elif regime == "IEV":
                (final_outs, n_samples_list, agreements,
                 _all_results, _all_programs, _refl_mask) = _generate_iev(
                    llm, sampling, qs_chunk, chunk, tokenizer,
                    k_samples=k_samples,
                    iev_reflect_threshold=iev_reflect_threshold,
                    iev_conservative=iev_conservative,
                    ev_temperature=ev_temperature,
                    ev_max_tokens=ev_max_tokens,
                    ev_type_gate=ev_type_gate,
                )
                chunk_rows = evaluate_ev_outputs(
                    qs_chunk, final_outs, n_samples_list, agreements,
                    regime="IEV",
                )
            else:
                outs = llm.generate(chunk, sampling, use_tqdm=False)
                chunk_rows = evaluate_outputs(qs_chunk, outs, regime)
            for r in chunk_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
            all_rows.extend(chunk_rows)
            n_done += len(chunk_rows)
            n_correct_c = sum(1 for r in chunk_rows if r["correct"])
            dt_c = time.perf_counter() - t_chunk
            avg_att = (sum(r.get("n_attempts", 1) for r in chunk_rows)
                         / max(1, len(chunk_rows)))
            print(f"[batched] chunk {ci+1}/{n_chunks} "
                  f"({n_done}/{len(rendered)}) "
                  f"acc={100*n_correct_c/max(1,len(chunk_rows)):.1f}% "
                  f"chunk_time={dt_c:.1f}s "
                  f"({len(chunk_rows)/max(1e-3,dt_c):.2f} q/s) "
                  f"avg_attempts={avg_att:.2f}",
                  flush=True)
    dt_gen = time.perf_counter() - t_gen
    rows = all_rows
    print(f"[batched] generation: {len(rows)} responses in {dt_gen:.1f}s "
          f"({len(rows)/dt_gen:.2f} q/s)", flush=True)
    n_correct = sum(1 for r in rows if r["correct"])
    print(f"[batched] accuracy on this batch: {n_correct}/{len(rows)} = "
          f"{100*n_correct/len(rows):.1f}%", flush=True)

    # If we're resuming, recompute metrics across the full file (existing+new)
    if done_qids:
        print(f"[batched] recomputing metrics across all "
              f"{len(done_qids)+len(rows)} rows...", flush=True)
        return _recompute_metrics(per_q_path, out_dir, regime, model_name,
                                       questions_path, dt_gen=dt_gen)

    metrics = scoring.aggregate([r["score"] for r in rows])
    metrics["regime"] = regime
    metrics["model"] = model_name
    metrics["questions_path"] = str(questions_path)
    metrics["wall_clock_s"] = dt_gen
    metrics["throughput_q_s"] = len(rows) / dt_gen if dt_gen else 0
    by_t = defaultdict(list); by_f = defaultdict(list)
    for r in rows:
        by_t[r["template"]].append(r["score"])
        by_f[r["family"]].append(r["score"])
    metrics["by_template"] = {k: scoring.aggregate(v) for k, v in by_t.items()}
    metrics["by_family"] = {k: scoring.aggregate(v) for k, v in by_f.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                          default=str))
    print(f"[batched] DONE  out_dir={out_dir}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, type=Path,
                      help="JSONL of questions to evaluate")
    ap.add_argument("--model-path", type=Path,
                      default=Path("./models/Qwen3-1.7B"))
    ap.add_argument("--regime", default="L0",
                       choices=["L0", "L1", "L2", "EV", "IEV"])
    ap.add_argument("--k-samples", type=int, default=5,
                       help="EV only: number of samples per question.")
    ap.add_argument("--iev-reflect-threshold", type=float, default=1.0,
                       help="IEV: trigger reflection when agreement < this. "
                            "1.0 = reflect on any non-unanimous (default, "
                            "strict). 0.66 = reflect only on truly contested "
                            "cases (1-1-1 split, no 2/3 majority). Lower "
                            "values = less reflection but higher precision.")
    ap.add_argument("--iev-conservative", action="store_true",
                       help="IEV: only commit reflected program if its "
                            "result matches one of the k original sample "
                            "results. Rejects reflections that produce "
                            "novel (likely hallucinated) results.")
    ap.add_argument("--ev-temperature", type=float, default=0.7,
                       help="EV/IEV: sampling temperature for k samples. "
                            "0.7 = high diversity (default); 0.3 = tighter "
                            "sampling (better for compositional G).")
    ap.add_argument("--ev-max-tokens", type=int, default=None,
                       help="EV/IEV: per-sample token budget. If not set, "
                            "falls back to --max-tokens (default 512). "
                            "Pass 384+ to fix truncation on compositional G.")
    ap.add_argument("--no-ev-type-gate", dest="ev_type_gate",
                       action="store_false",
                       help="Disable EV/IEV's type-validity gate. By "
                            "default, samples whose result has the wrong "
                            "type for q['answer_type'] are treated as None.")
    ap.set_defaults(ev_type_gate=True)
    ap.add_argument("--cot-prefix", action="store_true",
                       help="Add chain-of-thought reasoning checklist to "
                            "the L1/EV/IEV prompt. Targets compositional G "
                            "where multi-step reasoning helps.")
    ap.add_argument("--cot-variant", type=int, default=0,
                       help="CoT prompt variant: 0=original 4-step checklist, "
                            "1=generic think-step-by-step, 2=brief 2-step.")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--few-shot", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--dtype", default="float16",
                      help="float16 (P100) or bfloat16 (A100+)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                      help="ignore existing per_question.jsonl, evaluate all questions")
    ap.add_argument("--force", action="store_true",
                      help="delete existing per_question.jsonl before starting")
    ap.add_argument("--k-retry", type=int, default=0,
                      help="L1 only: number of execution-feedback retries "
                           "(0 = single-shot grammar-constrained, 1 = one "
                           "retry on failed execution, ...)")
    ap.add_argument("--max-turns", type=int, default=8,
                      help="L2 only: max ReAct turns per question.")
    ap.add_argument("--vllm-seed", type=int, default=None,
                      help="RNG seed for vLLM sampling (for multi-seed "
                           "reproducibility). If unset, vLLM uses its "
                           "internal non-deterministic state.")
    ap.add_argument("--guidance-backend", default="guidance",
                      choices=["auto", "xgrammar", "guidance",
                               "outlines", "lm-format-enforcer"],
                      help="Structured-outputs backend. 'guidance' "
                           "(LLGuidance) is ~3-5x faster CPU-side than "
                           "xgrammar; output distribution is equivalent.")
    ap.add_argument("--dsl-workers", type=int, default=8,
                      help="Number of CPU workers for parallel DSL "
                           "execution. Slow chunks (with multiple "
                           "10s SIGALRM timeouts) drop from sum(t) → "
                           "max(t) wall time. 0 = serial (legacy).")
    ap.add_argument("--enforce-eager", action="store_true",
                      help="Disable CUDA graph compile in vLLM. Use when "
                           "running multiple vLLM processes on the same "
                           "node: avoids torch_compile_cache contention "
                           "that can deadlock at safetensors load 0%.")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    # Configure the multiprocess DSL pool BEFORE run() (so workers are
    # forked while parent state is still small).
    global _DSL_POOL_SIZE
    _DSL_POOL_SIZE = max(0, int(args.dsl_workers))

    # Plumb the vLLM seed into make_sampling()'s module-level slot.
    make_sampling._vllm_seed = args.vllm_seed

    run(questions_path=args.questions, model_path=args.model_path,
         regime=args.regime, max_examples=args.max_examples,
         batch_size=args.batch_size, n_few_shot=args.few_shot,
         max_tokens=args.max_tokens, dtype=args.dtype,
         gpu_memory_utilization=args.gpu_memory_utilization,
         max_model_len=args.max_model_len, out_root=args.out_root,
         resume=args.resume, force=args.force, k_retry=args.k_retry,
         max_turns=args.max_turns,
         k_samples=args.k_samples,
         guidance_backend=args.guidance_backend,
         iev_reflect_threshold=args.iev_reflect_threshold,
         iev_conservative=args.iev_conservative,
         ev_temperature=args.ev_temperature,
         ev_max_tokens=args.ev_max_tokens,
         ev_type_gate=args.ev_type_gate,
         cot_prefix=args.cot_prefix,
         cot_variant=args.cot_variant,
         enforce_eager=args.enforce_eager)


if __name__ == "__main__":
    main()
