"""Post-process question JSONLs to re-render `question` text using the
current paraphrase list in 04_generate_questions.py.

Why this exists:
  When small wording fixes are made to a template's paraphrase_list
  (e.g., the 2026-05-04 Hb1 swap from "Report the CA-CA distance ..." to
  "How far apart are the alpha-carbons of residues ...") AFTER questions
  have already been generated for a long-running session, we don't want
  to regenerate the entire 360K-question dataset to pick up the change.

  Instead: each question already stores `paraphrase_id` and `params`.
  We can simply re-call the template's `render_question(params, id)`
  and overwrite the `question` field on disk. The gold `program` and
  `answer` are unchanged because they don't depend on the paraphrase.

What this script does:
  1. For each {species}/{template}.jsonl file under benchmark/questions/,
     load the latest Template instance.
  2. For each row, re-render the question text. If it differs from the
     stored text, update the row.
  3. Write the updated JSONL atomically (write to .tmp, then rename).
  4. Report per-template counts: total / changed / unchanged.

Idempotent: re-running on an already-updated file produces zero changes.

Restricted to specified templates by default (since most paraphrase
edits are surgical). Pass --all to walk every template.

Usage:
    # Apply only the Hb1 wording fix (2026-05-04)
    python benchmark/08_post_process_paraphrases.py --templates Hb1

    # Re-render every question against the current paraphrase pool
    python benchmark/08_post_process_paraphrases.py --all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

QUESTIONS_ROOT = HERE / "benchmark" / "questions"
SPECIES = ["human", "mouse", "fly", "chicken"]


def _load_gen_module():
    spec = importlib.util.spec_from_file_location(
        "_gen_questions",
        HERE / "benchmark" / "04_generate_questions.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_gen_questions"] = mod
    spec.loader.exec_module(mod)
    return mod


def _all_templates(gen) -> dict[str, "gen.Template"]:
    out: dict = {}
    for tpls in gen.TEMPLATES_BY_FAMILY.values():
        for t in tpls:
            out[t.name] = t
    return out


def repair_jsonl(path: Path, tpl) -> dict:
    """Re-render every row's question with the current paraphrase pool.
    Returns counts {total, changed, unchanged, errors}."""
    counts = {"total": 0, "changed": 0, "unchanged": 0, "errors": 0}
    if not path.exists():
        return counts

    tmp = path.with_suffix(path.suffix + ".tmp")
    n_paraphrase_pool = tpl.n_paraphrases()

    with path.open() as fh, tmp.open("w", encoding="utf-8") as out:
        for line in fh:
            counts["total"] += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                counts["errors"] += 1
                out.write(line)
                continue
            try:
                new_text = tpl.render_question(
                    d["params"],
                    d["paraphrase_id"] % max(1, n_paraphrase_pool),
                )
            except Exception:
                counts["errors"] += 1
                out.write(line)
                continue
            if new_text != d["question"]:
                d["question"] = new_text
                counts["changed"] += 1
            else:
                counts["unchanged"] += 1
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    tmp.replace(path)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="Hb1",
                      help="comma-separated template names to update "
                           "(default: 'Hb1')")
    ap.add_argument("--all", action="store_true",
                      help="walk every template, ignoring --templates")
    ap.add_argument("--species", default=None,
                      help="comma-separated subset (default: all)")
    args = ap.parse_args()

    gen = _load_gen_module()
    all_templates = _all_templates(gen)

    if args.all:
        target_names = sorted(all_templates.keys())
    else:
        target_names = [s.strip() for s in args.templates.split(",")]
        bad = [n for n in target_names if n not in all_templates]
        if bad:
            raise SystemExit(f"unknown templates: {bad}")

    species_keys = (args.species.split(",") if args.species else SPECIES)
    species_keys = [s.strip() for s in species_keys]

    print(f"=== Post-processing paraphrases ===")
    print(f"  templates: {target_names}")
    print(f"  species:   {species_keys}")
    print()

    grand = {"total": 0, "changed": 0, "unchanged": 0, "errors": 0}
    for sp in species_keys:
        for name in target_names:
            tpl = all_templates[name]
            jf = QUESTIONS_ROOT / sp / f"{name}.jsonl"
            if not jf.exists():
                continue
            t0 = time.perf_counter()
            counts = repair_jsonl(jf, tpl)
            dt = time.perf_counter() - t0
            for k in grand:
                grand[k] += counts[k]
            print(f"  {sp:8s}/{name:5s}  total={counts['total']:6d}  "
                  f"changed={counts['changed']:6d}  "
                  f"unchanged={counts['unchanged']:6d}  "
                  f"errors={counts['errors']}  ({dt:.1f}s)")

    print()
    print(f"=== TOTAL  total={grand['total']}  changed={grand['changed']}  "
          f"unchanged={grand['unchanged']}  errors={grand['errors']} ===")


if __name__ == "__main__":
    main()
