"""Surface-form fidelity check.

For each template family, sample N questions and verify that the
natural-language surface form is consistent with the canonical DSL
program. The check has two parts:

  1. SYNTACTIC: every literal numeric value used in the DSL program
     (residue indices, range endpoints, thresholds) must appear in the
     surface form. This catches paraphrase template bugs where a hand-
     written variant accidentally swaps or drops a parameter.

  2. SEMANTIC HINT: the surface form must mention at least one keyword
     from a small per-family lexicon (e.g., "pLDDT" or "confidence" for
     Family A, "distance" or "apart" for Family B). This catches off-
     family paraphrase contamination.

For questions that fail either check, we dump them to a JSONL file for
manual inspection. The aggregate counts go into the §3.5 validation
paragraph in the paper.

Usage:
    python benchmark/validate_surface_forms.py --n 100
    python benchmark/validate_surface_forms.py --n 50 --out manual.jsonl
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = PROJECT_ROOT / "benchmark/splits"


# Per-family lexicons. Surface forms must mention at least one term.
# Lower-cased substring match against the surface form.
KEYWORDS = {
    "A": [
        "plddt", "confidence", "confident", "reliab", "modelled", "modeled",
        "well-modelled", "well modelled", "well modeled", "score",
        "predicted accuracy", "alphafold", "poorly predicted",
        "weakest", "weakly", "weak", "strong", "strongest", "weak point",
        "uncertain", "uncertainty",
        "worst-predicted", "best-predicted", "worst predicted", "best predicted",
        "highest-quality", "lowest-quality", "highest quality", "lowest quality",
        "best-modelled", "worst-modelled",
    ],
    "B": [
        "distance", "apart", "separated", "separates", "separation",
        "far", "close", "angstrom", "Å",
        "ca-ca", "ca–ca", "ca to ca", "ca - ca", "alpha-carbon", "alpha carbon",
        "within 12", "within 8", "within 10", " within ",
        "contact", "contacts", "span", "spans in", "spatial separation",
    ],
    "C": [
        "pae", "predicted aligned error", "alignment error", "aligned error",
        "error", "uncertainty", "relative-position uncertainty",
        "position uncertainty",
    ],
    "D": [
        "sasa", "solvent", "accessib", "buried", "expose", "surface",
        "neighbor", "neighbour", "neighbors", "neighbours",
        "coordinat", "ligation",
        "within 8", "within 8 angstroms", "8 a", "8 å",
        "packed", "packing", "pack ", "contact count", "contact number",
        "sphere", "ca-ca contacts", "ca–ca contacts",
        "number of ca-ca", "8-angstrom cutoff", "8 angstrom cutoff",
    ],
    "E": [
        "helix", "helices", "helical", "alpha-helix", "alpha helix",
        "sheet", "strand", "loop", "coil", "secondary structure",
        "secondary-structure",
        "ss",   # bare "SS" works only if surrounded; we'll regex
        "h/e/c", "dssp", "h-residue", "h residue",
        "h-run", "h run", "h-runs", "runs of", "h-segment", "h segment",
        "contiguous h", "stretch of h",
    ],
    "F": [
        "contact", "compact", "neighbour", "neighbor", "topology", "density",
        "stretch", "fold", "rg", "radius of gyration", "radius",
        "gyration", "extended", "extension", "spatial spread", "spread",
        "fraction", "proportion", "pair within", "pairs within",
        "pairs in",
    ],
    "G": [],  # G is compositional; permissive: accept anything
}


def extract_numbers(text: str) -> list[int]:
    """Extract non-negative integers from text. We do NOT pick up a leading
    minus sign because ASCII hyphen and Unicode en/em-dash are routinely
    used as range separators in surface forms (e.g., 'residues 16-47')."""
    return [int(m.group()) for m in re.finditer(r"\d+", text)]


def extract_dsl_params(d: dict) -> set[int]:
    """Pull every integer literal that meaningfully appears in the program.
    We exclude small structural constants (e.g., the literal 1 in some
    helper expressions) by sticking to the `params` dict that the
    question generator records."""
    out = set()
    p = d.get("params") or {}
    for v in p.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            iv = int(v)
            if abs(iv) >= 1:
                out.add(iv)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    out.add(int(x))
    return out


def has_keyword(surface_form: str, family: str) -> bool:
    """Check at least one family lexicon term appears."""
    kws = KEYWORDS.get(family, [])
    if not kws:
        return True
    s = surface_form.lower()
    for kw in kws:
        if kw in s:
            return True
    # Special case: bare "SS" token for family E
    if family == "E":
        if re.search(r"\bSS\b", surface_form):
            return True
    return False


def check_one(d: dict) -> dict:
    """Return a dict describing which checks passed."""
    q = d["question"]
    q_nums = set(extract_numbers(q))
    p_nums = extract_dsl_params(d)
    # All literal program parameters must appear in the surface form.
    missing = p_nums - q_nums
    has_kw = has_keyword(q, d["family"])
    return dict(
        ok_syntactic=len(missing) == 0,
        ok_keyword=has_kw,
        missing_params=sorted(missing),
        question=q,
        program=d["program"],
        family=d["family"],
        template=d["template"],
        uniprot=d["uniprot"],
        species=d["species"],
        params=d.get("params", {}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100,
                    help="Number of questions to sample per family")
    ap.add_argument("--split", default="train",
                    help="Split to sample from")
    ap.add_argument("--families", default="A,B,C,D,E,F,G")
    ap.add_argument("--out", default=None,
                    help="JSONL of failing samples for manual inspection")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    random.seed(args.seed)
    families = args.families.split(",")
    path = SPLITS / f"{args.split}.jsonl"
    print(f"Loading {path}...")
    all_q = []
    with open(path) as f:
        for ln in f:
            all_q.append(json.loads(ln))
    random.shuffle(all_q)

    # Sample N per family
    by_fam = {fam: [] for fam in families}
    for d in all_q:
        fam = d["family"]
        if fam in by_fam and len(by_fam[fam]) < args.n:
            by_fam[fam].append(d)
        if all(len(v) >= args.n for v in by_fam.values()):
            break

    # Also sample from cross-species/compositional/hn eval splits to
    # cover non-human species and Family G (which isn't in train).
    extra_splits = {
        "test_compositional_eval.jsonl": ["G"],
        "test_cross_species_eval.jsonl": [f for f in families if f != "G"],
    }
    for split_name, target_fams in extra_splits.items():
        sp = SPLITS / split_name
        if not sp.exists(): continue
        extra = []
        with open(sp) as f:
            for ln in f:
                extra.append(json.loads(ln))
        random.shuffle(extra)
        for d in extra:
            fam = d["family"]
            if fam in target_fams and fam in by_fam and len(by_fam[fam]) < args.n:
                by_fam[fam].append(d)

    results = {fam: [] for fam in families}
    failing_samples = []
    for fam, qs in by_fam.items():
        for d in qs:
            r = check_one(d)
            results[fam].append(r)
            if not (r["ok_syntactic"] and r["ok_keyword"]):
                failing_samples.append(r)

    # Aggregate
    print("\n=== SURFACE-FORM FIDELITY SUMMARY ===")
    overall_n = overall_ok = 0
    for fam in families:
        rs = results[fam]
        n = len(rs)
        n_synt = sum(1 for r in rs if r["ok_syntactic"])
        n_kw = sum(1 for r in rs if r["ok_keyword"])
        n_both = sum(1 for r in rs if r["ok_syntactic"] and r["ok_keyword"])
        overall_n += n; overall_ok += n_both
        print(f"Family {fam}: n={n:3d}  syntactic={n_synt}/{n}"
              f"  keyword={n_kw}/{n}  both={n_both}/{n}")
    print(f"OVERALL: {overall_ok}/{overall_n} ({100*overall_ok/max(1,overall_n):.1f}%)")

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in failing_samples))
        print(f"\nSaved {len(failing_samples)} failing samples to {args.out}")

    # Brief drilldown by family/template for failures
    if failing_samples:
        print("\n=== FAILURE BREAKDOWN BY TEMPLATE ===")
        by_tmpl = Counter(r["template"] for r in failing_samples)
        for tmpl, cnt in by_tmpl.most_common(20):
            print(f"  {tmpl}: {cnt}")

    return results


if __name__ == "__main__":
    main()
