# Results

Anonymized aggregate accuracy data for ProtStructQA baselines on the
paper-evaluation subsample (32,357 questions across four splits).

## Files

| File | What it contains | Size |
|---|---|---|
| `consolidated_table1.csv` | Table 1 of the paper as a single CSV (16 rows × 5 method columns). EV/EV+CoT are 3-seed means; Standard/CoT/ReAct are seed-0. | 1 KB |
| `bootstrap_cis.json` | Per-cell bootstrap 95% CIs (1,000 resamples by question index). Same content as Appendix J Table. | 24 KB |
| `all_results.json` | Per-cell metrics (accuracy_overall, by_template, by_family, n_total, regime) for all 144 (model, split, method-seed) combinations. | 425 KB |

## `all_results.json` structure

```python
import json
results = json.load(open('all_results.json'))

# results[model][split][method_seed]
results["Qwen3-8B"]["compositional"]["CoT"]["accuracy_overall"]
# → 0.8612

# Three EV seeds for 3-seed mean:
import statistics
ev_compositional_17 = [
    results["Qwen3-1.7B"]["compositional"][f"EV_seed{s}"]["accuracy_overall"]
    for s in [0, 1, 2]
]
statistics.mean(ev_compositional_17) * 100
# → 13.17 (matches Table 1)
```

### Index keys

- **Models**: `Qwen3-0.6B`, `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`
- **Splits**: `iid`, `compositional`, `cross_species`, `hn`
- **Method-seeds**:
  - `Standard` (seed 0)
  - `CoT` (seed 0)
  - `EV_seed0`, `EV_seed1`, `EV_seed2` (3 seeds for 3-seed mean)
  - `EV_CoT_seed0`, `EV_CoT_seed1`, `EV_CoT_seed2`
  - `ReAct` (seed 0)

### Per-cell fields

- `accuracy_overall`: float in [0, 1], multiply by 100 for percent
- `n_total`: number of evaluated questions in that cell
- `by_family`: dict mapping family letter (A to G) to per-family accuracy
- `by_template`: dict mapping template name (e.g., `A1`, `B2`) to accuracy
- `regime`: `L0` (Standard or CoT), `EV` (EV or EV+CoT), `L2` (ReAct)

## What's NOT included

Per-question prediction files (`per_question.jsonl`, ~1.5 GB) are
omitted because of size. They can be regenerated deterministically by
running the released baselines on the released splits.

## Verifying paper tables

- **Table 1**: read `consolidated_table1.csv` directly, or compute means
  from `all_results.json` for EV/EV+CoT.
- **Appendix B (per-family at 8B)**: `all_results.json["Qwen3-8B"]["iid"]["CoT"]["by_family"]`.
- **Appendix F (per-template counts)**: `["by_template"]` field.
- **Appendix J (bootstrap CIs)**: `bootstrap_cis.json`.
