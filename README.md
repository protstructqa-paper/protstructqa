# ProtStructQA

Executable benchmark for protein structural question answering. Each
natural-language question is generated from a hidden typed DSL program;
the gold answer is obtained by executing that program on an
AlphaFold-predicted structure.

The release contains 382,200 questions over 10,000 proteins from four
species (human, mouse, fly, chicken), spanning seven question families
(confidence, distance, PAE, solvent exposure, secondary structure,
topology, compositional) and a 52.2K hard-negative robustness pool.

## Structure

```
benchmark/splits/   gzipped JSONL splits
benchmark/results/  per-cell metrics, bootstrap CIs, Table 1 CSV
benchmark/*.py      data-pipeline scripts
baselines/          Standard, CoT, EV, EV+CoT, ReAct
dsl/                grammar, executor, ProteinView
scripts/            statistical analysis
```

## Splits

| File | Questions | Role |
|---|---|---|
| `train.jsonl.gz` | 96,000 | Prompt-example pool (A–F) |
| `dev.jsonl.gz` | 12,000 | Prompt dev / sanity (A–F) |
| `test_iid.jsonl.gz` | 12,000 | Parameter-OOD eval (A–F) |
| `test_compositional.jsonl.gz` | 30,000 | Held-out compositional (G) |
| `test_compositional_eval.jsonl.gz` | 6,000 | Paper subsample (G) |
| `test_cross_species.jsonl.gz` | 180,000 | Cross-proteome shift (A–F) |
| `test_cross_species_eval.jsonl.gz` | 10,000 | Paper subsample |
| `test_hn.jsonl.gz` | 52,200 | HN robustness pool |
| `test_hn_eval.jsonl.gz` | 4,357 | Paper HN subsample |

## Quickstart

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open('benchmark/splits/test_iid.jsonl.gz', 'rt')]
```

Each row contains: `qid`, `uniprot`, `species`, `family`, `template`,
`question`, `program`, `answer`, `answer_type`, `params`,
`paraphrase_id`.

## License

Code: MIT (see `LICENSE`). Dataset (`benchmark/splits/*`): CC BY 4.0
(see `DATA_LICENSE`), matching upstream UniProt and AFDB licenses.
