# OpenDDE × FoldBench protein-protein — baseline reproduction

Reproduce one published number, with nothing of our own in the measurement path:

> **OpenDDE reaches 0.769 on FoldBench protein-protein.**
> — *Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine*
> (Aureka AI Research, [arXiv:2607.03787](https://arxiv.org/abs/2607.03787)),
> Appendix B.1 / Figure 8.

No training happens in this repository. The fine-tuned arms come later; this is
the ruler they will be measured against, so it has to be somebody else's ruler.

## What is reproduced, and from where

| Piece | Source | Modified? |
|---|---|---|
| Model weights | `opendde.pt`, sha256 `7b8266…07cc` | no — hash matches the published manifest |
| Sampling | `opendde pred` (OpenDDE CLI) | no |
| MSA search | `runner/msa_search.py` (OpenDDE) | no |
| Benchmark targets, ground truths, AF3 inputs | FoldBench | no |
| AF3-JSON → model input | FoldBench `algorithms/Protenix/preprocess.py` | **no — vendored verbatim** |
| Model output → mmCIF + `prediction_reference.csv` | same plugin's `postprocess.py` | two output-path templates only |
| Scoring (OpenStructure / DockQ) | FoldBench `evaluate.py` | no |
| Score tables | FoldBench `task_score_summary.py` | no |
| Target selection + stage sequencing | `src/main.py` | ours |

Upstream copies live in `vendor/foldbench-protenix-plugin/`, so any drift from
them is a two-file diff away from being visible.

## Protocol

Taken from FoldBench's own reference plugin, not chosen by us. It coincides with
OpenDDE's documented defaults (`docs/supported_models.md`).

| | |
|---|---|
| Recycles (`N_cycle`) | 10 |
| Diffusion steps (`N_step`) | 200 |
| Samples per seed | 5 |
| Seeds | 42, 66, 101, 2024, 8888 |
| Candidates per target | 25 |
| MSA / templates | **on** |
| Targets | 279 interfaces across 239 assemblies |
| Metric | DockQ success rate (> 0.23); lDDT reported alongside |

MSA and templates being on is the whole point. An earlier run of ours with them
off scored 10.5% on the antibody–antigen task against a published 70.0%, and
OpenDDE itself warns at runtime that turning MSA off "might degrade performance
significantly".

## Stages

Three stages because the work has three homes.

```bash
# login node — needs the network (ColabFold MMseqs2 service), caches every MSA
python -m src.main --stage prepare  ...

# GPU node — runs offline against the cached MSAs
python -m src.main --stage predict  ...

# CPU — OpenStructure
python -m src.main --stage evaluate ...
```

Paths for the RIKYU deployment are in `config/run/baseline-pretrained.yaml`.
Use `--limit` to run the pilot subset before committing to all 239 assemblies:
the full sweep is roughly 150× the per-target cost of our earlier low-budget
run, so cost is measured before it is spent.

## Recorded alongside the headline number

- **oracle success rate** — best of the 25 candidates by ground-truth DockQ.
  Free to compute here, and it is what sizes any future work on ranking.
- **seed-to-seed variance** — the minimum effect a later comparison could
  detect. Without it no arm can be called better than another.
- **dropped targets** — counted and named, never silently absorbed.
