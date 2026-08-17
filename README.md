# OpenDDE × FoldBench protein-protein — baseline reproduction

Reproduce one published number, with nothing of our own in the measurement path:

> **OpenDDE reaches 0.769 on FoldBench protein-protein.**
> — *Folding, Reasoning, and Scaling with Open-source Drug Discovery Engine*
> (Aureka AI Research, [arXiv:2607.03787](https://arxiv.org/abs/2607.03787)),
> Appendix B.1 / Figure 8.

No training happens in this repository. The fine-tuned arms come later; this is
the ruler they will be measured against, so it has to be somebody else's ruler.

## What is reproduced, and from where

Upstream projects, cited once:

- **OpenDDE** — <https://github.com/aurekaresearch/OpenDDE> ·
  [arXiv:2607.03787](https://arxiv.org/abs/2607.03787) ·
  [weights](https://huggingface.co/aurekaresearch/OpenDDE)
- **FoldBench** — <https://github.com/BEAM-Labs/FoldBench> ·
  [doi:10.1038/s41467-025-67127-3](https://doi.org/10.1038/s41467-025-67127-3)

| Piece | Source | Ours? |
|---|---|---|
| Model weights | `opendde.pt`, sha256 `7b8266…07cc` | no — hash matches OpenDDE's published manifest |
| Sampling | `opendde pred` — [OpenDDE CLI](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md) | no |
| MSA search | [`runner/msa_search.py`](https://github.com/aurekaresearch/OpenDDE/blob/main/runner/msa_search.py) | no — but see MSA note below |
| Benchmark targets, ground truths, AF3 inputs | [FoldBench](https://github.com/BEAM-Labs/FoldBench) | no |
| AF3-JSON → model input | FoldBench `algorithms/Protenix/preprocess.py` | no — **vendored verbatim** |
| Model output → mmCIF + `prediction_reference.csv` | the same plugin's `postprocess.py` | two output-path templates only |
| Scoring (OpenStructure / DockQv2) | FoldBench `evaluate.py` | no |
| Score tables | FoldBench `task_score_summary.py` | no |
| Plugin entry point | `algorithms/OpenDDE/make_predictions.sh` | **ours**, to FoldBench's five-argument contract |
| Target selection, stage sequencing, sharding, drop-detectors | `src/main.py` | **ours** |
| MSA pacing / depth verification / re-fetch | `src/msa_prefetch.py` | **ours**, around OpenDDE's own search |
| Environment | `docker/Dockerfile.predict` | **ours** |

Every source file carries a `PROVENANCE:` header saying which of these it is.
Pristine upstream copies live in `vendor/` (see `vendor/README.md`), so "we
changed only X" is a `diff` away from being checked rather than believed.

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

## The MSA protocol, in full

The OpenDDE report does not say how it built MSAs for its FoldBench evaluation —
its only mentions of MSA search concern *training* data. So this is our protocol,
recorded because the number depends on it. It is OpenDDE's own client at its own
settings ([`msa_service_client.py`](https://github.com/aurekaresearch/OpenDDE/blob/main/opendde/data/msa/msa_service_client.py)):

| | Unpaired (every chain) | Paired (multi-chain only) |
|---|---|---|
| Endpoint | `api.colabfold.com/ticket/msa` | `.../ticket/pair` |
| `use_env` | `true` | `false` |
| `use_filter` | `true` | — |
| Pairing strategy | — | `greedy` |
| Derived mode | `env` | `pairgreedy` |

Two things are **not** ours to pin, and are reproducibility caveats:

- **Which databases `env` maps to** is decided by the ColabFold server, not by
  OpenDDE. We observed `uniref` and `bfd.mgnify30.metaeuk30.smag30` in the
  returned artefacts; a server-side change would change our MSAs.
- **Returned depth** is likewise server-side. It matters less than it looks:
  the model reads at most `msa_depth = 1280` sequences
  ([`opendde/config/data.py`](https://github.com/aurekaresearch/OpenDDE/blob/main/opendde/config/data.py)),
  so the real distinction is *an MSA* versus *no MSA*.

**Why depth is verified.** OpenDDE's search catches every exception and falls
back to a query-only MSA "so inference can still run". Batching all 239 targets
at the shared public service — which its
[docs](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/msa_template_pipeline.md)
explicitly warn against — made that fallback fire for 111 of them. Nothing
downstream could tell: the JSON had a path, the file existed, and the run
proceeded in effectively single-sequence mode. `src/msa_prefetch.py` therefore
counts what came back and fails rather than passing an empty MSA on.

**Templates are off**, which is OpenDDE's shipped default
(`use_template: False`). They would also be impossible here: no chain carries a
`templatesPath`, and template search needs HMMER plus a
`pdb_seqres_2022_09_28.fasta` database that is not installed.

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

Paths for the RIKYU deployment live in `docker/Dockerfile.predict` as `FB_*`
environment variables, and nowhere else. There is deliberately no config file:
the execution platform runs a committed image's `CMD` verbatim and ignores
per-run flags, so a YAML would not be read — and an unread config drifts. One
did, here: it claimed `use_template: true` for a run that had templates off.
Only what executes is allowed to describe the run.

`FB_LIMIT` runs a pilot subset before committing to all 239 assemblies.

## Recorded alongside the headline number

- **oracle success rate** — best of the 25 candidates by ground-truth DockQ.
  Free to compute here, and it is what sizes any future work on ranking.
- **seed-to-seed variance** — the minimum effect a later comparison could
  detect. Without it no arm can be called better than another.
- **dropped targets** — counted and named, never silently absorbed.
