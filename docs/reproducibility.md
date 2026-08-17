# What is pinned, what is not, and what that costs

Recorded because the honest answer differs by claim. Reusing this baseline to
say *"method X moved the score by Y"* needs one thing; letting a stranger
re-derive the number from nothing needs another, and only one of those is
currently true.

## Versions

| Component | Version | How it is fixed |
|---|---|---|
| Model weights | `opendde.pt`, sha256 `7b8266…07cc` | Matches OpenDDE's published manifest; HF revision `eddd563ce96571f784012edd8f045181c8f8627d` |
| OpenDDE | 1.0.3 | The shared RIKYU install. Not a git checkout, so the version string is the identifier |
| FoldBench code | — | Not a git checkout either; the four plugin files we build on are vendored in `vendor/` |
| FoldBench targets | 279 interfaces / 239 assemblies | Released dataset |
| Ground truth | `ground_truth_20250520` | Named by date |
| OpenStructure | **2.11.1** | **See the mismatch below** |
| Sampling | N_cycle 10, N_step 200, 5 samples, seeds 42/66/101/2024/8888 | `algorithms/OpenDDE/make_predictions.sh` |
| Evaluation Python | scipy/networkx/pandas/numpy/biopython 1.85/tqdm/parallelbar | Not version-locked — see below |
| **MSAs** | 323 chains, depth 2 / 1209 / 21549 | **Cannot be re-derived.** Pinned by manifest only |

## Two known deviations

**OpenStructure 2.11.1, where FoldBench declares 2.8.0.** FoldBench's
`environment.yml` pins `aivant::openstructure=2.8.0`; the binary installed on
RIKYU is 2.11.1. DockQ and lDDT are computed by that binary, and their
implementations are free to change across three minor versions. If our
reproduction lands away from the published 0.769, this is a prime suspect
alongside the MSAs. It is recorded rather than silently accepted because a
number is only comparable to the extent its evaluator is.

**The evaluation environment is not version-locked.** FoldBench pins
`python=3.9`; `docs/evaluation-env.md` explains why this repository builds on
3.12 with current resolutions instead, and that this is a real difference.

## The MSA problem

Every other input above is a file with a checksum or a released artefact. The
MSAs are not. They came from a shared public service searching databases we do
not control, on a particular day. Re-running `prepare` next month returns
different alignments, and the score moves for reasons that have nothing to do
with the model.

So the cache is a dependency this reproduction cannot itself reproduce. Running
it again does not fix that. Three responses exist, and they buy different things:

### A. Pin the artefact — what is implemented

`.research/msa_manifest.json` records, for all 323 chains, the sha256, byte
size and sequence depth of both a3m files. Written by `src/msa_manifest.py`,
which also verifies:

```bash
python -m src.msa_manifest verify \
    --input   .../outputs/input/OpenDDE/inputs.json \
    --manifest .research/msa_manifest.json
```

This makes three things true that were not before:

- a later arm can *prove* it read the same MSAs the baseline did, which is what
  makes "fine-tuning changed the score by X" a statement about fine-tuning;
- a reader can check whether their cache matches ours before wondering why
  their number differs;
- losing or overwriting the cache becomes a detectable event.

It does **not** make the MSAs re-derivable. The a3m files themselves are 1.4 GB
and live at `/data1/rkp00041/rku00122/foldbench-ppi-baseline/outputs/msa`.

### B. Pin the databases — not done

Point `MMSEQS_SERVICE_HOST_URL` at a self-hosted MMseqs2 with a downloaded,
versioned ColabFold database (~1.5 TB; RIKYU has 9.7 PB free). MSAs then become
re-derivable from a pinned corpus, and the external dependency disappears along
with the rate limiting that cost this project a day.

### C. Accept and document

What the field usually does, and what OpenDDE itself did.

## Which is enough

| Claim | Needs |
|---|---|
| "Method X moved the score by Y" | **A**, and A is also *necessary* — every arm must read identical MSAs or the delta measures the weather |
| "Anyone can reproduce 0.769 from scratch" | **B** |

The claim this project exists to support is the first. The second is worth
noting for context: OpenDDE's report documents no inference-time MSA procedure
at all, so adopting B would leave this reproduction more reproducible than the
result it reproduces.
