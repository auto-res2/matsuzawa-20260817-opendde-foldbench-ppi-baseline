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
| FoldBench targets | 279 interfaces / 239 assemblies | Released dataset, and **verified to be the same set the report scored** — see below |
| Ground truth | `ground_truth_20250520` | Named by date |
| OpenStructure | **2.11.1** | **See the mismatch below** |
| Sampling | N_cycle 10, N_step 200, 5 samples, seeds 42/66/101/2024/8888 | Not our choice: FoldBench defines 5 seeds × 5 samples × 10 recycles, and N_step 200 is OpenDDE's own CLI default — see below |
| Evaluation Python | scipy/networkx/pandas/numpy/biopython 1.85/tqdm/parallelbar | Not version-locked — see below |
| **MSAs** | 323 chains, depth 2 / 1209 / 21549 | **Cannot be re-derived.** Pinned by manifest only |

## What was checked against the sources, rather than assumed

Everything in this section was read out of a published source; none of it is
inference from our own run. It is recorded because the alternative — assuming
the benchmark we run is the benchmark the report scored — is exactly the
assumption that makes a reproduction meaningless when it turns out to be wrong.

**The evaluation set is the same one.** The OpenDDE report's Figure 8 is
labelled `n=279` for protein-protein, and its AlphaFold 3 entry reads 0.729,
matching FoldBench's own published AlphaFold 3 number for that task. The report
did not re-cut the set; it dropped its model into FoldBench's harness and
carried FoldBench's baseline column across.

**The selection rule is ranking-based, not oracle.** Figure 8 reports 0.700 for
antigen-antibody, which is the report's own *ranked* FoldBench-AB figure; its
oracle figure for the same task is 0.819. So `summary_rank.csv` is the table to
compare against 0.769, and `summary_best.csv` is not.

**FoldBench defines the set precisely**, and we apply no filter of our own:
PDB entries from 2023-01-13 (AlphaFold 3's validation cutoff) to 2024-11-01,
non-NMR, resolution < 4.5 Å, and — for protein-protein specifically — only
complexes with TM-score < 0.5 against the training set, which is what cut an
initial 501 targets down to 279. Antibody-antigen is a separate task and is not
included. The 279 split into 195 homomeric and 84 heteromeric interfaces.
`resolve_targets()` reads `interface_protein_protein.csv` and de-duplicates
`pdb_id`; every filter above was already applied when the set was released.

**The sampling budget is the benchmark's, not ours.** FoldBench states that
predictions for every model were generated with a 5×5 strategy (5 seeds × 5
samples) and 10 recycles. `N_step = 200` is not stated there, but OpenDDE's
`docs/supported_models.md` lists `N_cycle = 10` and `N_step = 200` as its
recommended inference defaults *and* as the current `opendde pred` CLI defaults
for `opendde_v1`. Departing from these would have taken a deliberate act.

**There is no train/test leakage.** OpenDDE's training data is filtered at a
cutoff of 2021-09 (report §2.2); FoldBench collects entries from 2023-01-13
onward. The two do not overlap. FoldBench's low-homology filter is moreover
defined against AlphaFold 3's training set, whose cutoff is 16 months *later*
than OpenDDE's, so the filter is if anything stricter than OpenDDE requires.

**What is still inferred, not stated.** The report never says in words that it
followed FoldBench's harness and default sampling for Figure 8. That it used
FoldBench's baseline numbers, and that every sampling parameter above is a
documented default, is the whole of the evidence. It is strong, but it is
evidence rather than a statement.

## The decision taken

Both deviations below are **accepted deliberately**, not overlooked. RIKYU has
OpenStructure 2.11.1 and no 2.8.0, and building one was judged not worth doing
before knowing whether it matters.

The test is the reproduction itself: **if the number lands near the published
0.769, the deviations did not matter enough to chase; if it lands clearly
below, they become the first place to look** — together with the MSAs. That is
a real decision rule and it is written here so the eventual number is read
against the rule that was set before it was known, rather than one invented
afterwards.

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
noting for context: OpenDDE's *report* records no inference-time MSA procedure,
and while the project's `docs/msa_template_pipeline.md` documents the procedure
we followed, a procedure that queries a live service does not pin a result. So
adopting B would leave this reproduction more reproducible than the result it
reproduces.
