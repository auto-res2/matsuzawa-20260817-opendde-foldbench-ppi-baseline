# vendor/ — upstream files, untouched

Nothing in this directory is ours and nothing in it is executed. These are
pristine copies of upstream code, kept so that any claim this repository makes
about "we changed only X" can be checked with `diff` rather than believed.

## foldbench-protenix-plugin/

FoldBench's own reference plugin for Protenix, copied from a FoldBench checkout:

- Upstream: <https://github.com/BEAM-Labs/FoldBench> — `algorithms/Protenix/`
- Paper: *Benchmarking all-atom biomolecular structure prediction with
  FoldBench*, Nature Communications 2025,
  [doi:10.1038/s41467-025-67127-3](https://doi.org/10.1038/s41467-025-67127-3)
- Copied from the RIKYU checkout at
  `/data1/rkp00041/plinder_lddt_pli_improvement/data/foldbench/FoldBench`

Our plugin under `algorithms/OpenDDE/` is derived from these four files. To see
exactly what we changed:

```bash
# identical -- OpenDDE and Protenix take the same input schema
diff vendor/foldbench-protenix-plugin/preprocess.py  algorithms/OpenDDE/preprocess.py

# two output-path templates, because OpenDDE's dumper omits the seed from
# artefact names; plus a provenance header
diff vendor/foldbench-protenix-plugin/postprocess.py algorithms/OpenDDE/postprocess.py

# ours, written to the same five-argument contract
diff vendor/foldbench-protenix-plugin/make_predictions.sh algorithms/OpenDDE/make_predictions.sh
```

`container.def` is kept for reference only; this repository builds its
environment from `docker/Dockerfile.predict` instead, because runs are launched
through a Slurm cluster rather than FoldBench's own `run.sh`.
