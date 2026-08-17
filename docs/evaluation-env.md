# The evaluation environment

Scoring needs a Python that is not OpenDDE's. This records why, and how to
rebuild it.

## Why a separate environment

FoldBench's `evaluate.py` opens with

```python
from evaluation import eval_by_dockqv2, eval_by_ost
```

and `eval_by_dockqv2` imports `parallelbar` at module scope. So a
protein–protein run, which only ever calls the OpenStructure path, still fails
at import unless the DockQv2 dependencies are present.

Installing them into OpenDDE's virtualenv would work and is the wrong move: this
repository reproduces a published number, and the environment the model runs in
should stay exactly as OpenDDE shipped it. The evaluator gets its own.

## What it contains

FoldBench declares its evaluation environment in `environment.yml`:

```yaml
name: foldbench
dependencies:
  - python=3.9.*
  - aivant::openstructure=2.8.0
  - pip: [scipy, networkx, pandas, numpy, biopython==1.85, tqdm, parallelbar]
```

Only the pip half is rebuilt here. OpenStructure is already installed on RIKYU
at `/data1/rkp00041/.local/bin/ost` and is invoked as an external command, so
there is nothing to duplicate:

```bash
python3 -m venv /data1/rkp00041/rku00122/foldbench-ppi-baseline/evalenv
/data1/rkp00041/rku00122/foldbench-ppi-baseline/evalenv/bin/pip install \
    scipy networkx pandas numpy biopython==1.85 tqdm parallelbar
```

## Deviation from upstream, stated

FoldBench pins `python=3.9`; this venv is built on the system Python (3.12), and
pip resolves current versions of the rest. That is a real difference from the
environment FoldBench specifies, and it is recorded here rather than hidden —
if a score ever looks wrong, this is one of the places to look. The pinned
`biopython==1.85` is honoured because FoldBench pins it.

The scoring itself does not run in this environment: `ost compare-structures`
is a separate binary. What Python does here is drive it and parse its JSON.

## How the stage finds all this

Two environment variables, both set in `docker/Dockerfile.evaluate`:

| | |
|---|---|
| `FB_EVAL_PYTHON` | the interpreter above |
| `FB_OST_BIN_DIR` | prepended to `PATH` so `ost` resolves |

`src/main.py` refuses to start the stage if `ost` is not resolvable, because
FoldBench runs it through bash with `check=False` — a missing binary would
otherwise leave every target unscored and read as a poor benchmark result.
