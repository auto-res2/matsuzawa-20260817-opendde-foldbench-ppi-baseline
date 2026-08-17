"""Reproduce OpenDDE's published FoldBench protein-protein number.

This is an orchestrator, not an implementation. Everything that decides a score
belongs to somebody else and is called, not reimplemented:

  * sampling            -> `opendde pred` (OpenDDE's own CLI, its own defaults)
  * MSA search          -> `runner/msa_search.py` (OpenDDE's own client)
  * input conversion    -> FoldBench's Protenix plugin `preprocess.py`, verbatim
  * output conversion   -> the same plugin's `postprocess.py`, two path templates changed
  * scoring             -> FoldBench's `evaluate.py` (OpenStructure) and
                           `task_score_summary.py`, both unmodified

What is left here is target selection and stage sequencing.

Stages exist because the work has three different homes. `prepare` needs the
network (the ColabFold MMseqs2 service) and so runs on the login node;
`predict` needs a GPU and runs offline against the MSAs `prepare` cached;
`evaluate` needs only CPU and the OpenStructure binary.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ALGORITHM = "OpenDDE"
# The task under study. FoldBench's evaluate.py defaults to a target list that
# does not include protein-protein, so it is always passed explicitly.
TARGET_TYPE = "interface_protein_protein"


def resolve_targets(targets_dir: Path, target_type: str, limit: int) -> list[str]:
    """Distinct assembly ids of one FoldBench task, in file order."""
    df = pd.read_csv(targets_dir / f"{target_type}.csv")
    ids = list(dict.fromkeys(df.pdb_id.tolist()))
    return ids[:limit] if limit > 0 else ids


def build_af3_inputs(
    target_ids: list[str], input_json_dir: Path, out_path: Path
) -> list[str]:
    """Collect FoldBench's per-entry AF3 JSONs into the one list run.sh expects.

    FoldBench ships `input_json/<pdb>.json` as a dict whose `name` is the bare
    uppercase PDB code, while every downstream join -- the targets CSV, the
    prediction_reference.csv, the score tables -- keys on the assembly id
    (`8tuz-assembly1`). The rename below is what makes those two agree; getting
    it wrong produces an empty merge and a silent zero, so unresolved ids are
    raised rather than skipped.
    """
    entries: list[dict] = []
    missing: list[str] = []
    for pdb_id in target_ids:
        stem = pdb_id.split("-assembly")[0].lower()
        src = input_json_dir / f"{stem}.json"
        if not src.exists():
            missing.append(pdb_id)
            continue
        data = json.loads(src.read_text())
        if isinstance(data, list):
            if len(data) != 1:
                raise ValueError(f"{src} holds {len(data)} entries, expected 1")
            data = data[0]
        data["name"] = pdb_id
        entries.append(data)

    if missing:
        raise FileNotFoundError(
            f"no input JSON for {len(missing)} target(s): {missing[:5]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2))
    logger.info("wrote %d AF3 inputs to %s", len(entries), out_path)
    return [e["name"] for e in entries]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def stage_prepare(args: argparse.Namespace) -> int:
    """Select targets, build the AF3 input list, and cache every MSA."""
    targets = resolve_targets(Path(args.targets_dir), args.target_type, args.limit)
    logger.info("%d %s assemblies selected", len(targets), args.target_type)

    input_dir = Path(args.input_dir)
    af3_json = input_dir / "alphafold3_inputs.json"
    build_af3_inputs(targets, Path(args.input_json_dir), af3_json)

    # Upstream conversion, then upstream MSA search. Caching here is the whole
    # point of a separate stage: the GPU stage then needs no network at all, and
    # a re-run of predict does not re-query the MSA service.
    run_cmd(
        [
            args.python,
            str(Path(args.algorithm_dir) / "preprocess.py"),
            f"--af3_input_json={af3_json}",
            f"--input_dir={input_dir}",
        ]
    )
    run_cmd(
        [
            args.python,
            str(Path(__file__).with_name("msa_prefetch.py")),
            "--input",
            str(input_dir / "inputs.json"),
            "--out-dir",
            str(Path(args.msa_dir)),
            "--opendde-src",
            str(Path(args.opendde_src)),
        ]
    )
    print(json.dumps({"stage": "prepare", "targets": len(targets)}), flush=True)
    return 0


def stage_predict(args: argparse.Namespace) -> int:
    """Sample with OpenDDE and convert the output for the evaluators."""
    script = Path(args.algorithm_dir) / "make_predictions.sh"
    run_cmd(
        [
            "bash",
            str(script),
            str(Path(args.input_dir) / "alphafold3_inputs.json"),
            str(args.input_dir),
            str(args.prediction_dir),
            str(args.evaluation_dir),
            str(args.gpu_id),
        ],
        cwd=Path(args.algorithm_dir),
    )
    return 0


def stage_evaluate(args: argparse.Namespace) -> int:
    """Score with FoldBench's own evaluator and summary table."""
    foldbench = Path(args.foldbench_dir)
    # evaluate.py appends the algorithm name to --evaluation_dir itself, so it
    # is handed the parent of the directory postprocess.py wrote into.
    run_cmd(
        [
            args.eval_python,
            "evaluate.py",
            "--targets_dir",
            str(Path(args.targets_dir)),
            "--evaluation_dir",
            str(Path(args.evaluation_dir).parent),
            "--algorithm_name",
            ALGORITHM,
            "--ground_truth_dir",
            str(Path(args.ground_truth_dir)),
            "--targets",
            args.target_type,
        ],
        cwd=foldbench,
    )
    run_cmd(
        [args.eval_python, "task_score_summary.py", "--algorithm_names", ALGORITHM],
        cwd=foldbench,
    )
    return 0


STAGES = {
    "prepare": stage_prepare,
    "predict": stage_predict,
    "evaluate": stage_evaluate,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--foldbench-dir", required=True)
    p.add_argument("--targets-dir", required=True)
    p.add_argument("--input-json-dir", required=True)
    p.add_argument("--ground-truth-dir", required=True)
    p.add_argument("--opendde-src", required=True)
    p.add_argument("--algorithm-dir", default="algorithms/OpenDDE")
    p.add_argument("--input-dir", default="outputs/input/OpenDDE")
    p.add_argument("--prediction-dir", default="outputs/prediction/OpenDDE")
    p.add_argument("--evaluation-dir", default="outputs/evaluation/OpenDDE")
    p.add_argument("--msa-dir", default="outputs/msa")
    p.add_argument("--target-type", default=TARGET_TYPE)
    # 0 means the whole task; the pilot uses a small positive number to measure
    # cost before committing to all 239 assemblies.
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--eval-python", default="python")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(STAGES[args.stage](args))


if __name__ == "__main__":
    main()
