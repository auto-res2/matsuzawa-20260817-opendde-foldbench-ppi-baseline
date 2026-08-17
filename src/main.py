"""Reproduce OpenDDE's published FoldBench protein-protein number.

PROVENANCE: this file is OURS. It is an orchestrator, not an implementation --
everything that decides a score belongs to somebody else and is called here
rather than reimplemented.

Upstream projects:
  OpenDDE    https://github.com/aurekaresearch/OpenDDE   (arXiv:2607.03787)
  FoldBench  https://github.com/BEAM-Labs/FoldBench      (Nat Commun,
             doi:10.1038/s41467-025-67127-3)

What each step defers to:
  * sampling          -> `opendde pred`, OpenDDE's own CLI at its own defaults
  * MSA search        -> `opendde msa`, OpenDDE's own CLI, one job at a time,
                         driven by src/msa_prefetch.py (which only paces and
                         verifies -- it imports nothing from OpenDDE)
  * input conversion  -> FoldBench's Protenix plugin preprocess.py, VERBATIM
                         (algorithms/OpenDDE/preprocess.py)
  * output conversion -> the same plugin's postprocess.py, two path templates
                         changed (algorithms/OpenDDE/postprocess.py)
  * scoring           -> FoldBench's own evaluate.py (OpenStructure/DockQv2)
                         and task_score_summary.py, both unmodified and invoked
                         as subprocesses in stage_evaluate

OURS, and therefore the part to distrust first: target selection, the AF3-JSON
name rewrite, stage sequencing, sharding, and the two drop-detectors
(report_dropped_targets, merge_shard_references).

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


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    logger.info("$ %s", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


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
            "--work-dir",
            str(input_dir / "msa_jobs"),
            "--opendde-cli",
            args.opendde_cli,
        ]
    )
    print(json.dumps({"stage": "prepare", "targets": len(targets)}), flush=True)
    return 0


def shard_of(index: int, count: int) -> tuple[int, int]:
    """This worker's (index, count), preferring explicit args over the launcher.

    Seyval exports RANK/WORLD_SIZE when it starts one process per GPU, which is
    how a sweep this size gets done at all: 239 assemblies at 25 candidates each
    do not fit one GPU inside the cluster's four-day wall clock.
    """
    import os

    if count is None:
        count = int(os.environ.get("WORLD_SIZE", "1"))
    if index is None:
        index = int(os.environ.get("RANK", "0"))
    if not 0 <= index < count:
        raise ValueError(f"shard {index} outside 0..{count - 1}")
    return index, count


def stage_predict(args: argparse.Namespace) -> int:
    """Sample with OpenDDE and convert the output for the evaluators."""
    index, count = shard_of(args.shard, args.num_shards)
    input_dir = Path(args.input_dir)

    if args.limit > 0:
        # Pilot: a prefix of the prepared inputs, so cost is measured before the
        # full sweep is committed to. Applied before sharding so a one-worker
        # pilot sees exactly --limit assemblies.
        full = json.loads((input_dir / "inputs.json").read_text())
        subset = full[: args.limit]
        input_dir = input_dir.parent / f"{input_dir.name}-limit{args.limit}"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "inputs.json").write_text(json.dumps(subset, indent=2))
        (input_dir / "alphafold3_inputs.json").write_text(json.dumps(subset, indent=2))
        logger.info("pilot: %d of %d assemblies", len(subset), len(full))

    if count > 1:
        # Round-robin rather than contiguous blocks: target cost tracks assembly
        # size, and the targets CSV is not shuffled, so contiguous slices would
        # hand one worker a run of the largest complexes.
        full = json.loads((input_dir / "inputs.json").read_text())
        mine = full[index::count]
        input_dir = input_dir.parent / f"{input_dir.name}-shard{index}of{count}"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "inputs.json").write_text(json.dumps(mine, indent=2))
        # Carried over only so the plugin's five-argument contract still holds;
        # with inputs.json present the plugin will not read it.
        (input_dir / "alphafold3_inputs.json").write_text(json.dumps(mine, indent=2))
        logger.info("shard %d/%d: %d of %d assemblies", index, count, len(mine), len(full))

    # Each shard writes its own prediction_reference.csv; the evaluate stage
    # concatenates them. One shared path would have the workers overwrite each
    # other's index. postprocess.py writes into this directory without creating
    # it, so it is created here.
    eval_dir = Path(args.evaluation_dir)
    if count > 1:
        eval_dir = eval_dir / f"shard{index}of{count}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Absolute, because cwd is the plugin directory itself: make_predictions.sh
    # invokes ./preprocess.py and ./postprocess.py relative to where it runs, so
    # the cwd has to stay there while the script path must not be resolved
    # against it a second time.
    algorithm_dir = Path(args.algorithm_dir).resolve()
    script = algorithm_dir / "make_predictions.sh"
    run_cmd(
        [
            "bash",
            str(script),
            str(input_dir / "alphafold3_inputs.json"),
            str(input_dir),
            str(args.prediction_dir),
            str(eval_dir),
            str(args.gpu_id),
        ],
        cwd=algorithm_dir,
    )
    return report_dropped_targets(
        Path(args.prediction_dir),
        input_dir,
        eval_dir / "dropped_targets.json",
    )


def report_dropped_targets(prediction_dir: Path, input_dir: Path, record_to: Path) -> int:
    """Fail if any target went missing, and write down WHICH ones.

    `opendde pred` moves a target it could not finish into <dump_dir>/ERR
    (runner/inference.py) and carries on. Those targets then have no prediction,
    FoldBench left-joins them away, and the success rate comes out lower with
    nothing saying why.

    Counts alone are not enough. An earlier version of this project reported
    only n_attempted / n_scored, so a run that lost targets could say how many
    but never which -- and once the job log aged out, the identity of the
    casualties was gone for good. There was no way to check whether the losses
    were random or concentrated in, say, the largest complexes, which is the
    difference between noise and bias. So the names go to a file that lives
    beside the results, not to a log line.
    """
    err_dir = prediction_dir / "ERR"
    quarantined = sorted(p.name for p in err_dir.iterdir()) if err_dir.is_dir() else []
    requested = [e["name"] for e in json.loads((input_dir / "inputs.json").read_text())]
    produced = {p.name for p in prediction_dir.iterdir() if p.is_dir() and p.name != "ERR"}
    absent = sorted(set(requested) - produced - set(quarantined))

    record = {
        "requested": len(requested),
        "produced": len(produced),
        "quarantined_targets": quarantined,
        "absent_targets": absent,
        "requested_targets": requested,
    }
    record_to.parent.mkdir(parents=True, exist_ok=True)
    record_to.write_text(json.dumps(record, indent=2))
    logger.info("targets: %d requested, %d produced, %d quarantined, %d absent (recorded in %s)",
                len(requested), len(produced), len(quarantined), len(absent), record_to)

    if quarantined:
        logger.error("OpenDDE quarantined %d target(s): %s", len(quarantined), quarantined)
    if absent:
        logger.error("%d target(s) produced no output at all: %s", len(absent), absent)
    return 1 if (quarantined or absent) else 0


def merge_shard_references(evaluation_dir: Path) -> int:
    """Concatenate the per-shard prediction indexes into the one FoldBench reads.

    Returns the number of prediction rows. A shard that produced no CSV at all
    is reported rather than skipped: FoldBench merges this file against the
    targets list with a left join, so a missing shard does not fail loudly, it
    just quietly lowers the success rate.
    """
    shards = sorted(evaluation_dir.glob("shard*/prediction_reference.csv"))
    if not shards:
        return 0
    expected = int(shards[0].parent.name.split("of")[-1])
    if len(shards) != expected:
        found = {p.parent.name for p in shards}
        raise RuntimeError(
            f"{len(shards)} of {expected} shard indexes present; missing "
            f"{sorted({f'shard{i}of{expected}' for i in range(expected)} - found)}"
        )
    merged = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    out = evaluation_dir / "prediction_reference.csv"
    merged.to_csv(out, index=False)

    # Which targets actually reached scoring, by name. FoldBench merges this
    # file against the targets list with a left join, so a target missing here
    # is not an error downstream -- it simply never appears in the score, and
    # counts alone would never reveal which one it was.
    scored = sorted(merged.pdb_id.unique().tolist())
    (evaluation_dir / "scored_targets.json").write_text(
        json.dumps({"n_rows": len(merged), "n_targets": len(scored), "targets": scored}, indent=2)
    )
    logger.info("merged %d shard indexes into %s (%d rows over %d targets)",
                len(shards), out, len(merged), len(scored))
    return len(merged)


def require_ost(extra_bin_dir: str | None) -> dict:
    """Return an environment where `ost` resolves, or refuse to score.

    FoldBench builds its command as the bare string "ost compare-structures ..."
    and runs it through bash with `check=False`, so a missing binary is not an
    error there -- the command fails, no JSON is written, and the target simply
    has no score. That reads downstream as a low benchmark result rather than as
    a broken evaluator, which is the one failure this repository keeps having to
    guard against.
    """
    import os
    import shutil

    env = dict(os.environ)
    if extra_bin_dir:
        env["PATH"] = f"{extra_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    resolved = shutil.which("ost", path=env.get("PATH"))
    if resolved is None:
        raise FileNotFoundError(
            "`ost` (OpenStructure) is not on PATH, and FoldBench's evaluator "
            "invokes it by bare name without checking the exit code -- scoring "
            "would silently produce nothing. Point --ost-bin-dir / FB_OST_BIN_DIR "
            "at the directory holding it."
        )
    logger.info("scoring with %s", resolved)
    return env


def stage_evaluate(args: argparse.Namespace) -> int:
    """Score with FoldBench's own evaluator and summary table."""
    foldbench = Path(args.foldbench_dir)
    env = require_ost(args.ost_bin_dir)
    merge_shard_references(Path(args.evaluation_dir))
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
        env=env,
    )
    # task_score_summary.py carries its own defaults -- ./examples paths and a
    # target list without protein-protein -- so every argument is passed, or it
    # silently reads somebody else's directory.
    #
    # Run twice. `rank` selects the top-1 candidate by the model's own ranking
    # score, which is the number OpenDDE published and the one to compare. `best`
    # selects by ground-truth DockQ, which is the oracle: the ceiling reachable
    # by fixing ranking alone, given these same 25 candidates. The gap between
    # them prices the ranking axis, and FoldBench computes both for free.
    eval_root = Path(args.evaluation_dir).parent
    for metric_type in ("rank", "best"):
        run_cmd(
            [
                args.eval_python,
                "task_score_summary.py",
                "--evaluation_dir", str(eval_root),
                "--target_dir", str(Path(args.targets_dir)),
                "--output_path", str(eval_root / f"summary_{metric_type}.csv"),
                "--algorithm_names", ALGORITHM,
                "--targets", args.target_type,
                "--metric_type", metric_type,
            ],
            cwd=foldbench,
            env=env,
        )
        logger.info("wrote %s", eval_root / f"summary_{metric_type}.csv")
    return 0


STAGES = {
    "prepare": stage_prepare,
    "predict": stage_predict,
    "evaluate": stage_evaluate,
}


def env_default(name: str, fallback: str | None = None):
    """Argparse default taken from the environment.

    The GPU stage is launched by the container's own CMD, which the execution
    platform runs verbatim, so per-run configuration cannot arrive as command
    line flags there -- it arrives as environment variables set on the image or
    the job. Every path therefore has an env fallback, and stays a flag for
    running the same code by hand.
    """
    import os

    value = os.environ.get(name, fallback)
    return {"default": value, "required": value is None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=sorted(STAGES), **env_default("FB_STAGE"))
    p.add_argument("--foldbench-dir", **env_default("FB_FOLDBENCH_DIR"))
    p.add_argument("--targets-dir", **env_default("FB_TARGETS_DIR"))
    p.add_argument("--input-json-dir", **env_default("FB_INPUT_JSON_DIR"))
    p.add_argument("--ground-truth-dir", **env_default("FB_GROUND_TRUTH_DIR"))
    p.add_argument("--opendde-src", **env_default("FB_OPENDDE_SRC"))
    p.add_argument("--algorithm-dir", **env_default("FB_ALGORITHM_DIR", "algorithms/OpenDDE"))
    p.add_argument("--input-dir", **env_default("FB_INPUT_DIR", "outputs/input/OpenDDE"))
    p.add_argument("--prediction-dir", **env_default("FB_PREDICTION_DIR", "outputs/prediction/OpenDDE"))
    p.add_argument("--evaluation-dir", **env_default("FB_EVALUATION_DIR", "outputs/evaluation/OpenDDE"))
    p.add_argument("--msa-dir", **env_default("FB_MSA_DIR", "outputs/msa"))
    p.add_argument("--target-type", default=TARGET_TYPE)
    # 0 means the whole task; the pilot uses a small positive number to measure
    # cost before committing to all 239 assemblies.
    p.add_argument("--limit", type=int, default=int(__import__("os").environ.get("FB_LIMIT", "0")))
    p.add_argument("--gpu-id", default="0")
    # Default to the launcher's RANK/WORLD_SIZE; set explicitly to shard by hand.
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--opendde-cli", **env_default("OPENDDE_CLI", "opendde"))
    p.add_argument("--eval-python", **env_default("FB_EVAL_PYTHON", "python"))
    p.add_argument("--ost-bin-dir", **env_default("FB_OST_BIN_DIR", None))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(STAGES[args.stage](args))


if __name__ == "__main__":
    main()
