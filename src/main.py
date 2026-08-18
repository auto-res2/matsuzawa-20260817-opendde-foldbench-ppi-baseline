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
import time
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


def run_label(args: argparse.Namespace) -> str:
    """A name for this run, unique per job unless one is given.

    Repeated runs of the same protocol are the whole point of running three
    times, and they only measure anything if their artefacts stay apart. One
    shared prediction tree would do worse than mix them up: the resume logic
    reads what is on disk to decide what still needs computing, so the second
    job would find the first job's structures, conclude the work was done, and
    exit having produced a run that is really a copy of another one.

    The default is the directory the repository was staged into, which the
    platform names after the job. With the `all` stage one job is one whole
    experiment, so that name identifies the experiment too, and three jobs
    dispatched at once separate themselves with nothing configured.

    That is the reason the stages are chained inside a job rather than split
    across three. A label carried between jobs would have to come from a
    platform environment variable, which is set per repository and holds one
    value at a time -- so the three experiments could not have run together.
    """
    if args.run_label:
        return args.run_label
    return Path(__file__).resolve().parent.parent.name


def run_dir(base: str, label: str) -> Path:
    return Path(base) / label


def inference_env(args: argparse.Namespace, extra: dict | None = None) -> dict:
    """The environment the sampler runs in, with the model's data root pinned.

    OpenDDE reads OPENDDE_ROOT_DIR from the environment at import time and hangs
    the checkpoint, the CCD component definitions and the template databases off
    it. Whatever arrives from outside is therefore discarded and replaced here:
    a variable registered on the execution platform had already replaced the
    image's own setting once, and pointed a run's weights at a personal
    directory. Reproducing an experiment means reading the inputs someone else
    can also read, so this value comes from a flag, which nothing can override.
    """
    import os as _os

    env = dict(_os.environ)
    env["OPENDDE_ROOT_DIR"] = str(args.opendde_root_dir)
    if extra:
        env.update(extra)
    return env


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


def residue_count(entry: dict) -> int:
    """Total residues an assembly asks the model to build.

    Peak memory is dominated by the token-pair representation, which grows with
    the square of the token count (OpenDDE report, Appendix C), so this is the
    quantity that decides whether a target fits one card. Counting residues
    rather than tokens is an approximation -- ligands and modified residues
    tokenize per atom -- but the protein-protein task has neither, so for this
    benchmark the two coincide.
    """
    total = 0
    for chain in entry.get("sequences", []):
        for value in chain.values():
            if isinstance(value, dict) and "sequence" in value:
                total += len(value["sequence"]) * int(value.get("count", 1))
    return total


def split_oversized(
    entries: list[dict], threshold: int, explicit: str | None
) -> tuple[list[dict], list[dict]]:
    """Partition into (fits one card, needs Fold-CP).

    The boundary is measured, not guessed. In the 239-assembly sweep the largest
    target that finished was 1,836 residues and the smallest that died of CUDA
    OOM was 1,872, with nothing in between -- so any threshold in that gap
    reproduces the observed split exactly. The default sits just under 1,872
    because the margin at 1,836 was thin: it fit 184 GB, but not by much, and
    allocator fragmentation could plausibly push it over on a later run.

    `explicit` overrides the size rule with a comma-separated list of names. That
    is how a target that OOMs anyway gets retried on the Fold-CP path without
    moving the threshold for everything else.
    """
    if explicit:
        wanted = {name.strip() for name in explicit.split(",") if name.strip()}
        unknown = wanted - {e["name"] for e in entries}
        if unknown:
            raise ValueError(f"not in the input set: {sorted(unknown)}")
        chosen = [e for e in entries if e["name"] in wanted]
        rest = [e for e in entries if e["name"] not in wanted]
        return rest, chosen

    rest, chosen = [], []
    for entry in entries:
        (chosen if residue_count(entry) >= threshold else rest).append(entry)
    return rest, chosen


def scratch_dir(args: argparse.Namespace) -> Path:
    """Where this run puts the input lists it derives for its own workers.

    Under the run's own output tree, never beside the inputs it read. These
    directories are named after a shard, not after a run, so three experiments
    running at once would write different target lists to the same path and read
    each other's -- and they would land in the shared input tree, where a
    reproduction's scratch has no business being.
    """
    return Path(args.prediction_dir) / "_worklists"


def write_input_dir(path: Path, entries: list[dict]) -> Path:
    """Materialise an input directory holding exactly `entries`.

    Both files are written because FoldBench's plugin contract passes
    alphafold3_inputs.json as its first argument; with inputs.json present the
    plugin keeps the latter (it carries the MSA paths) and never reads the
    former, but the argument still has to point at something.
    """
    path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, indent=2)
    (path / "inputs.json").write_text(payload)
    (path / "alphafold3_inputs.json").write_text(payload)
    return path


def visible_gpus() -> list[str]:
    """The GPU ids this process may use, in the launcher's order."""
    import os as _os

    raw = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids or ["0"]


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
        input_dir = write_input_dir(scratch_dir(args) / f"limit{args.limit}", subset)
        logger.info("pilot: %d of %d assemblies", len(subset), len(full))

    # Hold back the targets that cannot fit one card, before sharding rather than
    # after. Sharding round-robins to spread cost evenly; leaving targets in that
    # are certain to die of OOM would both waste the slot and skew that balance.
    # They are run separately by the predict-oversized stage, which gives each
    # one several GPUs.
    full = json.loads((input_dir / "inputs.json").read_text())
    fits, oversized = split_oversized(full, args.foldcp_threshold, args.foldcp_targets)
    if oversized:
        logger.info(
            "deferred to Fold-CP (>= %d residues): %s",
            args.foldcp_threshold,
            ", ".join(f"{e['name']}({residue_count(e)})" for e in oversized),
        )

    # Targets that already have their candidates are skipped, which makes this
    # stage resumable: dispatching it again after a partial run fills the holes
    # instead of recomputing eight hours of work that is already on disk. The
    # threshold above only decides what never enters the sweep; what has
    # actually been produced decides the rest.
    todo = incomplete_targets(fits, Path(args.prediction_dir))
    done = len(fits) - len(todo)
    if done:
        logger.info("%d of %d already complete; %d to run", done, len(fits), len(todo))
    if not todo:
        logger.info("sweep already complete")
        return 0
    input_dir = write_input_dir(scratch_dir(args) / "todo", todo)

    if count > 1:
        # Round-robin rather than contiguous blocks: target cost tracks assembly
        # size, and the targets CSV is not shuffled, so contiguous slices would
        # hand one worker a run of the largest complexes.
        full = json.loads((input_dir / "inputs.json").read_text())
        mine = full[index::count]
        input_dir = write_input_dir(
            scratch_dir(args) / f"shard{index}of{count}", mine
        )
        logger.info("shard %d/%d: %d of %d assemblies", index, count, len(mine), len(full))

    # Each shard writes its own prediction_reference.csv; the evaluate stage
    # concatenates them. One shared path would have the workers overwrite each
    # other's index. postprocess.py writes into this directory without creating
    # it, so it is created here.
    eval_dir = Path(args.evaluation_dir)
    if count > 1:
        eval_dir = eval_dir / f"shard{index}of{count}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Stamp who is writing here. Output directories carry no record of which run
    # produced them, and that cost us a wrong conclusion: a single-GPU probe was
    # still running when an 8-GPU sweep was launched, the probe recreated the
    # directories and kept writing, and its structures were read as proof that
    # the sweep worked. It had already failed. A marker makes the question
    # answerable from disk.
    import os as _os
    import subprocess as _sp

    def _git(*args: str) -> str | None:
        """The commit this code is, read from the checkout it is running from."""
        try:
            return _sp.run(["git", *args], cwd=Path(__file__).resolve().parent.parent,
                           capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance must not break the run
            return None

    (eval_dir / "run_marker.json").write_text(json.dumps({
        # Which code produced these structures. Without this the artefacts on
        # disk cannot be tied to a version: predictions made under one commit
        # sit in the same tree as predictions made under a later one, and
        # nothing on disk says which is which. `git_dirty` matters as much as
        # the hash -- a clean hash on a modified tree is a false claim.
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "slurm_job_id": _os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": _os.environ.get("SLURM_STEP_ID"),
        "hostname": _os.environ.get("SLURMD_NODENAME") or _os.uname().nodename,
        "rank": index,
        "world_size": count,
        "local_rank_env": _os.environ.get("LOCAL_RANK"),
        "cuda_visible_devices": _os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2))

    # Absolute, because cwd is the plugin directory itself: make_predictions.sh
    # invokes ./preprocess.py and ./postprocess.py relative to where it runs, so
    # the cwd has to stay there while the script path must not be resolved
    # against it a second time.
    algorithm_dir = Path(args.algorithm_dir).resolve()
    script = algorithm_dir / "make_predictions.sh"
    logger.info("OpenDDE data root: %s", args.opendde_root_dir)

    # One GPU per worker, chosen by the launcher. Asking for 16 GPUs starts 16
    # processes, each shown all four cards of its node and distinguished by
    # LOCAL_RANK, which is exactly what OpenDDE selects its device by. Splitting
    # the work again here would put four samplers on every card.
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
        env=inference_env(args),
    )
    return report_dropped_targets(
        Path(args.prediction_dir),
        input_dir,
        eval_dir / "dropped_targets.json",
    )


EXPECTED_CANDIDATES = 25  # 5 seeds x 5 samples, FoldBench's protocol


def incomplete_targets(entries: list[dict], prediction_dir: Path) -> list[dict]:
    """The entries that do not yet have their full set of candidates on disk.

    This is the residual, and it is read from the artefacts rather than
    predicted from a rule. A target is unfinished for reasons a size threshold
    cannot enumerate -- it ran out of memory, a rank died, a seed failed on its
    own, the job hit its wall clock -- and all of them look identical here,
    which is what makes re-running converge instead of guessing.

    The denominator of the published metric is fixed at the task's interfaces,
    so a run is not a smaller run when it loses targets; it is a run that cannot
    be compared with any other. Finishing this list is therefore not clean-up,
    it is the condition for the run existing at all.
    """
    return [e for e in entries
            if count_candidates(prediction_dir, e["name"]) != EXPECTED_CANDIDATES]


def count_candidates(prediction_dir: Path, name: str) -> int:
    """Sampled structures on disk for one target.

    `*_postprocessed.cif` is excluded deliberately. postprocess.py writes a
    converted copy beside every prediction, so counting all CIFs doubles the
    total -- an earlier version of this project read that doubled number as
    progress and believed a sweep was twice as far along as it was.
    """
    root = prediction_dir / name
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*.cif") if not p.name.endswith("_postprocessed.cif"))


def stage_predict_oversized(args: argparse.Namespace) -> int:
    """Run the targets that do not fit one card, one at a time, over several GPUs.

    Separate from `predict` because the two need different machines, not just
    different flags. The sharded sweep puts one worker on each GPU of a node;
    Fold-CP needs all of a node's GPUs for a single target, so the two cannot
    share a node. Running this as its own stage keeps that visible instead of
    letting the two contend for the same cards.

    Each target gets its own invocation with a one-entry input directory. That
    is not a convenience: OpenDDE's inference sampler shards by world_size, so a
    multi-entry input would hand each rank a different target and the Fold-CP
    collectives would then disagree about tensor shapes.
    """
    import os as _os

    input_dir = Path(args.input_dir)
    entries = json.loads((input_dir / "inputs.json").read_text())
    prediction_dir = Path(args.prediction_dir)

    if args.foldcp_targets:
        _, oversized = split_oversized(entries, args.foldcp_threshold, args.foldcp_targets)
    else:
        # Whatever is unfinished, not whatever is large. The size threshold is a
        # prediction about which targets will not fit; this is a measurement of
        # which ones did not get produced, and only the second one converges. A
        # target that the sweep lost to something other than memory is picked up
        # here too, and running it under Fold-CP is safe because Fold-CP does not
        # change inference semantics -- it is the same computation on more cards.
        oversized = incomplete_targets(entries, prediction_dir)

    # Smallest first: the cheapest target answers "does this path work at all"
    # for the least GPU time, and a memory ceiling then shows up as the largest
    # ones failing at the end rather than as the first failing and saying
    # nothing about the rest.
    oversized.sort(key=residue_count)
    if not oversized:
        logger.info("every target has its %d candidates; nothing to do", EXPECTED_CANDIDATES)
        return 0
    logger.info("%d of %d targets unfinished: %s", len(oversized), len(entries),
                ", ".join(f"{e['name']}({count_candidates(prediction_dir, e['name'])}/{EXPECTED_CANDIDATES})"
                          for e in oversized))

    # One target needs a whole node, so several nodes can take several targets at
    # once. The launcher's RANK/WORLD_SIZE identify this process among the
    # node-level workers; unset, shard_of yields (0, 1) and the list runs
    # serially, which is the right behaviour on a single node.
    index, count = shard_of(args.shard, args.num_shards)
    if count > 1:
        mine = oversized[index::count]
        logger.info("worker %d/%d takes %d of %d targets: %s",
                    index, count, len(mine), len(oversized),
                    ", ".join(e["name"] for e in mine))
        oversized = mine
        if not oversized:
            return 0

    algorithm_dir = Path(args.algorithm_dir).resolve()
    script = algorithm_dir / "make_predictions.sh"
    eval_root = Path(args.evaluation_dir)
    failures: list[str] = []

    for position, entry in enumerate(oversized, start=1):
        name = entry["name"]
        logger.info(
            "[%d/%d] %s (%d residues) with Fold-CP size_cp=%d",
            position, len(oversized), name, residue_count(entry), args.foldcp_size_cp,
        )
        one = write_input_dir(scratch_dir(args) / f"foldcp-{name}", [entry])
        eval_dir = eval_root / f"foldcp-{name}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        env = inference_env(args, {
            "FB_FOLDCP_MODE": "distributed",
            "FB_FOLDCP_SIZE_CP": str(args.foldcp_size_cp),
            # Per-module timing and peak memory, which only Fold-CP emits. Written
            # per target so a failure does not cost the measurements of the ones
            # that already ran.
            "FB_FOLDCP_METRICS": str(eval_dir / "foldcp_metrics.jsonl"),
        })
        try:
            run_cmd(
                [
                    "bash", str(script),
                    str(one / "alphafold3_inputs.json"),
                    str(one),
                    str(args.prediction_dir),
                    str(eval_dir),
                    str(args.gpu_id),
                ],
                cwd=algorithm_dir,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            # One target failing must not cost the rest of the list. Which ones
            # failed is recorded and re-raised at the end, so the stage still
            # exits non-zero and the caller cannot mistake a partial run for a
            # complete one.
            logger.error("%s failed (exit %s)", name, exc.returncode)
            failures.append(name)
            continue

        # Exiting zero is not the same as having produced a target. A rank that
        # dies partway, or a seed that runs out of memory on its own, can leave
        # some of the 25 candidates written and still return success -- and a
        # target scored on 15 candidates is not comparable with one scored on 25,
        # while nothing downstream would notice the difference.
        produced = count_candidates(Path(args.prediction_dir), name)
        if produced != EXPECTED_CANDIDATES:
            logger.error("%s produced %d candidates, expected %d",
                         name, produced, EXPECTED_CANDIDATES)
            failures.append(name)
        else:
            logger.info("%s: %d candidates", name, produced)

    # Per worker, because several of them write here at once and one shared path
    # would have them overwrite each other's record of what ran.
    summary = f"foldcp_summary{f'_shard{index}of{count}' if count > 1 else ''}.json"
    (eval_root / summary).write_text(json.dumps({
        "threshold_residues": args.foldcp_threshold,
        "size_cp": args.foldcp_size_cp,
        "expected_candidates": EXPECTED_CANDIDATES,
        "attempted": [e["name"] for e in oversized],
        "candidates": {e["name"]: count_candidates(Path(args.prediction_dir), e["name"])
                       for e in oversized},
        "failed": failures,
    }, indent=2))

    # The stage's verdict is the state of the whole task, not of this pass. A
    # pass that finished its own list while other targets are still short has
    # not produced a comparable run, and saying "0 failures" there would be the
    # difference between a run that can be quoted and one that cannot.
    remaining = incomplete_targets(entries, prediction_dir)
    if remaining:
        logger.error(
            "%d target(s) still short of %d candidates; dispatch this stage "
            "again to continue: %s",
            len(remaining), EXPECTED_CANDIDATES,
            ", ".join(f"{e['name']}({count_candidates(prediction_dir, e['name'])})"
                      for e in remaining),
        )
        if failures:
            logger.error("failed this pass: %s", ", ".join(failures))
        return 1

    logger.info("all %d targets complete with %d candidates each",
                len(entries), EXPECTED_CANDIDATES)
    return 0


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

    # Recorded, but NOT fatal here. Exiting non-zero from one shard makes srun
    # cancel the whole step, and that is what it did: a single shard's shortfall
    # killed fifteen healthy workers three minutes into a sweep. A shard is the
    # wrong place to decide the sweep is unusable -- it can only see its own
    # slice, and it cannot know whether another worker is still running.
    #
    # The refusal belongs in stage_evaluate, which sees every shard at once and
    # runs when nothing is left to lose. dropped_targets.json carries the names
    # there.
    return 0


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

    # The refusal the predict stage deliberately does not make. Here every
    # shard's record is present, so "which targets never got predicted" is
    # finally answerable -- and scoring an incomplete sweep would produce a
    # number that looks like a poor result rather than a partial one.
    missing: list[str] = []
    for record in sorted(Path(args.evaluation_dir).glob("shard*/dropped_targets.json")):
        data = json.loads(record.read_text())
        missing += data.get("quarantined_targets", []) + data.get("absent_targets", [])
    still_missing = sorted(set(missing) - set(
        pd.read_csv(Path(args.evaluation_dir) / "prediction_reference.csv").pdb_id.unique()
    ))
    if still_missing:
        logger.error("%d target(s) have no prediction and would silently lower "
                     "the score: %s", len(still_missing), still_missing)
        return 1
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


def stage_all(args: argparse.Namespace) -> int:
    """One experiment, start to finish, inside one job.

    Three experiments are three jobs rather than nine, and that is not only
    tidiness: the label that keeps their artefacts apart comes from the job, so
    stages split across jobs would need it configured, and the platform carries
    such settings per repository rather than per job -- which would have forced
    the three experiments to run one after another. As one job each they run at
    the same time, and the wall clock is one experiment rather than three.

    The residual pass runs even when the sweep reports failures, because that is
    exactly what it is for. Scoring runs only once per experiment, on the worker
    that owns the first shard, and only after every worker has finished: a score
    computed while structures are still being written would be a score over a
    partial run, which this project refuses to produce.
    """
    import os as _os

    index, count = shard_of(args.shard, args.num_shards)
    local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    node = _os.environ.get("SLURMD_NODENAME") or _os.uname().nodename
    markers = Path(args.evaluation_dir) / "_progress"

    def signal(name: str) -> None:
        markers.mkdir(parents=True, exist_ok=True)
        (markers / name).write_text("")

    def wait_for(names: list[str], what: str) -> None:
        logger.info("waiting for %s (%d)", what, len(names))
        while not all((markers / n).exists() for n in names):
            time.sleep(30)

    if (code := stage_predict(args)) != 0:
        logger.warning("sweep reported %s; continuing to the residual pass", code)
    signal(f"swept_{index}")

    # Fold-CP wants every card on a node for one target, so only one worker per
    # node may run it, and only once its node-mates have stopped sampling. The
    # others wait rather than exit: the job is not finished until the residual
    # is, and a worker that exited early would let the platform call the job
    # complete while targets were still short.
    wait_for([f"swept_{i}" for i in range(count)], "all workers to finish sampling")

    if local_rank == 0:
        # Only one worker per node reaches here, so the residual is divided
        # between nodes rather than between all workers. Sharding it over the
        # full worker count would hand slices to workers that never run it, and
        # those targets would simply never be sampled.
        per_node = max(1, len(visible_gpus()))
        args.shard, args.num_shards = index // per_node, max(1, count // per_node)
        logger.info("node %s: residual pass as node %d of %d, on %d GPUs",
                    node, args.shard, args.num_shards, per_node)
        if (code := stage_predict_oversized(args)) != 0:
            logger.error("targets are still unfinished after the residual pass")
            signal(f"filled_{index}")
            return code
    else:
        logger.info("node %s worker %d: the residual pass is local rank 0's",
                    node, index)
    signal(f"filled_{index}")

    wait_for([f"filled_{i}" for i in range(count)], "the residual pass")
    if index != 0:
        return 0

    # Scored once, by one worker, after every structure exists. Scoring while
    # sampling is still running would score a partial run.
    return stage_evaluate(args)


STAGES = {
    "prepare": stage_prepare,
    "predict": stage_predict,
    "predict-oversized": stage_predict_oversized,
    "evaluate": stage_evaluate,
    "all": stage_all,
}


def env_default(name: str, fallback: str | None = None, optional: bool = False):
    """Argparse default taken from the environment.

    The GPU stage is launched by the container's own CMD, which the execution
    platform runs verbatim, so per-run configuration cannot arrive as command
    line flags there -- it arrives as environment variables set on the image or
    the job. Every path therefore has an env fallback, and stays a flag for
    running the same code by hand.

    `optional` exists because "has no value" and "must be supplied" are not the
    same thing, and conflating them cost a 16-task run: --ost-bin-dir is set
    only in the scoring image, so treating its absence as an error made the
    predict stage refuse to start over a flag it never reads. A stage must not
    be required to supply what it does not use.
    """
    import os

    value = os.environ.get(name, fallback)
    return {"default": value, "required": value is None and not optional}


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
    # Residues at or above this go to the Fold-CP stage instead of the sweep. The
    # measured boundary is 1,836 (largest that fit) / 1,872 (smallest that OOMed);
    # see split_oversized for why the default sits below the observed failure.
    p.add_argument("--foldcp-threshold", type=int,
                   default=int(__import__("os").environ.get("FB_FOLDCP_THRESHOLD", "1850")))
    p.add_argument("--foldcp-size-cp", type=int,
                   default=int(__import__("os").environ.get("FB_FOLDCP_SIZE_CP", "4")))
    # Comma-separated assembly names, overriding the size rule. For retrying a
    # target that OOMed anyway without moving the threshold for everything else.
    p.add_argument("--foldcp-targets", **env_default("FB_FOLDCP_TARGETS", None, optional=True))
    # Names this run's artefacts. Defaults to the staging directory, which the
    # platform names per job, so three dispatched jobs separate themselves. Give
    # it explicitly to continue a run that was interrupted.
    p.add_argument("--run-label", **env_default("FB_RUN_LABEL", None, optional=True))
    p.add_argument("--python", default=sys.executable)
    # Where OpenDDE finds the checkpoint and the CCD data it builds features
    # from. Passed explicitly rather than left to OPENDDE_ROOT_DIR in the
    # environment: a registered platform variable overrode that and pointed a
    # run at a personal directory instead of the shared install. The bytes
    # happened to be identical, which is not the point -- an experiment whose
    # inputs live in one person's directory cannot be repeated by anyone else.
    p.add_argument("--opendde-root-dir", **env_default("FB_OPENDDE_ROOT_DIR"))
    p.add_argument("--opendde-cli", **env_default("OPENDDE_CLI", "opendde"))
    p.add_argument("--eval-python", **env_default("FB_EVAL_PYTHON", "python"))
    p.add_argument("--ost-bin-dir", **env_default("FB_OST_BIN_DIR", None, optional=True))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Resolved once, here, rather than at each of the places that write: every
    # stage of one run has to agree on where that run's artefacts are, and a
    # stage that disagreed would silently score one run's structures as another
    # run's. Inputs are deliberately not labelled -- all three runs read the same
    # inputs, which is what makes them comparable at all.
    label = run_label(args)
    args.prediction_dir = str(run_dir(args.prediction_dir, label))
    args.evaluation_dir = str(run_dir(args.evaluation_dir, label))
    logger.info("run %s", label)
    logger.info("  predictions -> %s", args.prediction_dir)
    logger.info("  evaluation  -> %s", args.evaluation_dir)

    raise SystemExit(STAGES[args.stage](args))


if __name__ == "__main__":
    main()
