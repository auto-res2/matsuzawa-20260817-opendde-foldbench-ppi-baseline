"""Score the selected candidates again for the metrics FoldBench does not ask for.

PROVENANCE: ours, but it computes nothing itself. OpenStructure does the work,
invoked exactly as FoldBench invokes it plus the flags FoldBench leaves off.

WHY A SECOND PASS. FoldBench's command is

    ost compare-structures -m .. -r .. --fault-tolerant
        --min-pep-length 4 --min-nuc-length 4
        --lddt --rigid-scores --tm-score --dockq

so `--ilddt` -- lDDT computed only over inter-chain contacts, which is the
interface quality measure that does not depend on superposition -- is never
requested and never lands in the CSV. Asking for it means running ost again.

WHY ONLY THE SELECTED CANDIDATES. The table this feeds describes the prediction
the benchmark reports, which is one candidate per interface. Re-scoring all 25
would cost twenty-five times as much to produce numbers describing the sampler
rather than the result. One candidate per assembly is 239 comparisons instead
of 6975.

The scorer is the one FoldBench pins, taken from the same directory the scoring
stage uses, so these numbers and the DockQ they sit beside come from one binary.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INTERFACE_KEYS = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]


def selected_candidates(raw_csv: Path, metric_type: str) -> pd.DataFrame:
    """One (assembly, seed, sample) per assembly, chosen as FoldBench chooses."""
    df = pd.read_csv(raw_csv)
    df = df[df["dockq_score"].notna()]
    column = "ranking_score" if metric_type == "rank" else "dockq_score"
    per_interface = df.loc[df.groupby(INTERFACE_KEYS)[column].idxmax()]
    # An assembly can own several interfaces, and under `rank` they all select
    # the same candidate because ranking_score is a property of the prediction.
    # Under `best` they need not, so the pick is made per assembly to keep one
    # comparison per structure either way, and the assembly's own best is used.
    return (per_interface.sort_values(column, ascending=False)
            .drop_duplicates(subset=["pdb_id"])[["pdb_id", "seed", "sample",
                                                 "prediction_path"]])


def score_one(ost: Path, model: Path, reference: Path, out: Path) -> dict | None:
    cmd = [
        str(ost), "compare-structures",
        "-m", str(model), "-r", str(reference), "-o", str(out),
        "--fault-tolerant", "--min-pep-length", "4", "--min-nuc-length", "4",
        "--lddt", "--ilddt", "--tm-score",
    ]
    subprocess.run(cmd, capture_output=True, check=False)
    if not out.is_file():
        return None
    try:
        return json.loads(out.read_text())
    except json.JSONDecodeError:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--ground-truth-dir", required=True)
    p.add_argument("--ost-bin-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--metric-type", default="rank", choices=["rank", "best"])
    p.add_argument("--target-type", default="interface_protein_protein")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ost = shutil.which("ost", path=args.ost_bin_dir)
    if ost is None:
        raise SystemExit(f"no `ost` in {args.ost_bin_dir}")
    logger.info("scoring with %s", ost)

    eval_dir = Path(args.evaluation_dir)
    raw_csv = eval_dir / "raw" / f"{args.target_type}_ost.csv"
    picks = selected_candidates(raw_csv, args.metric_type)
    logger.info("%s: %d assemblies to re-score", args.label, len(picks))

    work = Path(args.out_dir) / f"ilddt_{args.label}"
    work.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []

    for n, (_, pick) in enumerate(picks.iterrows(), start=1):
        pdb_id, seed, sample = pick["pdb_id"], pick["seed"], pick["sample"]
        model = (Path(args.prediction_dir) / pdb_id / f"seed_{seed}" / "predictions"
                 / f"{pdb_id}_sample_{sample}_postprocessed.cif")
        reference = Path(args.ground_truth_dir) / f"{pdb_id}.cif"
        if not model.is_file() or not reference.is_file():
            failed.append(f"{pdb_id} (missing file)")
            continue
        data = score_one(Path(ost), model, reference, work / f"{pdb_id}.json")
        if data is None or data.get("status") != "SUCCESS":
            failed.append(f"{pdb_id} ({(data or {}).get('exception', 'no output')})")
            continue
        rows.append({"run": args.label, "pdb_id": pdb_id, "seed": seed,
                     "sample": sample, "ilddt": data.get("ilddt"),
                     "lddt": data.get("lddt"), "tm_score": data.get("tm_score")})
        if n % 40 == 0:
            logger.info("  %d/%d", n, len(picks))

    frame = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / f"ilddt_{args.metric_type}_{args.label}.csv"
    frame.to_csv(out_csv, index=False)
    if failed:
        # Named, not counted: an interface metric missing from a tenth of the
        # set still averages to something plausible.
        logger.warning("%d assembly/assemblies did not score: %s",
                       len(failed), failed[:5])
    if not frame.empty:
        values = pd.to_numeric(frame["ilddt"], errors="coerce").dropna()
        logger.info("%s: ilddt n=%d mean %.4f median %.4f",
                    args.label, len(values), values.mean(), values.median())
    logger.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()
