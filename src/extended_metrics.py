"""Pull the metrics FoldBench computes but does not tabulate.

PROVENANCE: ours, and only in the sense that it reads. Every number here was
already computed -- by OpenStructure when it scored a candidate, or by OpenDDE
when it wrote that candidate's confidence file. Nothing is recomputed and no
structure is touched, so running this cannot change a score.

WHY IT EXISTS. FoldBench's CSV carries seven of the columns its own scorer
produces (dockq_score, irmsd, lrmsd, len_dockq, lddt, tm_score, gdt_ts, rmsd)
and drops the rest on the floor. Among the ones it drops are fnat and fnonnat --
the two halves DockQ is built from, and the pair that says whether a wrong
interface is wrong by omission or by invention -- along with the clash list and
the contact counts. The confidence files are not read at all beyond
ranking_score, so pLDDT, pTM and ipTM never reach a table either.

Read the same candidate FoldBench reads. `rank` selection means one candidate
per interface, the one the model scored highest, and the metrics below describe
that candidate and no other. Reporting a mean fnat over all 25 would describe
the sampler rather than the prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

INTERFACE_KEYS = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]

# Per interface: computed by ost on the pair being scored.
INTERFACE_METRICS = ["dockq_score", "fnat", "fnonnat", "interface_f1",
                     "irmsd", "lrmsd", "nnat", "nmdl", "interface_clashes"]
# Per assembly: properties of the whole prediction, not of one interface.
ASSEMBLY_METRICS = ["lddt", "tm_score", "oligo_gdtts", "oligo_gdtha", "rmsd",
                    "plddt", "ptm", "iptm", "disorder", "ranking_score",
                    "total_clashes"]


def interface_f1(fnat: float | None, fnonnat: float | None) -> float | None:
    """Harmonic mean of recall and precision over native contacts.

    fnat is recall: of the contacts the reference has, the fraction the model
    reproduced. 1 - fnonnat is precision: of the contacts the model made, the
    fraction that are real. A model can score well on either alone -- predict
    everything and recall is high, predict one correct contact and precision is
    -- so the pair is summarised the way precision and recall usually are.
    """
    if fnat is None or fnonnat is None:
        return None
    recall, precision = fnat, 1.0 - fnonnat
    if recall + precision <= 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def clash_counts(clashes: list | None, chain_1: str, chain_2: str) -> tuple[int, int]:
    """(clashes across this interface, clashes anywhere in the model).

    ost reports each clash as {'a1': 'A.47..O', 'a2': 'B.55..NH1', ...}, so the
    chain is the part before the first dot. A clash counts as the interface's
    when its two atoms sit on the two chains being scored; the same atoms inside
    one chain are a folding problem rather than a docking one, and are counted
    only in the total.
    """
    if not clashes:
        return 0, 0
    pair, total = 0, 0
    want = {chain_1, chain_2}
    for clash in clashes:
        total += 1
        try:
            c1 = str(clash["a1"]).split(".", 1)[0]
            c2 = str(clash["a2"]).split(".", 1)[0]
        except (KeyError, TypeError):
            continue
        if c1 != c2 and {c1, c2} == want:
            pair += 1
    return pair, total


def find_interface(data: dict, chain_1: str, chain_2: str) -> int | None:
    """Index of the scored interface matching a reference chain pair.

    Membership rather than comparison, because ost's dockq_interfaces entries
    name the model pair and the reference pair together, in an order that is not
    the target list's.

    The LAST match, not the first, because that is what FoldBench takes:

        for i, interface in enumerate(data['dockq_interfaces']):
            if native_chain_id_1 in interface and native_chain_id_2 in interface:
                dockq = data['dockq'][i]      # eval_by_ost.py:53-56, no break

    The loop has no break, so a later match overwrites an earlier one. On a
    homomer this is not hypothetical: 8qjk's reference chains are A, A-2, A-3
    and A-4, several of its four interfaces contain both members of a given
    pair, and first-match would report a different DockQ than the benchmark
    does. Matching the benchmark matters more here than picking the entry one
    might argue is the right one.
    """
    found = None
    for i, interface in enumerate(data.get("dockq_interfaces") or []):
        if chain_1 in interface and chain_2 in interface:
            found = i
    return found


def read_ost(detail: Path, pdb_id: str, seed, sample) -> dict | None:
    matches = sorted(detail.glob(f"{pdb_id}_{seed}_{sample}_*_ost.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_confidence(prediction_dir: Path, pdb_id: str, seed, sample) -> dict | None:
    path = (prediction_dir / pdb_id / f"seed_{seed}" / "predictions"
            / f"{pdb_id}_summary_confidence_sample_{sample}.json")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect(raw_csv: Path, detail: Path, prediction_dir: Path,
            metric_type: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    # FoldBench's own selection: null scores leave before the choice is made, so
    # the candidate picked here is the candidate its summary describes.
    scored = df[df["dockq_score"].notna()]
    column = "ranking_score" if metric_type == "rank" else "dockq_score"
    chosen = scored.loc[scored.groupby(INTERFACE_KEYS)[column].idxmax()]

    rows, missing = [], []
    for _, row in chosen.iterrows():
        pdb_id = row["pdb_id"]
        c1, c2 = row["interface_chain_id_1"], row["interface_chain_id_2"]
        seed, sample = row["seed"], row["sample"]

        rec = {"pdb_id": pdb_id, "interface_chain_id_1": c1,
               "interface_chain_id_2": c2, "seed": seed, "sample": sample,
               "dockq_score": row["dockq_score"]}

        ost = read_ost(detail, pdb_id, seed, sample)
        if ost is None:
            missing.append(f"{pdb_id} seed{seed} sample{sample} (ost)")
        else:
            idx = find_interface(ost, c1, c2)
            if idx is not None:
                for key in ("fnat", "fnonnat", "irmsd", "lrmsd", "nnat", "nmdl"):
                    values = ost.get(key)
                    rec[key] = values[idx] if values and idx < len(values) else None
            rec["interface_f1"] = interface_f1(rec.get("fnat"), rec.get("fnonnat"))
            rec["interface_clashes"], rec["total_clashes"] = clash_counts(
                ost.get("model_clashes"), c1, c2)
            for key in ("lddt", "tm_score", "oligo_gdtts", "oligo_gdtha", "rmsd"):
                rec[key] = ost.get(key)
            rec["ost_version"] = ost.get("ost_version")

        conf = read_confidence(prediction_dir, pdb_id, seed, sample)
        if conf is None:
            missing.append(f"{pdb_id} seed{seed} sample{sample} (confidence)")
        else:
            for key in ("plddt", "ptm", "iptm", "disorder", "ranking_score",
                        "has_clash", "num_recycles"):
                rec[key] = conf.get(key)
        rows.append(rec)

    if missing:
        # Loudly: a metric absent from a third of the interfaces would still
        # produce a plausible mean, and the mean is what gets quoted.
        logger.warning("%d candidate(s) had no file to read: %s",
                       len(missing), missing[:5])
    return pd.DataFrame(rows)


def describe(frame: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        out[column] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True, action="append",
                   help="a run's evaluation tree (holds raw/ and detail/); repeatable")
    p.add_argument("--prediction-dir", required=True, action="append",
                   help="the matching prediction tree; repeatable, same order")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--metric-type", default="rank", choices=["rank", "best"])
    p.add_argument("--target-type", default="interface_protein_protein")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if not (len(args.evaluation_dir) == len(args.prediction_dir) == len(args.labels)):
        raise SystemExit("--evaluation-dir, --prediction-dir and --labels must "
                         "have the same number of values, in the same order")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, tables = {}, []

    for label, eval_dir, pred_dir in zip(args.labels, args.evaluation_dir,
                                         args.prediction_dir):
        eval_dir, pred_dir = Path(eval_dir), Path(pred_dir)
        raw_csv = eval_dir / "raw" / f"{args.target_type}_ost.csv"
        frame = collect(raw_csv, eval_dir / "detail", pred_dir, args.metric_type)
        logger.info("%s: %d interfaces, %d assemblies, ost %s",
                    label, len(frame), frame["pdb_id"].nunique(),
                    frame.get("ost_version", pd.Series(["?"])).dropna().iloc[0]
                    if "ost_version" in frame else "?")
        report[label] = {
            "n_interfaces": int(len(frame)),
            "n_assemblies": int(frame["pdb_id"].nunique()),
            # Per interface and per assembly are reported apart because their
            # denominators differ: a confidence score belongs to a prediction,
            # and several interfaces can share one.
            "per_interface": describe(frame, INTERFACE_METRICS),
            "per_assembly": describe(
                frame.drop_duplicates(subset=["pdb_id", "seed", "sample"]),
                ASSEMBLY_METRICS),
        }
        table = frame.copy()
        table.insert(0, "run", label)
        tables.append(table)

    combined = pd.concat(tables, ignore_index=True)
    csv_path = out_dir / f"extended_{args.metric_type}.csv"
    combined.to_csv(csv_path, index=False)
    json_path = out_dir / f"extended_{args.metric_type}.json"
    json_path.write_text(json.dumps({"metric_type": args.metric_type,
                                     "per_run": report}, indent=2))
    logger.info("wrote %s and %s", csv_path, json_path)


if __name__ == "__main__":
    main()
