"""Turn FoldBench's per-interface scores into the numbers a paper can quote.

PROVENANCE: this file is OURS, but it decides nothing about scoring. It reads
the per-interface table FoldBench's evaluate.py writes
(<evaluation_dir>/raw/interface_protein_protein_ost.csv) and reproduces
FoldBench's own candidate selection before summarising -- rank picks the
candidate with the highest ranking_score per interface, best picks the highest
DockQ, which is task_score_summary.get_best_rows verbatim in behaviour.

Nothing here re-implements statistics either. Intervals come from scipy:
binomtest(...).proportion_ci for the binomial ones and stats.bootstrap for the
clustered one. A hand-rolled Wilson interval agrees with scipy to four decimals,
which is exactly why it is not worth carrying.

What this adds over summary_rank.csv, which reports one number:

  * success rates at every DockQ threshold, not only 0.23. FoldBench's
    acceptable / medium / high cutoffs are 0.23 / 0.49 / 0.80, and a method can
    move one without moving another.
  * an interval on each rate. 279 interfaces put roughly +-0.05 around a rate
    near 0.77 whatever the method does, and a reproduction that does not say so
    invites its own noise to be read as a result.
  * a second interval that respects clustering. The 279 interfaces sit in 239
    assemblies and interfaces of one assembly come from one prediction, so they
    are not independent draws and the binomial interval is optimistic.
  * medians and quartiles. Success rate is a threshold count and hides where the
    distribution actually sits.
The denominator is fixed, and this file refuses to move it. A run that scored
fewer than the task's interfaces is not summarised at all -- not summarised with
a caveat, not summarised over what survived. Two runs whose rates are computed
over different denominators cannot be compared, and the difference between them
would read as method noise while actually being coverage. Completing the missing
targets is the fix; there is no summary to write until that has happened.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# FoldBench's CAPRI-derived cutoffs. 0.23 is the one the published number uses;
# the other two are reported so a change in quality is visible even when the
# headline rate does not move.
THRESHOLDS = {"acceptable": 0.23, "medium": 0.49, "high": 0.80}
INTERFACE_KEYS = ["pdb_id", "interface_chain_id_1", "interface_chain_id_2"]


def select_candidates(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """One row per interface, chosen the way FoldBench chooses.

    `rank` takes the candidate the model itself scored highest, which is the
    setting the published 0.769 was measured in. `best` takes the candidate with
    the highest DockQ, an oracle that cannot be achieved at inference and is
    reported only as the ceiling of what the sampling produced.

    Idempotent: applying it to a table that is already one row per interface
    returns that table.
    """
    df = df[df["dockq_score"].notna()]
    column = "ranking_score" if metric_type == "rank" else "dockq_score"
    return df.loc[df.groupby(INTERFACE_KEYS)[column].idxmax()]


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% binomial interval, from scipy rather than from arithmetic here."""
    if total == 0:
        return (float("nan"), float("nan"))
    ci = stats.binomtest(int(successes), int(total)).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    return (float(ci.low), float(ci.high))


def clopper_pearson(successes: int, total: int) -> tuple[float, float]:
    """The conservative interval, reported alongside so the choice is visible."""
    if total == 0:
        return (float("nan"), float("nan"))
    ci = stats.binomtest(int(successes), int(total)).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return (float(ci.low), float(ci.high))


def cluster_bootstrap(rows: pd.DataFrame, threshold: float) -> tuple[float, float]:
    """95% interval that resamples assemblies rather than interfaces.

    Two interfaces of one assembly are scored from the same predicted structure
    built on the same MSA, so they succeed and fail together far more often than
    two interfaces drawn at random. Treating all 279 as independent therefore
    claims more precision than the data supports. Resampling the 239 assemblies
    -- carrying whichever interfaces each one owns -- keeps that dependence.

    Returns NaN when every assembly agrees, which is not a failure: the
    statistic has no variance to estimate and scipy would raise instead of
    saying so.
    """
    hit = (rows["dockq_score"] >= threshold).astype(float)
    per_assembly = [g.to_numpy() for _, g in hit.groupby(rows["pdb_id"])]
    if len(per_assembly) < 2:
        return (float("nan"), float("nan"))

    index = np.arange(len(per_assembly))

    def rate(sample_index: np.ndarray) -> float:
        # scipy hands back a 2-D array of indices when vectorized; flatten one
        # resample at a time so the statistic stays a plain scalar.
        picked = np.concatenate([per_assembly[i] for i in sample_index.astype(int)])
        return float(picked.mean())

    try:
        result = stats.bootstrap(
            (index,), rate, confidence_level=0.95,
            n_resamples=10000, method="percentile", vectorized=False,
        )
    except Exception:  # noqa: BLE001 - a degenerate sample is a result, not a crash
        return (float("nan"), float("nan"))
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def summarise(rows: pd.DataFrame) -> dict:
    """Every number for one run under one selection rule."""
    scored = len(rows)
    out: dict = {
        "n_interfaces_scored": scored,
        "n_assemblies_scored": int(rows["pdb_id"].nunique()),
        "rates": {},
        "distribution": {},
    }

    for name, threshold in THRESHOLDS.items():
        hits = int((rows["dockq_score"] >= threshold).sum())
        out["rates"][name] = {
            "threshold": threshold,
            "successes": hits,
            "denominator": scored,
            "rate": hits / scored if scored else float("nan"),
            "ci_wilson": wilson(hits, scored),
            "ci_clopper_pearson": clopper_pearson(hits, scored),
            "ci_cluster_bootstrap": cluster_bootstrap(rows, threshold),
        }

    for column in ("dockq_score", "lddt", "irmsd", "lrmsd"):
        if column not in rows:
            continue
        values = rows[column].dropna()
        if values.empty:
            continue
        out["distribution"][column] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def cumulative_curve(rows: pd.DataFrame, steps: int = 101) -> list[dict]:
    """Fraction of interfaces at or above each DockQ cutoff, for the figure."""
    values = rows["dockq_score"].dropna().to_numpy()
    return [
        {"cutoff": float(c), "fraction": float((values >= c).mean())}
        for c in np.linspace(0.0, 1.0, steps)
    ]


def across_runs(per_run: dict[str, dict], tidy: pd.DataFrame) -> dict:
    """How much the result moves between runs of the same protocol.

    This is the only quantity repeated runs can answer. The intervals above do
    not shrink by running again -- their width comes from the benchmark holding
    279 interfaces -- so what several runs buy is a measurement of the noise
    floor, which is what a later arm's difference has to clear to mean anything.

    Two views, because the aggregate one can hide everything. The rates can
    agree to three decimals while different interfaces succeed each time: a
    sibling experiment saw two checkpoints post an identical 10.53% on disjoint
    sets of interfaces. So the per-interface view is reported alongside, and it
    is the one that says whether the pipeline is deterministic.
    """
    out: dict = {"aggregate": {}, "per_interface": {}}

    for name in THRESHOLDS:
        rates = [r["rates"][name]["rate"] for r in per_run.values() if name in r["rates"]]
        if not rates:
            continue
        out["aggregate"][name] = {
            "runs": len(rates),
            "rates": rates,
            "mean": float(np.mean(rates)),
            "median": float(np.median(rates)),
            # Sample sd: population sd would understate the spread of a handful.
            "stdev": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "range": float(max(rates) - min(rates)),
        }

    n_runs = tidy["run"].nunique()
    if n_runs < 2:
        return out

    wide = tidy.pivot_table(index=INTERFACE_KEYS, columns="run", values="dockq_score")
    wide = wide.dropna()
    for name, threshold in THRESHOLDS.items():
        hit = wide >= threshold
        always = int((hit.sum(axis=1) == n_runs).sum())
        never = int((hit.sum(axis=1) == 0).sum())
        out["per_interface"][name] = {
            "n_interfaces": int(len(wide)),
            "always_success": always,
            "always_failure": never,
            # The ones that changed verdict between runs. Zero here means the
            # pipeline reproduces itself exactly and any later difference is the
            # method; anything else is the floor a claim has to clear.
            "flipped": int(len(wide) - always - never),
        }

    spread = wide.max(axis=1) - wide.min(axis=1)
    out["per_interface"]["dockq_spread"] = {
        "n_interfaces": int(len(wide)),
        "identical": int((spread == 0).sum()),
        "median": float(spread.median()),
        "q3": float(spread.quantile(0.75)),
        "max": float(spread.max()),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", nargs="+", required=True,
                   help="interface_protein_protein_ost.csv, one per run")
    p.add_argument("--labels", nargs="*", default=None,
                   help="names for the runs, in the same order as --raw")
    p.add_argument("--metric-type", choices=("rank", "best"), default="rank",
                   help="rank reproduces the published setting; best is the oracle ceiling")
    p.add_argument("--expected-interfaces", type=int, default=279,
                   help="interfaces the task defines. A run that scored a "
                        "different number is refused, not summarised")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="summarise anyway. For inspecting a run in progress; "
                        "never for a number that will be compared with another")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    labels = args.labels or [Path(r).parent.parent.name or f"run{i}"
                             for i, r in enumerate(args.raw)]
    if len(labels) != len(args.raw):
        raise SystemExit(f"{len(labels)} labels for {len(args.raw)} inputs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run, per_interface, short = {}, [], []
    for label, raw in zip(labels, args.raw):
        raw_df = pd.read_csv(raw)
        reached = raw_df.groupby(INTERFACE_KEYS).ngroups
        rows = select_candidates(raw_df, args.metric_type)
        logger.info("%s: %d of %d interfaces reached scoring, %d scorable, "
                    "%d assemblies", label, reached, args.expected_interfaces,
                    len(rows), rows["pdb_id"].nunique())

        # Two different questions, and only the first is ours to answer.
        #
        # Whether all 279 interfaces reached the scorer is this pipeline's
        # responsibility: a target lost to a crashed shard or an index that was
        # never merged is our defect, and a rate computed without it is the
        # wrong number quietly. That is what is checked here.
        #
        # Whether a scored interface yields a DockQ is FoldBench's business.
        # task_score_summary.py drops rows the metric is null for before it
        # divides (`df = df[df[metric].notna()]`, task_score_summary.py:89), so
        # an interface the model failed to form -- no contacts, nothing for
        # DockQ to be defined on -- leaves the denominator. select_candidates
        # applies the same filter, so the rate here matches the benchmark's own
        # rather than a variant of it. The count that leaves is reported, not
        # decided.
        if reached != args.expected_interfaces:
            short.append(f"{label}: {reached} of {args.expected_interfaces} "
                         f"interfaces reached scoring")
        if (unscorable := reached - len(rows)):
            logger.warning(
                "%s: %d of %d interfaces have no DockQ and so leave the "
                "denominator, which is what FoldBench does with them",
                label, unscorable, reached)
        per_run[label] = summarise(rows)
        per_run[label]["n_interfaces_reached_scoring"] = int(reached)
        per_run[label]["n_interfaces_unscorable"] = int(unscorable)
        per_run[label]["cumulative"] = cumulative_curve(rows)

        keep = [c for c in INTERFACE_KEYS + ["dockq_score", "lddt", "irmsd", "lrmsd",
                                             "ranking_score", "seed", "sample"]
                if c in rows.columns]
        tidy = rows[keep].copy()
        tidy.insert(0, "run", label)
        per_interface.append(tidy)

    if short and not args.allow_incomplete:
        # Refusing rather than reporting a caveat. A rate over a smaller
        # denominator is not a worse version of the number, it is a different
        # number, and the targets that go missing are the ones that were hardest
        # to compute -- so the incomplete run reads as the better one. Finish
        # them and come back.
        raise SystemExit(
            "interfaces are missing from the scorer's input, so nothing was "
            "summarised:\n  "
            + "\n  ".join(short)
            + "\nThis is about targets that never reached scoring, not about "
              "targets FoldBench could not score. Complete them, or pass "
              "--allow-incomplete to inspect a run that will not be compared "
              "with anything."
        )

    tidy_all = pd.concat(per_interface)
    report = {
        "metric_type": args.metric_type,
        "thresholds": THRESHOLDS,
        "expected_interfaces": args.expected_interfaces,
        "complete": not short,
        "per_run": per_run,
        "across_runs": across_runs(per_run, tidy_all),
    }
    (out_dir / f"summary_{args.metric_type}.json").write_text(json.dumps(report, indent=2))
    tidy_all.to_csv(out_dir / f"per_interface_{args.metric_type}.csv", index=False)

    for label, r in per_run.items():
        a = r["rates"]["acceptable"]
        lo, hi = a["ci_wilson"]
        clo, chi = a["ci_cluster_bootstrap"]
        logger.info(
            "%s  DockQ>=0.23  %.3f (%d/%d)  Wilson [%.3f, %.3f]  cluster [%.3f, %.3f]  median DockQ %.3f",
            label, a["rate"], a["successes"], a["denominator"], lo, hi, clo, chi,
            r["distribution"].get("dockq_score", {}).get("median", float("nan")),
        )
    agg = report["across_runs"]["aggregate"].get("acceptable")
    if agg and agg["runs"] > 1:
        logger.info("across %d runs  mean %.4f  sd %.4f  range %.4f",
                    agg["runs"], agg["mean"], agg["stdev"], agg["range"])
        flips = report["across_runs"]["per_interface"].get("acceptable", {})
        logger.info("per interface  always %d / never %d / flipped %d",
                    flips.get("always_success"), flips.get("always_failure"),
                    flips.get("flipped"))
    logger.info("wrote %s", out_dir / f"summary_{args.metric_type}.json")


if __name__ == "__main__":
    main()
