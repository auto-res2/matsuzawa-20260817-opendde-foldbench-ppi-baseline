"""Per-chain CA RMSD: how well each chain folds, with the assembly taken apart.

PROVENANCE: ours. It is the one metric in this report that no upstream tool
already produces, because no upstream tool asks the question.

WHAT IT SEPARATES. `rmsd` in FoldBench's table comes from superposing the whole
complex at once, so a model that folds both chains perfectly and docks them
wrongly scores badly on it, and the two failures are not distinguishable. Here
each chain is superposed on its own counterpart alone. A large per-chain RMSD
means the fold is wrong; a small one beside a poor DockQ means the folds are
right and the placement is not.

The maximum over an assembly's chains is what gets reported, being the worst
fold in the prediction rather than an average that a well-folded partner can
hide.

Superposition is OpenStructure's, through its Python API, so the alignment
rules are the scorer's and not a second implementation of them.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def per_chain_rmsd(model_path: str, reference_path: str) -> dict:
    """CA RMSD for every chain the two structures share, after aligning each alone."""
    from ost import io
    from ost.mol.alg import SuperposeSVD

    mdl = io.LoadMMCIF(model_path, fault_tolerant=True)
    ref = io.LoadMMCIF(reference_path, fault_tolerant=True)

    out = {}
    ref_by_name = {c.name: c for c in ref.chains}
    for mdl_chain in mdl.chains:
        ref_chain = ref_by_name.get(mdl_chain.name)
        if ref_chain is None:
            continue
        # Residue numbers rather than a fresh alignment: both structures come
        # from the same benchmark entry, and FoldBench's own comparison relies
        # on the same correspondence.
        ref_res = {r.number.num: r for r in ref_chain.residues}
        mdl_view = mdl.CreateEmptyView()
        ref_view = ref.CreateEmptyView()
        paired = 0
        for residue in mdl_chain.residues:
            counterpart = ref_res.get(residue.number.num)
            if counterpart is None:
                continue
            a1 = residue.FindAtom("CA")
            a2 = counterpart.FindAtom("CA")
            if not (a1.IsValid() and a2.IsValid()):
                continue
            mdl_view.AddAtom(a1)
            ref_view.AddAtom(a2)
            paired += 1
        # Three points is the minimum a superposition is defined on, which is
        # also where OpenStructure's own rigid scores give up.
        if paired < 3:
            out[mdl_chain.name] = {"n_ca": paired, "rmsd": None}
            continue
        # SuperposeSVD rather than Superpose: the atoms are already paired one
        # to one above, so no matching strategy is wanted and none should be
        # guessed. apply_transform=False leaves the loaded structure alone --
        # every chain is superposed against the same untouched model.
        result = SuperposeSVD(mdl_view, ref_view, False)
        out[mdl_chain.name] = {"n_ca": paired, "rmsd": float(result.rmsd)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--picks", required=True,
                   help="CSV with pdb_id, seed, sample (from interface_lddt's selection)")
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--ground-truth-dir", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import csv
    rows, failed = [], []
    with open(args.picks) as handle:
        picks = list(csv.DictReader(handle))
    logger.info("%s: %d assemblies", args.label, len(picks))

    for n, pick in enumerate(picks, start=1):
        pdb_id, seed, sample = pick["pdb_id"], pick["seed"], pick["sample"]
        model = (Path(args.prediction_dir) / pdb_id / f"seed_{seed}" / "predictions"
                 / f"{pdb_id}_sample_{sample}_postprocessed.cif")
        reference = Path(args.ground_truth_dir) / f"{pdb_id}.cif"
        if not (model.is_file() and reference.is_file()):
            failed.append(f"{pdb_id} (missing file)")
            continue
        try:
            chains = per_chain_rmsd(str(model), str(reference))
        except Exception as exc:                       # noqa: BLE001 - recorded
            failed.append(f"{pdb_id} ({exc.__class__.__name__}: {exc})")
            continue
        values = [c["rmsd"] for c in chains.values() if c["rmsd"] is not None]
        rows.append({
            "run": args.label, "pdb_id": pdb_id, "seed": seed, "sample": sample,
            "n_chains_compared": len(values),
            "max_chain_ca_rmsd": max(values) if values else None,
            "mean_chain_ca_rmsd": (sum(values) / len(values)) if values else None,
            "per_chain": chains,
        })
        if n % 40 == 0:
            logger.info("  %d/%d", n, len(picks))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"label": args.label, "rows": rows,
                                          "failed": failed}, indent=2))
    if failed:
        logger.warning("%d assembly/assemblies did not compare: %s",
                       len(failed), failed[:5])
    got = [r["max_chain_ca_rmsd"] for r in rows if r["max_chain_ca_rmsd"] is not None]
    if got:
        got.sort()
        logger.info("%s: max-chain CA RMSD n=%d mean %.3f median %.3f",
                    args.label, len(got), sum(got) / len(got), got[len(got) // 2])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
