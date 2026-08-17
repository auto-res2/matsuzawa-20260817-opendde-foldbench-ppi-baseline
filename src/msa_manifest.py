"""Pin the MSA cache: record what it contains, and check it has not moved.

PROVENANCE: ours. Nothing upstream produces this; it exists because the cache is
the one input to this benchmark that cannot be re-derived.

Every other input is fixed. The weights have a published sha256. FoldBench's
targets and ground truths are a released dataset. The sampling budget is
written down. But the MSAs came from a shared public service that searches
databases we do not control, on a day we do not get to repeat: re-running
`prepare` next month would return different alignments, and the score would
move for reasons having nothing to do with the model.

That makes the cache a dependency the reproduction cannot reproduce. It cannot
be fixed by rerunning harder. What it can be is *pinned*: hashed, counted, and
described, so that

  * a later arm can prove it read the same MSAs the baseline did, which is the
    only way "fine-tuning changed the score by X" means anything;
  * a reader can tell whether their cache matches ours before wondering why
    their number differs;
  * losing the cache is a detectable event rather than a silent one.

The manifest is small enough to commit. The a3m files are not, and live on
RIKYU; see README.

    python -m src.msa_manifest write  --input inputs.json --out msa_manifest.json
    python -m src.msa_manifest verify --input inputs.json --manifest msa_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def file_facts(path_str: str | None) -> dict | None:
    """sha256, byte size and sequence count for one a3m."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    data = p.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "depth": data.count(b"\n>") + (1 if data.startswith(b">") else 0),
    }


def build(entries: list[dict]) -> dict:
    """Describe every protein chain's MSA, keyed by target and chain index."""
    chains: dict[str, dict] = {}
    for entry in entries:
        name = entry.get("name", "?")
        idx = 0
        for seq in entry.get("sequences", []):
            chain = seq.get("proteinChain")
            if chain is None:
                continue
            chains[f"{name}#{idx}"] = {
                "unpaired": file_facts(chain.get("unpairedMsaPath")),
                "paired": file_facts(chain.get("pairedMsaPath")),
            }
            idx += 1

    depths = sorted(
        c["unpaired"]["depth"] for c in chains.values() if c["unpaired"]
    )
    return {
        # Recorded so a mismatch can be attributed rather than merely noticed.
        "source": "https://api.colabfold.com (opendde msa, mode env / pairgreedy)",
        "n_targets": len({k.split("#")[0] for k in chains}),
        "n_chains": len(chains),
        "depth_min": depths[0] if depths else None,
        "depth_median": depths[len(depths) // 2] if depths else None,
        "depth_max": depths[-1] if depths else None,
        "chains": chains,
    }


def verify(entries: list[dict], manifest: dict) -> int:
    """Compare the cache on disk against the manifest, and name every drift."""
    current = build(entries)
    old, new = manifest["chains"], current["chains"]

    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(
        k for k in set(old) & set(new)
        if (old[k]["unpaired"] or {}).get("sha256") != (new[k]["unpaired"] or {}).get("sha256")
    )

    for label, items in (("missing from disk", missing), ("not in manifest", added),
                         ("content changed", changed)):
        if items:
            logger.error("%d chain(s) %s: %s", len(items), label, items[:10])

    if missing or added or changed:
        logger.error(
            "The MSA cache is not the one this baseline was measured on. Any "
            "score produced against it is not comparable with the recorded run."
        )
        return 1
    logger.info("MSA cache matches the manifest: %d chains over %d targets",
                current["n_chains"], current["n_targets"])
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["write", "verify"])
    p.add_argument("--input", required=True, help="OpenDDE inputs.json")
    p.add_argument("--out", help="manifest to write (mode=write)")
    p.add_argument("--manifest", help="manifest to check against (mode=verify)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    entries = json.loads(Path(args.input).read_text())

    if args.mode == "write":
        manifest = build(entries)
        Path(args.out).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        logger.info("wrote %s: %d chains over %d targets, depth %s/%s/%s",
                    args.out, manifest["n_chains"], manifest["n_targets"],
                    manifest["depth_min"], manifest["depth_median"], manifest["depth_max"])
        raise SystemExit(0)

    raise SystemExit(verify(entries, json.loads(Path(args.manifest).read_text())))


if __name__ == "__main__":
    main()
