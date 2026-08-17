"""Fill an OpenDDE input JSON with MSAs, ahead of the GPU stage, and verify them.

`runner.msa_search` is a library, not a command, so a shim is needed either way.
The verification around it is not optional decoration -- it is the whole point.

OpenDDE's own documentation says so plainly (docs/msa_template_pipeline.md):

    `opendde msa` uses the public ColabFold MMseqs2 API ... The service is
    shared and rate-limited; for batch runs, provide precomputed A3M files.

Firing 239 targets at that service as fast as they would go did not fail loudly.
It returned HTTP 429, the client retried, the retry "succeeded", and the a3m
written to disk contained the query sequence and nothing else. 128 of 239
targets came back that way -- 54% -- and every downstream step accepted them:
the JSON had a valid path, the file existed, and the prediction ran in what was
effectively single-sequence mode. Nothing in the pipeline could tell the
difference between a deep MSA and an empty one.

So this module treats a returned MSA as a claim to be checked. Each target is
fetched on its own with a pause between requests, the depth of what came back is
counted, and anything at or below the query-only floor is re-fetched with a
longer pause. A target that will not come back deep after every round is named
and the run fails, rather than being carried into a benchmark as a silent zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# An a3m holding only ">query" is what a rate-limited miss looks like on disk.
# Anything at this depth carries no evolutionary signal at all.
QUERY_ONLY_DEPTH = 1


def msa_depth(path: str | None) -> int:
    """Sequence count in an a3m, or -1 when the file is absent."""
    if not path:
        return -1
    p = Path(path)
    if not p.exists():
        return -1
    with p.open() as fh:
        return sum(1 for line in fh if line.startswith(">"))


def chain_msa_state(entry: dict) -> list[tuple[str, int]]:
    """(unpairedMsaPath, depth) for every protein chain of one job."""
    state = []
    for seq in entry.get("sequences", []):
        chain = seq.get("proteinChain")
        if chain is None:
            continue
        state.append((chain.get("unpairedMsaPath"), msa_depth(chain.get("unpairedMsaPath"))))
    return state


def entry_is_good(entry: dict) -> bool:
    """True when every protein chain has an MSA deeper than the query alone."""
    state = chain_msa_state(entry)
    return bool(state) and all(depth > QUERY_ONLY_DEPTH for _, depth in state)


def clear_msa_paths(entry: dict) -> None:
    """Drop MSA fields so OpenDDE's `need_msa_search` will look again.

    Without this a re-run is a no-op: the failed entries still carry a path to
    the empty a3m, so the search is considered already done.
    """
    for seq in entry.get("sequences", []):
        chain = seq.get("proteinChain")
        if chain is None:
            continue
        for field in ("pairedMsa", "unpairedMsa", "pairedMsaPath", "unpairedMsaPath", "msa"):
            chain.pop(field, None)


def fetch_one(entry: dict, out_dir: Path, update_seq_msa) -> None:
    """Search one job's MSAs with OpenDDE's own per-task routine."""
    name = entry.get("name", "unnamed")
    update_seq_msa(entry, str(out_dir / name / "msa"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="OpenDDE inputs.json to fill")
    p.add_argument("--out-dir", required=True, help="where a3m artefacts are cached")
    p.add_argument("--opendde-src", required=True, help="OpenDDE checkout to import")
    # Pacing exists because the service is shared. The first pass is polite; a
    # target that came back empty is retried more slowly still, on the theory
    # that an empty return means we are being throttled rather than that the
    # protein has no homologs.
    p.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    p.add_argument("--rounds", type=int, default=4, help="passes over the failures")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, args.opendde_src)
    from runner.msa_search import update_seq_msa

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    entries = json.loads(src.read_text())
    logger.info("%d jobs in %s", len(entries), src)

    for rnd in range(1, args.rounds + 1):
        todo = [e for e in entries if not entry_is_good(e)]
        if not todo:
            logger.info("round %d: every job already has a real MSA", rnd)
            break
        delay = args.delay * rnd
        logger.info("round %d: %d job(s) need an MSA, %.1fs between requests",
                    rnd, len(todo), delay)
        for i, entry in enumerate(todo, 1):
            clear_msa_paths(entry)
            try:
                fetch_one(entry, out_dir, update_seq_msa)
            except Exception as exc:  # noqa: BLE001 - one target must not stop the sweep
                logger.warning("%s: search raised %s", entry.get("name"), exc)
            depths = [d for _, d in chain_msa_state(entry)]
            logger.info("round %d [%d/%d] %s depth=%s", rnd, i, len(todo),
                        entry.get("name"), depths)
            # Written every round so a killed prefetch keeps what it earned.
            src.write_text(json.dumps(entries, indent=4))
            time.sleep(delay)

    bad = [e.get("name") for e in entries if not entry_is_good(e)]
    depths = sorted(d for e in entries for _, d in chain_msa_state(e))
    logger.info("MSA depth: min=%s median=%s max=%s over %d chains",
                depths[0], depths[len(depths) // 2], depths[-1], len(depths))

    if bad:
        logger.error("%d job(s) still have a query-only MSA after %d rounds: %s",
                     len(bad), args.rounds, bad)
        logger.error("Do not run the benchmark on these -- an empty MSA scores "
                     "like single-sequence mode and silently lowers the result.")
        raise SystemExit(1)

    logger.info("all %d jobs carry a real MSA", len(entries))


if __name__ == "__main__":
    main()
