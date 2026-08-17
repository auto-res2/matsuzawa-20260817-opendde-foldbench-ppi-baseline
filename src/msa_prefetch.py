"""Fill an OpenDDE input JSON with MSAs, one target at a time, and verify them.

PROVENANCE: this file is OURS, but it searches nothing itself. Every MSA is
fetched by shelling out to OpenDDE's documented command:

    opendde msa -i <one-job.json> -o <msa_dir>

  CLI      https://github.com/aurekaresearch/OpenDDE
           docs/inference_instructions.md
  Pipeline https://github.com/aurekaresearch/OpenDDE
           docs/msa_template_pipeline.md

The public command is used rather than importing `runner.msa_search` so that
nothing here reaches into OpenDDE's internals: the CLI is the contract, and a
reproduction should stand on the interface upstream documents. What we add is
strictly outside it -- pacing, depth verification, and re-fetching.

That addition is not decoration. OpenDDE's search catches every exception and
falls back to a query-only MSA "so inference can still run", and its own docs
warn that the public ColabFold service "is shared and rate-limited; for batch
runs, provide precomputed A3M files". Handing it all 239 targets as fast as they
would go made that fallback fire for 111 of them, silently: the JSON had a valid
path, the a3m existed, and prediction proceeded in what was effectively
single-sequence mode. Nothing downstream could tell the difference.

So a returned MSA is treated as a claim to be checked. Each target is fetched on
its own with a pause, the depth of what came back is counted, and a query-only
answer is re-fetched with a longer pause -- an empty return being read as
throttling rather than as a protein with no homologs. Anything still empty after
every round is named, and the stage exits non-zero rather than letting a silent
zero into the benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# An a3m holding only ">query" is what a rate-limited miss looks like on disk.
QUERY_ONLY_DEPTH = 1

# The fields `opendde msa` fills in, and that a re-fetch has to clear first --
# OpenDDE skips the search when a path is already present, so without this a
# re-run of a failed target does nothing at all.
MSA_FIELDS = ("pairedMsa", "unpairedMsa", "pairedMsaPath", "unpairedMsaPath", "msa")


def msa_depth(path: str | None) -> int:
    """Sequence count in an a3m, or -1 when the file is absent."""
    if not path:
        return -1
    p = Path(path)
    if not p.exists():
        return -1
    with p.open() as fh:
        return sum(1 for line in fh if line.startswith(">"))


def protein_chains(entry: dict) -> list[dict]:
    return [s["proteinChain"] for s in entry.get("sequences", []) if "proteinChain" in s]


def depths(entry: dict) -> list[int]:
    return [msa_depth(c.get("unpairedMsaPath")) for c in protein_chains(entry)]


def search_ran(entry: dict) -> bool:
    """True when the service actually answered, whatever it found.

    Depth alone cannot decide this, and using it would deadlock the run. A
    query-only a3m has two very different causes:

      fallback fired   `<msa>/unpaired/` is EMPTY -- the request never landed,
                       and the exception handler wrote the query out so
                       inference could proceed. Retry this.
      genuinely sparse `<msa>/unpaired/` holds uniref.a3m, the environmental
                       a3m, msa.sh and the manifest -- both databases were
                       searched and returned almost nothing. 8bbt-assembly1 is
                       a real example: 241 residues, one UniRef hit. Retrying
                       forever would never improve it.

    So the artefacts decide whether the search happened; depth is reported but
    never used to reject.
    """
    for chain in protein_chains(entry):
        path = chain.get("unpairedMsaPath")
        if not path:
            return False
        # <msa_dir>/<name>/msa/0/non_pairing.a3m -> <msa_dir>/<name>/msa/unpaired/
        unpaired = Path(path).parent.parent / "unpaired"
        if not (unpaired / "uniref.a3m").exists():
            return False
    return True


def entry_is_good(entry: dict) -> bool:
    """True when every protein chain has been searched for successfully."""
    return bool(protein_chains(entry)) and search_ran(entry)


def fetch_one(entry: dict, work_dir: Path, msa_dir: Path, cli: str) -> None:
    """Run `opendde msa` for a single job and copy the result back into `entry`.

    The CLI takes a file and writes `<stem>-update-msa.json` beside it, so a
    one-job file is written per target and the filled-in chains are merged back.
    """
    name = entry.get("name", "job")
    work_dir.mkdir(parents=True, exist_ok=True)
    job_file = work_dir / f"{name}.json"

    for chain in protein_chains(entry):
        for field in MSA_FIELDS:
            chain.pop(field, None)
    job_file.write_text(json.dumps([entry], indent=2))

    subprocess.run(
        [cli, "msa", "-i", str(job_file), "-o", str(msa_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    updated = job_file.with_name(f"{job_file.stem}-update-msa.json")
    if not updated.exists():
        # No update file means the CLI decided there was nothing to search.
        # Left as-is, the depth check below will catch it.
        logger.warning("%s: no -update-msa.json written", name)
        return

    filled = json.loads(updated.read_text())[0]
    for chain, new_chain in zip(protein_chains(entry), protein_chains(filled)):
        for field in MSA_FIELDS:
            if field in new_chain:
                chain[field] = new_chain[field]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="OpenDDE inputs.json to fill")
    p.add_argument("--out-dir", required=True, help="where a3m artefacts are cached")
    p.add_argument("--work-dir", required=True, help="scratch for per-job JSON")
    p.add_argument("--opendde-cli", default="opendde", help="the `opendde` executable")
    # Pacing exists because the service is shared. Failures are retried more
    # slowly still, on the theory that an empty return means throttling.
    p.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    p.add_argument("--rounds", type=int, default=4, help="passes over the failures")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    src = Path(args.input)
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
            name = entry.get("name")
            try:
                fetch_one(entry, Path(args.work_dir), Path(args.out_dir), args.opendde_cli)
            except subprocess.CalledProcessError as exc:
                logger.warning("%s: `opendde msa` exited %d: %s",
                               name, exc.returncode, (exc.stderr or "")[-300:])
            logger.info("round %d [%d/%d] %s depth=%s", rnd, i, len(todo), name, depths(entry))
            # Written every round so a killed prefetch keeps what it earned.
            src.write_text(json.dumps(entries, indent=4))
            time.sleep(delay)

    bad = [e.get("name") for e in entries if not entry_is_good(e)]
    alld = sorted(x for e in entries for x in depths(e))
    logger.info("MSA depth over %d chains: min=%s median=%s max=%s",
                len(alld), alld[0], alld[len(alld) // 2], alld[-1])

    # Reported, never rejected: a searched-but-sparse chain is a fact about the
    # protein, not a failure of ours. It belongs in the record so the eventual
    # score can be read against it.
    sparse = sorted(
        (min(depths(e)), e.get("name")) for e in entries
        if entry_is_good(e) and min(depths(e)) <= 10
    )
    if sparse:
        logger.info("%d job(s) searched successfully but have <=10 sequences: %s",
                    len(sparse), sparse[:15])

    if bad:
        logger.error("%d job(s) were never successfully searched after %d rounds: %s",
                     len(bad), args.rounds, bad)
        logger.error("These carry a fallback MSA, which scores like "
                     "single-sequence mode and silently lowers the result.")
        raise SystemExit(1)

    logger.info("all %d jobs were searched successfully", len(entries))


if __name__ == "__main__":
    main()
