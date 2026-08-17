"""Fill an OpenDDE input JSON with MSAs, ahead of the GPU stage.

`runner.msa_search` is a library, not a command, so this is the missing shim and
nothing more: it calls OpenDDE's own `update_infer_json` and then puts the
result back under the original filename.

The rename is what keeps the plugin honest. `update_infer_json` writes its
output beside the input as `<stem>-update-msa.json`; moving that over
`inputs.json` means `make_predictions.sh` names one file whether or not this
stage ever ran. With the MSAs already present, `--use_msa true` finds nothing to
search (`need_msa_search` is False) and the GPU stage touches no network -- which
is the only reason the two stages are split, since RIKYU's compute nodes are not
assumed to have egress.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="OpenDDE inputs.json to fill")
    p.add_argument("--out-dir", required=True, help="where a3m artefacts are cached")
    p.add_argument("--opendde-src", required=True, help="OpenDDE checkout to import")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, args.opendde_src)
    from runner.msa_search import update_infer_json

    src = Path(args.input)
    updated, changed = update_infer_json(str(src), args.out_dir, use_msa=True)

    if not changed:
        # Either every entry already carried an MSA, or none was searched. The
        # second case would silently produce a no-MSA run, so say which it is.
        logger.warning("no MSA search performed; %s already complete?", src)

    updated_path = Path(updated)
    if updated_path != src:
        shutil.move(str(updated_path), str(src))
        logger.info("MSAs merged into %s", src)


if __name__ == "__main__":
    main()
