#!/bin/bash
# =============================================================================
# PROVENANCE: OURS, written to somebody else's contract.
#
# The five-argument contract and the overall shape are FoldBench's, copied from
# its own Protenix plugin; only the inference command is different.
#
#   Contract spec : https://github.com/BEAM-Labs/FoldBench
#                   (algorithms/README.md -- the four-file plugin interface)
#   Shape copied  : vendor/foldbench-protenix-plugin/make_predictions.sh
#   Inference CLI : https://github.com/aurekaresearch/OpenDDE
#                   docs/inference_instructions.md
#
# Nothing about the sampling budget below is our choice. It is what FoldBench
# runs its reference model at, and it coincides with OpenDDE's own documented
# recommendation (docs/supported_models.md: N_cycle 10, N_step 200).
# =============================================================================
set -euo pipefail

af3_input_json=$1
input_dir=$2
prediction_dir=$3
evaluation_dir=$4
gpu_id=$5

PYTHON_PATH="${OPENDDE_PYTHON:-python}"
OPENDDE_CLI="${OPENDDE_CLI:-opendde}"

# AF3-dialect JSON -> the proteinChain/dnaSequence/ligand schema that both
# Protenix and OpenDDE consume. preprocess.py is upstream code, unmodified.
#
# Skipped when inputs.json is already there, because by then it is not merely a
# converted file: the prepare stage has written every chain's unpairedMsaPath
# into it. Re-running preprocess would regenerate it from the AF3 JSON and drop
# those paths, and the run would silently fall back to searching MSAs on a
# compute node -- or, with no egress, to no MSA at all. Delete inputs.json to
# force reconversion.
if [ -f "${input_dir}/inputs.json" ]; then
    echo "inputs.json exists; keeping it (MSA paths from the prepare stage)"
else
    $PYTHON_PATH ./preprocess.py --af3_input_json="$af3_input_json" --input_dir="$input_dir"
fi

# Leave the launcher's GPU wiring alone. srun exposes every card on the node to
# every task (CUDA_VISIBLE_DEVICES=0,1,2,3 with four GPUs) and sets LOCAL_RANK to
# the task's index on that node; OpenDDE then selects device LOCAL_RANK
# (opendde/utils/environment.py, select_torch_device). That pairing is correct
# and is what spreads four tasks across four cards.
#
# The multi-GPU sweeps died because this script used to overwrite
# CUDA_VISIBLE_DEVICES with a hardcoded 0, cutting each task down to one visible
# card, after which OpenDDE asked for device 1, 2 or 3 and got
#   "CUDA device index N is unavailable; detected 1 CUDA device(s)".
# Single-GPU runs survived only because their LOCAL_RANK was 0, which made the
# fault look like a concurrency ceiling and sent us halving GPU counts for
# nothing.
#
# Forcing LOCAL_RANK=0 would be the opposite mistake: every task on a node would
# pile onto card 0 and leave three idle.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_sample=5
N_step=200
N_cycle=10
# The five seeds FoldBench's reference plugin uses. postprocess.py enumerates
# exactly these, so the two must not drift apart.
seeds="42,66,101,2024,8888"

# MSA on, templates off -- both are OpenDDE's own defaults
# (opendde/config/inference_defaults.py: use_msa True, use_template False), and
# neither is ours to pick. MSA is what separates a real FoldBench number from a
# crippled one, and OpenDDE warns at runtime when it is off. Templates are off
# because the shipped default has them off, the OpenDDE report does not say it
# evaluated with them, and FoldBench's own reference plugin passes no template
# flag either. Turning them on additionally needs a kalign binary.
#
# The sampling arguments below are identical on both paths. Only the launcher
# differs, and that is the whole point: Fold-CP is a way of executing this same
# command, not a different command.
PRED_ARGS=(
    -i "${input_dir}/inputs.json"
    -o "${prediction_dir}"
    -s "${seeds}"
    -c "${N_cycle}"
    -p "${N_step}"
    -e "${N_sample}"
    -n opendde_v1
    --use_msa true
    --use_template "${FB_USE_TEMPLATE:-false}"
)

if [ "${FB_FOLDCP_MODE:-single}" = "distributed" ]; then
    # ---- Fold-CP: ONE target spread across several GPUs -------------------
    #
    # For the assemblies that do not fit a single card. Six of them (1,872-2,304
    # residues) drove a single process past the 184 GB on one GB200 and were
    # quarantined. The report's Table 3 measures the same trade -- 2,000 tokens
    # OOMs on one card and fits in ~79 GiB under CP4 -- and Appendix C states
    # Fold-CP "does not change model weights, architecture, or inference
    # semantics". So this runs the six at the same sampling budget as the other
    # 233 rather than lowering N_sample, which would have made their numbers
    # incomparable.
    #
    # Two constraints, both discovered by reading OpenDDE rather than by running
    # it, and both fatal if ignored:
    #
    #   1. It must be torchrun with size_dp * size_cp processes
    #      (opendde/distributed/foldcp/config.py, launch_hint). A single process
    #      makes runner/inference.py raise "Distributed Fold-CP must be launched
    #      with torchrun using N processes". RANK/WORLD_SIZE must therefore NOT
    #      be pinned here -- torchrun assigns them per process, and the single
    #      path's RANK=0 WORLD_SIZE=1 would be read as world_size 1.
    #
    #   2. inputs.json must hold exactly ONE target. The sampler shards by
    #      world_size, not by size_dp
    #      (opendde/data/inference/infer_dataloader.py), so at one entry its
    #      padding hands every rank the same target -- which is what Fold-CP
    #      needs -- while at two or more each rank gets a different target and
    #      the collectives deadlock or crash. src/main.py enforces this by
    #      building a one-entry input directory per oversized target.
    #
    # Only rank 0 writes structures; OpenDDE suppresses the others because all
    # ranks compute the same sample (runner/inference.py, the foldcp guard on
    # dumper.dump).
    CP=${FB_FOLDCP_SIZE_CP:-4}
    n_entries=$($PYTHON_PATH -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${input_dir}/inputs.json")
    if [ "${n_entries}" -ne 1 ]; then
        echo "Fold-CP needs exactly one target per invocation; ${input_dir}/inputs.json holds ${n_entries}." >&2
        exit 2
    fi
    # torchrun owns the rendezvous and the per-process rank assignment. Leaving a
    # stale RANK/WORLD_SIZE in the environment would be read before torchrun's.
    unset RANK WORLD_SIZE
    TORCHRUN="${OPENDDE_TORCHRUN:-torchrun}"
    echo "Fold-CP: ${CP} processes for 1 target"
    $TORCHRUN --standalone --nnodes=1 --nproc_per_node="${CP}" \
        -m runner.batch_inference pred \
        "${PRED_ARGS[@]}" \
        --foldcp_mode distributed \
        --foldcp_size_dp 1 \
        --foldcp_size_cp "${CP}" \
        ${FB_FOLDCP_METRICS:+--foldcp_metrics_jsonl "${FB_FOLDCP_METRICS}"}
else
    # ---- Single card: one process handles this whole shard ------------------
    #
    # `opendde pred` reads RANK/WORLD_SIZE and distributes the input JSON across
    # them by itself -- the run log shows "[Rank 1] ... [Rank 13]" from a single
    # invocation. Since src/main.py has already split the targets and handed this
    # process its own shard, leaving those variables set makes the work be
    # divided twice: a 15-target shard was split again 16 ways and the process
    # ran "0/1". Every shard then finished after roughly one target, and the
    # missing-target check fired and took the whole job down with it.
    #
    # So the model is told it is alone. The sharding stays where it is visible
    # and ours, in stage_predict.
    export RANK=0
    export WORLD_SIZE=1
    $OPENDDE_CLI pred "${PRED_ARGS[@]}"
fi

# OpenDDE output -> mmCIF that OpenStructure/DockQv2 accept, plus the
# prediction_reference.csv FoldBench's evaluator reads.
$PYTHON_PATH ./postprocess.py \
    --input_dir="$input_dir" \
    --prediction_dir="$prediction_dir" \
    --evaluation_dir="$evaluation_dir"
