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

# Slurm hands each task its own GPU and already exports CUDA_VISIBLE_DEVICES
# for it, so the device is index 0 from inside every task. Only override that
# when the launcher has not set it -- clobbering Slurm's value would point a
# task at a card it was not given.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

# OpenDDE selects its device by LOCAL_RANK (opendde/utils/environment.py,
# select_torch_device). Under srun that variable is the task's index on its
# node -- 0..3 with four tasks per node -- so tasks 1-3 asked for CUDA device
# 1, 2 and 3 and died with "device index N is unavailable; detected 1 CUDA
# device(s)". Each task can only ever see one card, and it is index 0.
#
# This is what actually killed the 16- and 8-GPU sweeps. The single-GPU runs
# survived because their LOCAL_RANK happened to be 0, which made it look like a
# concurrency limit and sent us halving the GPU count for nothing.
export LOCAL_RANK=0
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
$OPENDDE_CLI pred \
    -i "${input_dir}/inputs.json" \
    -o "${prediction_dir}" \
    -s "${seeds}" \
    -c ${N_cycle} \
    -p ${N_step} \
    -e ${N_sample} \
    -n opendde_v1 \
    --use_msa true \
    --use_template "${FB_USE_TEMPLATE:-false}"

# OpenDDE output -> mmCIF that OpenStructure/DockQv2 accept, plus the
# prediction_reference.csv FoldBench's evaluator reads.
$PYTHON_PATH ./postprocess.py \
    --input_dir="$input_dir" \
    --prediction_dir="$prediction_dir" \
    --evaluation_dir="$evaluation_dir"
