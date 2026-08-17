#!/bin/bash
# FoldBench plugin entry point for OpenDDE.
#
# Same five-argument contract and same shape as FoldBench's own Protenix plugin
# (vendor/foldbench-protenix-plugin/make_predictions.sh); only the inference
# command differs. The sampling budget below is not ours to choose -- it is the
# budget FoldBench runs its reference model at, and it coincides with OpenDDE's
# own documented defaults (docs/supported_models.md: N_cycle 10, N_step 200).
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
$PYTHON_PATH ./preprocess.py --af3_input_json="$af3_input_json" --input_dir="$input_dir"

export CUDA_VISIBLE_DEVICES=$gpu_id
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_sample=5
N_step=200
N_cycle=10
# The five seeds FoldBench's reference plugin uses. postprocess.py enumerates
# exactly these, so the two must not drift apart.
seeds="42,66,101,2024,8888"

# --use_msa/--use_template are what separate a real FoldBench number from a
# crippled one; OpenDDE warns at runtime when MSA is off and its own defaults
# have it on (opendde/config/inference_defaults.py).
$OPENDDE_CLI pred \
    -i "${input_dir}/inputs.json" \
    -o "${prediction_dir}" \
    -s "${seeds}" \
    -c ${N_cycle} \
    -p ${N_step} \
    -e ${N_sample} \
    -n opendde_v1 \
    --use_msa true \
    --use_template true

# OpenDDE output -> mmCIF that OpenStructure/DockQv2 accept, plus the
# prediction_reference.csv FoldBench's evaluator reads.
$PYTHON_PATH ./postprocess.py \
    --input_dir="$input_dir" \
    --prediction_dir="$prediction_dir" \
    --evaluation_dir="$evaluation_dir"
