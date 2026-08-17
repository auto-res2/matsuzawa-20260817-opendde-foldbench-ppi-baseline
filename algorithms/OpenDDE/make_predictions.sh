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

export CUDA_VISIBLE_DEVICES=$gpu_id
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
