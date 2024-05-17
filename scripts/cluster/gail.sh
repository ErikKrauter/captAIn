#!/bin/bash
source /scratch_net/got/ekrauter/conda/etc/profile.d/conda.sh
conda activate lab_mani_skill2
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0

export WANDB_CACHE_DIR='/scratch_net/got/ekrauter/Masterthesis/wandb_cache_dir'
echo "WANDB_CACHE_DIR: ${WANDB_CACHE_DIR}"
#export MASTER_ADDR=$(hostname)
#echo "MASTER_ADDR: ${MASTER_ADDR}"

# this is needed to add the project root to the python path, else the import of Evaluation.trajectory_analysis will fail
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')

set -o errexit

# Construct the command to run python script
PYTHON_CMD="python -u ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py ManiSkill2-Learn/configs/mfrl/gail/turnfaucet_pn.py  $@"

# Run python script
echo "Running command: ${PYTHON_CMD}"
${PYTHON_CMD}

echo "Finished at:     $(date)"
exit 0
