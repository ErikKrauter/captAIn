#!/bin/bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0

export WANDB_CACHE_DIR='/home/erik/Masterthesis/wandb_cache_dir'
echo "WANDB_CACHE_DIR: ${WANDB_CACHE_DIR}"
#export MASTER_ADDR=$(hostname)
#echo "MASTER_ADDR: ${MASTER_ADDR}"

# this is needed to add the project root to the python path, else the import of Evaluation.trajectory_analysis will fail
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')

set -o errexit

# Construct the command to run python script
PYTHON_CMD="python -u ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py ManiSkill2-Learn/configs/vat-mart/cluster/closed_loop_train_continuouslearning_cluster.py $@"

# Run python script
echo "Running command: ${PYTHON_CMD}"
${PYTHON_CMD}

echo "Finished at:     $(date)"
exit 0
