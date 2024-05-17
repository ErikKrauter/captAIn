#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/TrajectoryGenerator"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_bc.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --cfg-options "agent_cfg.mode=trajectoryGenerator" "agent_cfg.use_dataset=True"

