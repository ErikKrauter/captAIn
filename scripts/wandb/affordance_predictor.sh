#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/AffordancePredictor"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_bc.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --evaluation \
  --use-wandb \
  --train-dataset-version v7 \
  --eval-dataset-version v5 \
  --wandb-group affordancePredictor \
  --cfg-options \
  "agent_cfg.mode=affordancePredictor" \
  "agent_cfg.use_dataset=True"

# the dataset versions are defined in the wandb webinterface
# checkpoints of the trained model will be automatically stored in wandb