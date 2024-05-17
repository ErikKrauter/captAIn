#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="DatasetsAndModels/DataCollectionAgent"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_train_rl.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --cfg-options "env_cfg.control_mode=pd_ee_target_delta_pose"
