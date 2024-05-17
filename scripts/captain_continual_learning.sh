#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/CaptAIn_ContinualLearning"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/closed_loop_train_continuouslearning.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --resume-from DatasetsAndModels/captAIn_Model/model_2400000.ckpt