#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/CaptAIn_ContinuousLearning"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/closed_loop_train_continuouslearning.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --clean-up \
  --use-wandb \
  --resume-from VAT_SAC_model_2400000:v0 \
  --wandb-group captAIn_CL

# the value for resume-from is the names of the model version
# these values can be found in the wandb web interface