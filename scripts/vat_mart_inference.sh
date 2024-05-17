#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/VATMart_Inference"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_inference.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --evaluation

# the individual model checkpoints are specified in the config file, but can be overwritten here by using the
# cfg-options flag