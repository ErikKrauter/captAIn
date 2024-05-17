#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/CaptAIn"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/closed_loop_train.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --clean-up \
  --use-wandb \
  --aff_model affordancePredictor_model_10240:v3 \
  --gen_model PoseTrajectoryGenerator_model_21504:v1 \
  --wandb-group captAIn

# the values for aff_model and gen_model are the names of the model version
# these values can be found in the wandb web interface
# the wandb-group denotes the tag which will automatically be assigned to this runs
# these tags can be used to filter and search runs later in the wandb web interface