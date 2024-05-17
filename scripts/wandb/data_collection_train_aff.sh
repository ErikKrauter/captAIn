#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="Runs/DataCollectionTrainAff"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_collect.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --evaluation \
  --use-wandb \
  --upload-dataset \
  --resume-from rl_model_1200:v0 \
  --wandb-group Dataset_affordance_predictor \
  --cfg-options \
  "env_cfg.control_mode=pd_ee_target_delta_pose" \
  "eval_cfg.affordance_predictor_data_set=True" \
  "eval_cfg.augment_dataset=True" \
  "eval_cfg.only_save_success_traj=True" \
  "eval_cfg.num=5000"

# the value for resume-from is the name of the model version of the data collection agent used to collect this dataset
# these values can be found in the wandb web interface
# upon completion the entire dataset is stored in wandb if the flag --upload-dataset is set