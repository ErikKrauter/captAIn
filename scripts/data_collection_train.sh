#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
WORK_DIR="DatasetsAndModels/DatasetTrain"

# Run the training command
python ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py \
  ManiSkill2-Learn/configs/vat-mart/vat-mart_collect.py \
  --work-dir $WORK_DIR \
  --gpu-ids 0 \
  --evaluation \
  --resume-from DatasetsAndModels/DataCollectionAgent_Model/model_100000.ckpt \
  --cfg-options "env_cfg.control_mode=pd_ee_target_delta_pose" "eval_cfg.affordance_predictor_data_set=False" \
  "eval_cfg.augment_dataset=True" "eval_cfg.only_save_success_traj=False" "eval_cfg.num=6500"
