#!/bin/bash

# Default reward mode
REWARD_MODE="dense"
OUTPUT_DIR="processedTrainingDemos"

# Check for command line argument for reward mode
if [ "$1" == "--reward-mode=sparse" ]; then
    REWARD_MODE="sparse"
    OUTPUT_DIR="processedTrainingDemosSparse"
fi

ENV="TurnFaucet-v0"

cd ManiSkill2-Learn

# Convert observation mode with the specified reward mode
if [ -f "../ManiSkill2/demos/v0/rigid_body/$ENV/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5" ]; then
 echo "observation converted trajectory already exists"
else
 echo "converting observation mode"
 cd ../ManiSkill2-Learn
 python tools/convert_state.py \
 --env-name=$ENV \
 --num-procs=10 \
 --traj-name=../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.h5 \
 --json-name=../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.json \
 --output-name=../ManiSkill2/demos/v0/rigid_body/$ENV/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
 --control-mode=pd_ee_delta_pose \
 --max-num-traj=-1 \
 --obs-mode=pointcloud \
 --reward-mode=$REWARD_MODE \
 --obs-frame=ee \
 --n-points=1200
fi

# Shuffle pointcloud demos
if [ -f "../ManiSkill2/demos/v0/rigid_body/$ENV/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5" ]; then
 echo "shuffled trajectory already exists"
else
 echo "shuffling trajectories"
 cd ../ManiSkill2-Learn
 python tools/shuffle_demo.py \
 --source-file ../ManiSkill2/demos/v0/rigid_body/$ENV/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
 --target-file ../ManiSkill2/demos/v0/rigid_body/$ENV/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5
fi

