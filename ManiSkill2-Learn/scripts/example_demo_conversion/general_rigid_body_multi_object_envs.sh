#!/bin/bash

# this script will merge all training demonstrations into one h5 file
# will convert the merged trajectories to control mode pd_ee_delta_pose
# will then convert the trajectory to pointcloud observations
# final demonstrations trajectories will be stored

# For general envs with multiple objects, where each object has its own demos,
# and goal information is not provided in the observation
# i.e. TurnFaucet-v0; for ManiSkill1 envs (OpenCabinetDoor/Drawer, PushChair, MoveBucket), please use maniskill1.sh instead of this script)
ENV="TurnFaucet-v0"

# assuming you are in ManiSkill2-Learn directory
if [ -d "../ManiSkill2/demos/v0/rigid_body/$ENV/" ]; then
 echo "demos already exist"
else
 echo "demo will be downloaded"
 python ../ManiSkill2/mani_skill2/utils/download_demo.py $ENV -o ../ManiSkill2/demos
fi


if [ -d "../ManiSkill2/data/partnet_mobility" ]; then
 echo "data for TurnFaucet environment already exists"
else
 echo "dataset for TurnFaucet environment will be downloaded"
 python ../ManiSkill2/mani_skill2/utils/download_asset.py "TurnFaucet-v0"
fi

#curate dataset (use the txt.files with trainingset and holdoutset for that)
python ../copyDemonstrations.py \
-i ../ManiSkill2/demos/v0/rigid_body/$ENV/ \
-o ../ManiSkill2/demos/v0/rigid_body/${ENV}_filteredTrainingDemos/ \
-f ../ManiSkill2/faucetTrainingDataSet.txt

# assuming you are in ManiSkill2-Learn directory
python tools/merge_trajectory.py \
-i ../ManiSkill2/demos/v0/rigid_body/${ENV}_filteredTrainingDemos/ \
-o ../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.h5 \
-p *.h5 # this can be replaced with other patterns


# Inside the ManiSkill2's directory, run replay_trajectory.py. See wiki page
# of ManiSkill2 for more information.
cd ../ManiSkill2
python mani_skill2/trajectory/replay_trajectory.py --num-procs 10 \
--traj-path demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.h5 \
--save-traj \
--target-control-mode pd_ee_delta_pose \
--obs-mode none \
--use-env-states

# Inside ManiSkill2-Learn's directory, run convert_state.py to generate visual observations
# for the demonstrations.
cd ../ManiSkill2-Learn
# Generate pointcloud demo
python tools/convert_state.py \
--env-name=$ENV \
--num-procs=10 \
--traj-name=../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.h5  \
--json-name=../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.json \
--output-name=../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200

# Generate rgbd demo 
#python tools/convert_state.py \
#--env-name=$ENV \
#--num-procs=12 \
#--traj-name=../ManiSkill2/demos/v0/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.h5 \
#--json-name=../ManiSkill2/demos/v0/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose.json \
#--output-name=../ManiSkill2/demos/v0/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd.h5 \
#--control-mode=pd_ee_delta_pose \
#--max-num-traj=-1 \
#--obs-mode=rgbd \
#--reward-mode=dense

# Shuffle pointcloud demos
python tools/shuffle_demo.py \
--source-file ../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
--target-file ../ManiSkill2/demos/v0/rigid_body/$ENV/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5

# Shuffle rgbd demos 
#python tools/shuffle_demo.py \
#--source-file ../ManiSkill2/demos/v0/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd.h5 \
#--target-file ../ManiSkill2/demos/v0/rigid_body/$ENV/trajectory_merged.none.pd_ee_delta_pose_rgbd_shuffled.h5
