#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo $SCRIPT_DIR
MASTERTHESIS_DIR=$(dirname "$(dirname "$SCRIPT_DIR")") #$(dirname "$SCRIPT_DIR")
echo $MASTERTHESIS_DIR

MANISKILL2_DIR="${MASTERTHESIS_DIR}/ManiSkill2"
MANISKILL2_LEARN_DIR="${MASTERTHESIS_DIR}/ManiSkill2-Learn"
CREATEDATASET_DIR=$SCRIPT_DIR  #"${MASTERTHESIS_DIR}/CreateDataset"

# this script will download assets for the specified environment
# will download all demonstrations
# will filter out only demonstrations from relevant faucets specified in faucetTrainingDataSet
# merge all training demonstrations into one h5 file
# will convert the merged trajectories to control mode pd_ee_delta_pose
# will then convert the trajectory to pointcloud observations
# final demonstrations trajectories will be stored

# Default reward mode
REWARD_MODE="dense"
OUTPUT_DIR="processedTrainingDemos"

# Check for command line argument for reward mode
if [ "$1" == "--reward-mode=sparse" ]; then
    REWARD_MODE="sparse"
    OUTPUT_DIR="processedTrainingDemosSparse"
fi

ENV="TurnFaucet-v0"

# ASSUMING YOU ARE IN MASTERTHESIS DIRECTORY


# download demos if necessary
if [ -d "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/" ]; then
 echo "demos already exist"
else
 echo "demo will be downloaded"
 python ${MANISKILL2_DIR}/mani_skill2/utils/download_demo.py ${ENV} -o ${MANISKILL2_DIR}/demos
fi

# download assets, i.e. Faucet models, if not already there
if [ -d "${MANISKILL2_DIR}/data/partnet_mobility" ]; then
 echo "data for TurnFaucet environment already exists"
else
 echo "dataset for TurnFaucet environment will be downloaded"
 python ${MANISKILL2_DIR}/mani_skill2/utils/download_asset.py "TurnFaucet-v0"
fi

#curate dataset (use the txt.files with trainingset and holdoutset for that)
if [ -d "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}_filteredTrainingDemos_MultiFaucet/" ]; then
 echo "directory with training demos already exists"
else
 echo "copying training demos into ${ENV}_filteredTrainingDemos_MultiFaucet"

 python ${CREATEDATASET_DIR}/copyDemonstrations.py \
 -i ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/ \
 -o ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}_filteredTrainingDemos_MultiFaucet/ \
 -f ${MANISKILL2_DIR}/faucetTrainingDataSet_MultiFaucet.txt
fi

# merge training trajectories into one h5 file if not already there
if [ -f "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.h5" ]; then
 echo "merged trajectory already exists"
else
 echo "merging trajectories"
 python ${MANISKILL2_LEARN_DIR}/tools/merge_trajectory.py \
 -i ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}_filteredTrainingDemos_MultiFaucet/ \
 -o ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.h5 \
 -p 5*.h5 # this can be replaced with other patterns
fi


# Inside the ${MANISKILL2_DIR}'s directory, run replay_trajectory.py. See wiki page
# of ${MANISKILL2_DIR} for more information.
if [ -f "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.h5" ]; then
 echo "control converted trajectory already exists"
else
 echo "converting control mode"

 python ${MANISKILL2_DIR}/mani_skill2/trajectory/replay_trajectory.py --num-procs 10 \
 --traj-path ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.h5 \
 --save-traj \
 --target-control-mode pd_ee_delta_pose \
 --obs-mode none \
 --use-env-states
fi

# Inside ${MANISKILL2_LEARN_DIR}'s directory, run convert_state.py to generate visual observations
# for the demonstrations.
if [ -f "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5" ]; then
 echo "observation converted trajectory already exists"
else
 echo "converting observation mode"
 python ${MANISKILL2_LEARN_DIR}/tools/convert_state.py \
 --env-name=${ENV} \
 --num-procs=10 \
 --traj-name=${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.h5  \
 --json-name=${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.json \
 --output-name=${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
 --control-mode=pd_ee_delta_pose \
 --max-num-traj=-1 \
 --obs-mode=pointcloud \
 --reward-mode=$REWARD_MODE \
 --obs-frame=ee \
 --with-next \
 --n-points=1200
fi

# Shuffle pointcloud demos
if [ -f "${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5" ]; then
 echo "shuffled trajectory already exists"
else
 echo "shuffling trajectories"
 python ${MANISKILL2_LEARN_DIR}/tools/shuffle_demo.py \
 --source-file ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud.h5 \
 --target-file ${MANISKILL2_DIR}/demos/v0/rigid_body/${ENV}/${OUTPUT_DIR}/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5
fi
