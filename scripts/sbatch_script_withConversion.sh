#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --error=sbatch_log/%j.err
#SBATCH  --gres=gpu:4
#SBATCH  --mem=50G
#SBATCH  --constraint='geforce_rtx_2080_ti|geforce_gtx_titan_x'
source /scratch_net/got/ekrauter/conda/etc/profile.d/conda.sh
conda activate lab_mani_skill2
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

set -o errexit
# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Run Demonstration Generation Script
cd ManiSkill2-Learn
bash scripts/example_demo_conversion/TurnFaucet_CreateDemos.sh

# Wait for the file to be created by TurnFaucet_CreateDemos.sh
FILE="../ManiSkill2/demos/v0/rigid_body/TurnFaucet-v0/processedTrainingDemos/training_trajectories_merged.none.pd_ee_delta_pose.json"
MAX_WAIT_TIME=60 # maximum time to wait in seconds
WAIT_INTERVAL=5  # how long to wait between checks in seconds
elapsed_time=0    # time elapsed

while [ ! -f "$FILE" ]; do
    if (( elapsed_time >= MAX_WAIT_TIME )); then
        echo "Error: Timeout waiting for file $FILE to be created."
        exit 1 # exit the script with an error code
    fi

    echo "Waiting for file $FILE to be created... ($elapsed_time seconds elapsed)"
    sleep $WAIT_INTERVAL
    ((elapsed_time += WAIT_INTERVAL))
done

echo "File $FILE found, proceeding to run the Python script."

# Run python script
python -u maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn_cluster.py --work-dir ../PPO_DAPG_Baseline_cluster --gpu-ids 0 1 --sim-gpu-ids 2 3

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
