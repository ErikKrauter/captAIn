#!/bin/bash

# Usage: ./submit_job.sh --num-gpus <NUM_GPUS> --script <script_name> --job-name <job_name> [additional_arguments]

# Default values
NUM_GPUS=""
SCRIPT_NAME=""
JOB_NAME=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num-gpus) NUM_GPUS="$2"; shift ;;
        --script) SCRIPT_NAME="$2"; shift ;;
        --job-name) JOB_NAME="$2"; shift ;;
        *) break ;;
    esac
    shift
done

# Check if required arguments are provided
if [ -z "$NUM_GPUS" ] || [ -z "$SCRIPT_NAME" ] || [ -z "$JOB_NAME" ]; then
    echo "Error: Required arguments not specified"
    echo "Usage: $0 --num-gpus <NUM_GPUS> --script <script_name> --job-name <job_name> [additional_arguments]"
    exit 1
fi

# Create a SLURM script with the specified number of GPUs and job name
cat << EOF > generated_slurm_script.sh
#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --error=sbatch_log/%j.err
#SBATCH --gres=gpu:$NUM_GPUS
#SBATCH --exclude=biwirender11,biwirender12
#SBATCH --mem=50G
#SBATCH --job-name=$JOB_NAME
#SBATCH --constraint='geforce_rtx_2080_ti|geforce_gtx_titan_x'

bash $SCRIPT_NAME $@
EOF

chmod +x generated_slurm_script.sh
chmod +x $SCRIPT_NAME

# Submit the generated script to SLURM
sbatch generated_slurm_script.sh
