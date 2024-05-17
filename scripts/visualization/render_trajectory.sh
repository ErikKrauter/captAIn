#!/bin/bash

# Ensure the script exits on any error
set -e
export PYTHONPATH=$PYTHONPATH:$(python -c 'import os; print(os.getcwd())')
# Define the working directory
FILE="Runs/VATMart_Inference/test/trajectory.h5"

# Run the training command
python Evaluation/render_trajectories.py \
  --file $FILE

# additional configurations are available inside the python script
