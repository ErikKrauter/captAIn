# Evaluation


This folder contains some helpful python files for visualizing trajectories and computing dataset metrics.

**render_trajectories.py**:
The render_trajectories.py file creates 3D interactive visualizations of the point cloud, alongside the predicted
affordance maps, trajectories, contact points, gripper orientation. It expects as input a h5 file containing trajectories.
The file is automatically created if save_traj=True is specified in the config file used for inference/rollout.

If the parent directory of the trajectory file contains a folder with videos, the script will automatically render the 
video of the rollout next to the interactive 3D visualization. The videos are automatically created during rollout/inference
if save_video=True is specified in the config file.

All options to adjust what elements to visualize, can be found in the script itself.

**trajectory_analysis.py**:
The trajectory_analysis.py file is useful for computing different metrics and plots given a dataset of trajectories. 
It expects as input a h5 file containing trajectories. The script is automatically executed after every data collection. 
It generates several matplotlib plots which are also automatically uploaded to Weights And Biases (wandb) if a valid wandb run was passed.
All plots and metrics are stored in a subfolder in the same parent directory as the dataset itself.

