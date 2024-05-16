import os.path as osp, os, cv2
import json
import numpy as np
from pathlib import Path
from maniskill2_learn.env.env_utils import build_env
from maniskill2_learn.utils.data import GDict
from maniskill2_learn.utils.meta import mkdir_or_exist, ConfigDict
from copy import deepcopy
from transforms3d.axangles import axangle2mat
import sapien.core as sapien
import copy
from scipy.spatial.transform import Rotation
import argparse


def get_reset_kwargs_from_json(json_name):
    with open(json_name, "r") as f:
        json_file = json.load(f)
    reset_kwargs = {}
    for d in json_file["episodes"]:
        episode_id = d["episode_id"]
        r_kwargs = d["reset_kwargs"]
        reset_kwargs[episode_id] = r_kwargs
    return reset_kwargs


def requires_rollout_w_actions(trajectory):
    keys = trajectory.keys()
    assert "env_states" in keys or "env_init_state" in keys
    return (not 'env_states' in keys)


def render_trajectories(trajectory_file, json_name, env_name, control_mode, video_dir):
    reset_kwargs = get_reset_kwargs_from_json(json_name)
    trajectories = GDict.from_hdf5(trajectory_file, wrapper=False)
    env = build_env(ConfigDict(
        {"type": "gym", "env_name": env_name, "control_mode": control_mode})
    )
    if not osp.exists(video_dir):
        os.makedirs(video_dir)

    for traj_name in trajectories:
        trajectory = trajectories[traj_name]
        traj_idx = eval(traj_name.split("_")[1])
        env.reset(**reset_kwargs[traj_idx])

        rrwa = requires_rollout_w_actions(trajectory)
        if rrwa:
            state = trajectory["env_init_state"]
            length = trajectory["actions"].shape[0] + 1
        else:
            state = trajectory["env_states"]
            length = state.shape[0]
        img = env.render()

        video_writer = cv2.VideoWriter(osp.join(video_dir, f"{traj_idx}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (img.shape[1], img.shape[0]))
        for j in range(length):
            if not rrwa:
                env.set_state(state[j])
            else:
                if j == 0:
                    pass# env.set_state(state)
                else:
                    _ = env.step(trajectory["actions"][j - 1])
            img = env.render()
            img = img[..., ::-1]
            video_writer.write(img)
        video_writer.release()


def render_with_o3d(trajectory_file, env_configs, json_name=None, traj_id=0):
    data = GDict.from_hdf5(trajectory_file)
    # data2 = GDict.from_hdf5("/home/erikk/MasterThesis/ManiSkill2/demos/v0/rigid_body/PickSingleYCB-v0/002_master_chef_can.h5")

    reset_kwargs = get_reset_kwargs_from_json(json_name)
    env = build_env(ConfigDict(**env_configs))

    from pynput import keyboard
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3,3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3,3]))
    vis.add_geometry(geometry)    

    trajectory = data[f'traj_{traj_id}']
    #trajectory_raw = data2[f'traj_{traj_id}']
    #rrwa = requires_rollout_w_actions(trajectory_raw)
    rrwa = requires_rollout_w_actions(trajectory)
    if not rrwa:
        env_states = trajectory['env_states']
        length = env_states.shape[0]
    else:
        env_states = trajectory['env_init_state']
        length = trajectory['actions'].shape[0] + 1

    env.reset(**reset_kwargs[traj_id])

    idx = 0
    def on_press(key):
        nonlocal idx
        if hasattr(key, 'char'):
            if key.char in ['n']:
                idx = idx + 1
            elif key.char in ['l']:
                if rrwa:
                    print("Cannot go back to the previous frame because env_states is not given for every step.")
                else:
                    idx = max(idx - 1, 0)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()     

    print("Press 'n' for next frame, 'l' for previous frame, 'h' for Open3d help")
    while idx < length:
        if not rrwa:
            env.set_state(env_states[idx])
        else:
            if idx == 0:
                env.set_state(env_states)
            else:
                env.step(trajectory['actions'][idx - 1])
        obs = env.get_obs()
        geometry.points = o3d.utility.Vector3dVector(obs['xyz'])
        geometry.colors = o3d.utility.Vector3dVector(obs['rgb'])
        vis.update_geometry(geometry)
        old_idx = idx
        while idx == old_idx:
            vis.poll_events()
            vis.update_renderer()


def clip_and_scale_action(action):
    def scale(action, low, high):
        """Clip action to [-1, 1] and scale according to a range [low, high]."""
        low, high = np.asarray(low), np.asarray(high)
        action = np.clip(action, -1, 1)
        return 0.5 * (high + low) + 0.5 * (high - low) * action

    pos_action = scale(action[:3], np.ones(3)*(-0.1), np.ones(3)*0.1)  # low and high are taken from Controller in ManiSkill
    rot_action = action[3:]
    rot_norm = np.linalg.norm(rot_action)
    if rot_norm > 1:
        rot_action = rot_action / rot_norm
    rot_action = rot_action * 0.1  # here also taken from ManiSkill
    return np.hstack([pos_action, rot_action])


def waypoint_to_point_and_rotationMatrix(waypoint):
    # Split the waypoint into position and axis-angle orientation
    position = waypoint[0]
    axis_angle = waypoint[1]

    # Convert axis-angle to a rotation matrix
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle if angle != 0 else axis_angle
    rotation_matrix = Rotation.from_rotvec(axis_angle).as_matrix()

    return position, rotation_matrix


# this is how the target pose is constructed from the actions in ManiSkill Controller
# when using ee_delta_pose controller
def ee_delta_pose(waypoints, ee_pose_in_base):

    transformed_wps = []

    for i, waypoint in enumerate(waypoints):
        ee_pose = sapien.Pose(ee_pose_in_base[i][:3], ee_pose_in_base[i][3:])
        scaled_wp = clip_and_scale_action(waypoint)
        delta_pos, delta_rot = scaled_wp[0:3], scaled_wp[3:]
        delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
        delta_pose = sapien.Pose(delta_pos, delta_quat)
        target_pose = ee_pose * delta_pose
        transformed_wp = [target_pose.p, Rotation.from_quat(target_pose.q[[1, 2, 3, 0]]).as_rotvec()]
        transformed_wps.append(transformed_wp)

    return transformed_wps


# this is how the target pose is constructed from the actions in ManiSkill Controller
# when using ee_target_delta_pose controller
def ee_target_delta_pose(waypoints, startTrafo):
    quat = Rotation.from_matrix(startTrafo[:3, :3]).as_quat()[[3, 0, 1, 2]]
    pos = startTrafo[:3, 3]
    start_pose = sapien.Pose(pos, quat)
    prev_target = copy.deepcopy(start_pose)

    transformed_wps = []
    for i, waypoint in enumerate(waypoints):
        scaled_wp = clip_and_scale_action(waypoint)
        delta_pos, delta_rot = scaled_wp[0:3], scaled_wp[3:]
        delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
        delta_pose = sapien.Pose(delta_pos, delta_quat)
        target_pose_base_frame = prev_target * delta_pose
        transformed_wp = [target_pose_base_frame.p, Rotation.from_quat(target_pose_base_frame.q[[1, 2, 3, 0]]).as_rotvec()]
        transformed_wps.append(transformed_wp)
        prev_target = copy.deepcopy(target_pose_base_frame)

    return transformed_wps


# this constructs the actual pose of the end effector, not the waypoints given by the agent
def ee_pose_to_waypoints(ee_pose_in_base):
    transformed_wps = []
    for i, ee_points in enumerate(ee_pose_in_base):
        ee_pose_base = sapien.Pose(ee_points[:3], ee_points[3:])  # sapien pose from vectorize ee pose
        trafo = ee_pose_base.to_transformation_matrix()  # construct transformation matrix
        rot_vec = Rotation.from_matrix(trafo[:3, :3]).as_rotvec()  # axis angle representation of rotation
        wp_pos = trafo[:3, 3]  # position
        transformed_wp = [wp_pos, rot_vec]  # construct waypoint
        transformed_wps.append(transformed_wp)

    return transformed_wps

# this is how the pose is constructed in the baseline. And according to ManiSkill documentation this should also be how
# the controller construct the next waypoint based on the action
def waypoint_base_from_start_pose(start_pose, waypoints):
    # start_pose is the transformation matrix from gripper/ee frame to base frame
    # the waypoints are given in ee frame!
    trafo = copy.deepcopy(start_pose)
    transformed_wps = []

    for i, waypoint in enumerate(waypoints):
        scaled_waypoint = clip_and_scale_action(waypoint)
        rotation_matr = trafo[:3, :3]
        # first transform translation from ee frame to base frame using the rotation part of trafo
        rotated_translation = rotation_matr @ np.array(scaled_waypoint[:3])
        # add translation to tranlational part of trafo
        trafo[:3, 3] += rotated_translation
        # the rotation part of waypoint is relative rotation of ee gripper
        r = Rotation.from_rotvec(scaled_waypoint[3:]).as_matrix()
        temp = np.eye(3)
        temp = trafo[:3, :3] @ r
        trafo[:3, :3] = temp

        # its important to take deepcopy else we overwrite the wp_pos
        wp_pos = copy.deepcopy(trafo[:3, 3])
        rot_mat = copy.deepcopy(temp)
        # convert to axis angle and store the transformed waypoint
        wps_rot = Rotation.from_matrix(rot_mat).as_rotvec()
        wp = [wp_pos, wps_rot]
        transformed_wps.append(wp)

    return transformed_wps

def percentile_scaling(aff):
    # Use percentiles to stretch the range of affordances
    print('USING PERCENTILE SCALING')
    low_percentile = np.percentile(aff, 5)  # e.g., 10th percentile
    high_percentile = np.percentile(aff, 95)  # e.g., 90th percentile

    affordances_normalized = (aff - low_percentile) / (high_percentile - low_percentile)
    affordances_normalized = np.clip(affordances_normalized, 0, 1)

    return affordances_normalized

def log_scaling(aff):
    print('USING LOGARITHMIC SCALING')
    epsilon = 0.001  # Small constant to avoid log(0)
    affordances_scaled = np.log(aff + epsilon)

    # Normalize to the range [0, 1]
    affordances_scaled = (affordances_scaled - np.min(affordances_scaled)) / (
                np.max(affordances_scaled) - np.min(affordances_scaled))

    return affordances_scaled


def render_traj_VAT(trajectory_file,
                    render_gt_ee_pose=True,
                    render_closed_loop_pose=True,
                    render_open_loop_pose=True,
                    control_type=None,
                    render_pointcloud=True,
                    render_cp=True,
                    render_start_pose=True,
                    render_gt_start_pose=True,
                    use_contact_normal=False,
                    ):

    data = GDict.from_hdf5(trajectory_file)
    print(data.keys())  # prints 'traj_0', 'traj_1', etc

    from pynput import keyboard
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3, 3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3, 3]))
    vis.add_geometry(geometry)

    for traj in data.keys():
        if traj == 'meta':
            continue
        trajectory = data[traj]
        # length = trajectory['actions'].shape[0] if isinstance(trajectory['actions'], np.ndarray) else trajectory['actions']['action'].shape[0]
        num_vis_steps = 0

        def on_press(key):
            nonlocal num_vis_steps
            if hasattr(key, 'char'):
                if key.char in ['n']:
                    num_vis_steps = num_vis_steps + 1

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        #if trajectory['infos']['is_contact_point'][-1]:
            #continue

        # print("Press 'n' for next frame, 'l' for previous frame, 'h' for Open3d help")
        while num_vis_steps < 1:

            if trajectory['infos'].get('pointcloud', None) is not None:
                # this means we are rendering a trajectory obtained through Data Collection with RL agent
                # in this case need to grab the information from the first time step, i.e. at idx = 0
                xyz = trajectory['infos']['pointcloud'][0]
                affordance = None
                first_index = 0
            else:
                # in this case the trajectory was obtained with the VAT agent
                # in the case the first action positions the gripper at the contact point
                # at idx 0 the gripper is in neutral position
                # for trajectory construct we thus need to infos from idx 1
                xyz = trajectory['obs']['xyz'][0]
                affordance = trajectory['infos']['affordance'][0]
                # affordance = log_scaling(affordance)
                # in VAT inference many values are set after the first action!
                first_index = 1


            success = trajectory['infos']['success'][-1]

            contact_point_base = trajectory['infos']['init_contact_point_base'][first_index]
            initial_tcp_pose = trajectory['infos']['tcp_pose'][first_index]

            not_contact_points_mask = ~(np.all(np.isclose(xyz, contact_point_base, atol=1e-9), axis=1))
            xyz = xyz[not_contact_points_mask]
            if affordance is not None:
                affordance = affordance[not_contact_points_mask]

            gripper_up_dir_base = trajectory['infos']['gripper_up_dir'][first_index]  # z axis
            gripper_forward_dir_base = trajectory['infos']['gripper_forward_dir'][first_index]  # x axis
            contact_normal_world = trajectory['infos']['contact_normal_world'][first_index]
            waypoints = trajectory['infos']['waypoints'][-1]
            #actions = trajectory['actions']['action']

            # some rearangement of the list for better handling afterward
            waypoint_dim = 6
            # if the trajectory was successful and the episode terminated early, we need to cut the zero padding
            zero_idx = len(waypoints)
            for idx, num in enumerate(waypoints):
                if num == 0:
                    zero_idx = idx
                    if sum(waypoints[idx:(idx + 3)]) == 0:
                        break
            waypoints = waypoints[:zero_idx]
            init_num_waypoints = int(len(waypoints) / waypoint_dim)
            # list of lists. Each sublist contains 6 numbers, first three are position, last three axis-angle
            waypoints = [waypoints[i * waypoint_dim:(i + 1) * waypoint_dim] for i in range(0, init_num_waypoints)]

            # these two metrics are constant all the time
            action_mode = 'pull' if trajectory['infos']['action_mode'][0] == 1 else 'push'
            base_pose = trajectory['infos']['base_pose'][0]
            to_base = sapien.Pose(base_pose[:3], base_pose[3:]).inv().to_transformation_matrix()

            start_pose = np.eye(4)
            # the start_pose is the transformation matrix from gripper/ee frame to base frame!
            # construct start pose from gripper directions and contact point
            # the gripper direction are given in world frame, which is the same as base frame, because they are
            # only translated but not rotated relative to another
            start_pose[:3, :3] = np.stack([gripper_forward_dir_base, np.cross(gripper_up_dir_base, gripper_forward_dir_base), gripper_up_dir_base], axis=1)
            start_pose[:3, 3] = contact_point_base

            if use_contact_normal:
                #contact_normal_base = to_base[:3, :3] @ contact_normal_world

                start_pose[:3, 3] += 0.05 * contact_normal_world
            else:
                # I could determine whether to do + or - base on distance to other points in pointcloud
                # I need to increase the distance to the faucet handle
                start_pose[:3, 3] -= 0.05 * gripper_forward_dir_base  # this is tricky because it could also be +=

            # construct start pose directly from robot proprioception
            # by transforming initial tcp pose from world to base coordinates
            # initial_tcp_pose_base = sapien.Pose(base_pose[:3], base_pose[3:]).inv().transform(sapien.Pose(initial_tcp_pose[:3], initial_tcp_pose[3:]))
            initial_tcp_pose_base = trajectory['infos']['ee_pose_at_base'][0]
            initial_tcp_pose_base = sapien.Pose(initial_tcp_pose_base[:3], initial_tcp_pose_base[3:])
            gt_start_pose = initial_tcp_pose_base.to_transformation_matrix()

            # this renders the gripper orientation axis at the start
            # we render two start poses. One derived from the contact point and gripper axes
            # the other one directly uses the initial tcp pose transformed to base frame
            start_visual_list = [[] for _ in range(3)]
            gt_start_visual_list = [[] for _ in range(3)]
            num_points = 8  # number of points to visualize each axis
            for j in range(3):  # we have three axes...
                for i in range(num_points):
                    delta = (i + 1) * 0.005
                    point = np.array([start_pose[0, 3] + delta * start_pose[0, j],
                                      start_pose[1, 3] + delta * start_pose[1, j],
                                      start_pose[2, 3] + delta * start_pose[2, j]])
                    gt_point = np.array([gt_start_pose[0, 3] + delta * gt_start_pose[0, j],
                                         gt_start_pose[1, 3] + delta * gt_start_pose[1, j],
                                         gt_start_pose[2, 3] + delta * gt_start_pose[2, j]])
                    start_visual_list[j].append(point)
                    gt_start_visual_list[j].append(gt_point)

            start_visual_list = np.vstack(start_visual_list)
            gt_start_visual_list = np.vstack(gt_start_visual_list)

            rgb_start_pose_forward = np.tile(np.array([1, 0, 0]), (start_visual_list.shape[0]//3, 1))  # r, x
            rgb_start_pose_right = np.tile(np.array([0, 1, 0]), (start_visual_list.shape[0]//3, 1))  # g, y
            rgb_start_pose_down = np.tile(np.array([0, 0, 1]), (start_visual_list.shape[0]//3, 1))  # b, z
            rgb_start_pose = np.vstack([rgb_start_pose_forward, rgb_start_pose_right, rgb_start_pose_down])

            rgb_gt_start_pose = np.tile(np.array([0, 1, 0]), (gt_start_visual_list.shape[0], 1))  # green

            # for debugging I also output the true end effector pose of the robot in base coordinates
            # this pose is different every time step, because the robot moves
            # in theory the ee_pose_at_base should align with the waypoints_at_base
            # because the waypoints mark the target pose the low-level controller aims to reach
            # the ee_pose_at_base is always one step 'behind' the waypoints, because it is
            # computed BEFORE the robot moves
            ee_pose_at_base = []#[trajectory['infos']['ee_pose_at_base'][0]
            for i in range(0, init_num_waypoints):
                initial_tcp_pose_base = sapien.Pose(base_pose[:3], base_pose[3:]).inv().transform(
                    sapien.Pose(trajectory['infos']['tcp_pose'][i][:3], trajectory['infos']['tcp_pose'][i][3:]))

                ee_pose_at_base.append(np.hstack([initial_tcp_pose_base.p, initial_tcp_pose_base.q]))
                #ee_pose_at_base.append(trajectory['infos']['ee_pose_at_base'][i])

            # here we compute the waypoints in base frame
            # we use different methods to see where the differences are and for debugging
            # the main observation so far is the ee_pose_as_base does NOT coincide with the waypoints
            # I suppose this is because the low-level controller does not reach the target pose
            # because the inverse kinematics do NOT take into account the forces through contact with the faucet
            # we use two different methods to compute the waypoint_pose:
            # 1. waypoint_base_from_start_pose: uses the start pose to incrementally add a
            # delta position and delta rotation. This is how the baseline does it.
            # 2. waypoint_base_from_ee_pose: uses the previous ee_pose_at_base and adds delta position and delta rotation
            # this is essentially a closed loop approach, because it uses the last true pose and adds the delta, instead
            # of adding all the delta on top of each other starting from the start pose. This is method aligns with
            # how the agent's actions are converted to target poses in the low-level controller when using pd_ee_delta_pose
            # I would have assumed that all three methods to compute the waypoints result in the same trajectory
            # BUT THEY DO NOT!

            waypoints_pose_base_frame = []
            colors = []
            if render_gt_ee_pose and success:
                waypoints_pose_base_frame += ee_pose_to_waypoints(ee_pose_at_base)
                colors.append(np.array([213, 230, 101]) / 255)  # lime
            if render_closed_loop_pose and control_type == 'ee_delta_pose':
                waypoints_pose_base_frame += ee_delta_pose(waypoints, ee_pose_at_base)
                colors.append(np.array([1, 0, 1]))  # magenta
            if render_open_loop_pose and control_type == 'ee_target_delta_pose':
                waypoints_pose_base_frame += ee_target_delta_pose(waypoints, gt_start_pose)
                colors.append(np.array([226, 125, 9]) / 255)  # oragange/brownish

            # convert the waypoints in list of rotation and positions
            # this is needed for the visualization logic
            wp_rot_list = []
            wp_pos_list = []
            for wp in waypoints_pose_base_frame:
                position, rotation_matrix = waypoint_to_point_and_rotationMatrix(wp)
                wp_rot_list.append(rotation_matrix)
                wp_pos_list.append(position)

            # as we concatenated the outputs of the waypoint calculations we now have more waypoints...
            num_waypoints = len(wp_rot_list)
            wp_visual_list = [[] for _ in range(num_waypoints)]

            # this renders waypoints
            for j in range(num_waypoints):  # loop over waypoints
                for k in range(3):  # loop over the three axes per waypoints
                    for i in range(num_points):  # every axis is rendered using num_points points
                        delta = (i + 1) * 0.005
                        point = np.array([wp_pos_list[j][0] + delta * wp_rot_list[j][0, k],
                                          wp_pos_list[j][1] + delta * wp_rot_list[j][1, k],
                                          wp_pos_list[j][2] + delta * wp_rot_list[j][2, k]])
                        wp_visual_list[j].append(point)
            if len(wp_visual_list) > 0:
                wp_visual_list = np.vstack(wp_visual_list)

            ## RGB VALUES
            rgb_pointcloud = (np.ones_like(xyz) / 2)  # faucet and robot
            if affordance is not None:
                """if action_mode == 'pull':
                    rgb_pointcloud[:, 0] = 1 - affordance[:, 0]  # Red channel proportional to affordance
                    rgb_pointcloud[:, 1] = 1 - affordance[:, 0]  # Green channel inversely proportional
                    rgb_pointcloud[:, 2] = affordance[:, 0]
                elif action_mode == 'push':
                    rgb_pointcloud[:, 0] = 1 - affordance[:, 0]
                    rgb_pointcloud[:, 1] = affordance[:, 0]
                    rgb_pointcloud[:, 2] = 1 - affordance[:, 0]"""

                # For low affordance, start with a base turquoise color
                # Adjust the base turquoise color according to your preference
                base_turquoise = np.array([0, 0.5, 0.8])  # Less red, equal parts of green and blue

                # Adjusting colors based on the action mode and affordance values
                if action_mode == 'pull':
                    # Transition from turquoise to purple/pink for pulling
                    # Purple/pink can be achieved with high red and blue, low green
                    rgb_pointcloud[:, 0] = base_turquoise[0] + affordance[:, 0] * 1.0  # Increase red towards purple/pink
                    rgb_pointcloud[:, 1] = base_turquoise[1] * (
                                1 - affordance[:, 0])  # Decrease green when affordance increases
                    rgb_pointcloud[:, 2] = base_turquoise[2] + affordance[:, 0] * 0.5  # Adjust blue as needed
                elif action_mode == 'push':
                    # Transition from turquoise to greenish for pushing
                    # Greenish colors have high green, lower red and blue
                    rgb_pointcloud[:, 0] = base_turquoise[0] * (
                                1 - affordance[:, 0])  # Decrease red when affordance increases
                    rgb_pointcloud[:, 1] = base_turquoise[1] + affordance[:, 0] * 1.0  # Increase green towards greenish
                    rgb_pointcloud[:, 2] = base_turquoise[2] * (
                                1 - affordance[:, 0])  # Decrease blue when affordance increases

            if action_mode == 'pull':
                rgb_cp_base = np.array([1, 0, 1])  # contact point at correct location, i.e. in base frame
            elif action_mode == 'push':
                rgb_cp_base = np.array([0, 1, 0])  # contact point at correct location, i.e. in base frame

            rgb_waypoints = []
            # we want each waypoint to have a color
            # we want waypoints from the same method (ee_pose_to_waypoints, waypoint_base_from_start_pose..) to have
            # same shade of color. We have colors array that hold the base color, we dim the color with each consecutive
            # waypoint to make the sequential order of waypoint better visible
            # each waypoint consists of three axes, each is rendered using num_points many points, thus we need
            # 3*num_points many rgb copies to color all points in the waypoint
            num_rgb_points_per_color = 3 * num_points
            for j in range(int(num_waypoints / init_num_waypoints)):
                for i in range(init_num_waypoints):
                    color = colors[j] / (1 + (0.5 * i))  # make color a little darker with every waypoint
                    rgb_wp = np.tile(color, (num_rgb_points_per_color, 1))
                    rgb_waypoints.append(rgb_wp)
            if len(rgb_waypoints) > 0:
                rgb_waypoints = np.vstack(rgb_waypoints)

            points = []
            rgb = []

            if render_pointcloud:
                rgb.append(rgb_pointcloud)
                points.append(xyz)
            if render_cp:
                rgb.append(rgb_cp_base)
                points.append(contact_point_base)
            if render_start_pose:
                rgb.append(rgb_start_pose)
                points.append(start_visual_list)
            if render_gt_start_pose:
                rgb.append(rgb_gt_start_pose)
                points.append(gt_start_visual_list)

            if len(wp_visual_list) > 0:
                rgb.append(rgb_waypoints)
                points.append(wp_visual_list)

            points = np.vstack(points)
            rgb = np.vstack(rgb)

            geometry.points = o3d.utility.Vector3dVector(points)
            geometry.colors = o3d.utility.Vector3dVector(rgb)

            vis.update_geometry(geometry)
            old_num_vis_steps = num_vis_steps
            while num_vis_steps == old_num_vis_steps:
                vis.poll_events()
                vis.update_renderer()

        print(f'next trajectory')


def render_with_o3d_random_trajectory(env_configs):
    env = build_env(ConfigDict(**env_configs))

    from pynput import keyboard
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3,3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3,3]))
    vis.add_geometry(geometry)
    env.reset()

    idx = 0
    def on_press(key):
        nonlocal idx
        if hasattr(key, 'char'):
            if key.char in ['n']:
                idx = idx + 1

    listener = keyboard.Listener(on_press=on_press)
    listener.start()     

    print("Press 'n' for next frame, 'h' for Open3d help")
    while True:
        env.step(env.action_space.sample())
        obs = env.get_obs()
        geometry.points = o3d.utility.Vector3dVector(obs['xyz'])
        geometry.colors = o3d.utility.Vector3dVector(obs['rgb'])
        vis.update_geometry(geometry)
        old_idx = idx
        while idx == old_idx:
            vis.poll_events()
            vis.update_renderer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render trajectories')
    parser.add_argument('--file', type=str, help='h5 file containing trajectories')
    parser.add_argument('--control_type', type=str, choices=['ee_target_delta_pose', 'ee_delta_pose'],
                        help='control mode the agent uses: pd_ee_target_delta_pose or pd_ee_delta_pose')
    parser.add_argument("--gt", type=int, choices=[0, 1], default=1, help="visualize ground truth ee pose")
    parser.add_argument("--open_loop", type=int, choices=[0, 1], default=1, help="visualize open loop action pose")
    parser.add_argument("--closed_loop", type=int, choices=[0, 1], default=1, help="visualize closed loop action pose")
    parser.add_argument("--render_pointcloud", type=int, choices=[0, 1], default=1, help="visualize closed loop action pose")
    parser.add_argument("--render_cp", type=int, choices=[0, 1], default=1, help="visualize closed loop action pose")
    parser.add_argument("--render_start_pose", type=int, choices=[0, 1], default=1, help="visualize closed loop action pose")
    parser.add_argument("--render_gt_start_pose", type=int, choices=[0, 1], default=1, help="visualize closed loop action pose")
    parser.add_argument("--use_contact_normal", type=int, choices=[0, 1], default=1, help="use contact normal to construct firs pose")

    args = parser.parse_args()

    render_traj_VAT(trajectory_file=args.file,
                    render_gt_ee_pose=bool(args.gt),
                    render_open_loop_pose=bool(args.open_loop),
                    render_closed_loop_pose=bool(args.closed_loop),
                    control_type=args.control_type,
                    render_pointcloud=bool(args.render_pointcloud),
                    render_cp=bool(args.render_cp),
                    render_start_pose=bool(args.render_start_pose),
                    render_gt_start_pose=bool(args.render_gt_start_pose),
                    use_contact_normal=bool(args.use_contact_normal)
                    )

