import open3d as o3d
import numpy as np
from maniskill2_learn.utils.data import GDict
import sapien.core as sapien
import copy
from scipy.spatial.transform import Rotation
import argparse
import os
from maniskill2_learn.utils.meta import Config
import cv2
import threading
import time

import multiprocessing as mp


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


def percentile_scaling(aff):
    if aff is None:
        return  None
    # Use percentiles to stretch the range of affordances
    print('USING PERCENTILE SCALING')
    low_percentile = np.percentile(aff, 5)  # e.g., 10th percentile
    high_percentile = np.percentile(aff, 95)  # e.g., 90th percentile

    affordances_normalized = (aff - low_percentile) / (high_percentile - low_percentile)
    affordances_normalized = np.clip(affordances_normalized, 0, 1)

    return affordances_normalized

def log_scaling(aff):
    if aff is None:
        return  None
    print('USING LOGARITHMIC SCALING')
    epsilon = 0.001  # Small constant to avoid log(0)
    affordances_scaled = np.log(aff + epsilon)

    # Normalize to the range [0, 1]
    affordances_scaled = (affordances_scaled - np.min(affordances_scaled)) / (
                np.max(affordances_scaled) - np.min(affordances_scaled))

    return affordances_scaled


class VideoPlayer:

    def __init__(self, trajectory_file):
        # Video playback attributes
        if os.path.exists(os.path.join(os.path.dirname(trajectory_file), 'videos/')):
            self.base_video_path = os.path.join(os.path.dirname(trajectory_file), 'videos/')
        else:
            self.base_video_path = None

        self.playback_speed = 1.0 #0.1  # Playback speed

        self.endless_loop = True
        self.index = 0

    def run(self, queue):
        if self.base_video_path is None:
            return
        def play_video(video_path_):
            cap = cv2.VideoCapture(video_path_)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path_}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = int((1 / fps) * 1000 / self.playback_speed)

            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.endless_loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                cv2.imshow('Video', frame)
                if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                    break

                if not queue.empty():
                    self.index = queue.get()
                    print('Video player received message')
                    print(f'Index changed to {self.index}')
                    break

            cap.release()
            #cv2.destroyAllWindows()
            # Notify the application that the video thread has finished

        self.index = queue.get()
        while True:
            video_path = self.base_video_path + f'{self.index}.mp4'
            print(f'Video path is: {video_path}')
            play_video(video_path)


class TrajectoryRenderer:

    def __init__(self, trajectory_file,
                 visualize_waypoints=True,
                 only_first_waypoint=True,
                 num_trajectories=100,
                 render_gt=True,
                 render_predicted_ee_poses=True,
                 render_affordances=True):

        # the data set
        self.data = GDict.from_hdf5(trajectory_file)

        self.data_keys = list(self.data.keys())
        self.current_traj_index = 0
        self.current_traj_key = self.data_keys[self.current_traj_index]
        if self.current_traj_key == 'meta':
            self.current_traj_index += 1
            self.current_traj_key = self.data_keys[self.current_traj_index]

        # Initialize the Open3D GUI application
        self.app = o3d.visualization.gui.Application.instance
        self.app.initialize()

        # the visualizer
        self.vis = o3d.visualization.O3DVisualizer("Trajectory Visualization", 1024, 768)
        self.vis.add_action("Next Trajectory", self.load_next_trajectory)
        self.vis.show_settings = True
        self.vis.show_skybox(False)
        self.vis.show_ground = True
        self.vis.show_axes = True
        # Normalize the color values and ensure the array is of type float32
        #background_color = np.array([98, 98, 98, 255], dtype=np.float32) / 255.0  # Normalize and convert to float32
        background_color = np.array([255, 255, 255, 255], dtype=np.float32) / 255.0  # Normalize and convert to float32
        background_color = background_color.reshape((4, 1))  # Reshape to (4, 1) as expected
        self.vis.set_background(background_color, None)

        # others
        self.scale_factor = 5  # uniformly scale the entire scene. Great for being able to zoom in closer to objects
        self.coordinate_systems_size = 0.03 * self.scale_factor
        self.contact_point_size = 0.005
        self.vis.line_width = 100 * self.scale_factor
        self.vis.point_size = 1 * self.scale_factor
        self.trajectory_opacity = 0.3
        self.best_trajectory_base_color_success = np.array([21, 255, 48]) / 255  # strong green
        self.best_trajectory_base_color_unsuccess = np.array([255, 34, 20]) / 255  # strong red
        self.trajectory_lines_base_color_success = np.array([123, 255, 136]) / 255  # weak green
        self.trajectory_lines_base_color_unsuccess = np.array([255, 98, 84]) / 255  # weak red
        # self.faucet_color = np.array([137, 137, 137]) / 255  # grey
        self.faucet_color = np.array([100, 100, 100]) / 255  # black
        #self.contact_point_color = np.array([96, 108, 255]) / 255
        self.contact_point_color_pull = np.array([231, 26, 239]) / 255  # pinkish
        self.contact_point_color_push = np.array([47, 72, 239]) / 255  # blueish

        self.render_first_time = True
        self.geometry_names = []

        # command line
        self.visualize_waypoints = visualize_waypoints
        self.render_only_first_waypoint = only_first_waypoint
        self.num_trajectories = num_trajectories

        # following attributes will be populated from the meta information of the trajectory file
        self.visualizing_VAT = None  # indicate whether trajectory was produced by VAT agent
        self.VAT_SAC = False
        self.waypoint_dim = None  # action space is either 4d or 6d
        self.num_waypoints = None  # number of waypoints can vary depending on the dataset used to train the agent
        self.read_meta_data(trajectory_file)

        self.render_gt = render_gt
        self.render_predicted_ee_poses = render_predicted_ee_poses
        self.render_affordances = render_affordances


    def read_meta_data(self, trajectory_file):

        if isinstance(trajectory_file, list):
            trajectory_file = trajectory_file[0]

        meta_path = os.path.join(os.path.dirname(trajectory_file), "meta.py")
        meta_cfg = Config.fromfile(meta_path)

        if meta_cfg.agent_cfg.type in ['VAT-Mart', 'VAT-SAC']:
            self.visualizing_VAT = True
            if meta_cfg.agent_cfg.type == 'VAT-SAC':
                self.VAT_SAC = True
        else:
            self.visualizing_VAT = False

        self.num_waypoints = meta_cfg.env_cfg.num_waypoints  # FOR VAT THIS SHOULD BE -1
        self.waypoint_dim = 4 if meta_cfg.env_cfg.restrict_action_space else 6


    def extract_from_sample_RL(self, trajectory):
        self.xyz = trajectory['infos']['pointcloud'][0]
        self.affordance = None
        # affordance = log_scaling(affordance)

        self.contact_point_base = trajectory['infos']['init_contact_point_base'][0]

        # the first waypoint is the initial pose of the gripper
        self.waypoints = trajectory['infos']['waypoints'][-1]  # (16,)

        self.action_mode = 'pull' if trajectory['infos']['action_mode'][0] == 1 else 'push'
        self.success = trajectory['infos']['success'][-1]
        self.target_angle = np.rad2deg(trajectory['infos']['target_angle'][-1])
        self.init_angle = np.rad2deg(trajectory['infos']['init_angle'][-1])

        self.contact_normal_world = trajectory['infos']['contact_normal_world'][0]
        self.gripper_up_dir_base = trajectory['infos']['gripper_up_dir'][0]  # z axis
        self.gripper_forward_dir_base = trajectory['infos']['gripper_forward_dir'][0]  # x axis

        self.tcp_pose_world_after_action = trajectory['infos']['tcp_pose']
        self.pose_base_before_action = trajectory['infos']['ee_pose_at_base']
        self.base_pose = trajectory['infos']['base_pose'][0]

    def extract_from_sample_VAT(self, trajectory):
        self.xyz = trajectory['obs']['xyz'][1]
        self.affordance = trajectory['infos'].get('affordance', None) if self.render_affordances else None
        # self.affordance = log_scaling(self.affordance)
        self.contact_point_base = trajectory['infos']['init_contact_point_base'][0]

        if sum(np.all(np.isclose(self.xyz, self.contact_point_base, atol=1e-9), axis=1)) == 0:
            print('contact point not in point cloud')
        #not_contact_points_mask = ~(np.all(np.isclose(self.xyz, self.contact_point_base, atol=1e-9), axis=1))
        #self.xyz = self.xyz[not_contact_points_mask]
        if self.affordance is not None:
            self.affordance = self.affordance[0]
            #self.affordance = self.affordance[not_contact_points_mask]

        # the first waypoint is the initial pose of the gripper
        if trajectory['infos'].get('waypoints', None) is not None:
            self.waypoints = trajectory['infos']['waypoints'][-1]  # (20,)
        else:
            self.waypoints = trajectory['actions']['action'].flatten()

        self.action_mode = 'pull' if trajectory['infos']['action_mode'][0] == 1 else 'push'
        self.success = trajectory['infos']['success'][-1]
        self.target_angle = np.rad2deg(trajectory['infos']['target_angle'][-1])
        self.init_angle = np.rad2deg(trajectory['infos']['init_angle'][-1])

        if self.num_trajectories > 0:
            # list of 100 trajectories, each 5,5,6 dimensional. For every waypoint its the exact same trajectory
            self.trajectories = np.array(trajectory['actions']['trajectories'])[0, :, :, :] # for all trajectories take first waypoint -> 100, 5, 6

        self.contact_normal_world = trajectory['infos']['contact_normal_world'][0]
        self.gripper_up_dir_base = trajectory['infos']['gripper_up_dir'][0]  # z axis
        self.gripper_forward_dir_base = trajectory['infos']['gripper_forward_dir'][0]  # x axis

        self.tcp_pose_world_after_action = trajectory['infos']['tcp_pose']
        self.pose_base_before_action = trajectory['infos']['ee_pose_at_base']
        self.base_pose = trajectory['infos']['base_pose'][0]
        if self.render_predicted_ee_poses:
            # only need from the last step, because all steps are identical
            if trajectory['actions'].get('pose_trajectory', None) is not None:
                self.predicted_ee_pose_traj = trajectory['actions']['pose_trajectory'][0]  # 9, 6
            elif trajectory['actions'].get('open_loop_trajectory', None) is not None:
                self.predicted_ee_pose_traj = trajectory['actions']['open_loop_trajectory'][0]
            else:
                print('No predicted end effector trajectory found')
                exit()
            if self.predicted_ee_pose_traj.ndim == 1:
                # meaning its a flat array and not an array of waypoints
                # this is the case for open_loop_trajectory
                self.predicted_ee_pose_traj = [self.predicted_ee_pose_traj[i*self.waypoint_dim:(i+1)*self.waypoint_dim]
                                               for i in range(len(self.predicted_ee_pose_traj)//self.waypoint_dim)]
                self.predicted_ee_pose_traj = np.array(self.predicted_ee_pose_traj)

    def extract_from_sample(self, traj):

        if self.visualizing_VAT:
            self.extract_from_sample_VAT(traj)
        else:
            self.extract_from_sample_RL(traj)

    def expand_dim(self, waypoints):

        waypoints_ = [waypoints[i * self.waypoint_dim:(i + 1) * self.waypoint_dim] for i in range(self.num_waypoints)]
        expanded_waypoints = []
        for ind, wp in enumerate(waypoints_):
            pos = wp[:3]
            yaw = wp[-1]

            axis_angle = np.array([0, 0, yaw])
            action = np.hstack([pos, axis_angle])
            expanded_waypoints.append(action)

        expanded_waypoints = np.array(expanded_waypoints).flatten()

        return expanded_waypoints

    def construct_ee_trajectory(self, pose, ground_truth=False):
        start_pose = None
        if not ground_truth:
            # the start pose is in base coordinates!
            start_pose = self.initia_waypoint_from_gripper_axis()

        # the waypoints are in world coordinates!
        num_waypoints = pose.shape[0]   # 9, 6
        transformed_wps = []
        for i in range(0, num_waypoints):
            if ground_truth:
                # this means the rotation part is already a quaternion
                quat = pose[i][3:]
            else:
                # this means the rotation part is a rot vec
                quat = Rotation.from_rotvec(pose[i][3:]).as_quat()[[3, 0, 1, 2]]
            pose_base = sapien.Pose(self.base_pose[:3], self.base_pose[3:]).inv().transform(sapien.Pose(pose[i][:3], quat))
            transformed_wp = [pose_base.p, Rotation.from_quat(pose_base.q[[1, 2, 3, 0]]).as_rotvec()]
            transformed_wps.append(transformed_wp)

        if start_pose is not None:
            start_wp = [start_pose.p, Rotation.from_quat(start_pose.q[[1, 2, 3, 0]]).as_rotvec()]
            transformed_wps[0] = start_wp

        self.waypoints = transformed_wps  # waypoints now in base frame

    def initia_waypoint_from_gripper_axis(self):
        start_trafo = np.eye(4)
        # the start_pose is the transformation matrix from gripper/ee frame to base frame!
        # construct start pose from gripper directions and contact point
        # the gripper direction are given in world frame, which is the same as base frame, because they are
        # only translated but not rotated relative to another
        gripper_right = np.cross(self.gripper_up_dir_base, self.gripper_forward_dir_base)
        start_trafo[:3, :3] = np.stack([self.gripper_forward_dir_base, gripper_right, self.gripper_up_dir_base],
                                       axis=1)
        start_trafo[:3, 3] = self.contact_point_base
        start_trafo[:3, 3] += 0.05 * self.contact_normal_world
        start_pose = sapien.Pose.from_transformation_matrix(start_trafo)

        return start_pose

    def construct_trajectory_VAT(self, waypoints):

        # some rearangement of the list for better handling afterward
        # if the trajectory was successful and the episode terminated early, we need to cut the zero padding
        zero_idx = len(waypoints)
        for i in range(len(waypoints)-1, -1, -1):
            if abs(waypoints[i]) > 0.005:
                zero_idx = i + 1
                break
        if zero_idx != len(waypoints):
            print('TRAJECTORY SHORTENED')
        waypoints = waypoints[:zero_idx]

        init_num_waypoints = int(len(waypoints) / 6)  # after expanding the waypoints we always have a dimension of 6
        # list of lists. Each sublist contains 6 numbers, first three are position, last three axis-angle
        waypoints_ = [waypoints[i * 6:(i + 1) * 6] for i in range(init_num_waypoints)]

        if self.waypoint_dim == 4:
            start_pose = self.initia_waypoint_from_gripper_axis()
        else:
            initial_waypoint = waypoints_[0]
            # initial_waypoint = clip_and_scale_action(initial_waypoint)
            quat = Rotation.from_rotvec(initial_waypoint[3:]).as_quat()[[3, 0, 1, 2]]
            pos = initial_waypoint[:3]
            start_pose = sapien.Pose(pos, quat)

        relative_waypoints = waypoints_[1:]

        self.waypoints = self.ee_target_delta_pose(relative_waypoints, start_pose)  # waypoints now in base frame

    def construct_trajectory_RL(self, waypoints):

        # some rearangement of the list for better handling afterward
        # if the trajectory was successful and the episode terminated early, we need to cut the zero padding
        zero_idx = len(waypoints)
        for i in range(len(waypoints) - 1, -1, -1):
            if abs(waypoints[i]) > 0.001:
                zero_idx = i + 1
                break
        if zero_idx != len(waypoints):
            print('TRAJECTORY SHORTENED')
        waypoints = waypoints[:zero_idx]

        init_num_waypoints = int(len(waypoints) / 6)
        # list of lists. Each sublist contains 6 numbers, first three are position, last three axis-angle
        waypoints_ = [waypoints[i * 6:(i + 1) * 6] for i in range(init_num_waypoints)]

        start_pose = self.initia_waypoint_from_gripper_axis()

        self.waypoints = self.ee_target_delta_pose(waypoints_, start_pose)  # waypoints now in base frame

    def construct_trajectory(self, waypoints, expand_dim=False):

        if self.waypoint_dim == 4 and expand_dim:
            waypoints = self.expand_dim(waypoints)

        if self.visualizing_VAT:
            self.construct_trajectory_VAT(waypoints)
        else:
            self.construct_trajectory_RL(waypoints)


    # this is how the target pose is constructed from the actions in ManiSkill Controller
    # when using ee_target_delta_pose controller
    # given a list of relative waypoints and an absolute start pose this method compute waypoints in absolut coordinates
    def ee_target_delta_pose(self, relative_waypoints, initial_pose):

        prev_target = copy.deepcopy(initial_pose)
        relative_waypoints = copy.deepcopy(relative_waypoints)
        transformed_wps = [[initial_pose.p, Rotation.from_quat(initial_pose.q[[1, 2, 3, 0]]).as_rotvec()]]

        for i, waypoint in enumerate(relative_waypoints):
            waypoint = clip_and_scale_action(waypoint)
            delta_pos, delta_rot = waypoint[0:3], waypoint[3:]
            delta_quat = Rotation.from_rotvec(delta_rot).as_quat()[[3, 0, 1, 2]]
            delta_pose = sapien.Pose(delta_pos, delta_quat)
            target_pose_base_frame = prev_target * delta_pose
            transformed_wp = [target_pose_base_frame.p,
                              Rotation.from_quat(target_pose_base_frame.q[[1, 2, 3, 0]]).as_rotvec()]
            transformed_wps.append(transformed_wp)
            prev_target = copy.deepcopy(target_pose_base_frame)

        return transformed_wps

    def construct_faucet(self):

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.xyz)
        if self.affordance is not None:
            # For low affordance, start with a base turquoise color
            # Adjust the base turquoise color according to your preference
            colors = np.tile(self.faucet_color, (len(self.xyz), 1))

            if self.action_mode == 'pull':
                # Define color transformation for 'pull' action
                color_transformation = np.array([1.0, -1.0, 0.5])
            elif self.action_mode == 'push':
                # Define color transformation for 'push' action
                color_transformation = np.array([-1.0, -1.0, 1.0])
            else:
                # Default to no color transformation if action_mode is undefined
                color_transformation = np.array([0.0, 0.0, 0.0])

                # Apply color transformation based on affordance values
            for i in range(3):  # Iterate over RGB channels
                colors[:, i] += (self.affordance * color_transformation[i]).flatten()
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        else:
            point_cloud.colors = o3d.utility.Vector3dVector([self.faucet_color for _ in range(len(self.xyz))])

        point_cloud = self.scale_geometry(point_cloud)
        return point_cloud

    def render_trajectory(self, ind=None, color=None):

        # Define a material and set its properties
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"  # Use a lit shader for better visual effects
        opacity = 1 if ind is None else self.trajectory_opacity

        if color is None:
            if self.success:
                trajectory_lines_base_color = self.best_trajectory_base_color_success if ind is None else self.trajectory_lines_base_color_success
                material.base_color = np.hstack([trajectory_lines_base_color, opacity])
            else:
                trajectory_lines_base_color = self.best_trajectory_base_color_unsuccess if ind is None else self.trajectory_lines_base_color_unsuccess
                material.base_color = np.hstack([trajectory_lines_base_color, opacity])
        else:
            trajectory_lines_base_color = color
            material.base_color = np.hstack([color, 1])

        waypoints = [waypoint[0] for waypoint in self.waypoints]
        lines = [[i, i + 1] for i in range(len(waypoints) - 1)]
        #colors = [trajectory_lines_base_color / (1 + (0.5 * i)) for i in range(len(lines))]
        colors = [trajectory_lines_base_color for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(waypoints),
                                        lines=o3d.utility.Vector2iVector(lines))
        #line_set.colors = o3d.utility.Vector3dVector(colors)

        name = 'trajectory' + f'_{ind}' if ind is not None else 'trajectory'
        line_set = self.scale_geometry(line_set)
        self.add_geometry(name, line_set, material=material)

        return line_set

    def render_waypoints(self, ind=None):

        if self.render_only_first_waypoint:
            waypoints = self.waypoints[0]
            waypoints = [waypoints]
        elif not self.visualize_waypoints:
            return
        else:
            waypoints = self.waypoints

        for ind_, wp in enumerate(waypoints):
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.coordinate_systems_size)
            # Convert axis-angle to rotation matrix
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array(wp[1]))
            # Apply rotation
            coord_frame.rotate(rotation_matrix, center=(0, 0, 0))
            # Translate to the waypoint's position
            coord_frame.translate(wp[0]*self.scale_factor, relative=False)
            name = f"coord_frame_{ind_}" + f'_{ind}' if ind is not None else f"coord_frame_{ind_}"
            self.add_geometry(name, coord_frame)

    def render_contact_point(self):
        contact_point_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=self.contact_point_size)
        contact_point_mesh.compute_vertex_normals()
        if self.action_mode == 'pull':
            color = self.contact_point_color_pull
        elif self.action_mode == 'push':
            color = self.contact_point_color_push
        contact_point_mesh.paint_uniform_color(color)
        contact_point_mesh.translate(self.contact_point_base, relative=False)
        contact_point_mesh = self.scale_geometry(contact_point_mesh)
        self.add_geometry("contact_point", contact_point_mesh)

        return contact_point_mesh

    def scale_geometry(self, geometry):
        # For point clouds
        if isinstance(geometry, o3d.geometry.PointCloud):
            np_points = np.asarray(geometry.points) * self.scale_factor
            geometry.points = o3d.utility.Vector3dVector(np_points)
        # For meshes
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            np_vertices = np.asarray(geometry.vertices) * self.scale_factor
            geometry.vertices = o3d.utility.Vector3dVector(np_vertices)
        # For linesets (trajectories)
        elif isinstance(geometry, o3d.geometry.LineSet):
            np_points = np.asarray(geometry.points) * self.scale_factor
            geometry.points = o3d.utility.Vector3dVector(np_points)
        return geometry

    def add_text(self):

        if self.success:
            self.vis.add_3d_label(self.sucess_label_position, "SUCCESSFUL TRIAL")
        else:
            self.vis.add_3d_label(self.sucess_label_position, "UNSUCCESSFUL TRIAL")

        target_angle = np.round(self.target_angle, 1)
        self.vis.add_3d_label(self.target_angle_label_position, f"TARGET ANGLE: {target_angle}°")
        init_angle = np.round(self.init_angle, 1)
        self.vis.add_3d_label(self.init_angle_label_position, f"INITIAL ANGLE: {init_angle}°")
        task = target_angle - init_angle
        self.vis.add_3d_label(self.task_angle_label_position, f"TASK: {task}°")


    def set_label_positions(self, faucet_center, camera_position, faucet_extent):
        self.sucess_label_position = faucet_center + [0, 0, faucet_extent]  # display above faucet
        self.target_angle_label_position = faucet_center + [0, 0, 1.2*faucet_extent]  # display above faucet
        self.init_angle_label_position = faucet_center + [0, 0, 1.3*faucet_extent]  # display above faucet
        self.task_angle_label_position = faucet_center + [0, 0, 1.4*faucet_extent]

    def render(self, trajectory):

        self.extract_from_sample(trajectory)
        self.construct_trajectory(self.waypoints, expand_dim=True)

        # display faucet
        point_cloud = self.construct_faucet()

        if self.render_first_time:
            self.render_first_time = False
            # Assuming 'point_cloud' is your point cloud variable
            bbox = point_cloud.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_max_extent()  # Maximum extent of the bounding box

            # Calculate a suitable distance to ensure the point cloud is fully visible
            distance = extent * 5  # This is arbitrary; adjust based on your needs
            camera_position = center + [-distance, 0, 0]  # Position the camera in +Z direction
            up_direction = [0, 0, 1]  # Assuming Y-axis is up

            # Setup camera with a field of view of 60 degrees
            self.vis.setup_camera(60.0, center, camera_position, up_direction)

            self.set_label_positions(center, camera_position, extent)

        else:
            self.clear_geometries()
            self.vis.clear_3d_labels()

        # self.add_text()
        self.render_trajectory()
        self.render_waypoints()
        # self.render_contact_point()
        self.add_geometry("faucet", point_cloud)

        if self.num_trajectories > 0 and self.visualizing_VAT:
            # self.trajectories does not need to be expanded, because the waypoints were already expanded
            # in the forward pass of the VAT agent
            trajectories = self.trajectories[:self.num_trajectories]
            for i, t in enumerate(trajectories):
                self.construct_trajectory(t.flatten(), expand_dim=False)
                self.render_trajectory(ind=i)
                # self.render_waypoints(ind=i)

        if self.render_gt:
            self.construct_ee_trajectory(pose=self.tcp_pose_world_after_action, ground_truth=True)
            self.render_trajectory(ind=1000, color=np.array([142, 76, 229]) / 255)  # purple
            # self.render_waypoints(ind=1000)

        if self.render_predicted_ee_poses:
            self.construct_ee_trajectory(pose=self.predicted_ee_pose_traj, ground_truth=False)
            self.render_trajectory(ind=2000, color=np.array([229, 123, 59]) / 255)  # orange
            # self.render_waypoints(ind=2000)

    def add_geometry(self, name, geometry, material=None):
        self.vis.add_geometry(name, geometry, material)
        self.geometry_names.append(name)

    def clear_geometries(self):
        for name in self.geometry_names:
            self.vis.remove_geometry(name)

    def load_next_trajectory(self, vis):
        self.current_traj_index += 1
        self.current_traj_key = self.data_keys[self.current_traj_index]
        trajectory = self.data[self.current_traj_key]
        self.render(trajectory)
        print(f'Current trajectory key is : {self.current_traj_key}')
        index = int(self.current_traj_key.split("_")[-1])
        q.put(index)

    def run(self):
        trajectory = self.data[self.current_traj_key]
        print(f'Current trajectory key is: {self.current_traj_key}')
        index = int(self.current_traj_key.split("_")[-1])
        q.put(index)
        self.render(trajectory)
        self.app.add_window(self.vis)
        self.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='render trajectories')
    parser.add_argument('--file', type=str, help='h5 file containing trajectories')

    q = mp.Queue()
    args = parser.parse_args()
    renderer = TrajectoryRenderer(trajectory_file=args.file)
    video_player = VideoPlayer(trajectory_file=args.file)

    for idx_process in range(1):
        p = mp.Process(target=video_player.run, args=(q,))
        p.start()

    renderer.run()
