import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from maniskill2_learn.env import build_replay
import wandb
import matplotlib as mpl
from maniskill2_learn.utils.meta import Config



plt.rc('font', serif='Times New Roman')  # If available on your system
plt.rc('font', size=13)  # Base text size
plt.rc('axes', labelsize=13)  # Axes labels
plt.rc('axes', titlesize=13)  # Axes title size
plt.rc('xtick', labelsize=11)  # X-tick label size
plt.rc('ytick', labelsize=11)  # Y-tick label size

# Define a custom color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

class MetricsManager:
    NUM_WAYPOINTS = 5

    def __init__(self, directory, wandb_run=None):

        self.directory = directory
        self.run = wandb_run  # wandb run used for logging

        self.metrics = dict(
            pushing=dict(
                clockwise=dict(positive=0,
                               negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
                anti_clockwise=dict(positive=0,
                                    negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
            ),
            pulling=dict(
                clockwise=dict(positive=0,
                               negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
                anti_clockwise=dict(positive=0,
                                    negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
            )
        )
        self.metrics_percent = dict(
            p_negative=0,
            p_positive=0,
            p_pushing=0,
            p_pulling=0,
            p_clockwise=0,
            p_anti_clockwise=0,
            p_overshoot=0,
            p_undershoot=0,
            p_opposite=0,
            p_no_motion=0,
            p_contact_point=0,
            p_non_contact_point=0,
            pushing=dict(
                clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0)),
                anti_clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0)),
            ),
            pulling=dict(
                clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0)),
                anti_clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0)),
            )
        )

        self.model_ids = dict()
        self.multi_faucet = False

        self.lengths = dict()
        self.lengths_unsuc = dict()

        self.task_suc = [0 for _ in range(18)]  # 0 -> 0-10 deg, 1 -> 10-20 deg ...
        self.task_unsuc = [0 for _ in range(18)]
        self.task_unsuc_cp = [0 for _ in range(18)]

        self.initial_angle_suc = [0 for _ in range(18)]  # 0 -> -90 - -80 deg, 1 -> -80 - -70 deg, 17 -> 80 - 90
        self.initial_angle_unsuc = [0 for _ in range(18)]
        self.initial_angle_unsuc_cp = [0 for _ in range(18)]

        self.mean_contact_point = np.zeros(3)
        self.contact_points = []

        # displaying contact point / non-contact points
        self.all_contact_points = []
        self.contact_and_non_contact_points = dict(non_contact_point=0, contact_point=0)
        self.contact_point_color = []

        self.augmented_dataset = False


    def actionMode(self, action_mode):

        return 'pulling' if action_mode == 1 else 'pushing'

    def direction(self, task_motion):

        return 'clockwise' if task_motion > 0 else 'anti_clockwise'


    def failCase(self, actual_motion, task_motion):

        # for clockwise motion the task_motion is positive
        if actual_motion == 0:
            return 'no_motion'
        elif np.sign(actual_motion) != np.sign(task_motion):
            return 'opposite'
        elif np.abs(actual_motion) < np.abs(task_motion):
            return 'undershoot'
        elif np.abs(actual_motion) > np.abs(task_motion) + np.deg2rad(4.9):
            return 'overshoot'
        else:
            return 'undefined'

    def compute_metrics(self, sample):

        if sample['infos'].get('model_id', None) is not None:
            model_id = sample['infos']['model_id'][:, 1, :]
            if not self.multi_faucet:
                self.multi_faucet = len(set(model_id.flatten())) > 1
        else:
            model_id = None

        if sample['infos'].get('task_motion', None) is not None:
            self.augmented_dataset = True
            task_motion = sample['infos']['task_motion']
        else:
            task_motion = sample['infos']['target_angle'] - sample['infos']['init_angle']

        task_motion = task_motion[:, -1, :]

        actual_motion = sample['infos']['current_angle'] - sample['infos']['init_angle']
        actual_motion = actual_motion[:, -1, :]

        initial_angle = sample['infos']['init_angle'][:, -1, :]

        success = sample['infos']['success'][:, -1, :]  # need success flag from last step
        action_mode = sample['infos']['action_mode'][:, -1, :]

        traj_lengths = []
        for b in range(sample['infos']['success'].shape[0]):
            if np.where(sample['infos']['success'][b] == True)[0].size != 0:
                length = np.array(np.where(sample['infos']['success'][b] == True)[0][0] + 1)
                traj_lengths.append(length)
            else:
                traj_lengths.append(self.NUM_WAYPOINTS)

        contact_point_base = sample['infos']['init_contact_point_base'][:, 0, :]
        succ_contact_points = contact_point_base[success.flatten()]

        # initialize mean_contact_point
        if succ_contact_points.size != 0 and np.alltrue(self.mean_contact_point == np.zeros(3)):
            self.mean_contact_point = np.array(sum(succ_contact_points) / len(succ_contact_points))
        else:
            self.mean_contact_point = np.array([0, 0, 0.29])

        if self.augmented_dataset:
            is_contact_point = sample['infos']['is_contact_point'][:, -1, :]
            # everything concerning contact point visualization
            contact_point_base = sample['infos']['init_contact_point_base'][:, 0, :]
            contact_point_world = sample['infos']['init_contact_point_world'][:, 0, :]
            on_movable_part = sample['infos']['non_contact_on_movable_part'][:, 0, :]
            init_target_link_pose = sample['infos']['target_link_pose'][:, 0, :]
            init_angle = sample['infos']['init_angle'][:, 0, :]
            current_angle = sample['infos']['current_angle'][:, 0, :]
            init_contact_point_local = sample['infos']['init_contact_point_local'][:, 0, :]
            ee_pose_at_base = sample['infos']['ee_pose_at_base'][:, 0, :]
        else:
            is_contact_point = np.ones_like(action_mode)

        batch_size = action_mode.shape[0]
        for i in range(batch_size):

            # compute the metrics depending on the action type
            if success[i] == 0:

                # for analyzing fail cases depending on task angle
                # we keep track of successful and unsuccessful trials depending on task angle
                self.task_unsuc[int(np.abs(task_motion[i]) // np.deg2rad(10))] += 1

                # for analyzing fail cases depending on initial angle of faucet handle
                # we keep track of unsuccessful trials depending on initial position of faucet handle
                # the intitial angle is between -90 and 90 degree
                # i need to map that range to 0 to 180 degree, or rather to index from [0,18]
                # the clip just needed for the one edge case where the initial angle is 90 degrees
                # for 90 deg ind would be 18, but we need it to maximally be 17
                ind = np.clip(np.interp(int(initial_angle[i] // np.deg2rad(10)), [-9, 9], [0, 18]), 0, 17)
                self.initial_angle_unsuc[int(ind)] += 1

                self.metrics[self.actionMode(action_mode[i])][self.direction(task_motion[i])]['negative'][
                    self.failCase(actual_motion[i], task_motion[i])] += 1

                # compute trajectory lengths
                if self.lengths_unsuc.get(f'len{traj_lengths[i]}_unsuc', None) is None:
                    # initializing the key
                    self.lengths_unsuc[f'len{traj_lengths[i]}_unsuc'] = 1
                else:
                    self.lengths_unsuc[f'len{traj_lengths[i]}_unsuc'] += 1

            elif success[i]:
                self.metrics[self.actionMode(action_mode[i])][self.direction(task_motion[i])]['positive'] += 1

                self.task_suc[int(np.abs(task_motion[i]) // np.deg2rad(10))] += 1

                ind = np.clip(np.interp(int(initial_angle[i] // np.deg2rad(10)), [-9, 9], [0, 18]), 0, 17)
                self.initial_angle_suc[int(ind)] += 1

                # for analyzing fail cases depending on predicted contact point
                # we check how many chosen contact points are actually on the movable object
                # We use a hack.
                # We check the z coordinate of the contact point to know whether its located on the height
                # of the faucet handle (around 0.29)

                # here we compute running mean contact point for successful interactions
                self.contact_points.append(contact_point_base[i])
                self.mean_contact_point = np.array(sum(self.contact_points) / len(self.contact_points))

                # compute trajectory lengths
                if self.lengths.get(f'len{traj_lengths[i]}', None) is None:
                    # initializing the key
                    self.lengths[f'len{traj_lengths[i]}'] = 1
                else:
                    self.lengths[f'len{traj_lengths[i]}'] += 1

            if self.augmented_dataset:
                # track contact and non-contact points for later visualization of affordance labels
                if is_contact_point[i]:
                    self.contact_and_non_contact_points['contact_point'] += 1
                    # self.contact_point_color.append(np.array([0, 1, 0]))
                    self.contact_point_color.append(np.array([44, 160, 44])/255)
                else:
                    self.contact_and_non_contact_points['non_contact_point'] += 1
                    # self.contact_point_color.append(np.array([1, 0, 0]))
                    self.contact_point_color.append(np.array([214, 39, 40])/255)

                # those values are later needed to reproject contact points onto canonical faucet orientation
                contact_point_dict = dict(
                    init_contact_point_base=contact_point_base[i],
                    init_contact_point_world=contact_point_world[i],
                    is_on_movable_part=on_movable_part[i],
                    init_target_link_pose=init_target_link_pose[i],
                    is_contact_point=is_contact_point[i],
                    success=success[i],
                    init_angle=init_angle[i][0],
                    current_angle=current_angle[i][0],
                    init_contact_point_local=init_contact_point_local[i],
                    ee_pose_at_base=ee_pose_at_base[i]
                )
                self.all_contact_points.append(contact_point_dict)
                self.base_pose = sample['infos']['base_pose'][0, 0, :]

            if self.multi_faucet:
                id = int(model_id[i])
                if id not in self.model_ids.keys():
                    self.model_ids[id] = dict(suc=0, unsuc=0)

                if success[i]:
                    self.model_ids[id]['suc'] += 1
                else:
                    self.model_ids[id]['unsuc'] += 1

    def sum_up(self, dictionary, name, inside=False):
        internal_sum = 0
        if name == "":
            inside = True

        if inside:
            if isinstance(dictionary, dict):
                for key, value in dictionary.items():
                    internal_sum += self.sum_up(value, name, inside=True)
            else:
                return dictionary
        else:
            if isinstance(dictionary, dict):
                for key, value in dictionary.items():
                    if key == name:
                        internal_sum += self.sum_up(value, name, inside=True)
                    else:
                        internal_sum += self.sum_up(value, name, inside=False)

        return internal_sum


    def compute_percentage(self, source, source_key, total):
        self.metrics_percent[f'p_{source_key}'] = round((self.sum_up(source, source_key) / total) * 100 if total else 0, 1)

    def consolidate_metrics(self):

        total = self.sum_up(self.metrics, '')

        print(f'TOTAL NUMBER OF TRAJECTORIES WITHOUT CONTACT POINT ERRORS {total}')
        print(f'TOTAL NUMBER OF TRAJECTORIES WITH CONTACT POINT ERRORS {sum(self.task_unsuc_cp)}')
        all_total = sum(self.task_unsuc_cp) + total
        print(f'TOTAL NUMBER OF TRAJECTORIES {all_total}')

        # success / unsuccess
        self.compute_percentage(self.metrics, 'negative', all_total)
        self.metrics_percent['p_negative'] = self.metrics_percent['p_negative'] + sum(self.task_unsuc_cp)/all_total
        self.compute_percentage(self.metrics, 'positive', all_total)

        # task mode
        self.compute_percentage(self.metrics, 'pushing', all_total)
        self.compute_percentage(self.metrics, 'pulling', all_total)

        # direction
        self.compute_percentage(self.metrics, 'clockwise', all_total)
        self.compute_percentage(self.metrics, 'anti_clockwise', all_total)

        # fail cases
        all_neg = self.sum_up(self.metrics, 'negative')
        self.compute_percentage(self.metrics, 'overshoot', all_neg)
        self.compute_percentage(self.metrics, 'undershoot', all_neg)
        self.compute_percentage(self.metrics, 'no_motion', all_neg)
        self.compute_percentage(self.metrics, 'opposite', all_neg)
        self.compute_percentage(self.metrics, 'undefined', all_neg)

        total_ = self.sum_up(self.contact_and_non_contact_points, '')
        self.compute_percentage(self.contact_and_non_contact_points, 'contact_point', total_)
        self.compute_percentage(self.contact_and_non_contact_points, 'non_contact_point', total_)

        total_l = self.sum_up(self.lengths, '')
        for k in self.lengths.keys():
            self.compute_percentage(self.lengths, k, total_l)

        total_l = self.sum_up(self.lengths_unsuc, '')
        for k in self.lengths_unsuc.keys():
            self.compute_percentage(self.lengths_unsuc, k, total_l)

        for key, value in self.metrics.items():  # key : pushing, pulling
            for key1, value1 in value.items():  # key1: clockwise, anti_clockwise
                for key2, value2 in value1.items():  # key2: pos, negative
                    tot_neg = sum(self.metrics[key][key1]['negative'].values())
                    if key2 == 'negative':
                        for key3, value3 in value2.items():  # fail cases
                            self.metrics_percent[key][key1][key2][key3] = round(
                                (self.metrics[key][key1][key2][key3] / tot_neg) * 100 if tot_neg else 0, 1)
                    elif key2 == 'positive':
                        total_ = (tot_neg + self.metrics[key][key1][key2])
                        self.metrics_percent[key][key1][key2] = round(
                            self.metrics[key][key1][key2] / total_ * 100 if total_ else 0, 1)

        # task histogram
        for bucket, amount in enumerate(self.task_suc):
            self.metrics_percent[f'task_' + str(bucket)] = round(
                (amount / sum(self.task_suc)) * 100 if sum(self.task_suc) else 0, 1)

        for bucket, amount in enumerate(self.task_unsuc):
            self.metrics_percent[f'unsuc_task_' + str(bucket)] = round(
                (amount / sum(self.task_unsuc)) * 100 if sum(self.task_unsuc) else 0, 1)

        if self.multi_faucet:
            all_success = self.sum_up(self.metrics, 'positive')  # all successful trial
            for k in self.model_ids.keys():
                amount = self.model_ids[k]['suc']  # successful trials for this id
                # what percentage of all successful trials are made up from this id
                self.metrics_percent[f'{k}_suc'] = round((amount / all_success) * 100 if all_total else 0, 1)



    def save_metrics_to_json(self):
        # Save the metrics and metrics_percent dictionaries to JSON files
        # Save the dictionary to a JSON file
        json_file_path = os.path.join(self.directory, 'metrics.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(self.metrics, json_file, indent=4)

        # Save the dictionary to a JSON file
        json_file_path2 = os.path.join(self.directory, 'metrics_percent.json')
        with open(json_file_path2, 'w') as json_file:
            json.dump(self.metrics_percent, json_file, indent=4)

        if self.run is not None:
            # upload to wandb
            self.run.save(json_file_path2)
            self.run.save(json_file_path)

    def load_metrics_from_json(self):
        # Load the metrics_percent dictionary from a JSON file
        # Read the dictionary back from the JSON file
        json_file_path = os.path.join(self.directory, 'metrics_percent.json')
        with open(json_file_path, 'r') as json_file:
            self.metrics_percent = json.load(json_file)

    def pie_chart(self, labels, data, title):
        # Filter out data entries that are zero and their corresponding labels
        filtered_data = [d for d, lbl in zip(data, labels) if d > 0]
        filtered_labels = [lbl for d, lbl in zip(data, labels) if d > 0]

        plt.figure(figsize=(5, 5))
        if filtered_data:  # Check if there's any data left to plot
            plt.pie(filtered_data, labels=filtered_labels, autopct='%1.1f%%', startangle=140)
            # plt.title(title)
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust these values as needed

            image_path = os.path.join(self.directory, f'{title.replace(" ", "")}.png')
            plt.savefig(image_path, bbox_inches='tight')

            if self.run is not None:
                self.run.log({f"{title}": wandb.Image(image_path)})
        else:
            print("No data to plot.")
        #plt.show()

    def bar_chart(self, labels, data, title, ylabel, xlabel=None, color=None, rotate=False):
        plt.figure(figsize=(7, 5))
        bars = plt.bar(labels, data, width=0.5, color='gray')
        x = np.arange(len(labels))
        #plt.title(title)
        if rotate:
            plt.xticks(x, labels, rotation=45, ha='right')
        else:
            plt.xticks(x, labels)
        ylim = plt.gca().get_ylim()  # Get the current y-axis limits
        offset = (ylim[1] - ylim[0]) * 0.1
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval - offset, round(yval, 1), ha='center', va='bottom')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.tight_layout()
        image_path = os.path.join(self.directory, f'{title.replace(" ", "")}.png')
        plt.savefig(image_path, bbox_inches='tight')

        if self.run is not None:
            self.run.log({f"{title}": wandb.Image(image_path)})

        #plt.show()

    def absolut_histogram(self, buckets, data, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(7, 5))
        width = 9.5

        data = np.array(data)
        zero_mask = data == 0
        data = data[~zero_mask]

        buckets = np.array(buckets)
        buckets = buckets[~zero_mask].flatten()
        extended_bins = np.append(buckets, [buckets[-1] + (buckets[1] - buckets[0])])

        bars = ax.bar(buckets, data, width=width, color='grey', align='edge')
        ylim = plt.gca().get_ylim()  # Get the current y-axis limits
        offset = (ylim[1] - ylim[0]) * 0.1
        for bar in bars:
            yval = bar.get_height()
            if (yval - offset )< ylim[0]:
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.33*offset, round(yval, 1), ha='center', va='bottom')
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, yval - offset, round(yval, 1), ha='center', va='bottom')

        # ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        ax.set_xticks(extended_bins)
        ax.set_xticklabels([f'{int(b)}°' for b in extended_bins], rotation=45)
        #ax.set_ylim(0, 100)  # Since we're dealing with percentages, limit y-axis to 100%
        plt.tight_layout()
        image_path = os.path.join(self.directory, f'{title.replace(" ","")}.pdf')
        plt.savefig(image_path)

        if self.run is not None:
            self.run.log({f"{title}": wandb.Image(image_path)})

        #plt.show()

    def combined_histogram(self, buckets, data1, label1, data1_color, data2, label2, data2_color, xlabel, ylabel, title):

        data1 = np.array(data1)
        data2 = np.array(data2)

        # For simplicity, let's assume the total number of trials per bin is the sum of successful and unsuccessful trials.
        total = data1 + data2
        zero_mask = total == 0

        # Calculate the percentage of successful and unsuccessful trials for each bin relative to the total trials per bin
        data1_percentage_per_bin = (data1[~zero_mask] / total[~zero_mask]) * 100
        data2_percentage_per_bin = (data2[~zero_mask] / total[~zero_mask]) * 100

        bins = buckets  # np.arange(0, len(task_histogram_suc)*10, 10)
        non_zero_bins = bins[~zero_mask]
        zero_bins = bins[zero_mask]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 7))
        width = 9.5

        # Stacked bar chart: First layer for successful trials
        bars_data1 = ax.bar(non_zero_bins, data1_percentage_per_bin, width=width, color=data1_color,
                            label=label1, align='edge')
        #ax.bar(zero_bins, np.zeros((len(zero_bins))), width=width, color=data1_color, align='edge')

        # Stacked bar chart: Second layer for unsuccessful trials
        # The bottom parameter is set to success_percentage_per_bin to stack the bars
        ax.bar(non_zero_bins, data2_percentage_per_bin, width=width, color=data2_color, bottom=data1_percentage_per_bin,
               label=label2, align='edge')
        #ax.bar(zero_bins, np.zeros((len(zero_bins))), width=width, color=data2_color, bottom=zero_bins, align='edge')

        # create text annoations for each bar
        for bar, pct in zip(bars_data1, data1_percentage_per_bin):
            # ax.text(bar.get_x() + bar.get_width() / 2, pct / 2, f'{pct:.1f}', ha='center', va='center', color='black')
            # Position the text inside the bar, centered
            text_position = bar.get_height() - pct / 2 if bar.get_height() > 5 else bar.get_height() + 5
            ax.text(bar.get_x() + bar.get_width() / 2, text_position, f'{pct:.1f}', ha='center', va='center',
                    color='black')  # , fontsize=9)

        # Manually set the x-axis ticks
        # Ensure the last tick (end of your buckets range) is included
        extended_bins = np.append(non_zero_bins, [non_zero_bins[-1] + (non_zero_bins[1] - non_zero_bins[0])])
        ax.set_xticks(extended_bins)
        ax.set_xticklabels([f'{int(b)}°' for b in extended_bins], rotation=45)

        # Set other plot properties
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.set_xlim(extended_bins[0], extended_bins[-1])
        ax.legend(loc='upper right', fancybox=True, framealpha=0.5)  # Automatic best location with semi-transparent background
        plt.tight_layout()

        image_path = os.path.join(self.directory, f'{title.replace(" ", "")}.png')
        plt.savefig(image_path)

        if self.run is not None:
            self.run.log({f"{title}": wandb.Image(image_path)})

        #plt.show()

    def visualize_contact_points(self):

        import sapien.core as sapien

        base_pose_p = self.base_pose[:3]
        base_pose_q = self.base_pose[3:]
        base_pose = sapien.Pose(base_pose_p, base_pose_q)
        base_to_world = base_pose.to_transformation_matrix()
        world_to_base = base_pose.inv().to_transformation_matrix()

        reference_point = self.all_contact_points[0]
        canonical_to_world = sapien.Pose(reference_point['init_target_link_pose'][:3],
                                         reference_point['init_target_link_pose'][3:]).to_transformation_matrix()

        canonical_contact_points = []
        rgbs = []
        for ind, point in enumerate(self.all_contact_points):
            # dont want to visualize the contact points twice
            # so I have to skip over the augmented fail cases
            if point['is_contact_point'] and not point['success']:
                continue
            if point['is_contact_point'] or point['is_on_movable_part']:
                p_local = point['init_contact_point_local']
                p_canonical_world = canonical_to_world[:3, :3] @ p_local + canonical_to_world[:3, 3]
                p_canonical_base = world_to_base[:3, :3] @ p_canonical_world + world_to_base[:3, 3]
            elif point['is_on_movable_part']:  # non-contact points on movable part
                # the issue is that init_target_link_pose is not correct, because it is the target link pose
                # after the first action has been applied. This means that the target link has moved
                # in order to correct for the movement I need to transform it back somehow

                p_world = point['init_contact_point_world']  # contact point in world coordinates

                tcp_pose_in_base = point['ee_pose_at_base']  # tcp pose before action was applied, in base coordinates
                tcp_pose_in_base = sapien.Pose(tcp_pose_in_base[:3], tcp_pose_in_base[3:])

                # transforming tcp pose to world coordinates using the base pose
                tcp_pose_in_world = base_pose.transform(tcp_pose_in_base)  # base to world

                # now we know the tcp pose before the action was applied, in world coordinates
                world_to_local = tcp_pose_in_world.inv().to_transformation_matrix()

                p_local = world_to_local[:3, :3] @ p_world + world_to_local[:3, 3]
                p_canonical_world = canonical_to_world[:3, :3] @ p_local + canonical_to_world[:3, 3]
                p_canonical_base = world_to_base[:3, :3] @ p_canonical_world + world_to_base[:3, 3]
            else: # non-contact points on static part
                p_canonical_base = point['init_contact_point_base']

            canonical_contact_points.append(p_canonical_base)
            rgbs.append(self.contact_point_color[ind])

        points = np.vstack(canonical_contact_points)  # N, 3
        rgb = np.vstack(rgbs)  # N, 3

        if self.run is not None:
            pointcloud = np.concatenate([points, rgb], axis=1)  # N, 6
            self.run.log({'affordance_labels': wandb.Object3D(pointcloud)})

        else:
            import open3d as o3d
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(points)
            geometry.colors = o3d.utility.Vector3dVector(rgb)

            vis.add_geometry(geometry)

            while True:
                vis.poll_events()
                vis.update_renderer()


    def run_visualizations(self):
        # Call visualization methods here
        labels = ['p_negative', 'p_positive', 'p_pushing', 'p_pulling', 'p_clockwise', 'p_anti_clockwise',
                  'p_contact_point', 'p_non_contact_point']
        data = [self.metrics_percent[label] for label in labels]
        labels = ['negative', 'positive', 'pushing', 'pulling', 'clockwise', 'anti_clockwise',
                  'contact_point', 'non_contact_point']
        title = 'Overall Percentages Distribution'
        ylabel = 'Percentage'
        colors = None #['blue', 'blue', 'orange', 'orange', 'purple', 'purple', 'red', 'red']
        self.bar_chart(labels, data, title, ylabel, colors, rotate=True)

        if self.multi_faucet:
            labels = [f'{k}' for k in self.model_ids.keys()]
            # self.metrics_percent[f'{k}_suc']
            data = [self.metrics_percent[f'{k}_suc'] for k in self.model_ids.keys()]
            title = 'Distribution of successful trials over faucet models'
            ylabel = 'Percentage of successful trials'
            xlabel = 'ID of faucet model'
            #colors = ['blue', 'blue', 'orange', 'orange', 'purple', 'purple', 'red', 'red']
            self.bar_chart(labels, data, title, ylabel, xlabel=xlabel, rotate=False)

        labels = ['bad contact point', 'other']
        failed_contact_points = round(sum(self.task_unsuc_cp) / sum(self.task_unsuc) * 100 if sum(self.task_unsuc) else 0, 1)
        failed_other = 100 - failed_contact_points
        data = [failed_contact_points, failed_other]
        title = 'Failure Cases'
        ylabel = 'Percentage over all failed trials'
        colors = None #['red', 'grey']
        self.bar_chart(labels, data, title, ylabel, colors)

        for action in ['pushing', 'pulling']:
            for direction in['clockwise', 'anti_clockwise']:
                labels = ['overshoot', 'undershoot', 'opposite', 'no_motion']
                data = [self.metrics_percent[action][direction]['negative'][label] for label in labels]
                title = f'Negative Outcomes for {action.capitalize()} {direction.capitalize()}'
                if sum(data) != 0:
                    self.pie_chart(labels, data, title)

        labels = ['overshoot', 'undershoot', 'opposite', 'no_motion', 'undefined']
        data = [self.metrics_percent['p_' + label] for label in labels]
        labels = ['overshoot', 'undershoot', 'opposite', 'no motion', 'undefined']
        title = f'Failure Cases'
        self.pie_chart(labels, data, title)

        labels = self.lengths.keys()
        data = [self.metrics_percent['p_'+label] for label in labels]
        title = f'Distribution of trajectory lengths for successful trials'
        self.pie_chart(labels, data, title)

        labels = self.lengths_unsuc.keys()
        data = [self.metrics_percent['p_' + label] for label in labels]
        title = f'Distribution of trajectory lengths for unsuccessful trials'
        self.pie_chart(labels, data, title)

        buckets = np.arange(0, len(self.task_suc)*10, 10)
        data1 = self.task_suc
        data2 = self.task_unsuc
        label1 = 'Successful Trials'
        label2 = 'Unsuccessful Trials'
        data1_color = 'grey'
        data2_color = 'darkred'
        xlabel = 'Absolute task angle'
        ylabel = 'Percentage of trials'
        title = 'Percentage of successful and unsuccessful trials per task'
        self.combined_histogram(buckets, data1, label1, data1_color, data2, label2, data2_color, xlabel, ylabel, title)

        ylabel = 'Absolut number of successful trials'
        title = 'Successful trials per task'
        self.absolut_histogram(buckets, data1, xlabel, ylabel, title)

        ylabel = 'Absolute number of trials'
        title = 'Trials per task'
        self.absolut_histogram(buckets, np.array(data1)+np.array(data2), xlabel, ylabel, title)

        buckets = np.arange(-90, 90, 10)
        data1 = self.initial_angle_suc
        data2 = self.initial_angle_unsuc
        label1 = 'Successful Trials'
        label2 = 'Unsuccessful Trials'
        data1_color = 'grey'
        data2_color = 'darkred'
        xlabel = 'Initial angle of faucet handle'
        ylabel = 'Percentage of trials'
        title = 'Percentage of successful and unsuccessful trials per initial angle of the handle'
        self.combined_histogram(buckets, data1, label1, data1_color, data2, label2, data2_color, xlabel, ylabel, title)

        ylabel = 'Absolut number of successful trials'
        title = 'Successful trials per initial angle of faucet handle'
        self.absolut_histogram(buckets, data1, xlabel, ylabel, title)

        buckets = np.arange(0, len(self.task_unsuc) * 10, 10)
        data1 = self.task_unsuc_cp
        data2 = np.array(self.task_unsuc) - np.array(self.task_unsuc_cp)
        label1 = 'Bad Contact Point'
        label2 = 'Other Failure Case'
        data1_color = 'grey'
        data2_color = 'darkred'
        xlabel = 'Absolute task angle'
        ylabel = 'Percentage of unsuccessful trials'
        title = 'Percentage unsuccessful trials due to bad contact per task'
        self.combined_histogram(buckets, data1, label1, data1_color, data2, label2, data2_color, xlabel, ylabel, title)

        buckets = np.arange(-90, 90, 10)
        data1 = self.initial_angle_unsuc_cp
        data2 = np.array(self.initial_angle_unsuc) - np.array(self.initial_angle_unsuc_cp)
        label1 = 'Bad Contact Point'
        label2 = 'Other Failure Case'
        data1_color = 'grey'
        data2_color = 'darkred'
        xlabel = 'Initial angle of faucet handle'
        ylabel = 'Percentage of unsuccessful trials'
        title = 'Percentage unsuccessful trials due to bad contact per initial angle of faucet handle'
        self.combined_histogram(buckets, data1, label1, data1_color, data2, label2, data2_color, xlabel, ylabel, title)

        if self.augmented_dataset:
            self.visualize_contact_points()

        #if self.run is not None:
            #self.log_videos()

    def log_videos(self):
        pass
        #self.run.log({"video": wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")})

class TrajectoryAnalyzer:
    def __init__(self, file, only_visualize=False, num_waypoints=None, wandb_run=None, vat=False):
        self.file = file  # file containing the trajectories
        directory = os.path.dirname(file)
        self.directory = os.path.join(directory, 'Trajectory_metrics')  # directory to save the visualizations at
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.run = wandb_run

        self.eval_replay = dict(
            type="ReplayMemory",
            sampling_cfg=dict(type="TStepTransition", horizon=-1, with_replacement=False),
            capacity=-1, #int(20),
            num_samples=-1,
            #cache_size=int(20),
            dynamic_loading=False,
            synchronized=True,
            deterministic_loading=True,
            num_procs=1,
            keys=["obs", "actions", "dones", "episode_dones", "infos"],
            buffer_filenames=[file])

        self.only_visualize = only_visualize
        self.num_waypoints = num_waypoints
        self.metrics_manager = MetricsManager(self.directory, wandb_run=wandb_run)
        self.metrics_manager.NUM_WAYPOINTS = num_waypoints - 1 if vat else num_waypoints
    def process(self):
        if not self.only_visualize:
            self.get_data_from_trajectories(self.num_waypoints)
        self.metrics_manager.load_metrics_from_json()
        self.metrics_manager.run_visualizations()


    def compute_trajectory_length_of(self, buffer):
        # access the meta file of the trajectory
        # get the num_waypoint value from the meta file
        # that num_waypoint value corresponds to the number of waypoints that where used during
        # collection of the data/trajectory

        # for VAT mart that already is the correct value. No need to do +1
        replay_buffer_path = self.eval_replay['buffer_filenames']
        if isinstance(replay_buffer_path, list):
            replay_buffer_path = replay_buffer_path[0]

        meta_path = os.path.join(os.path.dirname(replay_buffer_path), "meta.py")
        meta_cfg = Config.fromfile(meta_path)

        num_waypoints = meta_cfg.env_cfg.num_waypoints

        if meta_cfg.agent_cfg.type == 'VAT-Mart':
            self.metrics_manager.NUM_WAYPOINTS = num_waypoints-1  # this is just so that the traj len computation is correct
        else:
            self.metrics_manager.NUM_WAYPOINTS = num_waypoints

        return num_waypoints

    def get_data_from_trajectories(self, num_waypoints=None):
        # Implement the data loading and metrics computation logic here
        # This method should use `self.metrics_manager.compute_metrics(sample)` for each sample
        replay = build_replay(self.eval_replay)

        if num_waypoints is None:
            num_waypoints = self.compute_trajectory_length_of(replay)

            print(f'num waypoints is {num_waypoints}')

        sample = replay.sample(20, auto_restart=False, traj_len=num_waypoints, drop_last=False)
        counter = 0

        while (sample is not None):
            self.metrics_manager.compute_metrics(sample)
            sample = replay.sample(20, auto_restart=False, traj_len=num_waypoints, drop_last=False)

            if counter % 1 == 0:
                print(f'sample {counter} sampled')

            counter += 1

        self.metrics_manager.consolidate_metrics()
        self.metrics_manager.save_metrics_to_json()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Render trajectories')
    parser.add_argument('--trajectories', type=str, help='h5 file containing trajectories')
    parser.add_argument('--only-visualize', type=int, choices=[0, 1], default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    script = TrajectoryAnalyzer(args.trajectories, bool(args.only_visualize))
    script.process()

