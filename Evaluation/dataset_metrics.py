from maniskill2_learn.utils.data import GDict
import numpy as np
import matplotlib.pyplot as plt
# I need to compute metrics on the dataset to ensure it contains all necessary information for the networks to learn
# something useful

#       For trajectory Scorer its essential to learn the following:
#       in order for a trajectory to be successful it must meet these three criteria:
#           1. if task has negative sign the trajectory must move clockwise, if positive sign anti-clockwise
#           2. if task has negative sign trajectory must start on right side of handle, else on left side of handle,
#               else no contact is made with the handle
#           3. trajectory must not overshoot or undershoot the target

#       For trajectory generator its essential to learn the following:
#       a trajectory has the following properties
#           1. the forward direction of the gripper points towards the contact point
#               1.1 there are certain cases where its not possible due to kinematic constraints.
#               In those cases I flip the robot gripper so the normal faces away from contact point
#               Should I skip those cases during simulation so they are not present in the dataset?
#           2. the progression of waypoints goes from right to left if task has negative sign -> clockwise
#           3. the progression of waypoints goes from left to right if task has positive sign -> anti-clockwise
#           4. the progression of waypoints goes toward the robot for pulling tasks -> can not directly be known from
#           task alone, for that the contact point position relative to the faucet handle must be taken into account
#               4.1 Should I additionally include a flag that indicates if its a pulling or a pushing task and make
#               the task input two-dimensional? Or will this hinder learning something from the cp and pc?
#               (first dimension is signed angle, second is 1 for pulling 0 for pushing)
#           5. the progression of waypoints goes away from robot for pushing tasks.
#           6. all the implicit information on how trajectories are shaped will be learned through reconstruction

#       For affordance predictor its essential to learn the following:
#           1. points on the handle have high affordance, while points not on the handle have low affordance
#           2. if task has negative sign the contact-point must be on right side of handle
#           3. if task has positive sign the contact-point must be on the left side of the handle


# in order to learn the above-mentioned things the dataset must contact certain types of interaction samples

# in general: 25% clockwise pushing, 25% anti-clockwise pushing, 25% clockwise pulling, 25% anti-clockwise pulling

#       For trajectory Scorer its essential to learn the following:
#       in order for a trajectory to be successful it must meet these three criteria:
#           1. if task has negative sign the trajectory must move clockwise, if positive sign anti-clockwise
#               1.1 Need positive and negative examples for that:
#                   - positive samples:
#                       - task negative sign with trajectory moving clockwise and success flag true
#                       - task positive sign with trajectory moving anti-clockwise and success flag true
#                   - negative samples:
#                       - task negative sign with trajectory moving anti-clockwise and success flag false
#                       - task positive sign with trajectory moving clockwise and success flag false
#           2. if task has negative sign trajectory must start on right side of handle, else on left side of handle,
#               else no contact is made with the handle
#                   1.1 Need positive and negative examples for that:
#                    - positive samples:
#                        - simply make target the actual-motion for both - tasks and + tasks -> success flag true
#                    - negative samples:
#                        - set - actual-task to + fake-task and vice versa -> success flag false
#                   THIS IS THE SAME DATA AS FOR 1.
#           3. trajectory must not overshoot or undershoot the target
#                   - positive data: just successful trajectories: set actual motion to task
#                       - for + tasks and - tasks
#                   - negative data: set fake-target to actual motion + offset in direction of movement (to overshoot)
#                       - for + tasks and - tasks!
#                   - negative data: set fake-target to actual motion + offset in opposite direction (to undershoot)
#                       - for + tasks and - tasks!
#           So all in all the dataset to train the trajectory Scorer should contain:
#               - for - tasks equally many pushing and pulling examples for:
#                   - just successful trajectories: set actual motion to task                               50%
#                   - overshooting set fake-target to actual motion + offset in direction of movement       10%
#                   - undershooting set fake-target to actual motion + offset in opposite direction         20%
#                   - task negative sign with trajectory moving anti-clockwise and success flag false       20%
#                       - set fake-target to actual motion + offset large enough to make fake target positive
#               - for + tasks equally many pushing and pulling examples for:
#                   - just successful trajectories: set actual motion to task                               50%
#                   - overshooting set fake-target to actual motion + offset in direction of movement       10%
#                   - undershooting set fake-target to actual motion + offset in opposite direction         20%
#                   - task positive sign with trajectory moving clockwise and success flag false            20%
#                       - set fake-target to actual motion - offset large enough to make fake target negative

#       For trajectory generator the dataset:
#           just examples where the target was replaced with the actual motion. Make sure that -,+, pulling, pushing is
#           balanced.


#       For affordance predictor the data set should contain:
#           1. points on the handle have high affordance, while points not on the handle have low affordance
#               - for that I need to include points sampled from non-handle part of faucet to the dataset and set
#               success flag to false. The issue is that for those samples I cannot use the trajectory scorer or
#               generator to create labels, because they can only deal with real contact points. So in oder to process
#               contact points from non-movable regions I will need to use the success flag from the dataset directly.
#           2. if task has negative sign the contact-point must be on right side of handle
#           3. if task has positive sign the contact-point must be on the left side of the handle
#               - for 2. and 3. I simply use the data points mentioned above.
#               - have examples where the task is - and the contact point is on left side w/ success flag negative
#               - and have examples with + task and contact point on right side w/ success flag negative
#                   - task negative sign with trajectory moving anti-clockwise and success flag false       20%
#                       - set fake-target to actual motion + offset large enough to make fake target positive


#               --------------------------- ALL IN ALL ------------------------------------
# in general: 25% clockwise pushing, 25% anti-clockwise pushing, 25% clockwise pulling, 25% anti-clockwise pulling
# for each of those four categories:
#
#                   - just successful trajectories: set actual motion to task                               50%
#                   - overshooting set fake-target to actual motion + offset in direction of movement       10%
#                       - not as important as undershooting
#                   - undershooting set fake-target to actual motion + offset in opposite direction         20%
#                   The next is very important for Scorer and Affordance predictor
#                   - task negative sign with trajectory moving anti-clockwise and success flag false       20%
#                       - for - tasks:
#                       - set fake-target to actual motion + offset large enough to make fake target positive
#                       - for + tasks:
#                       - set fake-target to actual motion - offset large enough to make fake target negative
#                   Only for affordance predictor have additionally
#                   - include points sampled from non-handle part of faucet to the dataset and set
#                       success flag to false


# Now finally: What metric do I need to check in my dataset?

# All the above-mentioned:
# % clockwise pushing 25%
#       - % success flag true       50%
#       - % success flag false      50%
#           - % overshooting        10%
#           - % undershooting       20%
#           - % opposite direction  20%
# % anti-clockwise pushing
# % clockwise pulling
# % anti-clockwise pulling

# % trajectories with length 4 3 2 1

NUM_WAYPOINTS = 4
eval_replay=dict(
    type="ReplayMemory",
    sampling_cfg=dict(type="TStepTransition", horizon=-1, traj_len=NUM_WAYPOINTS, with_replacement=False),
    capacity=-1, #int(20),
    num_samples=-1,
    #cache_size=int(20),
    dynamic_loading=False,
    synchronized=True,
    num_procs=10,
    keys=["obs", "actions", "dones", "episode_dones", "infos"],
    buffer_filenames=[
                #'RLTrajectories/RL_NonRandomizedDirection_5Waypoints/test/trajectory.h5'
                'DatasetsAndModels/RLTrajectories/RL_NonRandomizedDirection/model_5M/test/trajectory.h5'
    ])


metrics = dict(
    pushing=dict(
        clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
        anti_clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
    ),
    pulling=dict(
        clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
        anti_clockwise=dict(positive=0, negative=dict(no_motion=0, overshoot=0, undershoot=0, opposite=0, undefined=0)),
        )
)

metrics_percent = dict(
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


lenghts = dict()

contact=dict(
    contact_point=0,
    non_contact_point=0,
)

all_contact_points = []
all_rgb = []
task_histogram = [0 for _ in range(19)]
task_histogram_unsuc = [0 for _ in range(19)]
contact_point = np.array(3)


def actionMode(action_mode):

    return 'pulling' if action_mode == 1 else 'pushing'


def direction(task_motion):

    return 'clockwise' if task_motion > 0 else 'anti_clockwise'


def failCase(actual_motion, task_motion):

    # for clockwise motion the task_motion is positive
    if actual_motion == 0:
        return 'no_motion'
    elif np.sign(actual_motion) != np.sign(task_motion):
        return 'opposite'
    elif np.abs(actual_motion) < np.abs(task_motion):
        return 'undershoot'
    elif np.abs(actual_motion) > np.abs(task_motion):
        return 'overshoot'
    else:
        return 'undefined'  # actual_motion and task_motion are 0


def computeMetrics(sample):

    init = sample['infos']['init_angle']  # batch x num_waypoints x 1
    current = sample['infos']['current_angle']

    task_motion = sample['infos']['task_motion']

    task_motion = task_motion[:, -1, :]

    actual_motion = current - init
    actual_motion = actual_motion[:, -1, :]

    success = sample['infos']['success']
    success = success[:, -1, :]  # need success flag from last step
    action_mode = sample['infos']['action_mode'][:, -1, :]

    traj_lengths = [np.array(np.where(sample['infos']['success'][b]==True)[0][0]+1) for b in range(sample['infos']['success'].shape[0]) if np.where(sample['infos']['success'][b]==True)[0].size != 0]
    if len(traj_lengths) == 0:
        max_len = sample['infos']['success'].shape[1]
    else:
        max_len = NUM_WAYPOINTS

    traj_lengths = traj_lengths + [np.array(max_len)]*(sample['infos']['success'].shape[0]-len(traj_lengths))
    traj_lengths = np.array(traj_lengths)

    if sample['infos'].get('is_contact_point', None) is not None:
        is_contact_point = sample['infos']['is_contact_point'][:, -1, :]
    else:
        is_contact_point = np.ones_like(action_mode)

    # everything concerning contact point visualization
    contact_point_base = sample['infos']['init_contact_point_base'][:, 0, :]
    contact_point_world = sample['infos']['init_contact_point_world'][:, 0, :]
    on_movable_part = sample['infos']['non_contact_on_movable_part'][:, 0, :]
    init_target_link_pose = sample['infos']['target_link_pose'][:, 0, :]

    batch_size = action_mode.shape[0]
    for i in range(batch_size):

        # compute the metrics depending on the action type
        if success[i] == 0 and is_contact_point[i]:
            metrics[actionMode(action_mode[i])][direction(task_motion[i])]['negative'][failCase(actual_motion[i], task_motion[i])] += 1
        elif success[i]:
            metrics[actionMode(action_mode[i])][direction(task_motion[i])]['positive'] += 1

        # compute trajectory lengths
        if lenghts.get(f'len{traj_lengths[i]}', None) is None:
            lenghts[f'len{traj_lengths[i]}'] = 0
        else:
            lenghts[f'len{traj_lengths[i]}'] += 1

        # track contact and non-contact points for later visualization of affordance labels
        if is_contact_point[i]:
            contact['contact_point'] += 1
            all_rgb.append(np.array([0, 1, 0]))
        else:
            contact['non_contact_point'] += 1
            all_rgb.append(np.array([1, 0, 0]))

        # those values are later needed to reproject contact points onto canonical faucet orientation
        contact_point_dict = dict(
            init_contact_point_base=contact_point_base[i],
            init_contact_point_world=contact_point_world[i],
            is_on_movable_part=on_movable_part[i],
            init_target_link_pose=init_target_link_pose[i],
            is_contact_point=is_contact_point[i]
        )
        all_contact_points.append(contact_point_dict)

        # for analyzing fail cases depending on task angle
        # we keep track of successful and unsuccessful trials depending on task angle
        if success[i]:
            task_histogram[int(np.abs(task_motion[i]) // np.deg2rad(10))] += 1
        else:
            task_histogram_unsuc[int(np.abs(task_motion[i]) // np.deg2rad(10))] += 1


def sumUp(dictionary, name, inside=False):
    internal_sum = 0
    if name == "":
        inside = True

    if inside:
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                internal_sum += sumUp(value, name, inside=True)
        else:
            return dictionary
    else:
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                if key == name:
                    internal_sum += sumUp(value, name, inside=True)
                else:
                    internal_sum += sumUp(value, name, inside=False)

    return internal_sum


def consolidateMetrics():

    total = sumUp(metrics, '')

    print(f'TOTAL NUMBER OF TRAJECTORIES {total}')

    metrics_percent['p_negative'] = round((sumUp(metrics, 'negative') / total) * 100 if total else 0, 1)

    metrics_percent['p_positive'] = round((sumUp(metrics, 'positive') / total) * 100 if total else 0, 1)

    metrics_percent['p_pushing'] = round((sumUp(metrics, 'pushing') / total) * 100 if total else 0, 1)

    metrics_percent['p_pulling'] = round((sumUp(metrics, 'pulling') / total) * 100 if total else 0, 1)

    metrics_percent['p_clockwise'] = round((sumUp(metrics, 'clockwise') / total) * 100 if total else 0, 1)

    metrics_percent['p_anti_clockwise'] = round((sumUp(metrics, 'anti_clockwise') / total) * 100 if total else 0, 1)

    metrics_percent['p_overshoot'] = round((sumUp(metrics, 'overshoot') / total) * 100 if total else 0, 1)

    metrics_percent['p_undershoot'] = round((sumUp(metrics, 'undershoot') / total) * 100 if total else 0, 1)

    metrics_percent['p_opposite'] = round((sumUp(metrics, 'opposite') / total) * 100 if total else 0, 1)

    metrics_percent['p_no_motion'] = round((sumUp(metrics, 'no_motion') / total) * 100 if total else 0, 1)

    for k in lenghts.keys():
        metrics_percent[k] = round((sumUp(lenghts, k) / total) * 100 if total else 0, 1)

    total = sumUp(contact, '')
    print(f'TOTAL NUMBER OF TRAJECTORIES WITH CP and NON-CP {total}')
    metrics_percent['p_contact_point'] = round((sumUp(contact, 'contact_point') / total) * 100 if total else 0, 1)
    metrics_percent['p_non_contact_point'] = round((sumUp(contact, 'non_contact_point') / total) * 100 if total else 0, 1)

    for key, value in metrics.items():  # key : pushing, pulling
        for key1, value1 in value.items():  # key1: clockwise, anti_clockwise
            for key2, value2 in value1.items():  # key2: pos, negative
                tot_neg = sum(metrics[key][key1]['negative'].values())
                if key2 == 'negative':
                    for key3, value3 in value2.items():  # fail cases
                        metrics_percent[key][key1][key2][key3] = round((metrics[key][key1][key2][key3] / tot_neg) * 100 if tot_neg else 0, 1)
                elif key2 == 'positive':
                    metrics_percent[key][key1][key2] = round(metrics[key][key1][key2] / (tot_neg + metrics[key][key1][key2]) * 100 if (tot_neg + metrics[key][key1][key2]) != 0 else 0, 1)

    # task histogram
    for bucket, amount in enumerate(task_histogram):
        metrics_percent[f'task_' + str(bucket)] = round((amount / sum(task_histogram)) * 100 if sum(task_histogram) else 0, 1)

    for bucket, amount in enumerate(task_histogram_unsuc):
        metrics_percent[f'unsuc_task_' + str(bucket)] = round((amount / sum(task_histogram_unsuc)) * 100 if sum(task_histogram_unsuc) else 0, 1)


def visualize_overall_percentages(metrics_percent, save_path='overall_percentages_suc.png'):
    labels = ['p_negative', 'p_positive', 'p_pushing', 'p_pulling', 'p_clockwise', 'p_anti_clockwise', 'p_contact_point', 'p_non_contact_point']
    sizes = [metrics_percent[label] for label in labels]

    plt.figure(figsize=(10, 7))
    plt.bar(labels, sizes, color=['blue', 'blue', 'orange', 'orange', 'purple', 'purple', 'red', 'red'])
    plt.title('Overall Percentages Distribution')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_action_type_distribution(metrics_percent, save_path='action_type_distribution_suc.png'):
    labels = ['pushing', 'pulling']
    sizes = [metrics_percent['p_pushing'], metrics_percent['p_pulling']]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, sizes, color=['blue', 'green'])
    plt.title('Action Type Distribution')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_detailed_negative_outcomes(metrics_percent, action, direction, save_path='detailed_negative_outcomes_suc.png'):
    labels = ['overshoot', 'undershoot', 'opposite', 'no_motion']
    sizes = [metrics_percent[action][direction]['negative'][label] for label in labels]

    plt.figure(figsize=(7, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Negative Outcomes for {action.capitalize()} {direction.capitalize()}')
    plt.tight_layout()
    save_path = f'{action}_{direction}_{save_path}'
    plt.savefig(save_path)
    plt.show()


def visualize_traj_len(metrics_percent, save_path='traj_lengs_suc.png'):
    global lenghts
    labels = lenghts.keys()
    sizes = [metrics_percent[label] for label in labels]

    plt.figure(figsize=(7, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of trajectory lengths')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_task_histogram():
    bins = np.arange(0, len(task_histogram)*10, 10)
    succ = [round(i/sum(task_histogram), 1) * 100 for i in task_histogram]
    unsucc = [round(i/sum(task_histogram_unsuc), 1) * 100 for i in task_histogram_unsuc]

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 9.5

    ax.bar(bins, succ, width=width, color='grey', label='Successful Interactions', align='edge')

    ax.set_title('Distribution of Successful Tasks')
    ax.set_ylabel('Percentage over all successful tasks')
    ax.set_xlabel('Task Angles (degrees)')
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{int(b)}°' for b in bins], rotation=45)
    ax.set_ylim(0, 100)  # Since we're dealing with percentages, limit y-axis to 100%
    plt.tight_layout()
    plt.savefig('task_histogram_suc.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bins, unsucc, width=width, color='darkred', label='Unsuccessful Interactions', align='edge')
    ax.set_title('Distribution of Unsuccessful Tasks')
    ax.set_ylabel('Percentage over all unsuccessful tasks')
    ax.set_xlabel('Task Angles (degrees)')
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{int(b)}°' for b in bins], rotation=45)
    ax.set_ylim(0, 100)  # Since we're dealing with percentages, limit y-axis to 100%
    plt.tight_layout()
    plt.savefig('task_histogram_unsuc.png')
    plt.close()


def visualize_combined_task_histogram(save_path='task_histogram_comb.png'):
    # Each bar in the histogram will now represent 100% of the trials for each bucket,
    # divided into successful and unsuccessful percentages.
    global task_histogram_unsuc
    global task_histogram
    task_histogram_suc = np.array(task_histogram)
    task_histogram_unsuc = np.array(task_histogram_unsuc)

    # For simplicity, let's assume the total number of trials per bin is the sum of successful and unsuccessful trials.
    total_trials_per_bin = task_histogram_suc + task_histogram_unsuc
    zero_mask = total_trials_per_bin == 0

    # Calculate the percentage of successful and unsuccessful trials for each bin relative to the total trials per bin
    success_percentage_per_bin = (task_histogram_suc[~zero_mask] / total_trials_per_bin[~zero_mask]) * 100
    unsuccess_percentage_per_bin = (task_histogram_unsuc[~zero_mask] / total_trials_per_bin[~zero_mask]) * 100

    bins = np.arange(0, len(task_histogram_suc)*10, 10)
    non_zero_bins = bins[~zero_mask]
    zero_bins = bins[zero_mask]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 9.5

    # Stacked bar chart: First layer for successful trials
    bars_suc = ax.bar(non_zero_bins, success_percentage_per_bin, width=width, color='grey', label='Successful Interactions', align='edge')
    ax.bar(zero_bins, np.zeros((len(zero_bins))), width=width, color='grey', align='edge')

    # Stacked bar chart: Second layer for unsuccessful trials
    # The bottom parameter is set to success_percentage_per_bin to stack the bars
    bars_unsuc = ax.bar(non_zero_bins, unsuccess_percentage_per_bin, width=width, color='darkred', bottom=success_percentage_per_bin,
           label='Unsuccessful Interactions', align='edge')
    ax.bar(zero_bins, np.zeros((len(zero_bins))), width=width, color='darkred', bottom=zero_bins, align='edge')

    for bar, pct in zip(bars_suc, success_percentage_per_bin):
        #ax.text(bar.get_x() + bar.get_width() / 2, pct / 2, f'{pct:.1f}', ha='center', va='center', color='black')
        # Position the text inside the bar, centered
        text_position = bar.get_height() - pct/2 if bar.get_height() > 5 else bar.get_height() + 5
        ax.text(bar.get_x() + bar.get_width() / 2, text_position, f'{pct:.1f}%', ha='center', va='center',
                color='black')#, fontsize=9)

    # Adding some plot aesthetics
    ax.set_xlabel('Task Angles (degrees)')
    ax.set_ylabel('Percentage of Trials (%)')
    ax.set_title('Success and Unsuccess Rate per Task Angle Bucket')
    ax.set_xticks(bins)
    ax.set_xticklabels([f'{int(b)}°' for b in bins], rotation=45)
    ax.set_ylim(0, 100)  # Since we're dealing with percentages, limit y-axis to 100%
    ax.legend()
    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()


def visualize_contact_points(base_pose):
    import open3d as o3d
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.random.random([3, 3]))
    geometry.colors = o3d.utility.Vector3dVector(np.ones([3, 3]))
    vis.add_geometry(geometry)

    base_pose_p = base_pose[:3]
    base_pose_q = base_pose[3:]
    base_pose = sapien.Pose(base_pose_p, base_pose_q)
    world_to_base = base_pose.inv().to_transformation_matrix()
    base_to_world = base_pose.to_transformation_matrix()

    reference_point = all_contact_points[0]
    canonical_to_world = sapien.Pose(reference_point['init_target_link_pose'][:3], reference_point['init_target_link_pose'][3:]).to_transformation_matrix()

    canonical_contact_points = []
    for point in all_contact_points:
        if point['is_on_movable_part'] or point['is_contact_point']:
            target_link_pose = point['init_target_link_pose']
            T_dynamic = sapien.Pose(target_link_pose[:3], target_link_pose[3:]).to_transformation_matrix()
            T_inv = canonical_to_world @ np.linalg.inv(T_dynamic)
            p_canonical_world = T_inv[:3, :3] @ point['init_contact_point_world'] + T_inv[:3, 3]
            p_canonical_base = world_to_base[:3, :3] @ p_canonical_world + world_to_base[:3, 3]
        else:
            p_canonical_base = point['init_contact_point_base']

        canonical_contact_points.append(p_canonical_base)

    points = np.vstack(canonical_contact_points)
    rgb = np.vstack(all_rgb)

    geometry.points = o3d.utility.Vector3dVector(points)
    geometry.colors = o3d.utility.Vector3dVector(rgb)
    vis.update_geometry(geometry)

    while True:
        vis.poll_events()
        vis.update_renderer()


def main():
    from maniskill2_learn.env import build_replay
    import json

    only_visualize = False

    if not only_visualize:

        replay = build_replay(eval_replay)
        sample = replay.sample(5, auto_restart=False)

        base_pose = sample['infos']['base_pose'][0, 0, :]

        counter = 0
        while (sample is not None):
            computeMetrics(sample)
            sample = replay.sample(5, auto_restart=False)

            if counter % 1 == 0:
                print(f'sample {counter} sampled')

            counter += 1

        print(json.dumps(metrics, indent=2))

        consolidateMetrics()

        print(json.dumps(metrics_percent, indent=2))

        # Save the dictionary to a JSON file
        with open('metrics_suc.json', 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

        # Save the dictionary to a JSON file
        with open('metrics_percent_suc.json', 'w') as json_file:
            json.dump(metrics_percent, json_file, indent=4)


    # Read the dictionary back from the JSON file
    with open('metrics_percent_suc.json', 'r') as json_file:
        loaded_metrics_percent = json.load(json_file)

    # Visualize overall percentages
    visualize_overall_percentages(loaded_metrics_percent)

    # Visualize lengths of trajectories
    visualize_traj_len(loaded_metrics_percent)

    # Visualize action type distribution
    visualize_action_type_distribution(loaded_metrics_percent)

    # Visualize histrogram of task angles
    visualize_task_histogram()

    visualize_combined_task_histogram()

    # Detailed negative outcomes for pushing clockwise as an example
    visualize_detailed_negative_outcomes(loaded_metrics_percent, 'pushing', 'clockwise')
    visualize_detailed_negative_outcomes(loaded_metrics_percent, 'pulling', 'clockwise')
    visualize_detailed_negative_outcomes(loaded_metrics_percent, 'pushing', 'anti_clockwise')
    visualize_detailed_negative_outcomes(loaded_metrics_percent, 'pulling', 'anti_clockwise')

    visualize_contact_points(base_pose)


if __name__ == "__main__":

    main()
