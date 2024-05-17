agent_cfg = dict(
    type='SAC',
    batch_size=1,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=False,
    alpha_optim_cfg=dict(type='Adam', lr=0.0003),
    actor_cfg=dict(
        type='ContinuousActor',
        head_cfg=dict(type='TanhGaussianHead', log_std_bound=[-20, 2]),
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=['obs_shape', 256, 256, 'action_shape * 2'],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        optim_cfg=dict(type='Adam', lr=0.0003)),
    critic_cfg=dict(
        type='ContinuousCritic',
        num_heads=2,
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            bias='auto',
            mlp_spec=['obs_shape + action_shape', 256, 256, 1],
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        optim_cfg=dict(type='Adam', lr=0.0003)))
env_cfg = dict(
    type='gym',
    env_name='VAT-TurnFaucet-v0',
    obs_mode='VAT',
    ignore_dones=False,
    num_waypoints=8,
    control_mode='pd_ee_target_delta_pose',
    reward_mode='dense',
    obs_frame='base',
    n_points=650,
    interpolate=False,
    render_robot=False,
    hand_held_cam=False,
    front_cam=True,
    back_cam=True,
    faucets='faucetTrainingDataSet.txt',
    max_task_angle_difference=180,
    min_task_angle_difference=30,
    randomize_initial_faucet_pose=False,
    restrict_action_space=True)
eval_cfg = dict(
    type='Evaluation',
    num_procs=10,
    num=6500,
    use_hidden_state=False,
    save_traj=True,
    save_video=False,
    log_every_step=False,
    only_save_success_traj=False,
    augment_dataset=True,
    affordance_predictor_data_set=False,
    sample_mode='sample')
work_dir = None
resume_from = None
replay_cfg = None
expert_replay_cfg = None
recent_traj_replay_cfg = None
rollout_cfg = None