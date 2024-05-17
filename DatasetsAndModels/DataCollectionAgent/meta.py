agent_cfg = dict(
    type='SAC',
    batch_size=2048,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    ignore_dones=False,
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
train_cfg = dict(
    on_policy=False,
    total_steps=10000000,
    warm_steps=0,
    n_eval=6000,
    n_checkpoint=500000,
    n_steps=200,
    n_updates=100,
    n_log=100,
    print_steps=50,
    ep_stats_cfg=dict(info_keys_mode=dict(success=[True, 'max', 'mean'])))
env_cfg = dict(
    type='gym',
    env_name='VAT-TurnFaucet-v0',
    obs_mode='state',
    ignore_dones=False,
    interpolate=False,
    num_waypoints=8,
    control_mode='pd_ee_target_delta_pose',
    reward_mode='dense',
    faucets='faucetTrainingDataSet.txt',
    hand_held_cam=False,
    front_cam=False,
    back_cam=False,
    max_task_angle_difference=180,
    min_task_angle_difference=30,
    restrict_action_space=True,
    randomize_initial_faucet_pose=False)
replay_cfg = dict(type='ReplayMemory', capacity=1000000)
rollout_cfg = dict(type='Rollout', num_procs=1, with_info=True)
eval_cfg = dict(
    type='DoubleEvaluation',
    num_procs=1,
    num=1,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    log_every_step=False,
    id_faucets='faucetID_HoldOutDataSet.txt',
    ood_faucets='faucetOOD_HoldOutDataSet.txt',
    sample_mode='sample')
work_dir = None
resume_from = None
expert_replay_cfg = None
recent_traj_replay_cfg = None
