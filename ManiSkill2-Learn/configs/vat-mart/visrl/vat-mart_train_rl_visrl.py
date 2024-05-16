agent_cfg = dict(
    type="SAC",
    batch_size=128, # 2048,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=False,
    ignore_dones=False,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4),
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-20, 2],
        ),
        nn_cfg=dict(
            type="LinearMLP",
            norm_cfg=None,
            mlp_spec=["obs_shape", 256, 256, "action_shape * 2"],
            bias="auto",
            inactivated_output=True,
            # zero_init_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="LinearMLP",
            norm_cfg=None,
            bias="auto",
            mlp_spec=["obs_shape + action_shape", 256, 256, 1],
            inactivated_output=True,
            # zero_init_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)

train_cfg = dict(
    on_policy=False,
    total_steps=int(5e6),
    warm_steps=1200,
    n_eval=int(0.25e6),
    n_checkpoint=int(0.5e6),
    n_steps=30,
    n_updates=30,
    n_log=60,
    print_steps=60,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name='VAT-TurnFaucet-v0', #"TurnFaucet-v0",
    obs_mode='state',
    # render_mode='human',
    ignore_dones=False,
    interpolate=False,
    num_waypoints=8,
    control_mode='pd_ee_target_delta_pose',
    reward_mode='dense',
    render_robot=False,
    hand_held_cam=False,
    front_cam=False,
    back_cam=False,
    faucets='faucetTrainingDataSet_Debug.txt',
    max_task_angle_difference=180,
    min_task_angle_difference=30,
    restrict_action_space=True,
)

replay_cfg = dict(
    type="ReplayMemory",
    capacity=int(1e6),
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=10,
    with_info=True,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=2,
    num=2,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    log_every_step=False,
    faucets="faucetTrainingDataSet_Debug.txt",
)

'''eval_cfg = dict(
    type="DoubleEvaluation",
    num_procs=5,
    num=10,
    use_hidden_state=False,
    save_traj=True,
    save_video=False,
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
    id_faucets="faucetID_HoldOutDataSet.txt",
    ood_faucets="faucetOOD_HoldOutDataSet.txt"
)'''