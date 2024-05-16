agent_cfg = dict(
    type="SAC",
    batch_size=1,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=False,
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

env_cfg=dict(
        type="gym",
        env_name='DataCollection-TurnFaucet-v0', #"TurnFaucet-v0",
        obs_mode='VAT',
        ignore_dones=False,
        num_waypoints=8,
        control_mode='pd_ee_target_delta_pose',
        reward_mode='dense',
        # render_mode='human',
        obs_frame='base',
        n_points=650,
        interpolate=False,
        render_robot=False,
        hand_held_cam=False,
        front_cam=True,
        back_cam=True,
        faucets="faucetTrainingDataSet.txt",
        # camera_cfgs={'add_segmentation': True},
        max_task_angle_difference=180,
        min_task_angle_difference=30,
        restrict_action_space=False,
        contact_point_offset=0.03

    )

eval_cfg = dict(
    type="Evaluation",
    num_procs=10,
    num=6500,
    use_hidden_state=False,
    save_traj=True,
    save_video=False,
    log_every_step=False,
    only_save_success_traj=True,
    augment_dataset=True,
    affordance_predictor_data_set=False,
    sample_mode='sample'  # eval or explore
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