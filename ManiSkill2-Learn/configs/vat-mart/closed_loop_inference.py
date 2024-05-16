agent_cfg = dict(
    type="VAT-SAC",
    batch_size=2,  # Using multiple gpus leads to larger effective batch size, which can be crucial for SAC training
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=False,
    shared_backbone=True,
    detach_actor_feature=True,
    ignore_dones=False,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4),
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="TanhGaussianHead",
            log_std_bound=[-20, 2],
        ),
        nn_cfg=dict(
            type="Visuomotor",
            # visual_nn_cfg=dict(type="PointNet2", hparams={'feat_dim': 128}, n_points='n_points', global_feature=True),
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape * 2"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}),
        # *Above removes visual_nn from actor optimizer; should only do so if shared_backbone=True and detach_actor_feature=True
        # *If either of the config options is False, then param_cfg={} should be removed, i.e. actor should also update the visual backbone.
        #   In addition, mlp_specs should be modified as well

        # *It is unknown if sharing backbone and detaching feature works well under the 3D setting. It is up to the users to figure this out.
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            # visual_nn_cfg=dict(type="PointNet2", hparams={'feat_dim': 128}, n_points='n_points', global_feature=True),
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1], inactivated_output=True, zero_init_output=True
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    affordance_predictor_cfg=dict(
        type="AffordancePredictor",
        affordance_predictor_checkpoint_path='',#'DatasetsAndModels/TrainedModels/VAT_modules/affordancePredictor/model_16384.ckpt',
        topk=5,
        backbone_cfg=dict(type="PointNet2", hparams={'feat_dim': 128}, n_points='n_points'),
        mlp_cp_cfg=dict(
            # in baseline this is just a linear layer with no activation
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=None,
            bias="auto",
            mlp_spec=[3, 32],  # input dim is contact point dim, output dim is cp_feat_dim=32
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        mlp_task_cfg=dict(
            # in baseline this is just a linear layer with no activation
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=None,
            bias="auto",
            mlp_spec=[1, 32],  # input dim is dim of task, namely angle of revolut joint, output is task_feat_dim=32
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        head_cfg=dict(
            # in Baseline this is called ActorEncoder. It uses a leaky_relu activation after first layer and a
            # relu activation after second layer, and sigmoid after last layer
            # the construction of LinearMLP does not allow to use different activations functions for each layer
            # so instead I will use relu after first and second, and no activation after last
            # the activation after last layer I will use during loss calculation instead
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=dict(type="LeakyReLU"), # dict(type="ReLU"),
            bias="auto",
            mlp_spec=[128+32+32, 128, 128, 1],  # feat_dim+cp_feat_dim+task_feat_dim, feat_dim
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
    ),
    trajectory_generator_cfg=dict(
        type="TrajectoryGenerator",
        trajectory_generator_checkpoint_path='',#'DatasetsAndModels/TrainedModels/VAT_modules/poseTrajectoryGenerator/model_final.ckpt',
        backbone_cfg=dict(type="PointNet2", hparams={'feat_dim': 128}, n_points='n_points'),
        mlp_cp_cfg=dict(
            # in baseline this is just a linear layer with no activation
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=None,
            bias="auto",
            mlp_spec=[3, 32],  # input dim is contact point dim, output dim is cp_feat_dim=32
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        mlp_task_cfg=dict(
            # in baseline this is just a linear layer with no activation
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=None,
            bias="auto",
            mlp_spec=[1, 32],  # input dim is dim of task, namely angle of revolut joint, output is task_feat_dim=32
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        # for some reason the trajectory feature dimension here is only 128 and in the other classes its 256
        mlp_traj_cfg=dict(  # three layer MLP, Called TrajEncoder in Baseline
            type="LinearMLP",
            norm_cfg=None,
            act_cfg=None,
            bias="auto",
            # 5 weil wir 4 wegpunkte haben und zwei richtungsvektoren
            mlp_spec=['trajectory_dim', 128, 128, 128],  # (num_steps+1) * waypoint dim, 128, 128, trajectory features dim
            inactivated_output=True,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        vae_cfg=dict(
            # this will create an encoder, two linear heads to predict mu, sigma, and a decoder
            # the encoder's oupute is the input to the linear heads
            # from mu, sigma a random latent is sampled
            # latent and other conditional features are inputs to the decoder
            type="CVAE",
            latent_dim=128,
            lbd_kl=1,
            lbd_recon_pos=30,
            lbd_recon_dir=30,
            lbd_init_dir=39,
            encoder_cfg=dict(  # two headed MLP, Called ActorEmcoder in Baseline
                type="LinearMLP",
                norm_cfg=None,
                act_cfg=dict(type="LeakyReLU"),
                bias="auto",
                mlp_spec=[128+128+32+32, 128, 128],  # num_steps * waypoint dim, 128, 128, trajectory features dim
                inactivated_output=True,
                linear_init_cfg=dict(
                    type="xavier_init",
                    gain=1,
                    bias=0,
                ),
            ),
            decoder_cfg=dict(  # three layer MLP, Called ActorDecoder in Baseline
                # in Baseline this is part of the Critic. A two layer mlp with leaky relu activation after first layer
                # and no activation after laster layer
                type="LinearMLP",
                norm_cfg=None,
                act_cfg=None,
                bias="auto",
                mlp_spec=[128+32+128+32, 512, 256, 'trajectory_dim'],  # feat_dim + task_feat_dim + traj_feat_dim + cp_feat_dim, 512, 256, num waypoints  * waypoint
                inactivated_output=True,
                linear_init_cfg=dict(
                    type="xavier_init",
                    gain=1,
                    bias=0,
                ),
            )
        ),
    )
)

env_cfg = dict(
    type="gym",
    env_name='ClosedLoop-TurnFaucet-v0',
    faucets='faucetTrainingDataSet.txt', #'faucetTrainingDataSet_Debug.txt', faucetID_HoldOutDataSet
    max_task_angle_difference=180,
    min_task_angle_difference=30,

    # observation related
    obs_mode='pointcloud',
    ignore_dones=False,
    use_contact_point_feature=True,  # this should only be set to true if the SAC backbone is pointnet++
    use_trajectory_follow_observation=True,
    use_contact_point_observation=True,
    use_contact_normal_observation=True,

    # action stuff
    restrict_action_space=False,
    interpolate=False,
    num_waypoints=80,
    control_mode='pd_ee_target_delta_pose',

    # rendering stuff
    # render_mode='human',
    obs_frame='base',
    n_points=1200,
    hand_held_cam=False,
    front_cam=True,
    back_cam=False,
    filter_robot_links=False,  # if set to true only the gripper is contained in the point cloud

    # reward related
    reward_mode='dense',
    # penalties
    penalize_step=0.05,  # give negative reward for every step
    distance_penalty=1.0,  # give negative reward if too far from contact point
    error_penalty=0.1,  # give penalty if IK solver could not find solution
    # reward terms
    use_contact_point_reward=True,
    use_trajectory_follow_reward=True,
    # everything related to trajectory following
    # mean_position,mean_position_closest_waypoints,
    # track_direction, track_position, contouring_reward
    trajectory_reward_mode='contouring_reward',
    use_rotational_distance=False,
    use_relative_trajectory_reward=True,
    trajectory_follow_scale_decay=0.0,
    trajectory_follow_scale=7.0,
    trajectory_follow_reward_weight=1.0,
    lag_to_contouring_ration=0.7,

    curriculum_half_time=0,

    #friction=0.1 * 5.0,
    #damping=2.0 * 5.0
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=10,   # uploading videos to wandb only works when using one process.
    num=300,
    use_hidden_state=False,
    save_traj=True,
    save_video=False,
    log_every_step=False,
    augment_dataset=False,
    only_save_success_traj=False,
    sample_mode='sample'
)

'''eval_cfg = dict(
    type="DoubleEvaluation",
    num_procs=1,
    num=5,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    log_every_step=False,
    id_faucets="faucetID_HoldOutDataSet.txt",
    ood_faucets="faucetOOD_HoldOutDataSet.txt",
    sample_mode='sample'
)'''