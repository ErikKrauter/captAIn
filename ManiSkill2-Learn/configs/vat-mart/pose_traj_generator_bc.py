agent_cfg = dict(
    type="PoseTrajectoryGenerator",
    batch_size=16,
    num_traj=100,
    trajectory_generator_cfg=dict(
        type="TrajectoryGenerator",
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
                # 5*6 because I have two direction vectors each 3 dim and 4 waypoints each 6 dimension
                mlp_spec=[128+32+128+32, 512, 256, 'trajectory_dim'],  # feat_dim + task_feat_dim + traj_feat_dim + cp_feat_dim, 512, 256, num waypoints  * waypoint
                inactivated_output=True,
                linear_init_cfg=dict(
                    type="xavier_init",
                    gain=1,
                    bias=0,
                ),
            )
        ),
        optim_cfg=dict(type="Adam", lr=0.001, weight_decay=1e-5),
        lr_scheduler_cfg=dict(type="StepLR", step_size=2000, gamma=0.9),
    )
)

train_cfg = dict(
    on_policy=False,
    total_steps=200000,
    warm_steps=0,
    n_eval=128,
    n_checkpoint=1024,
    n_steps=0,
    n_updates=32,
    n_log=16,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

replay_cfg=dict(
    type="ReplayMemory",
    sampling_cfg=dict(type="TStepTransition", horizon=-1, with_replacement=False),
    capacity=-1,  #int(1e3),
    num_samples=-1,
    # cache_size=int(1e3),
    dynamic_loading=False,
    synchronized=False,
    num_procs=5,
    keys=["obs", "actions", "dones", "episode_dones", "infos"],
    buffer_filenames=[
                "DatasetsAndModels/PathToTrainTrajectoryFile/trajectory.h5"
    ])


eval_cfg = dict(
    type="SupervisedEvaluation",
    eval_replay=dict(
        type="ReplayMemory",
        sampling_cfg=dict(type="TStepTransition", horizon=-1, with_replacement=False),
        capacity=-1,  # int(1e3),
        num_samples=-1,
        # cache_size=int(1e3),
        dynamic_loading=False,
        synchronized=False,
        num_procs=5,
        keys=["obs", "actions", "dones", "episode_dones", "infos"],
        buffer_filenames=[
            "DatasetsAndModels/PathToEvalTrajectoryFile/trajectory.h5"
        ]),
)
