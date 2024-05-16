
agent_cfg = dict(
    type="TD3",
    batch_size=1,  # Using multiple gpus leads to larger effective batch size, which can be crucial for SAC training
    gamma=0.99,
    update_coeff=0.005,
    actor_update_interval=2,
    action_noise=0.1,
    noise_clip=0.5,
    shared_backbone=False,
    detach_actor_feature=False,
    ignore_dones=True,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            # type="SimpleDeterministicHead",
            type="TanhHead",
            noise_std=0.1,  # this is what they use in VAT
        ),
        nn_cfg=dict(
            type="Visuomotor",
            #visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            visual_nn_cfg=dict(type="PointNet2", hparams={'feat_dim': 512}),  # feat_dim is the ouput dim of the pointnet, it must align with mlp_spec below
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
    # if shared_backbone is true, the visual_nn_cfg in critic_cfg is not used, and instead the actor's visual_nn_cfg is used
    critic_cfg=dict(
        type="ContinuousCritic",
        num_heads=2,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet2", hparams={'feat_dim': 128}),
            #visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1], inactivated_output=True, zero_init_output=True
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
)


train_cfg = dict(
    on_policy=False,
    total_steps=int(5e6),
    warm_steps=100,
    n_eval=int(5e6),
    n_checkpoint=int(1e6),
    n_steps=1000,
    n_updates=1000,
    n_log=1000,
    print_steps=2,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name="PickCube-v0",  #"TurnFaucet-v0",
    obs_mode='pointcloud',
    ignore_dones=True,
    n_points=1200,
    control_mode='pd_ee_delta_pose',
    reward_mode='dense',
    obs_frame='ee',
    trainFaucets='faucetTrainingDataSet.txt'
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=int(1e6),
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=1,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=3,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
    trainFaucets="faucetTrainingDataSet.txt",
    holdOutFaucets="faucetHoldOutDataSet.txt"
)