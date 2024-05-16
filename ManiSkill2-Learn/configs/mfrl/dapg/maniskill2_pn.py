
agent_cfg = dict(
    type="PPO",
    gamma=0.95,
    lmbda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0,
    critic_clip=False,
    obs_norm=False,
    rew_norm=True,
    adv_norm=True,
    recompute_value=True,
    num_epoch=2,
    critic_warmup_epoch=4,
    batch_size=128, #330
    detach_actor_feature=False,
    max_grad_norm=0.5,
    eps_clip=0.2,
    max_kl=0.1,
    dual_clip=None,
    shared_backbone=True,
    ignore_dones=True,
    dapg_lambda=0.1,
    dapg_damping=0.995,
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="GaussianHead",
            init_log_std=-1,
            clip_return=True,
            predict_std=False,
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape", 256, 256, 1], inactivated_output=True, zero_init_output=True
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    demo_replay_cfg=dict(
        type="ReplayMemory",
        capacity=int(2e4),
        num_samples=-1,
        cache_size=int(2e4),
        dynamic_loading=True,
        synchronized=False,
        keys=["obs", "actions", "dones", "episode_dones"],
        buffer_filenames=[
            "ManiSkill2/demos/v0/rigid_body/TurnFaucet-v0/processedTrainingDemosOverFit/training_trajectories_merged.none.pd_ee_delta_pose_pointcloud_shuffled.h5",
            # this path is relative to the directory from which the training script is started which is ManiSkill2-Learn
        ],
    ),

)


train_cfg = dict(
    on_policy=True,
    total_steps=int(5e6),
    warm_steps=0,
    n_steps=int(2e4),
    n_updates=1,
    n_eval=int(5e6),  # will perform eval every n_eval steps
    n_log=1000,  # will log data every n_log steps to tensorboard
    n_checkpoint=int(1e6),  # will save model every n_checkpoint steps
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)


env_cfg = dict(
    type="gym",
    env_name="TurnFaucet-v0",
    obs_mode='pointcloud',
    ignore_dones=True,
    n_points=1200,
    control_mode='pd_ee_delta_pose',
    reward_mode='dense',
    obs_frame='ee',
    trainFaucets='faucetTrainingDataSet_OverFit.txt'
)


rollout_cfg = dict(
    type="Rollout",
    num_procs=1, #5
    with_info=True,
    multi_thread=False,
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=int(2e4),
    sampling_cfg=dict(type="OneStepTransition", with_replacement=False),
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
    trainFaucets="faucetTrainingDataSet_OverFit.txt",
    holdOutFaucets="faucetHoldOutDataSet.txt"
)
