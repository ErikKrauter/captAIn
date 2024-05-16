
agent_cfg = dict(
    type="SAC",
    batch_size=1,
    gamma=0.95,
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
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
    total_steps=8000000,
    warm_steps=0,
    n_eval=160,
    n_checkpoint=1000000,
    n_log=10,
    n_steps=80,
    n_updates=1,
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name="TurnFaucet-v0",
    obs_mode='state',
    ignore_dones=True,
    faucets='faucetTrainingDataSet.txt'
)


replay_cfg = dict(
    type="ReplayMemory",
    capacity=1000000,
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=8,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=5,
    num=5,
    use_hidden_state=False,
    save_traj=False,
    save_video=False,
    log_every_episode=True,
    env_cfg=dict(ignore_dones=False, faucets='faucetTrainingDataSet.txt'),
)
