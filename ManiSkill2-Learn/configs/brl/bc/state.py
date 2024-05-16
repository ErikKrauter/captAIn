agent_cfg = dict(
    type="BC",
    batch_size=256,
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
)

env_cfg = dict(
    type="gym",
    env_name="VAT-TurnFaucet-v0",
    obs_mode='state',
    control_mode='pd_ee_delta_pose',
    unwrapped=False,
    num_waypoints=4,
    trainFaucets='faucetTrainingDataSet_Debug.txt'
)


replay_cfg = dict(
    type="ReplayMemory",
    sampling_cfg=dict(type="TStepTransition", horizon=-1),
    capacity=-1,
    num_samples=-1,
    keys=["obs", "actions", "dones", "episode_dones", "infos"],
    buffer_filenames=[
        "Evaluation/EvalDebugTraj/test/trajectory.h5",
    ],
)


train_cfg = dict(
    on_policy=False,
    total_steps=50000,
    warm_steps=0,
    n_steps=0,
    n_updates=500,
    n_eval=50000,
    n_checkpoint=50000,
)

eval_cfg = dict(
    type="Evaluation",
    num=10,
    num_procs=1,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    use_log=False,
    trainFaucets="faucetTrainingDataSet_Debug.txt",
    holdOutFaucets="faucetHoldOutDataSet.txt"
)
