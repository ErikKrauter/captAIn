agent_cfg = dict(
    type='PoseTrajectoryGenerator',
    batch_size=64,
    num_traj=100,
    trajectory_generator_cfg=dict(
        type='TrajectoryGenerator',
        backbone_cfg=dict(
            type='PointNet2', hparams=dict(feat_dim=128), n_points='n_points'),
        mlp_cp_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            act_cfg=None,
            bias='auto',
            mlp_spec=[3, 32],
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        mlp_task_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            act_cfg=None,
            bias='auto',
            mlp_spec=[1, 32],
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        mlp_traj_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            act_cfg=None,
            bias='auto',
            mlp_spec=['trajectory_dim', 128, 128, 128],
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
        vae_cfg=dict(
            type='CVAE',
            latent_dim=128,
            lbd_kl=1,
            lbd_recon_pos=30,
            lbd_recon_dir=30,
            lbd_init_dir=39,
            encoder_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU'),
                bias='auto',
                mlp_spec=[320, 128, 128],
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0)),
            decoder_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                act_cfg=None,
                bias='auto',
                mlp_spec=[320, 512, 256, 'trajectory_dim'],
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0))),
        optim_cfg=dict(type='Adam', lr=0.001, weight_decay=1e-05),
        lr_scheduler_cfg=dict(type='StepLR', step_size=2000, gamma=0.9)),
    mode='trajectoryGenerator',
    waypoint_dim=6,
    num_waypoints=8)
train_cfg = dict(
    on_policy=False,
    total_steps=200000,
    warm_steps=0,
    n_eval=128,
    n_checkpoint=1024,
    n_steps=0,
    n_updates=32,
    n_log=16,
    ep_stats_cfg=dict(info_keys_mode=dict(success=[True, 'max', 'mean'])))
replay_cfg = dict(
    type='ReplayMemory',
    sampling_cfg=dict(
        type='TStepTransition', horizon=-1, with_replacement=False),
    capacity=-1,
    num_samples=-1,
    dynamic_loading=False,
    synchronized=False,
    num_procs=5,
    keys=['obs', 'actions', 'dones', 'episode_dones', 'infos'],
    buffer_filenames=['Experiments/PoseGenerator/train_dataset/trajectory.h5'])
eval_cfg = dict(
    type='SupervisedEvaluation',
    eval_replay=dict(
        type='ReplayMemory',
        sampling_cfg=dict(
            type='TStepTransition', horizon=-1, with_replacement=False),
        capacity=-1,
        num_samples=-1,
        dynamic_loading=False,
        synchronized=False,
        num_procs=5,
        keys=['obs', 'actions', 'dones', 'episode_dones', 'infos'],
        buffer_filenames=[
            'Experiments/PoseGenerator/eval_dataset/trajectory.h5'
        ]))
work_dir = None
env_cfg = None
resume_from = None
expert_replay_cfg = None
recent_traj_replay_cfg = None
rollout_cfg = None
