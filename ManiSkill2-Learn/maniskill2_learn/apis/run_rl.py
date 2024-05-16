import argparse
import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from copy import deepcopy
from pathlib import Path
import wandb
import sys

import gymnasium as gym
import numpy as np


np.set_printoptions(3)
warnings.simplefilter(action="ignore")


from maniskill2_learn.utils.data import is_not_null, is_null, num_to_str
from maniskill2_learn.utils.meta import (
    Config,
    DictAction,
    add_dist_var,
    add_env_var,
    collect_env,
    colored_print,
    get_dist_info,
    get_logger,
    get_world_rank,
    is_debug_mode,
    set_random_seed,
    log_meta_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified API for Training and Evaluation")
    # Configurations
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--cfg-options",
        "--opt",
        nargs="+",
        action=DictAction,
        help="Override some settings in the configuration file. The key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overridden is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    # Parameters for log dir
    parser.add_argument("--work-dir", help="The directory to save logs and models")
    parser.add_argument('--use-wandb', action="store_true", default=False, help="whether to use W&B for logging")
    parser.add_argument('--wandb-group', help="which name to assign to the W&B group")
    parser.add_argument('--upload-dataset', action="store_true", default=False, help='whether to upload the collected dataset to wandb')
    parser.add_argument('--train-dataset-version', help='version of the dataset to use. For newest specify "latest"')
    parser.add_argument('--eval-dataset-version', help='version of the dataset to use. For newest specify "latest"')
    parser.add_argument('--aff_model', default='affordancePredictor_model_final:latest', help='version of the affordance predictor model to use')
    parser.add_argument('--scor_model', default='trajectoryScorer_model_final:latest', help='version of the trajectory scorer model to use')
    parser.add_argument('--gen_model', default='trajectoryGenerator_model_final:latest', help='version of the trajectory generator model to use')
    parser.add_argument('--pose_gen_model', default='', help='version of the pose trajectory generator model to use')


    parser.add_argument("--dev", action="store_true", default=False, help="Add timestamp to the name of work-dir")
    parser.add_argument("--with-agent-type", default=False, action="store_true", help="Add agent type to work-dir")
    parser.add_argument(
        "--agent-type-first",
        default=False,
        action="store_true",
        help="When work-dir is None, we will use agent_type/config_name or config_name/agent_type as work-dir",
    )
    parser.add_argument("--clean-up", help="Clean up the work-dir", action="store_true")


    # Evaluation mode
    parser.add_argument("--evaluation", "--eval", help="Evaluate a model, instead of training it", action="store_true")
    parser.add_argument("--reg-loss", help="Measure regression loss during evaluation", action="store_true")
    parser.add_argument("--test-name",
        help="Subdirectory name under work-dir to save the test result (if None, use {work-dir}/test)", default=None)

    # Resume checkpoint model
    parser.add_argument("--resume-from", default=None, nargs="+", help="A specific checkpoint file to resume from")
    parser.add_argument(
        "--auto-resume",
        help="Auto-resume the checkpoint under work-dir. If --resume-from is not specified, --auto-resume is set to True", action="store_true"
    )
    parser.add_argument("--resume-keys-map", default=None, nargs="+", action=DictAction, help="Specify how to change the model keys in checkpoints")

    # Specify GPU
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument("--num-gpus", default=None, type=int, help="Number of gpus to use")
    group_gpus.add_argument("--gpu-ids", default=None, type=int, nargs="+", help="ids of gpus to use")
    parser.add_argument("--sim-gpu-ids", default=None, type=int, nargs="+", help="ids of gpus to do simulation on; if not specified, this equals --gpu-ids")

    # Torch and reproducibility settings
    parser.add_argument("--seed", type=int, default=None, help="Set torch and numpy random seed")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Whether to use benchmark mode in cudnn.")

    # Distributed parameters
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    # parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()

    # Merge cfg with args.cfg_options
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        print('args.cfg_options: ')
        print(args.cfg_options)
        for key, value in args.cfg_options.items():
            try:
                value = eval(value)
                args.cfg_options[key] = value
            except:
                pass
        cfg.merge_from_dict(args.cfg_options)

    args.with_agent_type = args.with_agent_type or args.agent_type_first
    for key in ["work_dir", "env_cfg", "resume_from", "eval_cfg", "replay_cfg", "expert_replay_cfg", "recent_traj_replay_cfg", "rollout_cfg"]:
        cfg[key] = cfg.get(key, None)
    if args.debug:
        os.environ["PYRL_DEBUG"] = "True"
    elif "PYRL_DEBUG" not in os.environ:
        os.environ["PYRL_DEBUG"] = "False"
    if args.seed is None:
        args.seed = 3 #np.random.randint(2**32 - int(1E8))
    args.mode = "eval" if args.evaluation else "train"
    return args, cfg


def build_work_dir():
    if is_null(args.work_dir):
        root_dir = "./work_dirs"
        env_name = cfg.env_cfg.get("env_name", None) if is_not_null(cfg.env_cfg) else None
        config_name = osp.splitext(osp.basename(args.config))[0]
        folder_name = env_name if is_not_null(env_name) else config_name
        if args.with_agent_type:
            if args.agent_type_first:
                args.work_dir = osp.join(root_dir, agent_type, folder_name)
            else:
                args.work_dir = osp.join(root_dir, folder_name, agent_type)
        else:
            args.work_dir = osp.join(root_dir, folder_name)
    elif args.with_agent_type:
        if args.agent_type_first:
            colored_print("When you specify the work dir path, the agent type cannot be at the beginning of the path!", level="warning")
        args.work_dir = osp.join(args.work_dir, agent_type)

    if args.dev:
        args.work_dir = osp.join(args.work_dir, args.timestamp)

    if args.clean_up:
        if args.evaluation or args.auto_resume or (is_not_null(args.resume_from) and os.path.commonprefix(args.resume_from) == args.work_dir):
            colored_print("We will ignore the clean-up flag, since we are either in the evaluation mode or resuming from the directory!", level="warning")
        else:
            shutil.rmtree(args.work_dir, ignore_errors=True)
    os.makedirs(osp.abspath(args.work_dir), exist_ok=True)


def find_checkpoint():
    logger = get_logger()
    if is_not_null(args.resume_from):
        if is_not_null(cfg.resume_from):
            colored_print(f"The resumed checkpoint from the config file is overwritten by {args.resume_from}!", level="warning")
        cfg.resume_from = args.resume_from

    if args.auto_resume or (args.evaluation and is_null(cfg.resume_from)):
        logger.info(f"Search model under {args.work_dir}.")
        model_names = list(glob.glob(osp.join(args.work_dir, "models", "*.ckpt")))
        latest_index = -1
        latest_name = None
        for model_i in model_names:
            index_str = osp.basename(model_i).split(".")[0].split("_")[1]
            if index_str == 'final':
                continue
            index = eval(index_str)
            if index > latest_index:
                latest_index = index
                latest_name = model_i

        if is_null(latest_name):
            colored_print(f"Find no checkpoints under {args.work_dir}!", level="warning")
        else:
            cfg.resume_from = latest_name
            cfg.train_cfg["resume_steps"] = latest_index
    if is_not_null(cfg.resume_from):
        if isinstance(cfg.resume_from, str):
            cfg.resume_from = [
                cfg.resume_from,
            ]
        logger.info(f"Get {len(cfg.resume_from)} checkpoint {cfg.resume_from}.")
        logger.info(f"Check checkpoint {cfg.resume_from}!")

        for file in cfg.resume_from:
            if not (osp.exists(file) and osp.isfile(file)):
                logger.error(f"Checkpoint file {file} does not exist!")
                exit(-1)


def get_python_env_info():
    env_info_dict = collect_env()
    num_gpus = env_info_dict["Num of GPUs"]
    if is_not_null(args.num_gpus) and is_not_null(args.gpu_ids):
        colored_print("Please use either 'num-gpus' or 'gpu-ids'!", level="error")
        exit(0)

    if is_not_null(args.num_gpus):
        assert args.num_gpus <= num_gpus, f"We do not have {args.num_gpus} GPUs on this machine!"
        args.gpu_ids = list(range(args.num_gpus))
        args.num_gpus = None
    if args.gpu_ids is None:
        args.gpu_ids = []

    if len(args.gpu_ids) == 0 and num_gpus > 0:
        args.gpu_ids = list(range(num_gpus))
        args.num_gpus = None
        # colored_print(f"We will use cpu to do training, although we have {num_gpus} gpus available!", level="warning")

    if args.evaluation and len(args.gpu_ids) > 1:
        colored_print(f"Multiple GPU evaluation is not supported; we will use the first GPU to do evaluation!", level="warning")
        args.gpu_ids = args.gpu_ids[:1]
    args.env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])


def init_torch(args):
    import torch

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    rank = get_world_rank()
    if args.gpu_ids is not None and len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[rank])
        torch.set_num_threads(1)

    if is_debug_mode():
        torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main_rl(rollout, evaluator, replay, args, cfg, run=None, expert_replay=None, recent_traj_replay=None):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from maniskill2_learn.apis.train_rl import train_rl
    from maniskill2_learn.env import save_eval_statistics
    from maniskill2_learn.methods.builder import build_agent
    from maniskill2_learn.utils.data.converter import dict_to_str
    from maniskill2_learn.utils.torch import BaseAgent, BasePerception, load_checkpoint, save_checkpoint
    logger = get_logger()
    logger.info("Initialize torch!")
    init_torch(args)
    logger.info("Finish Initialize torch!")
    world_rank, world_size = get_dist_info()
    logger.info(f'current working directory: {os.getcwd()}')
    logger.info(f'PYTHONPATH: {sys.path}')

    from Evaluation.trajectory_analysis import TrajectoryAnalyzer

    if is_not_null(cfg.agent_cfg.get("batch_size", None)) and isinstance(cfg.agent_cfg.batch_size, (list, tuple)):
        assert len(cfg.agent_cfg.batch_size) == len(args.gpu_ids)
        cfg.agent_cfg.batch_size = cfg.agent_cfg.batch_size[world_rank]
        logger.info(f"Set batch size to {cfg.agent_cfg.batch_size}!")

    logger.info("Build agent!")
    agent = build_agent(cfg.agent_cfg)
    assert agent is not None, f"Agent type {cfg.agent_cfg.type} is not valid!"

    logger.info("Build perception!")
    perception = build_agent(cfg.get("perception_cfg", None))
    # assert perception is not None, f"Agent type {cfg.perception.type} is not valid!"

    logger.info(agent)
    logger.info(
        f'Num of parameters: {num_to_str(agent.num_trainable_parameters, unit="M")}, Model Size: {num_to_str(agent.size_trainable_parameters, unit="M")}'
    )
    if perception:
        logger.info('=========================================================')
        logger.info(perception)
        logger.info(
            f'Num of parameters: {num_to_str(perception.num_trainable_parameters, unit="M")}, Model Size: {num_to_str(perception.size_trainable_parameters, unit="M")}'
        )
    device = "cpu" if len(args.gpu_ids) == 0 else "cuda"
    #logger.error(f'device is: {device}')
    agent = agent.float().to(device)
    assert isinstance(agent, (BaseAgent, BasePerception)), "The agent object should be an instance of BaseAgent!"
    #logger.error(f'agent device is: {next(agent.parameters()).device}')

    #if isinstance(agent, BasePerception):
        #agent.checkDevice()

    if is_not_null(cfg.resume_from):
        logger.info("Resume agent with checkpoint!")
        for file in cfg.resume_from:
            if cfg.agent_cfg.type == 'VAT-SAC': # and cfg.agent_cfg.get('continuous_learning', None) is not None:
                args.resume_keys_map = {r'poseTrajectoryGenerator\.': 'perception.poseTrajectoryGenerator.',
                                        r'affordancePredictor\.': 'perception.affordancePredictor.'}
            load_checkpoint(agent, file, device, keys_map=args.resume_keys_map, logger=logger)


    #backend = cfg.train_cfg.pop('backend', 'nccl')
    if len(args.gpu_ids) > 1:
        logger.info("Setting DDP!")
        assert not args.evaluation, "We do not support multi-gpu evaluation!"

        logger.info(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
        logger.info(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
        logger.info("Initialize process group")
        #logger.info(f"BACKEND: {backend}")
        dist.init_process_group(backend='nccl', rank=world_rank, world_size=world_size)
        # from maniskill2_learn.utils.torch import ExtendedDDP
        agent = nn.SyncBatchNorm.convert_sync_batchnorm(agent)
        try:
            from torchsparse.nn.modules import SyncBatchNorm as SpSyncBatchNorm

            agent = SpSyncBatchNorm.convert_sync_batchnorm(agent)
        except:
            pass
        is_ppo = "ppo" in cfg.agent_cfg.type.lower()
        logger.info("agent to DDP")
        agent.to_ddp(device_ids=["cuda"], find_unused_parameters=not is_ppo)
        #if isinstance(agent, BasePerception):
            #agent.checkDevice()

    logger.info(f"Work directory of this run {args.work_dir}")
    if len(args.gpu_ids) > 0:
        logger.info(f"Train over GPU {args.gpu_ids}!")
    else:
        logger.info(f"Train over CPU!")

    if not args.evaluation:
        train_rl(
            agent,
            rollout,
            evaluator,
            replay,
            work_dir=args.work_dir,
            eval_cfg=cfg.eval_cfg,
            expert_replay=expert_replay,
            recent_traj_replay=recent_traj_replay,
            run=run,
            **cfg.train_cfg,
        )
    else:
        agent.eval()
        agent.set_mode("test")

        if is_not_null(replay) and args.reg_loss:
            loss_dict = agent.compute_test_loss(replay)
            logger.info(dict_to_str(loss_dict))
        if is_not_null(evaluator):
            # For RL
            lens, rewards, finishes = evaluator.run(agent, work_dir=args.work_dir, wandb_run=run, **cfg.eval_cfg)
            save_eval_statistics(args.work_dir, lens, rewards, finishes)

            # the TrajectoryAnalyzer will create plots and compute metrics for the dataset and sync them to
            # wandb
            from maniskill2_learn.apis.wandb_setup import find_file_in_dir
            dataset_file = find_file_in_dir(args.work_dir, ending='.h5')
            # this is needed, so I can treat the num_waypoints param correctly inside the TrajectoryAnalyzer
            vat = cfg.agent_cfg.type == 'VAT-Mart'
            analyzer = TrajectoryAnalyzer(file=dataset_file, num_waypoints=cfg.env_cfg.num_waypoints, wandb_run=run, vat=vat)
            analyzer.process()
            if run is not None and args.upload_dataset:
                from maniskill2_learn.apis.wandb_setup import upload_dataset
                upload_dataset(dataset_file, cfg, args, run)

        agent.train()
        agent.set_mode("train")

    if len(args.gpu_ids) > 1:
        print("destroying process group")
        dist.destroy_process_group()

def run_one_process(rank, world_size, port, args, cfg):
    import numpy as np

    if rank == 0 and args.use_wandb:
        # The main process continues the run and logs data
        # I re-init the config, because the values have changed since I populated them with the meta information form
        # the artifacts
        # I use that config as the meta data when downloading the artifact produced by this run
        run = wandb.init(project="MasterThesis", resume="must", reinit=True, sync_tensorboard=True, dir=args.work_dir, config=cfg)
    else:
        # Other processes either log silently or not at all
        wandb.init(mode="disabled")
        run = None

    np.set_printoptions(3)
    args.seed += (rank * 2022)

    add_dist_var(rank, world_size, free_port=port)
    set_random_seed(args.seed)

    if is_not_null(cfg.env_cfg) and len(args.gpu_ids) > 0:
        if args.sim_gpu_ids is not None:
            assert len(args.sim_gpu_ids) == len(args.gpu_ids), "Number of simulation gpus should be the same as the number of training gpus!"
        else:
            args.sim_gpu_ids = args.gpu_ids
        cfg.env_cfg.device = f"cuda:{args.sim_gpu_ids[rank]}"

    work_dir = args.work_dir
    #logger_file = osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.log")
    logger_file = osp.join(work_dir, f"run.log")
    logger = get_logger(name=None, log_file=logger_file, log_level=cfg.get("log_level", "INFO"))

    if is_debug_mode():
        dash_line = "-" * 60 + "\n"
        logger.info("Environment info:\n" + dash_line + args.env_info + "\n" + dash_line)

    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Set random seed to {args.seed}")

    # Create replay buffer for RL
    if is_not_null(cfg.replay_cfg) and (not args.evaluation or (args.reg_loss and cfg.replay_cfg.get("buffer_filenames", None) is not None)):
        logger.info(f"Build replay buffer!")
        from maniskill2_learn.env import build_replay
        replay = build_replay(cfg.replay_cfg)
        expert_replay, recent_traj_replay = None, None
        if is_not_null(cfg.expert_replay_cfg):
            assert cfg.expert_replay_cfg.buffer_filenames is not None
            expert_replay = build_replay(cfg.expert_replay_cfg)
        if is_not_null(cfg.recent_traj_replay_cfg):
            recent_traj_replay = build_replay(cfg.recent_traj_replay_cfg)
    else:
        replay = None
        expert_replay = None
        recent_traj_replay = None

    # Create rollout module for online methods
    if not args.evaluation and is_not_null(cfg.rollout_cfg):
        from maniskill2_learn.env import build_rollout

        logger.info(f"Build rollout!")
        rollout_cfg = cfg.rollout_cfg
        rollout_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
        rollout_cfg['seed'] = args.seed #np.random.randint(0, int(1E9))
        rollout_cfg['work_dir'] = args.work_dir
        # this is the creation of the environment and all its wrappers
        rollout = build_rollout(rollout_cfg)
    else:
        rollout = None

    # Build evaluation module
    # we will do multi-GPU evaluation when we train affordancePredictor, else onlu GPU 0 will do evaluation
    # the reason is that the affordancePredictor uses the other two networks during its evaluation
    # this leads to deadlocks when only one GPU does the evaluation.
    # to avoid deadlocks all GPUs need to do it.
    if is_not_null(cfg.eval_cfg) and (rank == 0 or cfg.agent_cfg.get('mode', None) == 'affordancePredictor'):
        # Only the first process will do evaluation
        from maniskill2_learn.env import build_evaluation

        logger.error(f"Build evaluation!")
        eval_cfg = cfg.eval_cfg
        # Evaluation environment setup can be different from the training set-up. (Like early-stop or object sets)
        if eval_cfg.get("env_cfg", None) is None:
            eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
        else:
            tmp = eval_cfg["env_cfg"]
            eval_cfg["env_cfg"] = deepcopy(cfg.env_cfg)
            eval_cfg["env_cfg"].update(tmp)
        get_logger().info(f"Building evaluation: eval_cfg: {eval_cfg}")
        eval_cfg['seed'] = args.seed + (100 * (rank+1)) # np.random.randint(0, int(1E9))
        get_logger().info(f"evaluation environment seed: {eval_cfg['seed']}")
        # eval_cfg['wandb_run'] = run
        evaluator = build_evaluation(eval_cfg)
    else:
        logger.error(f"evaluator is None")
        evaluator = None

    # Get environments information for agents
    obs_shape, action_shape = None, None
    trajectory_dim = None
    n_points = None
    env_params = None
    if is_not_null(cfg.env_cfg):
        # For RL which needs environments
        logger.info(f"Get obs shape!")
        from maniskill2_learn.env import get_env_info

        if rollout is not None:
            env_params = get_env_info(cfg.env_cfg, rollout.vec_env)
        elif hasattr(evaluator, 'vec_env'):
            env_params = get_env_info(cfg.env_cfg, evaluator.vec_env)
        else:
            env_params = get_env_info(cfg.env_cfg)

        obs_shape = env_params["obs_shape"]
        action_shape = env_params["action_shape"]

        env_params["control_mode"] = cfg.env_cfg['control_mode']  # this is needed for VAT inference!
        if cfg.env_cfg.get('num_waypoints', None) is not None:
            if cfg.agent_cfg.type == 'VAT-Mart':  # VAT inference!
                env_params["num_waypoints"] = cfg.env_cfg['num_waypoints'] - 1
                initial_pose_dim = 6
                n_points = cfg.env_cfg.get('n_points', 650)
                waypoint_dim = action_shape if action_shape < 7 else 6
                trajectory_dim = initial_pose_dim + waypoint_dim * env_params["num_waypoints"]
            else:
                # env_params["num_waypoints"] = cfg.env_cfg['num_waypoints'] - 1
                # the following is needed to closed loop approach that utilizes the trajectory generator
                initial_pose_dim = 6
                waypoint_dim = 6
                trajectory_dim = initial_pose_dim + waypoint_dim * 8
                n_points = cfg.env_cfg.get('n_points', 1200)  # number of points in the point_cloud

        logger.info(f'State shape:{env_params["obs_shape"]}, action shape:{env_params["action_shape"]},'
                    f' trajectory dimension: {trajectory_dim}, n_points: {n_points}')

    elif is_not_null(replay):
        obs_shape = None
        action_shape = None
        trajectory_dim = None
        for obs_key in ["inputs", "obs"]:
            if obs_key in replay.memory:
                obs_shape = replay.memory.slice(0).shape[obs_key]
                break
        if 'actions' in replay.memory:
            action_shape = replay.memory.slice(0).shape['actions']  # will give correct output of RL agent action dim
        if 'infos' in replay.memory:
            # Training Perception Modules on data set
            if 'waypoints' in replay.memory['infos']:
                if cfg.agent_cfg.type == 'VAT-Mart':
                    # training baseline VAT
                    len_waypoints = replay.memory.slice(0).shape['infos']['waypoints']
                    assert len_waypoints % action_shape == 0, 'Length of waypoint array is not a multiple of action dimension!'
                    num_wp = int(len_waypoints / action_shape)
                    cfg.agent_cfg['waypoint_dim'] = action_shape
                    cfg.agent_cfg['num_waypoints'] = num_wp
                    # this is done to replace the 'action_shape' in the config file for the VAT networks
                    initial_pose_dim = 6
                    trajectory_dim = initial_pose_dim + action_shape * num_wp
                    n_points = 650
                elif cfg.agent_cfg.type == 'PoseTrajectoryGenerator':
                    # Training Trajectory Generator for closed loop agent
                    len_waypoints = replay.memory.slice(0).shape['infos']['waypoints']
                    assert len_waypoints % action_shape == 0, 'Length of waypoint array is not a multiple of action dimension!'
                    initial_pose_dim = 6
                    action_shape = 6
                    num_wp = 8
                    cfg.agent_cfg['waypoint_dim'] = action_shape
                    cfg.agent_cfg['num_waypoints'] = num_wp
                    n_points = 650
                    trajectory_dim = initial_pose_dim + action_shape * num_wp

        logger.info(f'State shape:{obs_shape}, action shape:{action_shape}, trajectory_dim: {trajectory_dim}')

    if is_not_null(obs_shape) or is_not_null(action_shape) or is_not_null(trajectory_dim) or is_not_null(n_points):
        from maniskill2_learn.networks.utils import get_kwargs_from_shape, replace_placeholder_with_args

        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape, trajectory_dim, n_points)
        cfg = replace_placeholder_with_args(cfg, **replaceable_kwargs)

        #if rank == 0:
            #cfg.dump(osp.join(args.work_dir, f"meta.py"))

    if env_params is not None:
        cfg.agent_cfg["env_params"] = env_params

    logger.info(f"Final agent config:\n{cfg.agent_cfg}")

    # Output version of important packages
    log_meta_info(logger)

    main_rl(rollout, evaluator, replay, args, cfg, run=run, expert_replay=expert_replay, recent_traj_replay=recent_traj_replay)

    if is_not_null(evaluator):
        evaluator.close()
        logger.info("Close evaluator object")
    if is_not_null(rollout):
        rollout.close()
        logger.info("Close rollout object")
    if is_not_null(replay):
        replay.close()
        logger.info("Delete replay buffer")

def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def main():
    if args.use_wandb:
        from maniskill2_learn.apis.wandb_setup import setup_wandb
        setup_wandb(args, cfg)
        print("USING WANDB")
    else:
        wandb.init(mode="disabled")

    find_checkpoint()
    if len(args.gpu_ids) > 1:
        import torch.multiprocessing as mp
        print("using multiple GPUs")
        port = find_free_port()
        world_size = len(args.gpu_ids)
        mp.spawn(run_one_process, args=(world_size, port, args, cfg), nprocs=world_size, join=True)
    else:
        print("using single GPU")
        port = 12355
        run_one_process(0, 1, port, args, cfg)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Remove mujoco_py lock
    mjpy_lock = Path(gym.__file__).parent.parent / "mujoco_py/generated/mujocopy-buildlock.lock"
    if mjpy_lock.exists():
        os.remove(str(mjpy_lock))

    add_env_var()

    args, cfg = parse_args()
    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    agent_type = cfg.agent_cfg.type

    build_work_dir()
    #find_checkpoint()
    get_python_env_info()

    work_dir = args.work_dir
    if args.evaluation:
        test_name = args.test_name if args.test_name is not None else "test"
        work_dir = osp.join(work_dir, test_name)
        # Always clean up for evaluation
        shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir

    '''# save meta-data as json
    meta_path = os.path.join(args.work_dir, 'meta.json')
    with open(meta_path, 'w') as json_file:
        json.dump(cfg, json_file, indent=4)'''

    logger_name = cfg.env_cfg.env_name if is_not_null(cfg.env_cfg) else cfg.agent_cfg.type
    args.name_suffix = f"{args.mode}"
    if args.test_name is not None:
        args.name_suffix += f"-{args.test_name}"
    os.environ["PYRL_LOGGER_NAME"] = f"{logger_name}-{args.name_suffix}"
    #cfg.dump(osp.join(work_dir, f"{args.timestamp}-{args.name_suffix}.py"))
    cfg.dump(osp.join(work_dir, f"meta.py"))

    main()
