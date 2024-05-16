import os
import wandb
import json
from maniskill2_learn.utils.meta import Config

def find_file_in_dir(directory, ending=".ckpt"):
    """
    Recursively searches for the first .h5 file in the given directory and its subdirectories.
    Returns the full path to the .h5 file if found, or None if no .h5 file is found.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ending):
                return os.path.join(root, file)
    return None


def data_collection_rl(args, cfg, run):
    # args.resume_from contains the name and version of the artifact to download
    rl_model = run.use_artifact(str(args.resume_from[0]), type='model')
    producer_run = rl_model.logged_by()
    # meta_data now contains the configuration under which the rl agent was trained
    meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])

    # rl model dir is the path at which the artifact is stored, namely work_dir/model
    rl_model_dir = rl_model.download(root=os.path.join(args.work_dir, 'model'))

    # now I need to overwrite args.resume_from with the location of the checkpoint
    # because later args.resume_from is used to load the model weights

    # rl_model_file is the path to the .ckpt
    rl_model_file = find_file_in_dir(rl_model_dir, ending='.ckpt')
    # we populate resume_from with the path to the checkpoint
    # this is a neat intergration in the existing workflow of loading and resuming training
    args.resume_from = rl_model_file

    # now we use the metadata of the rl agent to setup the correct hyperparameters for data collection

    # we must align the action space for data collection with the action space the RL agent was trained with
    restrict_action_space = meta_data.env_cfg.restrict_action_space
    cfg.env_cfg["restrict_action_space"] = restrict_action_space

    # if the number of waypoints is NOT specified in the config file for data collection
    # we will use the number of waypoints that the RL agent was trained with
    num_waypoints = meta_data.env_cfg.num_waypoints
    if cfg.env_cfg.get("num_waypoints", None) is None:
        cfg.env_cfg['num_waypoints'] = num_waypoints


def resume_training_rl(args, cfg, run):
    # args.resume_from contains the name and version of the artifact to download
    rl_model = run.use_artifact(str(args.resume_from[0]), type='model')
    producer_run = rl_model.logged_by()
    # meta_data now contains the configuration under which the rl agent was trained
    meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])

    # rl model dir is the path at which the artifact is stored, namely work_dir/model
    rl_model_dir = rl_model.download(root=os.path.join(args.work_dir, 'model'))

    # now I need to overwrite args.resume_from with the location of the checkpoint
    # because later args.resume_from is used to load the model weights

    # rl_model_file is the path to the .ckpt
    rl_model_file = find_file_in_dir(rl_model_dir, ending='.ckpt')
    # we populate resume_from with the path to the checkpoint
    # this is a neat intergration in the existing workflow of loading and resuming training
    args.resume_from = rl_model_file

    for k in cfg.keys():
        cfg[k] = meta_data[k]


def inference_vat(args, cfg, run):

    if args.resume_from is not None:
        # need to only download affordance predictor
        affordancePredictor = run.use_artifact(args.aff_model, type='model')
        affordancePredictor.download(root=args.work_dir)
        affordancePredictor_file = find_file_in_dir(args.work_dir, ending=".ckpt")

        # during construction of VAT agent for inference, the cfg.agent_cfg.affordance_predictor_checkpoint_path
        # is used to load the model
        if affordancePredictor_file is not None:
            cfg.agent_cfg.affordance_predictor_checkpoint_path = affordancePredictor_file
        else:
            print('Downloading Or Finding Affordance Predictor Checkpoint Failed')
            exit()
    else:
        # need to download all three models seperately
        affordancePredictor = run.use_artifact(args.aff_model, type='model')
        trajectoryGenerator = run.use_artifact(args.gen_model, type='model')
        trajectoryScorer = run.use_artifact(args.scor_model, type='model')

        aff_dir = affordancePredictor.download(root=os.path.join(args.work_dir, 'affordancePredictor'))
        gen_dir = trajectoryGenerator.download(root=os.path.join(args.work_dir, 'trajectoryGenerator'))
        scor_dir = trajectoryScorer.download(root=os.path.join(args.work_dir, 'trajectoryScorer'))

        affordancePredictor_file = find_file_in_dir(aff_dir, ending=".ckpt")
        trajectoryGenerator_file = find_file_in_dir(gen_dir, ending=".ckpt")
        trajectoryScorer_file = find_file_in_dir(scor_dir, ending=".ckpt")

        # during construction of VAT agent for inference, the cfg.agent_cfg.affordance_predictor_checkpoint_path ...
        # are used to load all perception networks separately

        if trajectoryScorer_file is not None:
            cfg.agent_cfg.trajectory_scorer_checkpoint_path = trajectoryScorer_file
        else:
            print('Downloading Or Finding Trajectory Scorer Checkpoint Failed')
            exit()
        if trajectoryGenerator_file is not None:
            cfg.agent_cfg.trajectory_generator_checkpoint_path = trajectoryGenerator_file
        else:
            print('Downloading Or Finding Trajectory Generator Checkpoint Failed')
            exit()
        if affordancePredictor_file is not None:
            cfg.agent_cfg.affordance_predictor_checkpoint_path = affordancePredictor_file
        else:
            print('Downloading Or Finding Affordance Predictor Checkpoint Failed')
            exit()

        if args.pose_gen_model != '':
            poseTrajectoryGenerator = run.use_artifact(args.pose_gen_model, type='model')
            pose_gen_dir = poseTrajectoryGenerator.download(root=os.path.join(args.work_dir, 'poseTrajectoryGenerator'))
            poseTrajectoryGenerator_file = find_file_in_dir(pose_gen_dir, ending=".ckpt")
            if poseTrajectoryGenerator_file is not None:
                cfg.agent_cfg.pose_trajectory_generator_checkpoint_path = poseTrajectoryGenerator_file
            else:
                print('Downloading Or Finding Affordance Predictor Checkpoint Failed')
                exit()

        # VAT inference
        # I must set the action dimension and number of waypoints
        # to the one from the data collection environment

        # During VAT inference I do not have direct access to the data set because its not needed for
        # inference. However, I do have the path to the VAT model, and they have the correct number of waypoints
        # and waypoint dimension in their metafile

        # we use the affordancePredictor, because it is used in both cases (all models separately and not)
        producer_run = affordancePredictor.logged_by()
        # meta_data now contains the configuration under which the rl agent was trained
        meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])

        waypoint_dim = meta_data.agent_cfg.waypoint_dim
        num_waypoints = meta_data.agent_cfg.num_waypoints
        restrict_action_space = True if waypoint_dim == 4 else False
        cfg.env_cfg["restrict_action_space"] = restrict_action_space
        # +1 is important here, because VAT has one action more that the RL agent used to collect trajectories
        cfg.env_cfg['num_waypoints'] = num_waypoints + 1


def vat_training(args, cfg, run):

    mode = cfg.agent_cfg.mode

    if mode == 'affordancePredictor':
        training_dataset = run.use_artifact(f'train_dataset_affordance_predictor:{args.train_dataset_version}',
                                            type='dataset')
        validation_dataset = run.use_artifact(f'eval_dataset_affordance_predictor:{args.eval_dataset_version}',
                                              type='dataset')
    else:
        training_dataset = run.use_artifact(f'train_dataset:{args.train_dataset_version}', type='dataset')
        validation_dataset = run.use_artifact(f'eval_dataset:{args.eval_dataset_version}', type='dataset')

    training_dataset_dir = training_dataset.download(root=os.path.join(args.work_dir, 'train_dataset'))
    validation_dataset_dir = validation_dataset.download(root=os.path.join(args.work_dir, 'eval_dataset'))

    train_file = find_file_in_dir(training_dataset_dir, ending='.h5')
    val_file = find_file_in_dir(validation_dataset_dir, ending='.h5')

    # light integration into the training pipeline
    # buffer_filenames is used during construction of replay buffer to load the datasets
    if train_file is not None and val_file is not None:
        cfg.eval_cfg.eval_replay.buffer_filenames = [val_file]
        cfg.replay_cfg.buffer_filenames = [train_file]
    else:
        print('Downloading or finding the datasets failed')
        exit()

    producer_run = training_dataset.logged_by()
    # meta_data now contains the configuration under which the data was collected
    meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])
    waypoint_dim = 4 if meta_data.env_cfg.restrict_action_space else 6
    num_wp = meta_data.env_cfg.num_waypoints
    # to correctly construct the network architecture of the VAT models,
    # I need to now the action dimension/ waypoint dimension and the number of waypoints
    cfg.agent_cfg['waypoint_dim'] = waypoint_dim
    cfg.agent_cfg['num_waypoints'] = num_wp


def closed_loop_training(args, cfg, run):
    affordancePredictor = run.use_artifact(args.aff_model, type='model')
    trajectoryGenerator = run.use_artifact(args.gen_model, type='model')

    aff_dir = affordancePredictor.download(root=os.path.join(args.work_dir, 'affordancePredictor'))
    gen_dir = trajectoryGenerator.download(root=os.path.join(args.work_dir, 'poseTrajectoryGenerator'))

    affordancePredictor_file = find_file_in_dir(aff_dir, ending=".ckpt")
    trajectoryGenerator_file = find_file_in_dir(gen_dir, ending=".ckpt")

    # during construction of VAT agent for inference, the cfg.agent_cfg.affordance_predictor_checkpoint_path ...
    # are used to load all perception networks separately

    if trajectoryGenerator_file is not None:
        cfg.agent_cfg.trajectory_generator_cfg.trajectory_generator_checkpoint_path = trajectoryGenerator_file
    else:
        print('Downloading Or Finding Trajectory Generator Checkpoint Failed')
        exit()
    if affordancePredictor_file is not None:
        cfg.agent_cfg.affordance_predictor_cfg.affordance_predictor_checkpoint_path = affordancePredictor_file
    else:
        print('Downloading Or Finding Affordance Predictor Checkpoint Failed')
        exit()

def pose_trajectory_generator_training(args, cfg, run):

    training_dataset = run.use_artifact(f'train_dataset:{args.train_dataset_version}', type='dataset')
    validation_dataset = run.use_artifact(f'eval_dataset:{args.eval_dataset_version}', type='dataset')

    training_dataset_dir = training_dataset.download(root=os.path.join(args.work_dir, 'train_dataset'))
    validation_dataset_dir = validation_dataset.download(root=os.path.join(args.work_dir, 'eval_dataset'))

    train_file = find_file_in_dir(training_dataset_dir, ending='.h5')
    val_file = find_file_in_dir(validation_dataset_dir, ending='.h5')

    # light integration into the training pipeline
    # buffer_filenames is used during construction of replay buffer to load the datasets
    if train_file is not None and val_file is not None:
        cfg.eval_cfg.eval_replay.buffer_filenames = [val_file]
        cfg.replay_cfg.buffer_filenames = [train_file]
    else:
        print('Downloading or finding the datasets failed')
        exit()

    producer_run = training_dataset.logged_by()
    # meta_data now contains the configuration under which the data was collected
    meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])
    waypoint_dim = 6
    num_wp = meta_data.env_cfg.num_waypoints
    # to correctly construct the network architecture of the VAT models,
    # I need to now the action dimension/ waypoint dimension and the number of waypoints
    cfg.agent_cfg['waypoint_dim'] = waypoint_dim
    cfg.agent_cfg['num_waypoints'] = num_wp


def inference_closed_loop(args, cfg, run):
    # args.resume_from contains the name and version of the artifact to download
    rl_model = run.use_artifact(str(args.resume_from[0]), type='model')
    producer_run = rl_model.logged_by()
    # meta_data now contains the configuration under which the rl agent was trained
    meta_data = Config(json.loads(producer_run.json_config)['_cfg_dict']['value'])

    # rl model dir is the path at which the artifact is stored, namely work_dir/model
    rl_model_dir = rl_model.download(root=os.path.join(args.work_dir, 'model'))

    # now I need to overwrite args.resume_from with the location of the checkpoint
    # because later args.resume_from is used to load the model weights

    # rl_model_file is the path to the .ckpt
    rl_model_file = find_file_in_dir(rl_model_dir, ending='.ckpt')
    # we populate resume_from with the path to the checkpoint
    # this is a neat intergration in the existing workflow of loading and resuming training
    args.resume_from = rl_model_file

    # make sure I use the same observatiosn and reward formulations when i do inference
    # or continuous learning
    faucets = cfg['env_cfg'].get('faucets', None)
    damping = cfg['env_cfg'].get('damping', None)
    friction = cfg['env_cfg'].get('friction', None)
    back_cam = cfg['env_cfg'].get('back_cam', None)
    randomize_physical_properties = cfg['env_cfg'].get('randomize_physical_properties', None)
    randomize_initial_faucet_pose = cfg['env_cfg'].get('randomize_initial_faucet_pose', None)
    cfg['env_cfg'].update(meta_data['env_cfg'])
    if faucets is not None:
        cfg['env_cfg']['faucets'] = faucets
    if damping is not None:
        cfg['env_cfg']['damping'] = damping
    if friction is not None:
        cfg['env_cfg']['friction'] = friction
    if back_cam is not None:
        cfg['env_cfg']['back_cam'] = back_cam
    if randomize_initial_faucet_pose is not None:
        cfg['env_cfg']['randomize_initial_faucet_pose'] = randomize_initial_faucet_pose
    if randomize_physical_properties is not None:
        cfg['env_cfg']['randomize_physical_properties'] = randomize_physical_properties

def upload_dataset(file, cfg, args, run):
    if cfg.agent_cfg.type != 'SAC':
        return

    # we want to collect the dataset of RL trajectories for training VAT later
    type = 'train' if args.seed == 3 else 'eval'
    if cfg.eval_cfg.get('affordance_predictor_data_set', False):
        name = f'{type}_dataset_affordance_predictor'
    else:
        name = f'{type}_dataset'

    dataset = wandb.Artifact(name=name, type="dataset")

    # also add meta.py file
    meta_file_path = os.path.join(os.path.dirname(file), 'meta.py')

    if file is not None and meta_file_path is not None:
        dataset.add_file(str(file))
        dataset.add_file(str(meta_file_path))
        run.log_artifact(dataset)
    else:
        print(f'Could Not Find Dataset To Upload or meta file of the dataset: file: {file}, meta: {meta_file_path}')
        exit()


def setup_wandb(args, cfg):
    run_id = wandb.util.generate_id()
    os.environ["WANDB_RUN_ID"] = run_id
    # Insert your wandb api key here...
    wandb.login(key="INSERT_YOUR_KEY")
    run = wandb.init(project="MasterThesis", sync_tensorboard=True, id=run_id, reinit=True,
                         save_code=True, dir=args.work_dir, config=cfg, notes="", tags=[args.wandb_group])  # name=args.wandb_group , group=args.wandb_group)

    if args.evaluation:
        if cfg.agent_cfg.type == 'SAC':
            # its data collection using rl agent
            data_collection_rl(args, cfg, run)
        elif cfg.agent_cfg.type == 'VAT-SAC':
            # its inference using the closed loop approach
            inference_closed_loop(args, cfg, run)
        elif cfg.agent_cfg.type == 'VAT-Mart':
            # its inference using VAT
            inference_vat(args, cfg, run)
    elif args.resume_from is not None:
        if cfg.agent_cfg.type == 'VAT-SAC':
            # its continous learning
            inference_closed_loop(args, cfg, run)
    else:
        if cfg.agent_cfg.type == 'VAT-Mart':
            # its VAT training so we need to download the dataset
            vat_training(args, cfg, run)
        elif cfg.agent_cfg.type == 'VAT-SAC':
            # its closed loop training so we need to dowload perception modules
            closed_loop_training(args, cfg, run)
        elif cfg.agent_cfg.type == 'PoseTrajectoryGenerator':
            # need to download the datasets
            pose_trajectory_generator_training(args, cfg, run)
        else:
            # its rl training so we only might need to download a model checkpoint to resume training
            if args.resume_from is not None:
                resume_training_rl(args, cfg, run)

    # update the meta.py after having populated the cfg with the infor from the artifacts
    cfg.dump(os.path.join(args.work_dir, f"meta.py"))