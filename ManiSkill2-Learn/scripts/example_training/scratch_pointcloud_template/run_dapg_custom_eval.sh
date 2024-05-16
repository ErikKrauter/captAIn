python maniskill2_learn/apis/run_rl.py configs/mfrl/dapg/maniskill2_pn.py \
            --work-dir PPO_DAPG_Baseline --gpu-ids 0 --evaluation --resume-from PPO_DAPG_Baseline/models/model_2000000.ckpt \
            --cfg-options "eval_cfg.save_video=True" "eval_cfg.num=20"
# To manually evaluate the model, add --evaluation and --resume-from YOUR_LOGGING_DIRECTORY/models/SOME_CHECKPOINT.ckpt 
# to the above commands.

# Using multiple GPUs will increase training speed; 
# Note that train_cfg.n_steps will also be multiplied by the number of gpus you use, so you may want to divide it by the number of gpus
