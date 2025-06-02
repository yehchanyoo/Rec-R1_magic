# NOTE: Currently buggy -- use this code at your peril

## Changes made:
# N_GPUS = 2 -> 1
# ROLLOUT_TP_SIZE = 2 -> 1
# CUDA_VISIBLE_DEVICES = 0,1 -> 0
# BASE_MODEL = Qwen/Qwen2.5-3B-Instruct -> Qwen/Qwen2.5-0.5B-Instruct
# EXPERIMENT_NAME = matching-qwen-0.5b-inst-ppo
export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export DATA_DIR=data/amazon_review/inst
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=matching-qwen0.5b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME="/home/rapids/.cache/huggingface"
export PROJECT_NAME="adv-ml-project"
export CUDA_VISIBLE_DEVICES=0

DATE=$(date '+%Y-%m-%d-%H-%M-%S')

# Start Ray head node (using 16 cpus/task and 1 L40S at the time)
ray start --head --num-cpus=16 --num-gpus=1

# Wait for Ray to initialize
sleep 5

## Changes made:
# rollout.n = 12 -> 6
# gpu_memory_utilization = 0.3 -> 0.7
# ppo_mini_batch_size = 128 -> 64
# grad_offload = True -> False
# optimizer_offload = True -> False
# total_epochs = 50 -> 5
# train_files and val_files changed to list of string
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files='["'"$DATA_DIR"'/train.parquet"]' \
    data.val_files='["'"$DATA_DIR"'/val.parquet"]' \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +trainer.num_gpus=1 \
    +trainer.num_cpus=16 \
    +actor_rollout_ref.model.tensor_parallel_size=1

# Stop ray processes (only my own)
ray stop
