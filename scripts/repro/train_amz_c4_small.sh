export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export DATA_DIR=data/matching/qwen-instruct
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=matching-qwen2.5-0.5b-inst-ppo
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HOME="~/.cache/huggingface"
export CUDA_VISIBLE_DEVICES=0

bash scripts/train/train_rec-amazon_c4.sh