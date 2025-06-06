#MODEL_NAME=checkpoints/Rec-R1-esci/esci-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_1400
MODEL_NAME=checkpoints/adv-ml-project/qwen2.5-3b-inst-ppo-esci_sparse-20250605_015919/actor/global_step_700
echo $MODEL_NAME

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks humaneval \
    --device cuda:0 \
    --batch_size 16 

# --model_args pretrained=checkpoints/Rec-R1-esci/esci-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_1400 \

# Qwen/Qwen2.5-3B-Instruct

# ifeval
