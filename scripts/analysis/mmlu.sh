lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct \
    --tasks ifeval \
    --device cuda:0 \
    --batch_size 32


# --model_args pretrained=checkpoints/Rec-R1-esci/esci-qwen2.5-3b-inst-grpo-2gpus/actor/global_step_1400 \