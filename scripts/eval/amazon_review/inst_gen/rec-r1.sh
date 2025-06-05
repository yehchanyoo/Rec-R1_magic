DOMAIN_NAME=$1
MODEL_PATH=checkpoints/adv-ml-project/matching-qwen3b-inst-ppo/actor/global_step_400
DATA_PATH=data/amazon_review/inst/test.parquet
SAVE_DIR=results/amazon_review
MODEL_NAME=Rec-r1-amazon-review


CUDA_VISIBLE_DEVICES=1 python src/eval/amazon_review/model_generate.py \
    --domain_name $DOMAIN_NAME \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME
