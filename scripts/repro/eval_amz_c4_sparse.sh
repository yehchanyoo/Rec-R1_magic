
DOMAINS=('Video_Games' 'Baby' 'Office' 'Sports')
MODEL_PATH=checkpoints/adv-ml-project/qwen2.5-3b-inst-ppo-amazon-c4-sparse-20250531_223909/actor/global_step_500
DATA_PATH=data/amazon_c4/inst/sparse/subset_other/test.parquet
MODEL_NAME=rec-r1
SAVE_DIR=results/repro/amazon_c4


for DOMAIN in "${DOMAINS[@]}"; do
    echo "Processing $DOMAIN"
    TEST_FILE_PATH="$SAVE_DIR/${MODEL_NAME}_${DOMAIN}.json"

    python src/eval/amazon_c4/model_generate.py \
        --domain_name $DOMAIN \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --save_dir $SAVE_DIR

    echo "$DOMAIN Score"
    python src/eval_search/BM25/amazon_c4.py \
        --res_path $TEST_FILE_PATH

done