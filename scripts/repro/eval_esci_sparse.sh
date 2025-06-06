DOMAINS=('Video_Games' 'Baby_Products' 'Office_Products' 'Sports_and_Outdoors')
MODEL_PATH=/home/rapids/Rec-R1_magic/checkpoints/adv-ml-project/qwen2.5-3b-inst-ppo-esci_sparse-20250605_015919/actor/global_step_700
DATA_PATH=data/esci/inst/sparse/subset/test.parquet
MODEL_NAME=rec-r1
SAVE_DIR=results/repro/esci


for DOMAIN in "${DOMAINS[@]}"; do
    echo "Processing $DOMAIN"
    TEST_FILE_PATH="$SAVE_DIR/${MODEL_NAME}_${DOMAIN}.json"

    python src/eval/esci/model_generate.py \
        --domain_name $DOMAIN \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --model_name $MODEL_NAME \
        --save_dir $SAVE_DIR

    echo "$DOMAIN Score"
    python src/eval_search/BM25/esci2.py \
        --res_path $TEST_FILE_PATH
done
