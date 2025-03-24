DOMAIN_NAME=$1
QUERY_GEN_MODEL_NAME=Qwen-inst
# QUERY_GEN_MODEL_NAME=gpt-4o
TEST_FILE_PATH=results/amazon_review/$QUERY_GEN_MODEL_NAME-amazon-review_$DOMAIN_NAME.json
INDEX_DIR=database/amazon_review/$DOMAIN_NAME/pyserini_index

python src/eval_search/BM25/amazon_review.py \
    --res_path $TEST_FILE_PATH \
    --index_dir $INDEX_DIR \