DOMAIN_NAME=$1
# QUERY_GEN_MODEL_NAME=Qwen-inst
# QUERY_GEN_MODEL_NAME=gpt-4o
# QUERY_GEN_MODEL_NAME=Qwen-sft
# QUERY_GEN_MODEL_NAME=Qwen-sft-no-reason
# QUERY_GEN_MODEL_NAME=Qwen-sft-no-json-no-reason
QUERY_GEN_MODEL_NAME=Qwen-sft-with-reason-no-json
TEST_FILE_PATH=results/esci/$QUERY_GEN_MODEL_NAME-esci_$DOMAIN_NAME.json


python src/eval_search/BM25/esci.py \
    --res_path $TEST_FILE_PATH