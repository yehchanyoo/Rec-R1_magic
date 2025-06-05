# Running from the root of the directory

# After running Part 2 of data prep code,
# start training
bash scripts/repro/train_rec-amazon_review_3b.sh

# After training is done, run eval code
bash scripts/eval/amazon_review/inst_gen/rec-r1.sh inductive
bash scripts/eval/amazon_review/inst_gen/rec-r1.sh transductive

# Then, run eval search code
bash scripts/eval_search/amazon_review/sparse/eval_search.sh inductive
bash scripts/eval_search/amazon_review/sparse/eval_search.sh transductive
