# Running from the root directory of the repo

# Amazon Review data prep code
# Part 1: Needs to be only run once
python src/dataset/amazon_review/process_amazon_2023.py
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz
gunzip meta_All_Beauty.jsonl.gz
mv meta_All_Beauty.jsonl data/amazon_review/raw
python src/dataset/amazon_review/built_corpus.py
