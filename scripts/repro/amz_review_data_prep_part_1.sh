# Running from the root directory of the repo

# Amazon Review data prep code
# Part 1: Needs to be only run once
# (Added echo lines to show progress due to long completion time)
echo "[Step 1/5] Running process_amazon_2023.py..."
python src/dataset/amazon_review/process_amazon_2023.py

echo "[Step 2/5] Downloading All Beauty meta file..."
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz

echo "[Step 3/5] Unzipping the downloaded All Beauty meta file..."
gunzip meta_All_Beauty.jsonl.gz

echo "[Step 4/5] Moving the file to appropriate directory..."
mkdir -p data/amazon_review/raw/All_Beauty
mv meta_All_Beauty.jsonl data/amazon_review/raw/All_Beauty/meta_All_Beauty.jsonl

echo "[Step 5/5] Running built_corpus.py..."
python src/dataset/amazon_review/built_corpus.py
echo "✅ Done! Now, run script/repro/amz_review_data_prep_part_2.sh."
