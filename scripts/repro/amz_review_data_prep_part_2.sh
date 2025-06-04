# Running from the root directory of the repo

# Amazon Review data prep code
# Part 2: Needs to be run every time we try to train

# Make the parquets writable by user
chmod u+w data/amazon_review/inst/*.parquet

# Zero out/delete the existing parquets
for f in data/amazon_review/inst/*.parquet; do
  : > "$f"
done
rm -rf data/amazon_review/inst/*.parquet

# Delete past Lucene files before rebuilding Lucene database
rm -rf database/amazon_review/All_Beauty/pyserini_index/
rm -f database/amazon_review/All_Beauty/jsonl_docs/pyserini.jsonl

# Run Amazon Beauty's version of subset_eq (changed to only do All Beauty)
python src/dataset/amazon_review/inst/sparse/amazon_beauty.py

# Build Lucene database
python src/Lucene/amazon_review/1_convert_format.py
bash src/Lucene/amazon_review/2_build_database.sh
python src/Lucene/amazon_review/search.py

# Make the parquets read-only
chmod a-w data/amazon_review/inst/*.parquet

# Train
bash scripts/repro/train_rec-amazon_review_3b.sh
