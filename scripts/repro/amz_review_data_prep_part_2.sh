# Running from the root directory of the repo

# Amazon Review data prep code
# Part 2: Needs to be run every time we try to train

# Zero out the existing parquets
for f in data/amazon_review/inst/*.parquet; do
  : > "$f"
done

# Run Amazon Beauty's version of subset_eq (changed to only do All Beauty)
python src/dataset/amazon_review/inst/sparse/amazon_beauty.py

# Build Lucene database
python src/Lucene/amazon_review/1_convert_format.py
bash src/Lucene/amazon_review/2_build_database.sh
python src/Lucene/amazon_review/search.py

# Train
bash scripts/repro/train_rec-amazon_review_3b.sh
