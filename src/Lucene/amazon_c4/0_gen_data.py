import os
import shutil
from huggingface_hub import hf_hub_download

# Step 1: Download the file to the cache (hf_hub_download returns the cached path)
filepath = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-C4',
    filename='sampled_item_metadata_1M.jsonl',
    repo_type='dataset'
)

# Step 2: Define your target path (relative to where the script is run)
target_path = os.path.join('data', 'amazon_c4', 'raw', 'sampled_item_metadata_1M.jsonl')

# Step 3: Create the target directory if it doesn't exist
os.makedirs(os.path.dirname(target_path), exist_ok=True)

# Step 4: Copy (or move) the file
shutil.copy(filepath, target_path)


