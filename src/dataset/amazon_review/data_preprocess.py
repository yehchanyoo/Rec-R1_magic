"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from collections import defaultdict, Counter
import random
import pdb


PROMPT = """Your task is to generate a search query for the user's next purchase based on their previous purchase history.
Below is the user's purchase history:
```{purchase_history}```

Then, generate a search query for the user's next purchase based on their previous purchase history.
"""

def make_prefix(dp, template_type):
    input_str = PROMPT.format(purchase_history=dp['purchase_history'])
    if template_type == 'base':
        input_str = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n""" + input_str
        input_str += """\nShow your work in <think> </think> tags. You should return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible. </answer>. 
Assistant: Let me solve this step by step. 
<think>"""
    elif template_type == 'qwen-instruct':
        input_str = """<|im_start|>system\nYou are an AI assistant specializing in analyzing user purchase behaviors and providing personalized recommendations.<|im_end|>\n<|im_start|>user\n""" + input_str
        input_str += """\nYour final response must be in JSON format within <answer> </answer> tags. The generated query should use Boolean operators (AND, OR) to structure your query logically. For example,
<answer>
{
    "query": "(xxx AND yyy) OR zzz AND (aaa OR bbb)"
}
</answer>.<|im_end|>
<|im_start|>assistant\n"""
    elif template_type == 'gpt':
        input_str += "Solve this step by step and return the final answer in <answer> </answer> tags, for example, <answer> Trial-level eligibility: 2) Eligible.  </answer>."
    else:
        raise NotImplementedError

    return input_str

def load_rec_dataset(data_dir):
    # read train/val/test.jsonl
    train_data = []
    with open(os.path.join(data_dir, 'train.jsonl'), 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    # select the last 10% of the training data as unseen test data
    train_data = train_data[:-int(0.1 * len(train_data))]

    val_data = []
    with open(os.path.join(data_dir, 'val.jsonl'), 'r') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    val_data = val_data[:-int(0.1 * len(val_data))]
    # further choose about 800 samples from the val data
    val_data = val_data[:800]

    test_data = []
    with open(os.path.join(data_dir, 'test.jsonl'), 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    test_unseen_data = test_data[-int(0.1 * len(test_data)):]
    test_seen_data = test_data[:-int(0.1 * len(test_data))]
    
    assert len(train_data) == len(test_seen_data)
    
    return train_data, val_data, test_seen_data, test_unseen_data


def load_item_dict(item_data_dir):

    def process_item(data):
        contents = (
            f"**Title:** {data.get('title', '')} "
            # f"**Store:** {data.get('store', '')} "
            # f'**Features:** {" | ".join(data.get("features", []))} '
            f'**Description:** {" ".join(data.get("description", ""))} '
            f"**Main Category:** {data.get('main_category', '')} "
            f"**Categories:** {', '.join(data.get('categories', []))} "
            # f"**Details:** {' | '.join(f'{k}: {v}' for k, v in data.get('details', {}).items())}"
        )

        return contents

    item_dict = {}
    with open(item_data_dir, 'r') as f:
        for line in f:
            item = json.loads(line)
            item_dict[item['parent_asin']] = process_item(item)
    return item_dict


def generate_purchase_history_json(user_data, asin_to_content):
    """
    Converts user purchase history into a structured JSON format.
    
    :param user_data: Dictionary containing user history and target product.
    :param asin_to_content: Dictionary mapping ASINs to product descriptions.
    :return: JSON string formatted for structured output.
    """
    history = user_data["history"]
    
    history_list = []

    for item in history:
        asin = item["parent_asin"]
        rating = item["rating"]
        title = item["title"]
        review_text = item["text"]
        # Retrieve product details from the ASIN-to-content dictionary
        product_info = asin_to_content.get(asin, "")
        
        # Construct a structured JSON entry
        history_entry = {
            "Product Info": product_info,
            "User Review": title + ' ' + review_text,
            "User Rating": rating,
        }

        history_list.append(history_entry)

    # Construct final JSON object
    final_output = history_list

    return json.dumps(final_output, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/amazon_review/split/Appliances')
    parser.add_argument('--review_data_dir', default='data/amazon_review/raw/Appliances/meta_Appliances.jsonl')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct', choices=['base', 'qwen-instruct', 'gpt'])
    parser.add_argument('--data_source', type=str, default='Appliances')
    parser.add_argument('--save_dir', type=str, default='data/amazon_review/inst_data/Appliances')

    args = parser.parse_args()
    
    data_source = args.data_source
    save_dir = os.path.join(args.save_dir, args.template_type, data_source)
    os.makedirs(save_dir, exist_ok=True)

    item_dict = load_item_dict(args.review_data_dir)
    
    train_data, val_data, test_seen_data, test_unseen_data = load_rec_dataset(args.local_dir)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_seen_dataset = Dataset.from_list(test_seen_data)
    test_unseen_dataset = Dataset.from_list(test_unseen_data)


    def make_map_fn(split):
        def process_fn(example, idx):
            purchase_history = generate_purchase_history_json(example, item_dict)
            example['purchase_history'] = purchase_history
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['target']['parent_asin'],
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "amazon_review",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)
    test_seen_dataset = test_seen_dataset.map(function=make_map_fn('test_seen'), with_indices=True)
    test_unseen_dataset = test_unseen_dataset.map(function=make_map_fn('test_unseen'), with_indices=True)

    # shuffle the dataset
    train_dataset = train_dataset.shuffle(seed=42)

    threshold = 3000
    
    def truncate(train_dataset, threshold):
        count = 0
        # for those that are exceeding the threshold, we can delete the text between "\nTitle:" to "\nInclusion criteria:"
        for i, d in enumerate(train_dataset):
            if len(d['prompt'][0]['content'].split()) > threshold:
                text = d['prompt'][0]['content']
                count += 1
                words = text.split()
                truncate_length = max(threshold - 200, 0)  # Ensure we don't end up with a negative index
                text = ' '.join(words[-truncate_length:])
                train_dataset[i]['prompt'][0]['content'] = text

        print(f"Truncated {count} examples")
    
        return train_dataset
    
    train_dataset = truncate(train_dataset, threshold=threshold)

    hdfs_dir = os.path.join(args.hdfs_dir, args.template_type) if args.hdfs_dir is not None else None

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(save_dir, 'val.parquet'))
    test_seen_dataset.to_parquet(os.path.join(save_dir, 'test_seen.parquet'))
    test_unseen_dataset.to_parquet(os.path.join(save_dir, 'test_unseen.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=save_dir, dst=hdfs_dir) 
