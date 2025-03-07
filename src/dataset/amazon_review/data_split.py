import json
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

import pdb

def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if json.loads(line)["verified_purchase"]]
    return data


def get_user_purchase_sequences(data):
    user_purchases = defaultdict(list)
    original_count = 0
    filtered_count = 0

    for review in data:
        user_purchases[review["user_id"]].append(review)
        original_count += 1

    for user_id in tqdm(user_purchases):
        sorted_purchases = sorted(user_purchases[user_id], key=lambda x: x["timestamp"])

        seen_titles = {}
        filtered_purchases = []
        for purchase in sorted_purchases:
            title = purchase["title"]
            if title not in seen_titles:
                seen_titles[title] = purchase
                filtered_purchases.append(purchase)

        user_purchases[user_id] = filtered_purchases
        filtered_count += len(filtered_purchases) 

    print(f"Original count: {original_count}, Filtered count: {filtered_count}")

    return user_purchases



def filter_and_split_sequences(user_purchases):
    filtered_sequences = {user: seq for user, seq in user_purchases.items() if len(seq) >= 5}
    
    train_data, val_data, test_data = [], [], []

    for user, seq in filtered_sequences.items():
        test_item = seq[-1]
        val_item = seq[-2]
        train_item = seq[-3]
        train_history = seq[:-3]

        train_data.append({
            "user_id": user,
            "history": train_history,
            "target": train_item
        })

        val_data.append({
            "user_id": user,
            "history": seq[:-2],
            "target": val_item
        })

        test_data.append({
            "user_id": user,
            "history": seq[:-1],
            "target": test_item
        })

    return train_data, val_data, test_data


def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

from collections import Counter

def count_train_sequence_lengths(train_data):
    sequence_lengths = [len(user_data["history"]) for user_data in train_data]
    length_counter = Counter(sequence_lengths)
    return length_counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_area_name', type=str, choices=['Beauty', 'Fashion', 'Appliances'], default='Beauty')
    parser.add_argument('--raw_review_path', type=str, default='data/amazon_review/raw/Beauty/All_Beauty.jsonl')
    parser.add_argument('--raw_item_path', type=str, default='data/amazon_review/raw/Beauty/meta_All_Beauty.jsonl')
    parser.add_argument('--output_dir', type=str, default='data/amazon_review/split/Beauty')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    review_data = read_json_file(args.raw_review_path)
    userseq = get_user_purchase_sequences(review_data)
    
    
    train_data, val_data, test_data = filter_and_split_sequences(userseq)

    stats = count_train_sequence_lengths(train_data)
    print(stats)

    save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(args.output_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(args.output_dir, "test.jsonl"))
    