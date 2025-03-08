import json
import argparse
import pandas as pd
from tqdm import tqdm
import pdb


def filter_reviews(file_path):
    # Initialize an empty list to store filtered reviews
    filtered_reviews = []

    # Read and process the JSONL file
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse each line as a JSON object
            review = json.loads(line.strip())

            # Apply filtering conditions
            if (
                review["rating"] == 5.0  # Check if the rating is 5.0
                and review["verified_purchase"] is True  # Check if purchase is verified
                and len(review["text"]) >= 100  # Check if the review text has at least 100 characters
            ):
                filtered_reviews.append(review)

    return filtered_reviews


def filter_out_existing(csv_file_path, filtered_reviews, item_metadata_dict):
    df = pd.read_csv(csv_file_path)
    existing_pairs = set(zip(df["item_id"], df["user_id"]))
    initial_count = len(filtered_reviews)
    filtered_reviews = [
        review for review in tqdm(filtered_reviews) if ((review["asin"], review["user_id"]) not in existing_pairs) and (review["asin"] in item_metadata_dict)
    ]
    removed_count = initial_count - len(filtered_reviews)
    print(f"Removed {removed_count} existing reviews from the dataset.")
    return filtered_reviews


file_path_dict = {
    'Appliances': 'data/amazon_review/raw/Appliances/Appliances.jsonl',
    'Fashion': 'data/amazon_review/raw/Fashion/Amazon_Fashion.jsonl',
    'Beauty': 'data/amazon_review/raw/Beauty/All_Beauty.jsonl',
    'Arts_Crafts': 'data/amazon_review/raw/Arts_Crafts_and_Sewing/Arts_Crafts_and_Sewing.jsonl',
}

def load_item_metadata_dict(meta_data_path):
    item_metadata_dict = {}
    with open(meta_data_path, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line.strip())
            item_metadata_dict[item["item_id"]] = item['metadata']
    
    return item_metadata_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_path', type=str, default='data/amazon_c4/raw/test.csv')
    parser.add_argument('--meta_data_path', type=str, default='data/amazon_c4/raw/sampled_item_metadata_1M.jsonl')
    args = parser.parse_args()

    item_metadata_dict = load_item_metadata_dict(args.meta_data_path)
    
    review_list = []
    for category, file_path in file_path_dict.items():
        print(f"Filtering reviews for {category} category")
        # Filter reviews
        filtered_reviews = filter_reviews(file_path)
        # Filter out existing reviews
        filtered_reviews = filter_out_existing(args.csv_file_path, filtered_reviews, item_metadata_dict)

        review_list.extend(filtered_reviews)

    # Save the filtered reviews to a JSONL file
    with open("data/amazon_c4/raw/filtered_reviews.jsonl", "w", encoding="utf-8") as file:
        for review in review_list:
            file.write(json.dumps(review) + "\n")
    
    # print counts of reviews
    print(f"Total reviews: {len(review_list)}")