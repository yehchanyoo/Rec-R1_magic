import argparse
import json
import glob
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    
    if args.split == 'train':
        # read all the files in data/amazon_c4/filtered/raw
        json_files = glob.glob("data/amazon_c4/filtered/raw/*.jsonl")

        all_data = []
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(data)
        
        for i, entry in enumerate(all_data):
            entry["qid"] = i
            entry["ori_rating"] = int(entry["ori_rating"])
        
        output_file = "data/amazon_c4/train.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
    
    elif args.split == 'test':
        # read data/amazon_c4/raw/test.csv
        import pandas as pd
        df = pd.read_csv('data/amazon_c4/raw/test.csv')
        json_data = df.to_dict(orient="records")

        output_json_path = "data/amazon_c4/test.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

