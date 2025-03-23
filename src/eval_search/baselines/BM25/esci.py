import argparse
import json
import os
import re
from tqdm import tqdm
import pdb

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k
from src.Lucene.esci.search import PyseriniMultiFieldSearch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, choices=['Video_Games', 'Baby_Products', 'Office_Products', 'Sports_and_Outdoors'], default='Video_Games')
    parser.add_argument('--test_data_dir', type=str, default='data/esci/test_subset')
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')
    
    # Load the test data
    test_data_path = os.path.join(args.test_data_dir, f"{args.domain}.json")
    with open(test_data_path, "r") as f:
        raw_test_data = json.load(f)
    

    test_data = []
    for entry in raw_test_data:
        query = entry['query']
        target = entry['item_id']
        scores = [1] * len(target)
        test_data.append({'query': query, 'target': target, 'scores': scores})
    
    
    ndcg = []
    batch_size = 100
    topk = 100
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [item['query'] for item in batch]
        targets = {item['query']: item['target'] for item in batch} 
        scores = {item['query']: item['scores'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=topk, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], topk, scores[query]))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")