import argparse
import json
import os
import re
from tqdm import tqdm
import pdb

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k
from src.Lucene.amazon_c4.search import PyseriniMultiFieldSearch


from src.eval_search.utils import extract_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/esci/gpt-4o_esci_Sports_and_Outdoors.json')
    args = parser.parse_args()

    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')

    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)
    
    
    test_data = []
    for _, value_dict in res_dict.items():
        query = value_dict['generated_text']
        try:
            query = extract_answer(query)
        except:
            query = query
        if isinstance(value_dict['target'], str):
            target = eval(value_dict['target'])
        else:
            target = value_dict['target']
        
        scores = [1] * len(target)
        test_data.append({'query': query, 'target': target, 'scores': scores})
    
    ndcg = []
    batch_size = 100
    topk = 100
    
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        queries = [str(item['query']) for item in batch]
        targets = {str(item['query']): item['target'] for item in batch} 
        scores = {str(item['query']): item['scores'] for item in batch}
        
        results = search_system.batch_search(queries, top_k=topk, threads=16)
        
        for query in queries:
            retrieved = [result[0] for result in results.get(query, [])]
            ndcg.append(ndcg_at_k(retrieved, targets[query], topk, scores[query]))
    
    print(f"Average NDCG@10: {sum(ndcg) / len(ndcg)}")