import argparse
import json
import os
import re 
from tqdm import tqdm

import sys
sys.path.append('./')

from src.eval_search.utils import ndcg_at_k
from src.Lucene.esci.search2 import PyseriniMultiFieldSearch 
from src.eval_search.utils import extract_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='results/esci/gpt-4o_esci_Sports_and_Outdoors.json', help="Path to the JSON file containing generated texts from model_generate.py")
    parser.add_argument('--save_path', type=str, default='results/esci/metric_results_gpt4o.json', help="Path to save the final evaluation metrics.")
    args = parser.parse_args()

    if not os.path.exists(args.res_path):
        print(f"Error: Results file not found at {args.res_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("Initializing search system...")
    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')
    print("Search system initialized.")

    print(f"Loading results from: {args.res_path}")
    with open(args.res_path, 'r') as f:
        res_dict = json.load(f)
    print(f"Loaded {len(res_dict)} items.")

    test_data = []
    print("Preparing test data...")
    for sample_id, value_dict in res_dict.items():
        query_from_model_output = value_dict['generated_text']
        
        processed_query = query_from_model_output 
        try:
            processed_query = extract_answer(query_from_model_output)
        except Exception as e:
            # print(f"Warning: extract_answer failed for sample_id {sample_id}, query: {query_from_model_output}. Error: {e}. Using raw generated_text.")
            pass

        target_raw = value_dict.get('target') 
        if isinstance(target_raw, str):
            try:
                target = eval(target_raw) 
                if not isinstance(target, list): 
                    target = [str(target)] 
            except Exception as e:
                target = [str(target_raw)] 
        elif isinstance(target_raw, list):
            target = target_raw
        elif target_raw is not None: 
             target = [str(target_raw)]
        else: 
            # print(f"Warning: Target is missing or None for sample_id {sample_id}. Using empty list.")
            target = []


        scores = [1] * len(target) 
        test_data.append({'id': sample_id, 'query': processed_query, 'target': target, 'scores': scores})
    print(f"Prepared {len(test_data)} items for evaluation.")

    batch_size = 100
    topk = 100
    results_dict = {}

    print("Starting batch search and evaluation...")
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating Batches"):
        batch = test_data[i:i + batch_size]
        
        queries_for_batch_search = [item['query'] for item in batch]
        ids_in_batch = [item['id'] for item in batch]
        targets_map = {item['id']: item['target'] for item in batch}
        scores_map = {item['id']: item['scores'] for item in batch}

        try:
            search_results_from_system = search_system.batch_search(queries_for_batch_search, top_k=topk, threads=16)
        except Exception as e:
            print(f"\nCRITICAL ERROR during search_system.batch_search: {e}")
            print("This might indicate an issue within the Pyserini call or query processing inside batch_search.")
            print(f"Problematic queries in this batch might be: {[q for q in queries_for_batch_search if isinstance(q, list)]}")
            search_results_from_system = {} # Create empty results to proceed
            for q_idx, original_q_item_in_error_batch in enumerate(queries_for_batch_search):
                key_for_dummy = tuple(original_q_item_in_error_batch) if isinstance(original_q_item_in_error_batch, list) else original_q_item_in_error_batch
                search_results_from_system[key_for_dummy] = []
            print("Continuing with dummy results for this failed batch.")


        for original_query_idx_in_batch, sample_id in enumerate(ids_in_batch):
            query_as_prepared = queries_for_batch_search[original_query_idx_in_batch]

            retrieval_key: object
            if isinstance(query_as_prepared, list):
                retrieval_key = tuple(query_as_prepared) # Convert list to tuple
            else:
                retrieval_key = query_as_prepared # It's already a string or other hashable type
            
            retrieved_hits_list = search_results_from_system.get(retrieval_key, [])
            
            retrieved_doc_ids = [item[0] for item in retrieved_hits_list] 

            query_as_string_for_results_key = str(query_as_prepared)

            current_target = targets_map.get(sample_id, []) 
            current_scores = scores_map.get(sample_id, []) 
            
            if not isinstance(current_target, list): current_target = []
            if not isinstance(retrieved_doc_ids, list): retrieved_doc_ids = []
            if len(current_scores) != len(current_target): 
                current_scores = [1] * len(current_target)


            results_dict[f"{sample_id}_{query_as_string_for_results_key}"] = {
                'id': sample_id,
                'query_used_for_retrieval': str(retrieval_key), 
                'retrieved': str(retrieved_doc_ids), 
                'target': str(current_target), 
                'ndcg@100': ndcg_at_k(retrieved_doc_ids, current_target, 100, current_scores),
            }

    print(f"Saving final metrics to: {args.save_path}")
    with open(args.save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print("Metrics saved.")

    if results_dict:
        ndcg_100_scores = [v['ndcg@100'] for v in results_dict.values() if 'ndcg@100' in v and isinstance(v['ndcg@100'], (int, float))]
        if ndcg_100_scores:
            print(f"Average NDCG@100: {sum(ndcg_100_scores) / len(ndcg_100_scores):.4f} ({len(ndcg_100_scores)} items)")
        else:
            print("No valid NDCG@100 scores to average.")
    else:
        print("Results dictionary is empty. No scores to average.")

    print("Script finished.")

