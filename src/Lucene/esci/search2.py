import json
from pyserini.search.lucene import LuceneSearcher
import time

class PyseriniMultiFieldSearch:
    def __init__(self, index_dir="pyserini_index"):
        """Initialize Pyserini MultiField Searcher"""
        self.searcher = LuceneSearcher(index_dir)
        # self.searcher.set_bm25(1.2, 0.75) 

    def search(self, query_str, top_k=10):
        """Perform search across multiple fields"""
        if isinstance(query_str, list):
            query_str_processed = " ".join(query_str) 
            # print(f"Warning: search method received a list, joined to: '{query_str_processed}'")
        else:
            query_str_processed = str(query_str)
        
        hits = self.searcher.search(query_str_processed, k=top_k)

        results = []
        for hit in hits:
            try:
                doc = json.loads(hit.lucene_document.get("raw"))
                doc_id = doc.get("id", "ID_NOT_FOUND")
                doc_title = doc.get("title", doc.get("contents", "TITLE_OR_CONTENTS_NOT_FOUND"))
                results.append((doc_id, doc_title, hit.score))
            except Exception as e:
                # print(f"Error processing hit for query '{query_str_processed}': {e}. Hit: {hit.docid[:50] if hit.docid else 'N/A'}")
                results.append(("ERROR_PROCESSING_HIT", str(e), 0.0))
        return results

    def batch_search(self, queries_input_list, top_k=10, threads=4): 
        """
        Perform parallel search across multiple fields using batch_search
        :param queries_input_list: List of query items. Each item can be a string or a list of strings.
        :param top_k: Number of results per query
        :param threads: Number of parallel threads for searching
        :return: Dictionary {actual_dict_key: [(doc_id, content, score), ...]}
                 where actual_dict_key is a string if original query was a string,
                 or a tuple if original query was a list.
        """
        
        processed_for_pyserini_queries = []
        for q_item in queries_input_list: 
            if isinstance(q_item, list):
                processed_q_str = " ".join(q_item)
            else:
                processed_q_str = str(q_item)
            processed_for_pyserini_queries.append(processed_q_str)

        qids_for_pyserini = [str(i) for i in range(len(queries_input_list))] 
        
        batch_results_from_pyserini = self.searcher.batch_search(
            processed_for_pyserini_queries, 
            qids_for_pyserini,             
            k=top_k,
            threads=threads
        )

        final_results = {} 
        
        
        for i, query_item_from_input in enumerate(queries_input_list): 
            pyserini_qid = str(i) 
            hits_for_this_query = batch_results_from_pyserini.get(pyserini_qid, [])

            formatted_results = [] 
            for hit in hits_for_this_query:
                try:
                    doc = json.loads(hit.lucene_document.get("raw"))
                    doc_id = doc.get("id", "ID_NOT_FOUND")
                    doc_content = doc.get("contents", "CONTENTS_NOT_FOUND") 
                    formatted_results.append((doc_id, doc_content, hit.score))
                except Exception as e:
                    # print(f"Error processing hit for query item {query_item_from_input} (index {i}): {e}.")
                    formatted_results.append(("ERROR_PROCESSING_HIT", str(e), 0.0))
            
            actual_dict_key: object
            if isinstance(query_item_from_input, list):
                actual_dict_key = tuple(query_item_from_input) # Convert list to tuple
            else:
                actual_dict_key = query_item_from_input             
            final_results[actual_dict_key] = formatted_results

        return final_results

if __name__ == "__main__":
    print("Running PyseriniMultiFieldSearch example...")
    search_system = PyseriniMultiFieldSearch(index_dir='database/esci/pyserini_index')

    example_queries = [
        "3-Pack Replacement for Whirlpool AND (Water Inlet OR brand new)", 
        ["baby", "activity", "center"], 
        "office chair ergonomic mesh" 
    ]
    
    print(f"\nTest Queries: {example_queries}")

    tic = time.time()
    search_results = search_system.batch_search(example_queries, top_k=3, threads=4)
    print(f"Batch search time: {time.time() - tic:.2f}s")
    
    print("\n--- Batch Search Results ---")
    for query_key, results_list in search_results.items():
        print(f"\nResults for Query Key: {query_key} (Type: {type(query_key)})")
        if results_list:
            for doc_id, content, score in results_list:
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Content Preview: {content_preview}")
        else:
            print("  No results found.")
    
    print("\n--- Testing single search method ---")
    single_query_test = "replacement water filter"
    if example_queries and isinstance(example_queries[0], list): 
        single_query_test = example_queries[0] 
    
    print(f"Single search for: {single_query_test}")
    single_results = search_system.search(single_query_test, top_k=2)
    print(f"Results for Single Query: {single_query_test}")
    if single_results:
        for doc_id, title, score in single_results:
            print(f"  Doc ID: {doc_id}, Score: {score:.4f}, Title: {title}")
    else:
        print("  No results found.")

    print("\nExample usage finished.")

