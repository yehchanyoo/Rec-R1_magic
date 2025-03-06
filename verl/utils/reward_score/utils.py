import numpy as np

def dcg_at_k(retrieved, target, k):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    """
    retrieved = retrieved[:k]
    gains = [1.0 if item == target else 0.0 for item in retrieved]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k)
    ideal_dcg = dcg_at_k([target], target, k)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_ndcg(retrieved_list, target_list, k=10):
    """
    Compute the average NDCG@k over multiple queries.
    
    retrieved_list: list of retrieved item lists
    target_list: list of corresponding ground truth targets
    k: Rank position up to which NDCG is computed
    """
    ndcg_scores = [ndcg_at_k(retrieved, target, k) for retrieved, target in zip(retrieved_list, target_list)]
    return np.mean(ndcg_scores)

if __name__ == '__main__':
    retrieved_items = [
        ["A", "B", "C", "D"] * 100 + ['Q'] + ['A', 'B'] * 200,  # retrieved list for first user
        ["X", "Y", "Z", "W"],  # retrieved list for second user
    ]

    target_items = [
        "Q",  # target item for first user
        "Y",  # target item for second user
    ]
    K = 401
    ndcg_score = ndcg_at_k(retrieved_items[0], target_items[0], k=K)
    print(f"NDCG@{K}: {ndcg_score:.4f}")