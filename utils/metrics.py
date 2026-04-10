"""
Similarity and Distance Metrics for Vector Retrieval

This module provides functions for computing similarity/distance scores between embeddings
and ranking results for retrieval tasks.

Key Educational Concepts:
-------------------------
1. **Similarity vs Distance**:
   - Similarity: Higher values = more alike (e.g., cosine similarity)
   - Distance: Lower values = more alike (e.g., Euclidean distance)

2. **Common Metrics**:
   - Cosine: Measures angle between vectors (direction matters, magnitude doesn't)
   - Euclidean: Straight-line distance in space (both direction and magnitude matter)
   - Manhattan: Grid-based distance (sum of absolute differences)
   - Sign Match: Custom binary metric (matching activation patterns)

3. **Metric Selection**:
   - Normalized embeddings → Cosine similarity
   - Raw features with scale → Euclidean distance
   - Sparse/count data → Manhattan distance
   - Binary patterns → Sign match

Why This Matters:
-----------------
Choosing the right metric is crucial for retrieval quality:
- Wrong metric → poor results (even with good embeddings)
- Right metric → captures the notion of "similar" you care about
- Different metrics → different ranking orders
"""

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)


def calculate_similarity(query_embedding, corpus_embeddings, metric="cosine"):
    """
    Calculate similarity or distance between a query and corpus of vectors.

    This is the core function for retrieval: given a query (e.g., user's search text),
    find the most similar items in the corpus (e.g., movie database).

    How Retrieval Works:
    --------------------
    1. Embed query: "funny movie" → [0.23, -0.45, ..., 0.12]
    2. Compare to all movies using chosen metric
    3. Rank movies by similarity/distance
    4. Return top-K results

    Metric Deep Dive:
    -----------------
    **Cosine Similarity**:
    - Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    - Range: [-1, 1] where 1 = identical direction, -1 = opposite
    - Interpretation: Measures angle between vectors
    - Best for: Normalized embeddings (like ours from Nomic/DINOv2)
    - Example: [1, 0] and [2, 0] have cosine = 1.0 (same direction, different length)

    **Euclidean Distance**:
    - Formula: sqrt(Σ(A_i - B_i)²)
    - Range: [0, ∞) where 0 = identical
    - Interpretation: Straight-line distance in space
    - Best for: Raw features where magnitude matters
    - Example: [1, 0] and [2, 0] have distance = 1.0 (different positions)

    **Manhattan Distance (L1)**:
    - Formula: Σ|A_i - B_i|
    - Range: [0, ∞) where 0 = identical
    - Interpretation: Walking distance on a grid
    - Best for: Count data, sparse features
    - Example: [1, 1] and [2, 2] have L1 = 2.0 (walk 1 right, 1 up)

    **Sign Match (Custom)**:
    - Formula: mean(sign(A) == sign(B))
    - Range: [0, 1] where 1 = all signs match
    - Interpretation: Fraction of dimensions with same sign
    - Best for: Binary activation patterns, hashing
    - Example: [0.5, -0.3] and [0.2, -0.9] have sign_match = 1.0

    Args:
        query_embedding: Query vector of shape (1, dim) or (dim,)
                        Example: User's search query embedding

        corpus_embeddings: Matrix of corpus vectors, shape (n_samples, dim)
                          Example: All movie embeddings in database

        metric: Which metric to use ('cosine', 'euclidean', 'l1', 'sign_match')
                Default: 'cosine' (works well with normalized embeddings)

    Returns:
        np.ndarray: 1D array of scores, shape (n_samples,)
                   One score per corpus item

                   For similarity metrics (cosine, sign_match):
                   - Higher values = more similar
                   - Use get_top_k() to get highest scores

                   For distance metrics (euclidean, l1):
                   - Lower values = more similar
                   - Use get_top_k() to get lowest scores

    Example Usage:
    --------------
    # Setup
    query = embedder.embed(["action movie"], task_type="search_query")
    corpus = df['embedding'].values  # All movie embeddings

    # Cosine similarity (higher = more similar)
    scores = calculate_similarity(query, corpus, metric="cosine")
    # scores might be: [0.95, 0.23, 0.87, ...]

    # Get top 5 most similar
    top_indices = get_top_k(scores, k=5, metric="cosine")
    top_movies = df.iloc[top_indices]

    Educational Note:
    -----------------
    The choice of metric embeds assumptions about your data:
    - Cosine: "direction matters, scale doesn't" → good for topic similarity
    - Euclidean: "absolute position matters" → good for feature matching
    - Manhattan: "grid distance" → robust to outliers, interpretable
    - Sign: "activation patterns" → extreme compression, binary logic

    Experiment with different metrics to see how rankings change!
    """
    # ====================================================================
    # ENSURE QUERY IS 2D FOR SKLEARN COMPATIBILITY
    # ====================================================================
    # sklearn expects 2D arrays: (n_samples, n_features)
    # If query is 1D (just features), reshape to (1, n_features)
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # ====================================================================
    # COMPUTE METRIC
    # ====================================================================
    if metric == "cosine":
        # Cosine similarity: Measures angle between vectors
        # Range: [-1, 1]. Higher is more similar.
        # Returns 2D array (n_queries, n_corpus), we take first row [0]
        return cosine_similarity(query_embedding, corpus_embeddings)[0]

    elif metric == "euclidean":
        # Euclidean distance: L2 norm ||A - B||
        # Range: [0, infinity). Lower is more similar.
        return euclidean_distances(query_embedding, corpus_embeddings)[0]

    elif metric == "l1":
        # Manhattan distance: L1 norm Σ|A_i - B_i|
        # Range: [0, infinity). Lower is more similar.
        # Also called "taxicab distance" or "city block distance"
        return manhattan_distances(query_embedding, corpus_embeddings)[0]

    elif metric == "sign_match":
        # Custom sign-based similarity metric
        # Treats vectors as binary codes based on sign (+/-)
        # Range: [0, 1]. Higher is more similar (1 = all signs match).

        # Educational: This is related to "semantic hashing" and
        # "locality-sensitive hashing" (LSH) concepts
        # It's an extreme form of quantization (768 floats → 768 bits)

        # Convert to signs: positive → +1, negative → -1, zero → 0
        q_sign = np.sign(query_embedding)
        c_sign = np.sign(corpus_embeddings)

        # Element-wise comparison: where do signs match?
        # (q_sign == c_sign) gives boolean array
        # .mean(axis=1) computes fraction of matches per corpus item
        # axis=1 means "average across dimensions" (across the features)
        matches = (q_sign == c_sign).mean(axis=1)

        return matches

    else:
        raise ValueError(
            f"Unknown metric: '{metric}'. "
            f"Supported metrics: 'cosine', 'euclidean', 'l1', 'sign_match'"
        )


def get_top_k(scores, k=5, metric="cosine"):
    """
    Get indices of top-K most similar/closest items from scores.

    This function handles the difference between:
    - Similarity metrics (higher = better) → return largest scores
    - Distance metrics (lower = better) → return smallest scores

    Why This Matters:
    -----------------
    Different metrics have opposite interpretations:
    - Cosine similarity 0.95 > 0.23 → 0.95 is MORE similar
    - Euclidean distance 2.1 < 5.7 → 2.1 is MORE similar

    This function abstracts away that difference so you can just say
    "give me the top K" regardless of which metric you used.

    Args:
        scores: Array of similarity/distance scores, shape (n_items,)
               From calculate_similarity()

        k: Number of results to return
          Default: 5 (typical for "top 5 recommendations")

        metric: Which metric was used to compute scores
               This determines whether to sort ascending or descending

    Returns:
        np.ndarray: Indices of top-K items, shape (k,)
                   These are indices into the original corpus

                   Example: [42, 17, 99, 3, 56]
                   → Items at positions 42, 17, 99, 3, 56 are most similar

    Example Usage:
    --------------
    # Compute similarities
    scores = calculate_similarity(query, corpus, metric="cosine")

    # Get top 5 indices
    top_idx = get_top_k(scores, k=5, metric="cosine")

    # Retrieve corresponding movies
    top_movies = df.iloc[top_idx]
    print(top_movies[['title', 'overview']])

    # Show scores for top results
    top_scores = scores[top_idx]
    for idx, score in zip(top_idx, top_scores):
        print(f"{df.iloc[idx]['title']}: {score:.3f}")

    Educational Note - Sorting:
    ---------------------------
    np.argsort(scores) returns indices that would sort the array:

    Example:
    scores = [0.23, 0.95, 0.45, 0.87]
    np.argsort(scores) = [0, 2, 3, 1]  # ascending order

    Interpretation: "Put index 0 first (0.23 is smallest),
                     then index 2 (0.45), then 3 (0.87),
                     then 1 (0.95 is largest)"

    For similarity (higher = better), we reverse with [::-1]:
    np.argsort(scores)[::-1] = [1, 3, 2, 0]  # descending order

    Then take first k elements with [:k]:
    np.argsort(scores)[::-1][:k]
    """
    # ====================================================================
    # DETERMINE SORT ORDER BASED ON METRIC TYPE
    # ====================================================================
    # Metrics where HIGHER score means MORE similar (similarity metrics)
    higher_is_better = ["cosine", "sign_match"]

    if metric in higher_is_better:
        # SIMILARITY METRICS: Return indices of LARGEST values
        # argsort() sorts ascending (small to large)
        # [::-1] reverses to descending (large to small)
        # [:k] takes first k elements (top k largest)
        return np.argsort(scores)[::-1][:k]
    else:
        # DISTANCE METRICS: Return indices of SMALLEST values
        # argsort() already sorts ascending (small to large)
        # [:k] takes first k elements (top k smallest = closest)
        return np.argsort(scores)[:k]
