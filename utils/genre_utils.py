"""
Genre Utilities for Multi-Label Classification

This module provides utilities for working with movie genres in multi-label classification.
It handles conversion between different genre representations and metric calculations.

Key Educational Concepts:
-------------------------
1. **Multi-Label Classification**: Unlike single-label (one class per sample), each movie
   can have multiple genres simultaneously (e.g., "Action Comedy Drama")

2. **Multi-Hot Encoding**: Binary vector where 1 indicates presence of genre
   Example: [1, 0, 1, 0, ...] means genres 0 and 2 are present

3. **Incomplete Labels**: In real-world data, absence of a label doesn't mean it's wrong
   - A movie tagged "Action" might also be "Thriller" but wasn't tagged
   - This motivates downweighting negative examples in the loss function

4. **Threshold-Based Prediction**: Converting probabilities to binary predictions
   - Higher threshold = more conservative (fewer predictions, higher precision)
   - Lower threshold = more liberal (more predictions, higher recall)
"""

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def extract_all_genres(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique genre names from the dataset.

    The dataset stores genres as JSON strings like:
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'

    We need to:
    1. Parse each movie's genre JSON
    2. Extract genre names
    3. Collect all unique genres across the dataset

    This gives us the complete "vocabulary" of genres for encoding.

    Args:
        df: DataFrame with 'genres' column containing JSON strings

    Returns:
        Sorted list of unique genre names

    Educational Note:
    -----------------
    Knowing all possible classes upfront is essential for multi-label classification:
    - Determines output layer size (num_genres)
    - Ensures consistent encoding across train/val/test
    - Allows us to create a fixed mapping (genre name -> index)
    """
    all_genres = set()

    for genres_str in df["genres"]:
        try:
            # Parse JSON string to list of dicts
            if isinstance(genres_str, str):
                genres_list = json.loads(genres_str)

                # Extract genre names
                for genre_dict in genres_list:
                    if "name" in genre_dict:
                        all_genres.add(genre_dict["name"])
        except (json.JSONDecodeError, TypeError):
            # Handle malformed or missing data gracefully
            continue

    # Return sorted for consistency (same order every time)
    # This is crucial so genre index 0 always means the same genre
    return sorted(list(all_genres))


def parse_genres(genres_str: str) -> List[str]:
    """
    Parse a single movie's genre JSON string into a list of genre names.

    Args:
        genres_str: JSON string like '[{"id": 28, "name": "Action"}]'

    Returns:
        List of genre names, e.g., ["Action", "Adventure"]

    Example:
    --------
    >>> parse_genres('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]')
    ["Action", "Adventure"]
    """
    try:
        if isinstance(genres_str, str):
            genres_list = json.loads(genres_str)
            return [g["name"] for g in genres_list if "name" in g]
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def genre_to_multihot(genre_names: List[str], all_genres: List[str]) -> np.ndarray:
    """
    Convert a list of genre names to a multi-hot binary vector.

    Multi-Hot Encoding Explained:
    ------------------------------
    Given genres ["Action", "Comedy"] and vocabulary ["Action", "Comedy", "Drama"]:
    Result: [1, 1, 0]
            ^ Action present
               ^ Comedy present
                  ^ Drama absent

    This is different from one-hot encoding where only ONE element is 1.
    In multi-hot, MULTIPLE elements can be 1 (multi-label).

    Args:
        genre_names: List of genre names present for this movie
        all_genres: Complete vocabulary of all possible genres (defines ordering)

    Returns:
        Binary numpy array of shape (num_genres,)

    Educational Note:
    -----------------
    Why binary vectors?
    - Neural networks need numeric inputs
    - Binary vectors work well with sigmoid + BCE loss
    - Each output neuron independently predicts one genre
    - Sigmoid gives us P(genre | movie) for each genre
    """
    # Create mapping: genre name -> index
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}

    # Initialize all zeros
    multihot = np.zeros(len(all_genres), dtype=np.float32)

    # Set 1 for present genres
    for genre in genre_names:
        if genre in genre_to_idx:
            multihot[genre_to_idx[genre]] = 1.0

    return multihot


def multihot_to_genres(
    multihot: np.ndarray, all_genres: List[str], threshold: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Convert model predictions (probabilities) to genre names with confidence scores.

    Thresholding Trade-offs:
    ------------------------
    - threshold = 0.5: Balanced (default for binary classification)
    - threshold = 0.3: More predictions (higher recall, lower precision)
    - threshold = 0.7: Fewer predictions (lower recall, higher precision)

    In multi-label classification, you can tune the threshold based on your use case:
    - Movie recommendation: Lower threshold (don't miss relevant genres)
    - Genre tagging: Higher threshold (only confident predictions)

    Args:
        multihot: Array of probabilities, shape (num_genres,)
        all_genres: List of genre names (same order as training)
        threshold: Minimum probability to predict a genre

    Returns:
        List of (genre_name, probability) tuples for genres above threshold,
        sorted by probability (highest first)

    Example:
    --------
    >>> probs = np.array([0.9, 0.6, 0.2, 0.4])
    >>> genres = ["Action", "Comedy", "Drama", "Thriller"]
    >>> multihot_to_genres(probs, genres, threshold=0.5)
    [("Action", 0.9), ("Comedy", 0.6)]
    """
    # Get indices where probability exceeds threshold
    indices = np.where(multihot >= threshold)[0]

    # Create (genre, probability) pairs
    results = [(all_genres[idx], float(multihot[idx])) for idx in indices]

    # Sort by probability (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def get_top_k_genres(
    multihot: np.ndarray, all_genres: List[str], k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get the top-K most probable genres regardless of threshold.

    This is useful for display purposes where we always want to show
    some predictions even if confidence is low.

    Args:
        multihot: Array of probabilities, shape (num_genres,)
        all_genres: List of genre names
        k: Number of top genres to return

    Returns:
        List of (genre_name, probability) tuples, sorted by probability

    Educational Note:
    -----------------
    Top-K vs Threshold:
    - Top-K: Always returns K predictions (good for UI)
    - Threshold: Variable number of predictions (good for metrics)

    You might use both:
    - Training: Threshold-based for loss/metrics
    - Display: Top-K for consistent UI
    """
    # Get indices of top-K probabilities
    top_indices = np.argsort(multihot)[-k:][::-1]  # Sort descending

    # Create (genre, probability) pairs
    results = [(all_genres[idx], float(multihot[idx])) for idx in top_indices]

    return results


def calculate_genre_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genre_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-genre precision, recall, and F1 scores.

    Multi-Label Metrics Explained:
    -------------------------------
    For each genre independently, we calculate:

    - Precision: Of all movies we predicted have this genre, how many actually do?
      P = TP / (TP + FP)
      High precision = few false positives

    - Recall: Of all movies that actually have this genre, how many did we find?
      R = TP / (TP + FN)
      High recall = few false negatives

    - F1 Score: Harmonic mean of precision and recall
      F1 = 2 * (P * R) / (P + R)
      Balances precision and recall

    Why Per-Genre Metrics?
    -----------------------
    - Some genres are common (Drama: 50% of movies)
    - Some genres are rare (Western: 2% of movies)
    - Overall accuracy is misleading (could just predict "Drama" always)
    - Per-genre metrics reveal if model learns all genres or just common ones

    Args:
        y_true: Ground truth labels, shape (n_samples, n_genres)
        y_pred: Predicted probabilities, shape (n_samples, n_genres)
        genre_names: List of genre names
        threshold: Threshold for converting probabilities to binary predictions

    Returns:
        Dictionary mapping genre names to {"precision": ..., "recall": ..., "f1": ...}

    Educational Note:
    -----------------
    Class Imbalance Challenge:
    - Common genres: Easy to get high recall (lots of training data)
    - Rare genres: Hard to learn (few examples)
    - Solution: Monitor per-genre metrics, potentially use class weights
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(np.float32)

    metrics = {}

    for idx, genre in enumerate(genre_names):
        # Extract binary vectors for this genre
        true_genre = y_true[:, idx]
        pred_genre = y_pred_binary[:, idx]

        # Calculate True/False Positives/Negatives
        tp = np.sum((true_genre == 1) & (pred_genre == 1))
        fp = np.sum((true_genre == 0) & (pred_genre == 1))
        fn = np.sum((true_genre == 1) & (pred_genre == 0))

        # Calculate metrics (avoid division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[genre] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(true_genre)),  # Number of true examples
        }

    return metrics


def calculate_overall_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate overall multi-label classification metrics.

    Aggregation Strategies:
    -----------------------
    1. **Micro-Average**: Pool all predictions together
       - Treats all predictions equally
       - Dominated by common classes
       - Good for overall system performance

    2. **Macro-Average**: Average per-class metrics
       - Treats all classes equally
       - Gives equal weight to rare classes
       - Good for checking if model learns all classes

    Args:
        y_true: Ground truth labels, shape (n_samples, n_genres)
        y_pred: Predicted probabilities, shape (n_samples, n_genres)
        threshold: Threshold for converting probabilities to binary

    Returns:
        Dictionary with 'micro_f1' and 'macro_f1' scores
    """
    # Convert to binary
    y_pred_binary = (y_pred >= threshold).astype(np.float32)

    # Micro-averaged metrics (treat each prediction independently)
    tp_micro = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp_micro = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn_micro = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision_micro = (
        tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    )
    recall_micro = (
        tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    )
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    # Macro-averaged metrics (average across classes)
    num_classes = y_true.shape[1]
    precisions = []
    recalls = []

    for idx in range(num_classes):
        true_class = y_true[:, idx]
        pred_class = y_pred_binary[:, idx]

        tp = np.sum((true_class == 1) & (pred_class == 1))
        fp = np.sum((true_class == 0) & (pred_class == 1))
        fn = np.sum((true_class == 1) & (pred_class == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = (
        2 * precision_macro * recall_macro / (precision_macro + recall_macro)
        if (precision_macro + recall_macro) > 0
        else 0.0
    )

    return {
        "micro_f1": float(f1_micro),
        "macro_f1": float(f1_macro),
        "micro_precision": float(precision_micro),
        "micro_recall": float(recall_micro),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
    }
