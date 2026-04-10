"""
Data Loading Utilities

This module provides functions for loading, saving, and managing data for different datasets.
It uses `utils.dataset_config` to determine file paths and schemas.

Key Educational Concepts:
-------------------------
1. **Data Persistence**: Saving processed results to avoid recomputation
2. **File Formats**: CSV vs Parquet trade-offs
3. **Caching Strategy**: Local caching to reduce network dependency
4. **Data Pipeline**: Raw → Processed → Split data flow
5. **Abstraction**: Handling multiple datasets with a unified interface
"""

import os

import pandas as pd
import streamlit as st
from datasets import load_dataset

from utils.dataset_config import get_dataset_config

# ========================================================================
# RAW DATA LOADING (TMDB Specific)
# ========================================================================
# Raw data loading is often dataset-specific (different sources, formats).
# For now, we keep the TMDB specific loader here. Airbnb loading is handled
# in its own processing script.

# Module-level constant for raw data path (can be imported by other scripts)
RAW_DATA_PATH = "data/raw/tmdb_5000_movies.csv"


def load_tmdb_raw():
    """
    Loads the TMDB 5000 Movie Dataset.
    """
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)

    print("Raw data not found locally.")
    print("Downloading TMDB 5000 dataset from Hugging Face...")

    try:
        dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
        df = dataset.to_pandas()

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)

        print(f"✓ Dataset downloaded and cached to {RAW_DATA_PATH}")
        return df

    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        print(f"Error details: {e}")
        return pd.DataFrame()


# ========================================================================
# PROCESSED DATA MANAGEMENT
# ========================================================================


def save_processed_data(df, dataset_name="tmdb"):
    """
    Saves the processed DataFrame with embeddings to disk.
    """
    config = get_dataset_config(dataset_name)
    path = config["path"]

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save to Parquet format
    df.to_parquet(path)

    print(f"✓ Processed data saved to {path}")


def load_processed_data(dataset_name="tmdb"):
    """
    Loads the processed DataFrame with embeddings for the specified dataset.
    """
    config = get_dataset_config(dataset_name)
    path = config["path"]

    if os.path.exists(path):
        return pd.read_parquet(path)

    # File doesn't exist - return None to signal "not processed yet"
    return None


# ========================================================================
# SPLITS MANAGEMENT
# ========================================================================


def get_splits_path(dataset_name):
    """Returns the path for the splits file for a given dataset."""
    # We'll store splits in the same directory as processed data,
    # but with a suffix or different name.
    # E.g. data/processed/tmdb_embedded.parquet -> data/processed/tmdb_splits.csv
    config = get_dataset_config(dataset_name)
    base_path = config["path"]
    # Replace extension and append _splits
    # This assumes path ends with .parquet
    return base_path.replace(".parquet", "_splits.csv")


def save_splits(df, dataset_name="tmdb"):
    """
    Saves train/validation/test split assignments.
    """
    path = get_splits_path(dataset_name)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save only essential columns to keep file small
    # We assume 'id' is the identifier column.
    # Note: Airbnb uses 'id' as well, so this works.
    if "id" in df.columns and "split" in df.columns:
        df[["id", "split"]].to_csv(path, index=False)
        print(f"✓ Split assignments saved to {path}")
    else:
        print("Error: DataFrame must contain 'id' and 'split' columns to save splits.")


def load_splits(dataset_name="tmdb"):
    """
    Loads train/validation/test split assignments.
    """
    path = get_splits_path(dataset_name)

    if os.path.exists(path):
        return pd.read_csv(path)

    return None
