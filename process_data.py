"""
TMDB Data Processing Script

This script is responsible for converting the raw TMDB movie dataset into
a machine learning-ready format with embeddings.

What are embeddings?
Embeddings are numerical vector representations of text that capture semantic
meaning. Instead of treating words as discrete symbols, we convert them into
points in a high-dimensional space where semantically similar items are close together.

This script performs the following steps:
1. Loads the raw TMDB dataset from Hugging Face
2. Preprocesses text fields (handles missing values, combines fields)
3. Generates embeddings using a pre-trained language model (Nomic AI)
4. Saves the processed data with embeddings to a Parquet file

Run this script once to prepare the data before using the Streamlit app.
"""

import numpy as np
from tqdm import tqdm  # Progress bar library for better user experience

from utils.data_loader import load_tmdb_raw, save_processed_data
from utils.embedding import get_embedder


def process_and_save():
    """
    Main data processing pipeline.

    Steps:
    1. Load raw TMDB dataset.
    2. Preprocess text fields (handle missing values, combine title + overview).
    3. Check if we can reuse existing embeddings (optimization).
    4. Generate embeddings for new data using the Nomic model.
    5. Save the final dataframe with embeddings to Parquet.
    """
    # ========================================================================
    # STEP 1: LOAD RAW DATA
    # ========================================================================
    print("Loading raw data...")
    df = load_tmdb_raw()

    # Check if the data loaded successfully
    if df.empty:
        print("Error: Could not load data.")
        return

    print(f"Loaded {len(df)} movies.")

    # ========================================================================
    # STEP 2: PREPROCESS TEXT DATA
    # ========================================================================
    # Preprocessing is crucial for ML pipelines. We need to:
    # - Handle missing values (NaN) that would break string operations
    # - Combine relevant text fields to give the model more context
    # - Clean and standardize the data format

    # Fill NaN values with empty strings to avoid errors during string concatenation
    # In pandas, NaN + "text" = NaN, which would lose information
    df["overview"] = df["overview"].fillna("")
    df["title"] = df["title"].fillna("")

    # Create a combined text field for embedding
    # Why combine title + overview?
    # - The title provides the main topic/entity
    # - The overview provides detailed context
    # - Together, they give the embedding model the full semantic picture
    # - The "Title: " and "Overview: " prefixes help the model distinguish between fields
    df["text_content"] = "Title: " + df["title"] + "; Overview: " + df["overview"]

    # Keep only relevant columns to reduce file size and memory usage
    # This is a best practice: only keep what you need
    cols_to_keep = [
        "id",
        "title",
        "overview",
        "text_content",
        "genres",
        "vote_average",
        "revenue",
        "popularity",
        "homepage",
    ]

    # Check if columns exist (dataset schema might vary across versions)
    # This defensive programming prevents KeyErrors
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]

    # ========================================================================
    # STEP 3: CHECK FOR EXISTING EMBEDDINGS (OPTIMIZATION)
    # ========================================================================
    # Generating embeddings is computationally expensive (can take minutes).
    # If we've already processed this data, we can reuse the embeddings.
    # This is especially useful during development when tweaking metadata.

    existing_df = None
    try:
        from utils.data_loader import load_processed_data

        existing_df = load_processed_data()
    except Exception:
        # If loading fails (file doesn't exist), that's okay - we'll generate new embeddings
        pass

    if existing_df is not None and len(existing_df) == len(df):
        print("Found existing processed data. Reusing embeddings...")

        # Ensure alignment: the existing embeddings must match our current data
        if "embedding" in existing_df.columns:
            # Verify that the movie IDs match exactly
            # This ensures we're not mixing up embeddings between different movies
            if existing_df["id"].equals(df["id"]):
                # Copy the pre-computed embeddings
                df["embedding"] = existing_df["embedding"]

                # Preserve other columns that might have been added (like image embeddings)
                # This allows us to incrementally add features without reprocessing everything
                for col in existing_df.columns:
                    if col not in df.columns:
                        df[col] = existing_df[col]

                print("Embeddings reused successfully.")
                print("Saving processed data...")
                save_processed_data(df)
                print("Done!")
                return
            else:
                print("Data mismatch. Regenerating embeddings...")

    # ========================================================================
    # STEP 4: GENERATE EMBEDDINGS
    # ========================================================================
    # If we reach here, we need to generate new embeddings

    print("Initializing embedder...")
    # The embedder loads a pre-trained transformer model (Nomic AI)
    # This model has learned to convert text into meaningful vectors
    embedder = get_embedder()

    print("Generating embeddings...")

    # Batch processing configuration
    # Why use batches?
    # 1. Memory efficiency: Processing all 5000 movies at once could exceed RAM
    # 2. GPU utilization: Modern GPUs are optimized for batch operations
    # 3. Progress tracking: We can show progress as batches complete
    batch_size = 32  # Process 32 movies at a time (a common batch size)

    embeddings = []  # Collect embeddings from all batches

    # Convert dataframe column to a list for easier batch processing
    texts = df["text_content"].tolist()

    # Process in batches to manage memory usage and show progress
    # tqdm shows a progress bar so we know how long the process will take
    for i in tqdm(range(0, len(texts), batch_size)):
        # Extract the current batch of texts
        batch_texts = texts[i : i + batch_size]

        # Generate embeddings for this batch
        # task_type="search_document" tells the model these are the items to be retrieved
        # (as opposed to "search_query" which is used for the query text)
        # This distinction helps the model optimize for asymmetric search
        batch_embeddings = embedder.embed(batch_texts, task_type="search_document")

        # Collect the batch results
        embeddings.append(batch_embeddings)

    # Concatenate all batch results into a single numpy matrix
    # vstack = vertical stack (stacks arrays row-wise)
    # Result shape: (n_movies, embedding_dimension)
    all_embeddings = np.vstack(embeddings)

    # ========================================================================
    # STEP 5: SAVE PROCESSED DATA
    # ========================================================================
    # Add embeddings to dataframe
    # We convert the numpy array to a list of arrays for Parquet compatibility
    # Parquet handles lists well, but numpy arrays can sometimes be tricky depending on the engine
    df["embedding"] = list(all_embeddings)

    print("Saving processed data...")
    # Save to Parquet format (faster and more efficient than CSV)
    save_processed_data(df)
    print("Done!")


# ========================================================================
# SCRIPT ENTRY POINT
# ========================================================================
# This pattern allows the file to be imported as a module OR run as a script
# When run as a script (python process_data.py), __name__ == "__main__"
# When imported (from process_data import ...), __name__ == "process_data"
if __name__ == "__main__":
    process_and_save()
