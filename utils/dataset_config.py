"""
Dataset Configuration Module

This module provides centralized configuration for all datasets used in the course.
It defines where data lives, what columns contain what information, and how to
display dataset-specific UI elements.

Why Centralize Configuration?
------------------------------
1. **Single Source of Truth**: Change a column name once, affects entire app
2. **Easy to Add Datasets**: Just add a new dictionary entry
3. **Type Safety**: Column names validated in one place
4. **UI Consistency**: Standardized terminology across lessons

Educational Design Pattern: Configuration Dictionary
----------------------------------------------------
Instead of hardcoding dataset details throughout the app, we use a dictionary
to map dataset names to their properties. This is a common pattern in:
- Web frameworks (Django settings, Flask config)
- Build tools (webpack.config.js, package.json)
- Infrastructure as Code (terraform, kubernetes)

Benefits for Students:
- Learn separation of config from code
- Understanding data schemas
- Designing flexible, extensible systems
"""

# ========================================================================
# DATASET DEFINITIONS
# ========================================================================
# Each dataset is defined as a dictionary with standardized keys.
# This allows our lessons to work with any dataset without modification.

DATASETS = {
    # ====================================================================
    # TMDB 5000 MOVIES DATASET
    # ====================================================================
    "tmdb": {
        # Display name shown in UI
        "name": "TMDB 5000 Movies",
        # File path to processed data (with embeddings)
        # Parquet format: compressed, fast, preserves datatypes
        "path": "data/processed/tmdb_embedded.parquet",
        # Column containing main text for embedding/display
        # For movies, this is the plot synopsis
        "text_col": "overview",
        # Column containing the title/name identifier
        "title_col": "title",
        # Column with path to image file (movie poster)
        "image_col": "local_poster_path",
        # Column with pre-computed image embeddings (768-d vectors from DINOv2)
        "image_embedding_col": "image_embedding",
        # Column with pre-computed text embeddings (768-d vectors from Nomic AI)
        "text_embedding_col": "embedding",
        # Columns that can be used as classification labels
        # For TMDB, we only have genres, but could add others (e.g., original_language)
        "label_cols": ["genres"],
        # Columns to display in data preview tables
        # Keep this small for UI performance (large text breaks tables)
        "metadata_cols": [
            "title",
            "overview",
            "genres",
            "release_date",
            "vote_average",
        ],
        # Default column for color-coding visualizations
        # E.g., in scatter plots, color points by genre
        "visualization_col": "genres",
        # Singular noun for UI text (e.g., "Choose a Movie")
        # This makes the UI read naturally for different datasets
        "item_name": "Movie",
    },
    # ====================================================================
    # NYC AIRBNB LISTINGS DATASET
    # ====================================================================
    "airbnb": {
        "name": "NYC Airbnb Listings",
        "path": "data/processed/airbnb_embedded.parquet",
        # For Airbnb, the main text is the listing description
        "text_col": "description",
        # Listings have names (e.g., "Cozy Brooklyn Apartment")
        "title_col": "name",
        # Path to property photos
        "image_col": "local_picture_path",
        "image_embedding_col": "image_embedding",
        "text_embedding_col": "embedding",
        # Airbnb has richer categorical data for classification
        # neighbourhood: Specific area (e.g., "Williamsburg")
        # neighbourhood_group: Borough (e.g., "Brooklyn")
        # property_type: Apartment, House, etc.
        # room_type: Entire home, Private room, Shared room
        "label_cols": [
            "neighbourhood",
            "neighbourhood_group",
            "property_type",
            "room_type",
        ],
        "metadata_cols": [
            "name",
            "description",
            "neighbourhood",
            "price",
            "review_scores_rating",
        ],
        # Color by neighbourhood for geographic patterns
        "visualization_col": "neighbourhood",
        "item_name": "Listing",
    },
}

# ========================================================================
# ACCESSOR FUNCTION
# ========================================================================


def get_dataset_config(dataset_name: str) -> dict:
    """
    Returns the configuration dictionary for a specific dataset.

    This function provides safe access to dataset configs with a fallback
    to TMDB if the requested dataset doesn't exist.

    Args:
        dataset_name: Dataset identifier (e.g., "tmdb", "airbnb")

    Returns:
        Configuration dictionary with all dataset properties

    Example:
        >>> config = get_dataset_config("tmdb")
        >>> print(config["name"])
        "TMDB 5000 Movies"
        >>> print(config["title_col"])
        "title"

    Design Note:
    ------------
    We use .get() with a default instead of indexing (DATASETS[dataset_name])
    to avoid KeyError exceptions. This makes the code more robust:
    - If someone types a wrong dataset name, we fall back to TMDB
    - The app doesn't crash, it just uses the default dataset
    - For production, you might want to log a warning instead
    """
    return DATASETS.get(dataset_name, DATASETS["tmdb"])


# ========================================================================
# FUTURE EXTENSIONS
# ========================================================================
# To add a new dataset:
#
# 1. Add an entry to the DATASETS dictionary above
# 2. Create a processing script (e.g., process_new_dataset.py)
# 3. Follow the same structure: generate embeddings, save to parquet
# 4. The lessons will automatically work with the new dataset!
#
# Example for a hypothetical "books" dataset:
#
# "books": {
#     "name": "Goodreads Books",
#     "path": "data/processed/books_embedded.parquet",
#     "text_col": "summary",
#     "title_col": "book_title",
#     "image_col": "cover_path",
#     "image_embedding_col": "cover_embedding",
#     "text_embedding_col": "embedding",
#     "label_cols": ["genres", "author_nationality"],
#     "metadata_cols": ["book_title", "author", "summary", "rating"],
#     "visualization_col": "genres",
#     "item_name": "Book",
# }
