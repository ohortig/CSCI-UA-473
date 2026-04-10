"""
TMDB Image Processing Script

This script extends the text-based movie dataset with visual information by:
1. Downloading movie poster images from TMDB
2. Generating image embeddings using a Vision Transformer (ViT) model
3. Adding image embeddings to the existing processed data

Why image embeddings?
Just like text embeddings capture semantic meaning of words, image embeddings
capture visual features of images. This allows us to:
- Find movies with similar poster designs
- Perform image-based search (upload a poster, find similar movies)
- Compare how text and image representations differ for the same movie

The script uses DINOv2, a state-of-the-art self-supervised vision transformer.

Run this script AFTER process_data.py to add image capabilities.
"""

import os

import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils.data_loader import load_processed_data, save_processed_data
from utils.image_embedding import get_image_embedder
from utils.image_utils import download_image, load_image

# ========================================================================
# CONFIGURATION
# ========================================================================
# Directory where poster images will be stored locally
POSTER_DIR = "data/posters"

# TMDB image server base URL
# TMDB provides images at different resolutions (w92, w154, w185, w342, w500, w780, original)
# We use w500 as a balance between quality and file size
BASE_URL = "https://image.tmdb.org/t/p/w500"

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================


def create_placeholder(path):
    """
    Generates a placeholder image for movies without posters.

    Why use placeholders?
    - Some movies don't have poster URLs in the dataset
    - We need consistent embeddings for all movies (can't have None)
    - A placeholder embedding represents "no visual information"

    Args:
        path (str): Where to save the placeholder image

    Returns:
        str: The path where the placeholder was saved
    """
    # Create a simple gray image (500x750 is a standard poster size)
    img = Image.new("RGB", (500, 750), color=(200, 200, 200))

    # Draw text on the placeholder so it's visually distinct
    d = ImageDraw.Draw(img)
    d.text((150, 350), "No Poster", fill=(0, 0, 0))

    # Save the placeholder for reuse
    img.save(path)
    return path


# ========================================================================
# MAIN PROCESSING FUNCTION
# ========================================================================


def process_images():
    """
    Main pipeline for downloading posters and generating image embeddings.

    Pipeline stages:
    1. Load existing data (requires process_data.py to have been run)
    2. Load poster URLs from an external dataset
    3. Download poster images
    4. Generate image embeddings using a vision transformer
    5. Save updated data with image embeddings
    """

    # ========================================================================
    # STAGE 1: LOAD EXISTING DATA
    # ========================================================================
    print("Loading existing processed data...")
    df = load_processed_data()

    # This script extends existing data, so we need the base data first
    if df is None:
        print("Please run process_data.py first.")
        return

    # ========================================================================
    # STAGE 2: LOAD POSTER URLs
    # ========================================================================
    # The TMDB 5000 dataset doesn't include poster URLs, so we merge with
    # another dataset (Pablinho/movies-dataset) that has this information
    print("Loading dataset with posters (Pablinho/movies-dataset)...")
    try:
        # Load the dataset containing poster URLs
        poster_ds = load_dataset("Pablinho/movies-dataset", split="train")
        poster_df = poster_ds.to_pandas()

        # Clean title strings for better matching
        # strip() removes leading/trailing whitespace that could break matches
        poster_df["Title"] = poster_df["Title"].astype(str).str.strip()
        df["title"] = df["title"].astype(str).str.strip()

        # IMPORTANT: The poster dataset may have duplicate titles
        # We need to deduplicate BEFORE merging to prevent creating duplicate rows
        # Keep the first occurrence of each title
        poster_df_dedup = poster_df[["Title", "Poster_Url"]].drop_duplicates(
            subset=["Title"], keep="first"
        )

        # Merge datasets on movie title
        # left join ensures we keep all our original movies even if no poster exists
        merged = df.merge(
            poster_df_dedup, left_on="title", right_on="Title", how="left"
        )

        # Report how many posters we found
        found_count = merged["Poster_Url"].notna().sum()
        print(f"Found posters for {found_count} out of {len(merged)} movies.")

    except Exception as e:
        print(f"Failed to load poster dataset: {e}")
        return

    # ========================================================================
    # STAGE 3: DOWNLOAD POSTERS
    # ========================================================================
    print("Downloading posters...")

    # Create directory to store poster images
    # exist_ok=True prevents errors if directory already exists
    os.makedirs(POSTER_DIR, exist_ok=True)

    # Create a placeholder image that will be used for movies without posters
    placeholder_path = os.path.join(POSTER_DIR, "placeholder.jpg")
    if not os.path.exists(placeholder_path):
        create_placeholder(placeholder_path)

    # Track local file paths for all posters
    local_paths = []

    # Download each poster (with progress bar via tqdm)
    for idx, row in tqdm(merged.iterrows(), total=len(merged)):
        movie_id = row["id"]
        poster_url = row["Poster_Url"]

        # Use movie ID as filename to avoid issues with special characters in titles
        save_path = os.path.join(POSTER_DIR, f"{movie_id}.jpg")

        # Skip download if we already have this poster
        if os.path.exists(save_path):
            local_paths.append(save_path)
            continue

        # If no URL is available, mark as None (we'll use placeholder later)
        if pd.isna(poster_url):
            local_paths.append(None)
            continue

        # Handle relative vs absolute URLs
        # TMDB API returns paths starting with '/' which need the base URL prepended
        if str(poster_url).startswith("/"):
            full_url = BASE_URL + poster_url
        else:
            full_url = poster_url

        # Attempt to download the image
        # download_image returns True on success, False on failure
        if download_image(full_url, save_path):
            local_paths.append(save_path)
        else:
            # Download failed (404, network error, etc.)
            local_paths.append(None)

    # Add local paths to our dataframe for future reference
    merged["local_poster_path"] = local_paths

    # ========================================================================
    # STAGE 4: GENERATE IMAGE EMBEDDINGS
    # ========================================================================
    print("Generating image embeddings...")

    # Initialize the vision transformer model
    # This is a DINOv2 model that converts images to fixed-size vectors
    embedder = get_image_embedder()

    # Pre-compute placeholder embedding
    # This is more efficient than computing it repeatedly for each missing poster
    # All movies without posters will share this same embedding
    placeholder_img = load_image(placeholder_path)
    placeholder_embedding = embedder.embed(placeholder_img)

    # Generate embeddings for all images
    image_embeddings = []

    for path in tqdm(local_paths):
        # Case 1: We have a valid poster image
        if path and os.path.exists(path):
            try:
                img = load_image(path)
                if img:
                    # Successfully loaded, generate embedding
                    emb = embedder.embed(img)
                    image_embeddings.append(list(emb))
                else:
                    # Image file is corrupt or invalid
                    # Use placeholder embedding to maintain data consistency
                    image_embeddings.append(list(placeholder_embedding))
            except Exception:
                # Any other error during embedding generation
                # Use placeholder embedding as fallback
                image_embeddings.append(list(placeholder_embedding))
        else:
            # Case 2: No poster available for this movie
            # Use placeholder embedding
            image_embeddings.append(list(placeholder_embedding))

    # Add image embeddings to the dataframe
    merged["image_embedding"] = image_embeddings

    # ========================================================================
    # STAGE 5: CLEANUP AND SAVE
    # ========================================================================
    # Remove temporary columns used only for merging
    if "Title" in merged.columns:
        del merged["Title"]
    if "Poster_Url" in merged.columns:
        del merged["Poster_Url"]

    # Save the updated dataframe with both text AND image embeddings
    print("Saving updated data...")
    save_processed_data(merged)
    print("Done!")


# ========================================================================
# SCRIPT ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    process_images()
