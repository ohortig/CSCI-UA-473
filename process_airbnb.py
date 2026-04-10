"""
Airbnb Data Processing Script

This script downloads and processes the NYC Airbnb dataset for the machine learning course.
It performs the following steps:
1. Downloads the listings.csv.gz file from Inside Airbnb.
2. Cleans and preprocesses the text data (name, description).
3. Downloads property images.
4. Generates text embeddings using the SentenceTransformer model.
5. Generates image embeddings using the DINOv2 model.
6. Saves the processed data to a Parquet file.
"""

import argparse
import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from utils.image_embedding import get_image_embedder
from utils.image_utils import download_image, load_image

# Configuration
DATA_URL = "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/data/listings.csv.gz"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
IMAGE_DIR = "data/airbnb_images"
OUTPUT_FILE = "airbnb_embedded.parquet"


def download_dataset():
    """Downloads the Airbnb dataset if it doesn't exist."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filename = os.path.join(RAW_DATA_DIR, "listings.csv.gz")

    if os.path.exists(filename):
        print(f"Dataset already exists at {filename}")
        return filename

    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
        return filename
    else:
        raise Exception(f"Failed to download dataset: {response.status_code}")


def process_text_embeddings(df):
    """Generates text embeddings for the name and description."""
    print("Generating text embeddings...")

    # Combine name and description for a richer representation
    # Handle missing values by replacing them with empty strings
    df["text_content"] = df["name"].fillna("") + ": " + df["description"].fillna("")

    try:
        # Use the shared embedder to ensure consistency (768 dimensions)
        from utils.embedding import get_embedder

        embedder = get_embedder()

        # Embed in batches with progress bar
        batch_size = 32
        texts = df["text_content"].tolist()
        embeddings = []

        print(f"Embedding {len(texts)} documents in batches of {batch_size}...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embedder.embed(batch_texts, task_type="search_document")
            embeddings.extend(batch_embeddings)

        df["embedding"] = list(embeddings)

    except Exception as e:
        print(f"Warning: Failed to generate text embeddings: {e}")
        print("Using dummy text embeddings (random vectors).")
        # Generate random embeddings (768 dimensions for Nomic)
        embeddings = np.random.randn(len(df), 768).astype(np.float32)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        df["embedding"] = list(embeddings)

    return df


def process_images(df):
    """Downloads images and generates image embeddings."""
    print("Processing images...")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    embedder = get_image_embedder()

    local_paths = []
    image_embeddings = []

    # Create a placeholder image
    placeholder_path = os.path.join(IMAGE_DIR, "placeholder.jpg")
    if not os.path.exists(placeholder_path):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (500, 333), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((200, 150), "No Image", fill=(0, 0, 0))
        img.save(placeholder_path)

    placeholder_img = load_image(placeholder_path)
    embedder.embed(placeholder_img)

    # Step 1: Download all images first
    print("Downloading images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        image_url = row["picture_url"]
        listing_id = row["id"]
        save_path = os.path.join(IMAGE_DIR, f"{listing_id}.jpg")

        # Download image if not exists
        if not os.path.exists(save_path):
            if pd.notna(image_url):
                download_image(image_url, save_path)
            else:
                save_path = None

        # Verify it exists after download attempt
        if save_path and os.path.exists(save_path):
            local_paths.append(save_path)
        else:
            local_paths.append(None)

    # Step 2: Embed in batches
    batch_size = 32
    print(f"Embedding images in batches of {batch_size}...")

    for i in tqdm(range(0, len(local_paths), batch_size), desc="Embedding batches"):
        batch_paths = local_paths[i : i + batch_size]
        batch_images = []

        # Load images for this batch
        for path in batch_paths:
            img = None
            if path:
                try:
                    img = load_image(path)
                except Exception:
                    pass

            if img:
                batch_images.append(img)
            else:
                batch_images.append(placeholder_img)

        # Generate embeddings for the batch
        if batch_images:
            batch_embs = embedder.embed(batch_images)
            image_embeddings.extend(list(batch_embs))

    df["local_picture_path"] = local_paths
    df["image_embedding"] = image_embeddings
    return df


def clean_data(df):
    """Cleans the dataframe."""
    print("Cleaning data...")

    # Convert price to numeric
    # Prices are often strings like "$100.00"
    if df["price"].dtype == "object":
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "")
            .str.replace(",", "")
            .astype(float)
        )

    # Ensure review_scores_rating is numeric
    df["review_scores_rating"] = pd.to_numeric(
        df["review_scores_rating"], errors="coerce"
    )

    # Fill missing values for categorical columns if needed
    df["neighbourhood"] = df["neighbourhood_cleansed"]  # Use cleansed version
    df["neighbourhood_group"] = df["neighbourhood_group_cleansed"]  # Use borough

    return df


def main():
    parser = argparse.ArgumentParser(description="Process NYC Airbnb dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full re-processing (ignore existing data).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image downloading and embedding.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Number of listings to sample (default: 5000).",
    )
    args = parser.parse_args()

    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)

    # Check for existing processed data
    if os.path.exists(output_path) and not args.force:
        print(f"Found existing processed data at {output_path}")
        print("Loading to update...")
        try:
            df = pd.read_parquet(output_path)
            print(f"Loaded {len(df)} listings.")

            # Always update text embeddings (fast and ensures correctness)
            # unless we add a specific flag to skip it, but for now this is the desired fix.
            df = process_text_embeddings(df)

            # Process images unless skipped
            if not args.skip_images:
                # Check if we need to process images (simple heuristic: check if column exists and has data)
                if (
                    "image_embedding" in df.columns
                    and df["image_embedding"]
                    .apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
                    .all()
                ):
                    print(
                        "Image embeddings already exist. Skipping image processing (use --force to regenerate)."
                    )
                else:
                    print(
                        "Image embeddings missing or incomplete. Processing images..."
                    )
                    df = process_images(df)
            else:
                print("Skipping image processing as requested.")

            print(f"Saving updated data to {output_path}...")
            df.to_parquet(output_path)
            print("Done!")
            return
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Falling back to full processing...")

    # Full processing pipeline
    print("Starting full processing pipeline...")

    # 1. Download
    gz_path = download_dataset()

    # 2. Load CSV
    print("Loading CSV...")
    df = pd.read_csv(gz_path)
    print(f"Loaded {len(df)} listings.")

    # Filter for active/relevant listings
    if len(df) > args.sample:
        print(f"Sampling {args.sample} listings...")
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    # 3. Clean
    df = clean_data(df)

    # 4. Text Embeddings
    df = process_text_embeddings(df)

    # 5. Image Embeddings
    if not args.skip_images:
        df = process_images(df)
    else:
        print("Skipping image processing as requested.")
        df["image_embedding"] = None
        df["local_picture_path"] = None

    # 6. Save
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
