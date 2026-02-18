#!/usr/bin/env python3
"""
Prepare the TMDB dataset used by Lab 3.

This script:
1. Downloads TMDB data from Hugging Face
2. Builds text fields for retrieval
3. Generates Nomic text embeddings
4. Saves `data/processed/tmdb_embedded.parquet`
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "AiresPucrs/tmdb-5000-movies"
DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TMDB data for Lab 3.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to load (defaults to train).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tmdb_embedded.parquet",
        help="Where to save the processed parquet file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model for embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit (useful for quick checks).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def choose_column(
    df: pd.DataFrame, candidates: list[str], required: bool = False
) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series, pd.DataFrame)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def normalize_genres(value: object) -> str:
    if is_missing_scalar(value):
        return ""

    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else ""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(str(name))
            elif item is not None:
                names.append(str(item))
        return ", ".join(names)

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        return normalize_genres(parsed)

    return text


def resolve_poster_url(value: object) -> str:
    if is_missing_scalar(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if not text.startswith("/"):
        text = f"/{text}"
    return f"{POSTER_BASE_URL}{text}"


def load_tmdb_dataframe(dataset_name: str, split: str) -> tuple[pd.DataFrame, str]:
    try:
        split_ds = load_dataset(dataset_name, split=split)
        return split_ds.to_pandas(), split
    except Exception as split_error:
        print(
            f"Could not load split '{split}' directly ({type(split_error).__name__}). "
            "Trying dataset splits."
        )
        ds_dict = load_dataset(dataset_name)
        if split in ds_dict:
            return ds_dict[split].to_pandas(), split
        selected_split = next(iter(ds_dict.keys()))
        print(f"Using split '{selected_split}'.")
        return ds_dict[selected_split].to_pandas(), selected_split


def build_output_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    title_col = choose_column(raw_df, ["title", "original_title"], required=True)
    overview_col = choose_column(
        raw_df, ["overview", "description", "plot", "plot_summary"], required=True
    )
    id_col = choose_column(raw_df, ["id", "movie_id", "tmdb_id"])
    genres_col = choose_column(raw_df, ["genres"])
    rating_col = choose_column(raw_df, ["vote_average", "rating", "imdb_rating"])
    revenue_col = choose_column(raw_df, ["revenue", "box_office"])
    poster_col = choose_column(raw_df, ["poster_path", "poster", "poster_url"])
    release_col = choose_column(raw_df, ["release_date", "release_year"])

    out = pd.DataFrame(index=raw_df.index)
    if id_col:
        out["movie_id"] = raw_df[id_col]
    else:
        out["movie_id"] = np.arange(len(raw_df))

    out["title"] = raw_df[title_col].fillna("").astype(str).str.strip()
    out["overview"] = raw_df[overview_col].fillna("").astype(str).str.strip()
    out["genres"] = (
        raw_df[genres_col].apply(normalize_genres)
        if genres_col
        else pd.Series("", index=raw_df.index)
    )
    out["vote_average"] = (
        pd.to_numeric(raw_df[rating_col], errors="coerce") if rating_col else np.nan
    )
    out["revenue"] = (
        pd.to_numeric(raw_df[revenue_col], errors="coerce") if revenue_col else np.nan
    )
    out["release_date"] = (
        raw_df[release_col].fillna("").astype(str).str.strip() if release_col else ""
    )
    out["poster_path"] = (
        raw_df[poster_col].fillna("").astype(str).str.strip() if poster_col else ""
    )
    out["poster_url"] = out["poster_path"].apply(resolve_poster_url)
    out["local_poster_path"] = ""

    out = out[out["title"] != ""].reset_index(drop=True)
    return out


def build_document_texts(df: pd.DataFrame) -> list[str]:
    documents: list[str] = []
    for row in df.itertuples(index=False):
        parts = [f"Title: {row.title}"]
        if row.genres:
            parts.append(f"Genres: {row.genres}")
        if row.overview:
            parts.append(f"Overview: {row.overview}")
        documents.append("search_document: " + " | ".join(parts))
    return documents


def main() -> None:
    args = parse_args()
    output_path = resolve_path(args.output)

    if output_path.exists() and not args.force:
        print(f"Output already exists: {output_path}")
        print("Use --force to overwrite.")
        return

    print(f"Loading dataset '{args.dataset}' (split='{args.split}')...")
    raw_df, selected_split = load_tmdb_dataframe(args.dataset, args.split)
    print(f"Loaded {len(raw_df)} rows from split '{selected_split}'.")
    print(f"Columns: {', '.join(raw_df.columns)}")

    df = build_output_dataframe(raw_df)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    if df.empty:
        raise ValueError("No rows available after preprocessing.")

    print(f"Preparing embeddings for {len(df)} movies...")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    doc_texts = build_document_texts(df)

    embeddings = model.encode(
        doc_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    df["embedding"] = [vec.astype(np.float32).tolist() for vec in embeddings]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} rows to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
