"""
Load paired text + image embeddings for Lab 9 (NYC Airbnb default).

Falls back to synthetic paired data if processed parquet is missing so the lab
UI still runs for grading scaffolding.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.data_loader import load_processed_data
from utils.dataset_config import get_dataset_config

TEXT_DIM = 768
IMAGE_DIM = 384


def _valid_emb(x, dim: int) -> bool:
    if isinstance(x, (list, np.ndarray)):
        return len(x) == dim
    return False


def load_paired_subset(
    dataset_name: str = "airbnb",
    max_items: int = 1200,
    seed: int = 42,
) -> tuple[pd.DataFrame | None, np.ndarray | None, np.ndarray | None, str | None]:
    """
    Returns (df, text_np, image_np, error_message).

    error_message is None on success. On failure, df/text_np/image_np may be
    None and error_message explains; caller can use synthetic data instead.
    """
    config = get_dataset_config(dataset_name)
    df = load_processed_data(dataset_name)
    if df is None or len(df) == 0:
        return None, None, None, f"Could not load processed data for '{dataset_name}'."

    tcol = config["text_embedding_col"]
    icol = config["image_embedding_col"]
    if icol not in df.columns:
        return (
            None,
            None,
            None,
            f"Column '{icol}' missing. Run image processing for this dataset.",
        )

    mask = df[tcol].apply(lambda x: _valid_emb(x, TEXT_DIM)) & df[icol].apply(
        lambda x: _valid_emb(x, IMAGE_DIM)
    )
    df = df[mask].reset_index(drop=True)
    if len(df) == 0:
        return (
            None,
            None,
            None,
            "No rows with both valid text (768-d) and image (384-d) embeddings.",
        )

    if len(df) > max_items:
        df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)

    text_np = np.stack(df[tcol].values).astype(np.float32)
    image_np = np.stack(df[icol].values).astype(np.float32)
    return df, text_np, image_np, None


def make_synthetic_paired(
    n: int = 256,
    shared_signal_dim: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic text/image embeddings that share a latent factor so contrastive
    learning has a learnable signal (used when real data is unavailable).
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(n, shared_signal_dim).astype(np.float32)
    noise_t = rng.randn(n, TEXT_DIM).astype(np.float32) * 0.3
    noise_i = rng.randn(n, IMAGE_DIM).astype(np.float32) * 0.3
    # Repeat latent across first dimensions of each modality
    t_part = np.repeat(
        z, (TEXT_DIM + shared_signal_dim - 1) // shared_signal_dim, axis=1
    )[:, :TEXT_DIM]
    i_part = np.repeat(
        z, (IMAGE_DIM + shared_signal_dim - 1) // shared_signal_dim, axis=1
    )[:, :IMAGE_DIM]
    text = t_part + noise_t
    image = i_part + noise_i
    return text.astype(np.float32), image.astype(np.float32)


def project_root() -> str:
    """Repo root (parent of labs/)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
