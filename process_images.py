#!/usr/bin/env python3
"""
Download TMDB poster images and update local poster paths.

This script reads `data/processed/tmdb_embedded.parquet`, downloads available
posters to `data/posters/`, and writes updated `local_poster_path` values.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
KNOWN_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TMDB poster images.")
    parser.add_argument(
        "--input",
        default="data/processed/tmdb_embedded.parquet",
        help="Input parquet file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet file (defaults to overwriting --input).",
    )
    parser.add_argument(
        "--poster-dir",
        default="data/posters",
        help="Directory to store downloaded posters.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of new downloads.",
    )
    parser.add_argument(
        "--max-lookups",
        type=int,
        default=None,
        help="Optional cap on poster-resolution lookups (API/scrape attempts).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if a local file already exists.",
    )
    parser.add_argument(
        "--tmdb-api-key",
        default=os.getenv("TMDB_API_KEY", ""),
        help=(
            "Optional TMDB API key (v3). If provided, missing poster paths can be "
            "resolved using movie IDs."
        ),
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def to_stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sanitize_name(value: object, fallback: str) -> str:
    raw = str(value) if value is not None else fallback
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return clean or fallback


def extension_from_url_or_type(url: str, content_type: str) -> str:
    path_ext = Path(urlparse(url).path).suffix.lower()
    if path_ext in KNOWN_IMAGE_EXTENSIONS:
        return path_ext
    if "png" in content_type.lower():
        return ".png"
    if "webp" in content_type.lower():
        return ".webp"
    return ".jpg"


def resolve_poster_url_values(
    poster_url_value: object, poster_path_value: object
) -> str:
    for value in (poster_url_value, poster_path_value):
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text.startswith("http://") or text.startswith("https://"):
            return text
        if not text.startswith("/"):
            text = f"/{text}"
        return f"{POSTER_BASE_URL}{text}"
    return ""


def local_file_exists(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.exists()


def normalize_movie_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def fetch_tmdb_poster_path(
    movie_id: str, api_key: str, session: requests.Session, timeout: float
) -> str:
    if not api_key:
        return ""

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        response = session.get(url, params={"api_key": api_key}, timeout=timeout)
    except requests.RequestException:
        return ""

    if response.status_code != 200:
        return ""

    try:
        payload = response.json()
    except ValueError:
        return ""

    poster_path = payload.get("poster_path")
    if not poster_path:
        return ""
    poster_text = str(poster_path).strip()
    if not poster_text:
        return ""
    if not poster_text.startswith("/"):
        poster_text = f"/{poster_text}"
    return poster_text


def extract_tmdb_poster_path_from_html(html: str) -> str:
    match = re.search(r'"poster_path"\s*:\s*"([^"]+)"', html)
    if not match:
        return ""

    poster_path = match.group(1).strip()
    if not poster_path:
        return ""
    poster_path = poster_path.replace("\\/", "/")
    try:
        poster_path = bytes(poster_path, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        pass

    if not poster_path or poster_path.lower() == "null":
        return ""
    if not poster_path.startswith("/"):
        poster_path = f"/{poster_path}"
    return poster_path


def extract_tmdb_og_image_url(html: str) -> str:
    match = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""

    image_url = match.group(1).strip().replace("&amp;", "&")
    if not image_url:
        return ""
    if image_url.startswith("//"):
        image_url = f"https:{image_url}"
    if image_url.startswith("/"):
        image_url = urljoin("https://www.themoviedb.org", image_url)
    return image_url


def poster_path_from_image_url(image_url: str) -> str:
    parsed = urlparse(image_url)
    path = parsed.path
    if "/t/p/" not in path:
        return ""
    match = re.search(r"/t/p/(?:[^/]+)/([^/?#]+)$", path)
    if not match:
        return ""
    filename = match.group(1).strip()
    if not filename:
        return ""
    return f"/{filename}"


def fetch_tmdb_poster_without_key(
    movie_id: str, session: requests.Session, timeout: float
) -> tuple[str, str]:
    movie_page_url = f"https://www.themoviedb.org/movie/{movie_id}"
    try:
        response = session.get(movie_page_url, timeout=timeout)
    except requests.RequestException:
        return "", ""

    if response.status_code != 200:
        return "", ""

    html = response.text

    poster_path = extract_tmdb_poster_path_from_html(html)
    if poster_path:
        return poster_path, f"{POSTER_BASE_URL}{poster_path}"

    image_url = extract_tmdb_og_image_url(html)
    if not image_url:
        return "", ""

    inferred_path = poster_path_from_image_url(image_url)
    if inferred_path:
        return inferred_path, f"{POSTER_BASE_URL}{inferred_path}"
    return "", image_url


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else input_path
    poster_dir = resolve_path(args.poster_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if "local_poster_path" not in df.columns:
        df["local_poster_path"] = ""
    if "poster_path" not in df.columns:
        df["poster_path"] = ""
    if "poster_url" not in df.columns:
        df["poster_url"] = ""

    poster_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    missing_poster_url = 0
    tmdb_resolved = 0
    tmdb_lookup_failed = 0
    no_key_resolved = 0
    no_key_lookup_failed = 0
    max_downloads = args.max_images if args.max_images is not None else float("inf")
    max_lookups = args.max_lookups if args.max_lookups is not None else float("inf")
    lookups_attempted = 0
    local_paths = df["local_poster_path"].fillna("").astype(str).tolist()
    poster_paths = df["poster_path"].fillna("").astype(str).tolist()
    poster_urls = df["poster_url"].fillna("").astype(str).tolist()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ua473-dataset-prep/1.0",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        }
    )

    for idx, row in df.iterrows():
        current_path = str(local_paths[idx]).strip()
        if current_path and not args.force and local_file_exists(current_path):
            skipped += 1
            continue

        if downloaded >= max_downloads:
            continue

        poster_url = resolve_poster_url_values(poster_urls[idx], poster_paths[idx])
        movie_id = normalize_movie_id(row.get("movie_id", idx))
        if not poster_url and movie_id is not None and lookups_attempted >= max_lookups:
            break

        if not poster_url and args.tmdb_api_key and movie_id is not None:
            lookups_attempted += 1
            resolved_path = fetch_tmdb_poster_path(
                movie_id=movie_id,
                api_key=args.tmdb_api_key,
                session=session,
                timeout=args.timeout,
            )
            if resolved_path:
                poster_paths[idx] = resolved_path
                poster_url = f"{POSTER_BASE_URL}{resolved_path}"
                poster_urls[idx] = poster_url
                tmdb_resolved += 1
            else:
                tmdb_lookup_failed += 1

        if not poster_url and movie_id is not None:
            lookups_attempted += 1
            scraped_path, scraped_url = fetch_tmdb_poster_without_key(
                movie_id=movie_id,
                session=session,
                timeout=args.timeout,
            )
            if scraped_url:
                if scraped_path:
                    poster_paths[idx] = scraped_path
                poster_urls[idx] = scraped_url
                poster_url = scraped_url
                no_key_resolved += 1
            else:
                no_key_lookup_failed += 1

        if not poster_url:
            missing_poster_url += 1
            continue

        filename_stem = sanitize_name(movie_id, fallback=f"movie_{idx}")

        try:
            response = session.get(poster_url, timeout=args.timeout)
            if response.status_code != 200:
                failed += 1
                continue

            content_type = response.headers.get("Content-Type", "")
            extension = extension_from_url_or_type(poster_url, content_type)
            poster_path = poster_dir / f"{filename_stem}{extension}"
            poster_path.write_bytes(response.content)

            local_paths[idx] = to_stored_path(poster_path)
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded} posters...")
        except requests.RequestException:
            failed += 1

    df["poster_path"] = poster_paths
    df["poster_url"] = poster_urls
    df["local_poster_path"] = local_paths
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Saved updated parquet to: {output_path}")
    print(
        f"Posters downloaded: {downloaded} | skipped existing: {skipped} | "
        f"failed: {failed} | missing URL: {missing_poster_url}"
    )
    if args.tmdb_api_key:
        print(
            f"TMDB lookups resolved: {tmdb_resolved} | "
            f"TMDB lookups failed: {tmdb_lookup_failed}"
        )
    print(
        f"No-key page lookups resolved: {no_key_resolved} | "
        f"No-key page lookups failed: {no_key_lookup_failed}"
    )
    if args.max_lookups is not None:
        print(f"Lookup attempts used: {lookups_attempted}/{args.max_lookups}")


if __name__ == "__main__":
    main()
