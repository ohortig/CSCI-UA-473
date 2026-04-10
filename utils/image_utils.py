"""
Image Download and Loading Utilities

This module provides helper functions for downloading movie posters from URLs
and loading them from disk.

Key Educational Concepts:
-------------------------
1. **Web Scraping**: Downloading images from internet URLs
   - HTTP requests with timeout (avoid hanging)
   - Status code checking (404 = not found, 200 = success)
   - Content verification (ensure it's a valid image)

2. **Error Handling**: Graceful degradation
   - Network failures (timeout, connection error)
   - Invalid URLs (malformed, non-existent)
   - Corrupted data (not a real image)
   - Missing files (poster not downloaded yet)

3. **PIL (Pillow)**: Python's image library
   - Open images from bytes or disk
   - Verify integrity (.verify() checks structure)
   - Convert formats (.convert("RGB") ensures 3 channels)
   - Save in different formats (JPEG, PNG, etc.)

Why This Matters:
-----------------
In real-world ML systems, data doesn't magically appear:
- You scrape/download from various sources
- You validate quality (broken images → bad embeddings)
- You handle errors gracefully (one failure shouldn't crash everything)
- You cache locally (respect bandwidth, enable offline work)
"""

import os
from io import BytesIO

import requests
from PIL import Image


def download_image(url, save_path):
    """
    Download an image from a URL and save it locally.

    This function implements robust image downloading with:
    - Timeout protection (avoid hanging on slow servers)
    - Error code checking (handle 404, 403, etc.)
    - Image validation (ensure downloaded data is a valid image)
    - Directory creation (prepare save location)

    Common Failure Modes:
    ---------------------
    1. **Network Issues**:
       - Timeout (server too slow)
       - ConnectionError (no internet, DNS failure)
       - SSL errors (certificate problems)

    2. **HTTP Errors**:
       - 404 Not Found (poster URL no longer exists)
       - 403 Forbidden (server blocks our request)
       - 500 Server Error (TMDB API having issues)

    3. **Data Issues**:
       - Not an image (HTML error page instead)
       - Corrupted image (incomplete download)
       - Unsupported format (obscure image type)

    Args:
        url: String URL pointing to an image file
             Example: "https://image.tmdb.org/t/p/w500/abc123.jpg"

        save_path: Local path where image should be saved
                  Example: "data/posters/12345.jpg"

    Returns:
        bool: True if download and save succeeded, False otherwise
             We return False (not raise exception) for graceful degradation

    Example Usage:
    --------------
    from utils.image_utils import download_image

    # Download single poster
    url = "https://image.tmdb.org/t/p/w500/poster.jpg"
    success = download_image(url, "data/posters/123.jpg")

    if success:
        print("Downloaded successfully!")
    else:
        print("Failed - will use placeholder")

    # Batch download with progress tracking
    from tqdm import tqdm

    failed_urls = []
    for movie_id, poster_url in tqdm(poster_urls.items()):
        save_path = f"data/posters/{movie_id}.jpg"
        if not download_image(poster_url, save_path):
            failed_urls.append((movie_id, poster_url))

    print(f"Failed to download {len(failed_urls)} posters")

    Educational Note - timeout Parameter:
    ------------------------------------
    timeout=10 means:
    - Wait max 10 seconds for server to respond
    - If no response, raise requests.Timeout exception
    - Prevents hanging indefinitely on dead servers

    Without timeout, a single slow URL could freeze your script for minutes!

    Always use timeouts for production web scraping.
    """
    try:
        # ====================================================================
        # STEP 1: DOWNLOAD IMAGE DATA
        # ====================================================================
        # Make HTTP GET request with timeout protection
        # timeout=10 means "give up if server doesn't respond in 10 seconds"
        response = requests.get(url, timeout=10)

        # Check if request succeeded
        # raise_for_status() raises HTTPError for bad status codes (4xx, 5xx)
        # Examples: 404 Not Found, 403 Forbidden, 500 Server Error
        response.raise_for_status()

        # ====================================================================
        # STEP 2: VALIDATE IT'S A REAL IMAGE
        # ====================================================================
        # Sometimes servers return HTML error pages instead of images
        # We need to verify the downloaded content is actually an image

        # Open image from bytes (in-memory, not saved yet)
        # BytesIO creates a file-like object from bytes
        # This allows PIL to read the image without saving to disk first
        image = Image.open(BytesIO(response.content))

        # Verify image integrity
        # .verify() checks the file structure is valid
        # Detects corrupted downloads, truncated files, etc.
        # Note: verify() closes the file internally, so we need to reopen
        image.verify()

        # ====================================================================
        # STEP 3: REOPEN AND SAVE
        # ====================================================================
        # Re-open the image (verify() closed it)
        # We need a fresh Image object to save
        image = Image.open(BytesIO(response.content))

        # Create directory if it doesn't exist
        # os.path.dirname(save_path) extracts directory part
        # Example: "data/posters/123.jpg" → "data/posters"
        # exist_ok=True prevents error if directory already exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the validated image to disk
        # PIL automatically detects format from file extension
        image.save(save_path)

        return True

    except requests.Timeout:
        # Server took too long to respond (> 10 seconds)
        # Common with slow or overloaded servers
        # print(f"Timeout downloading {url}")
        return False

    except requests.HTTPError:
        # HTTP error codes (404, 403, 500, etc.)
        # print(f"HTTP error {e.response.status_code} for {url}")
        return False

    except requests.RequestException:
        # Generic network errors (no internet, DNS failure, etc.)
        # print(f"Network error downloading {url}")
        return False

    except Exception:
        # Catch-all for other issues (corrupted image, disk full, permissions)
        # We use broad except to ensure download failures never crash the pipeline
        # In production, you'd log these for debugging
        # print(f"Failed to download {url}: {e}")
        return False


def load_image(path):
    """
    Load an image from local disk.

    This is a simple wrapper around PIL.Image.open() with:
    - Existence checking (avoid FileNotFoundError)
    - Error handling (corrupted files, wrong format)
    - RGB conversion (ensure 3 channels for embedding model)

    Why Convert to RGB?
    -------------------
    Images can have different color modes:
    - RGB: 3 channels (Red, Green, Blue) - what we need
    - RGBA: 4 channels (RGB + Alpha transparency)
    - L: 1 channel (grayscale)
    - CMYK: 4 channels (for printing)

    DINOv2 expects RGB (3 channels), so we convert all images to RGB.
    This ensures consistent input regardless of source format.

    Args:
        path: Local file path to image
             Example: "data/posters/123.jpg"

    Returns:
        PIL.Image or None:
            - PIL.Image object if load succeeded (RGB mode)
            - None if file doesn't exist or is corrupted

    Example Usage:
    --------------
    from utils.image_utils import load_image
    from utils.image_embedding import get_image_embedder

    # Load and embed
    image = load_image("data/posters/123.jpg")
    if image is not None:
        embedder = get_image_embedder()
        embedding = embedder.embed(image)
    else:
        print("Image not found - using placeholder")
        embedding = np.zeros(384)

    # Batch processing
    poster_paths = df['local_poster_path'].values
    embeddings = []

    for path in poster_paths:
        img = load_image(path)
        if img is not None:
            emb = embedder.embed(img)
        else:
            emb = np.zeros(384)  # Placeholder for missing posters
        embeddings.append(emb)

    Educational Note - Error Handling Philosophy:
    ---------------------------------------------
    We return None instead of raising exceptions because:
    - Missing posters shouldn't crash the entire pipeline
    - Caller can decide how to handle (placeholder, skip, retry)
    - Enables graceful degradation (show app with missing images)

    Alternative: Raise exception and let caller handle
    Trade-off: Explicit errors vs ease of use

    For educational purposes, returning None is simpler to understand.
    """
    try:
        # Check if file exists before trying to open
        # Avoids FileNotFoundError exception
        if os.path.exists(path):
            # Open image from disk
            # PIL supports JPEG, PNG, GIF, BMP, and many others
            image = Image.open(path)

            # Convert to RGB mode (3 channels)
            # This ensures consistency for embedding models
            # Handles: RGBA → RGB, L (grayscale) → RGB, etc.
            # If already RGB, this is a no-op (fast)
            return image.convert("RGB")

    except Exception:
        # Catch any errors (corrupted file, permission denied, etc.)
        # We silently fail and return None for graceful degradation
        # In production, you might want to log these errors
        pass

    # File doesn't exist or failed to load
    return None
