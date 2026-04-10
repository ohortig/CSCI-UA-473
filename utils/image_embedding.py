"""
Image Embedding Utilities using DINOv2

This module provides a wrapper around Meta's DINOv2 model for converting
images into dense vector representations.

What is DINOv2?
----------------
DINOv2 is a self-supervised vision transformer (ViT) trained by Meta AI.
Unlike text embeddings which learned from language, image embeddings learn
visual patterns from millions of images.

Key Features:
- No labels needed during training (self-supervised learning)
- Captures visual semantics (colors, shapes, textures, objects)
- Transfer learning: trained on general images, works on movie posters
- Small variant: Balances quality vs speed/memory

Architecture:
-------------
Vision Transformer (ViT):
1. Split image into patches (e.g., 14×14 pixel squares)
2. Each patch becomes a token (like words in text)
3. Transformer processes patch tokens with self-attention
4. CLS token aggregates information from all patches
5. CLS vector becomes the image embedding

Educational Comparison:
-----------------------
Text Embeddings (Nomic):
- Input: Sequence of words → tokens
- Model: Transformer encoder
- Output: 768-D vector from CLS token/ mean pooling

Image Embeddings (DINOv2):
- Input: Image → grid of patches
- Model: Vision Transformer (same architecture!)
- Output: 384-D vector from CLS token

→ Same fundamental architecture, different modality!
"""

import numpy as np
import torch

# Try to import transformers, handle failure gracefully
try:
    from transformers import AutoImageProcessor, AutoModel

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import transformers: {e}")
    TRANSFORMERS_AVAILABLE = False
except RuntimeError as e:
    print(
        f"Warning: Runtime error importing transformers (likely torchvision issue): {e}"
    )
    TRANSFORMERS_AVAILABLE = False

# ========================================================================
# MODEL CONFIGURATION
# ========================================================================
# DINOv2 comes in multiple sizes:
# - dinov2-small: 384-D embeddings (~50M parameters) ← We use this
# - dinov2-base: 768-D embeddings
# - dinov2-large: 1024-D embeddings
# - dinov2-giant: 1536-D embeddings
#
# Small version is chosen for:
# - Faster inference (~100ms vs ~500ms for giant)
# - Lower memory usage ( ~200MB vs ~2GB)
# - Still high quality for our use case (movie poster similarity)

MODEL_NAME = "facebook/dinov2-small"


class ImageEmbedder:
    """
    Wrapper for DINOv2 image embedding model with educational documentation.

    Design Pattern: Singleton (via get_image_embedder() function)
    Similar to MovieEmbedder, we only create one instance to save memory.

    Components:
    -----------
    1. **Processor**: Preprocesses images for the model
       - Resizes to 224×224 (DINOv2's expected input size)
       - Normalizes pixel values (mean/std from ImageNet)
       - Converts to tensor format

    2. **Model**: The DINOv2 Vision Transformer
       - Splits image into 16×16 = 256 patches (for 224×224 input)
       - Processes with 12 transformer layers
       - Outputs embeddings for each patch + CLS token

    Why Normalize?
    --------------
    Like text embeddings, we normalize image embeddings to unit length.
    This ensures:
    - Cosine similarity = dot product (faster computation)
    - All images have equal "magnitude" (only direction matters)
    - Consistent scale across different images
    """

    def __init__(self):
        """
        Initialize the image embedding model.

        This loads both the preprocessing pipeline and the neural network.

        What Gets Loaded:
        -----------------
        1. Image Processor:
           - Configuration for image preprocessing
           - Normalization statistics (mean, std)
           - Expected input size (224×224)

        2. DINOv2 Model:
           - Vision Transformer weights (~50MB for small variant)
           - Patch embedding layers
           - 12 transformer encoder layers
           - Layer normalization parameters

        Model Evaluation Mode:
        ----------------------
        .eval() tells PyTorch we're doing inference, not training:
        - Disables dropout (no random neuron dropping)
        - Uses running averages for batch norm (if any)
        - Ensures deterministic behavior (same input → same output)

        First run: ~20 seconds (download model from HuggingFace)
        Subsequent runs: ~5 seconds (load from cache)
        """
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available. Using dummy embedder.")
            return

        print(f"Loading image embedding model: {MODEL_NAME}")

        # Load preprocessing pipeline
        # AutoImageProcessor automatically selects the right processor for the model
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

        # Load the DINOv2 vision transformer
        # AutoModel automatically loads the correct architecture
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        # Set to evaluation mode (disable training-specific layers)
        self.model.eval()

        print("✓ Image model loaded successfully")

    def embed(self, images):
        """
        Generate embedding for a single image or a list of images.

        Pipeline:
        ---------
        1. Preprocess: PIL Image(s) → Tensor
        2. Forward pass: Tensor → Model → Hidden states
        3. Extract CLS: Hidden states → CLS token embedding
        4. Normalize: CLS token → Unit vector

        Args:
            images: PIL Image object or List[PIL Image]

        Returns:
            np.ndarray: Normalized embedding vector(s)
                       Shape (384,) for single image
                       Shape (N, 384) for list of images
        """
        if not TRANSFORMERS_AVAILABLE:
            # Return dummy embedding (random unit vector)
            if isinstance(images, list):
                vecs = np.random.randn(len(images), 384).astype(np.float32)
                return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            else:
                vec = np.random.randn(384).astype(np.float32)
                return vec / np.linalg.norm(vec)

        # Handle single image case by wrapping it
        is_single = False
        if not isinstance(images, list):
            images = [images]
            is_single = True

        # ====================================================================
        # STEP 1: PREPROCESS IMAGE
        # ====================================================================
        # Convert PIL Image to model-ready tensor
        # return_tensors="pt" means PyTorch tensors
        inputs = self.processor(images=images, return_tensors="pt")

        # ====================================================================
        # STEP 2: FORWARD PASS THROUGH MODEL
        # ====================================================================
        # Disable gradient computation (we're not training)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # ====================================================================
        # STEP 3: EXTRACT CLS TOKEN
        # ====================================================================
        # last_hidden_state: (batch, num_tokens, hidden_dim)
        last_hidden_states = outputs.last_hidden_state

        # Extract CLS token: first token (index 0) of each item in batch
        # Shape: (batch_size, 384)
        cls_tokens = last_hidden_states[:, 0, :]

        # ====================================================================
        # STEP 4: NORMALIZE TO UNIT LENGTH
        # ====================================================================
        # L2 normalization: divide by vector's magnitude
        embeddings = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)

        # Convert from PyTorch tensor to NumPy array
        result = embeddings.numpy()

        # Return single vector if input was single image
        if is_single:
            return result[0]
        return result


# ========================================================================
# SINGLETON PATTERN IMPLEMENTATION
# ========================================================================
# Global variable holding the single ImageEmbedder instance
_image_embedder = None


def get_image_embedder():
    """
    Factory function to get the global image embedder instance (singleton).

    Same pattern as get_embedder() for text - ensures we only load the model once.

    Returns:
        ImageEmbedder: The global image embedder instance

    Example Usage:
    --------------
    # First call anywhere in the app
    embedder = get_image_embedder()  # Loads model

    # Later calls reuse the same model
    embedder = get_image_embedder()  # Returns cached instance
    """
    global _image_embedder

    if _image_embedder is None:
        # First access - create the model
        print("Initializing image embedder (first use)...")
        _image_embedder = ImageEmbedder()
    else:
        # Subsequent access - reuse existing model
        print("Reusing existing image embedder instance")

    return _image_embedder
