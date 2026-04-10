"""
Text Embedding Utilities using Sentence Transformers

This module provides a wrapper around the Nomic AI embedding model for converting
movie descriptions into dense vector representations.

What are Embeddings?
--------------------
Embeddings are numerical vector representations of text that capture semantic meaning.
Instead of treating words as discrete symbols, we convert them into points in a
high-dimensional space where semantically similar items are close together.

Example:
--------
"Action movie with explosions" → [0.23, -0.45, 0.67, ..., 0.12]  (768 numbers)
"Thriller with car chases"     → [0.25, -0.42, 0.70, ..., 0.09]  (close in space!)
"Romantic comedy about dating" → [-0.87, 0.32, -0.15, ..., 0.43] (far away)

Key Properties:
---------------
1. **Semantic Similarity**: Similar meanings → similar vectors
2. **Fixed Dimensionality**: All texts become same-size vectors (e.g., 768-D)
3. **Dense Representation**: Every dimension has meaning (vs sparse like one-hot)
4. **Transferable**: Learned on billions of texts, works on our movie dataset

Why Nomic AI Model?
-------------------
- High quality: Trained on diverse web text, optimized for retrieval
- Long context: Handles up to 8192 tokens (most models: 512)
- Asymmetric search: Distinguishes between queries and documents
- Open source: MIT licensed, can inspect and modify

Model Details:
--------------
- Name: nomic-ai/nomic-embed-text-v1.5
- Architecture: Transformer encoder (BERT-style)
- Embedding dimension: 768
- Training: Contrastive learning on text pairs
- Normalization: Outputs are unit vectors (cosine similarity = dot product)
"""

from sentence_transformers import SentenceTransformer

# ========================================================================
# MODEL CONFIGURATION
# ========================================================================
# We use Nomic AI's embedding model which provides:
# - High quality embeddings for retrieval tasks
# - Support for long documents (up to 8K tokens)
# - Distinction between search queries and documents (asymmetric search)
# - Open source and free to use

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


class MovieEmbedder:
    """
    A wrapper for the Nomic AI embedding model with educational documentation.

    Design Pattern: Singleton
    --------------------------
    This class uses an implicit singleton pattern (via get_embedder() function).
    We only create ONE instance of the model because:

    1. **Memory Efficiency**: The model weighs ~500MB
       - Loading multiple copies would waste RAM
       - Typical: single model serves entire application

    2. **Loading Time**: Model initialization takes ~10 seconds
       - Downloads weights from Hugging Face (first time)
       - Loads into RAM and potentially GPU
       - Creating one instance avoids repeated loading

    3. **Consistency**: Same model for all embeddings
       - Ensures all vectors come from the same embedding space
       - Prevents subtle bugs from model version mismatches

    Why Sentence Transformers?
    ---------------------------
    The sentence-transformers library provides:
    - Easy interface to 1000+ pre-trained models
    - Automatic batching and GPU utilization
    - Consistent API across different model architectures
    - Built-in normalization and pooling options

    Educational Note:
    -----------------
    In production systems, you'd typically:
    - Load model once at startup (singleton or global)
    - Cache embeddings in a database (avoid recomputing)
    - Use a vector database (Pinecone, Weaviate) for fast similarity search
    - Potentially deploy model as separate embedding service (microservices)
    """

    def __init__(self):
        """
        Initialize the embedding model.

        This loads the model weights from Hugging Face and prepares it for inference.

        Parameters Explained:
        ---------------------
        - MODEL_NAME: The model identifier on Hugging Face Hub
          Format: {organization}/{model-name}

        - trust_remote_code=True:
          Some models use custom Python code for architecture/preprocessing.
          This flag allows executing that code.

          ⚠️  Security Note: Only use with models from trusted sources!
          The code runs on your machine and could theoretically be malicious.
          Nomic AI is a reputable organization, so this is safe here.

        What happens during initialization:
        -----------------------------------
        1. Check local cache (~/.cache/torch/sentence_transformers/)
        2. If not found, download model weights from HuggingFace (~500MB)
        3. Load model into RAM
        4. Compile model for efficient inference
        5. Move to GPU if available (automatic detection)

        First run: ~30 seconds (download + load)
        Subsequent runs: ~10 seconds (load from cache)
        """
        print(f"Loading embedding model: {MODEL_NAME}")
        print("This may take a moment on first run (downloading model weights)...")

        # Load the pre-trained model
        # The SentenceTransformer class handles all the complexity:
        # - Tokenization (text → token IDs)
        # - Model inference (tokens → hidden states)
        # - Pooling (hidden states → single vector)
        # - Normalization (ensure unit length)
        self.model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

        print("✓ Model loaded successfully")

    def embed(self, texts, task_type="search_document"):
        """
        Generate embeddings for a list of text strings.

        This is the core function that converts text into numbers.

        How It Works:
        -------------
        1. **Tokenization**: Text → Integer IDs
           "Action movie" → [101, 2895, 3185, 102]
           (Special tokens: 101=CLS, 102=SEP)

        2. **Embedding Lookup**: Token IDs → Initial Vectors
           Each token gets a learned vector from a vocabulary

        3. **Transformer Layers**: Process and contextualize
           Tokens attend to each other (self-attention)
           "movie" embedding is influenced by "action"
           Multiple layers of processing (typically 12)

        4. **Pooling**: Multiple token vectors → Single document vector
           Common strategies: mean pooling, CLS token, max pooling
           Nomic uses mean pooling of last layer

        5. **Normalization**: Scale to unit length
           Important for cosine similarity search

        Args:
            texts: List of strings to embed
                   Can be titles, descriptions, queries, etc.
                   Each becomes an independent embedding

            task_type: Embedding mode, affects the model's behavior

                       'search_document': For items in your corpus
                       - Example: Movie descriptions to be searched
                       - Model optimizes for being retrieved

                       'search_query': For user search queries
                       - Example: "funny movies about dogs"
                       - Model optimizes for finding relevant documents

                       Why distinguish?
                       ----------------
                       Asymmetric search: queries ≠ documents
                       - Queries are short, specific
                       - Documents are long, descriptive
                       - Model learned these are different tasks
                       - Adding task prefix helps model provide better embeddings

        Returns:
            np.ndarray: Embedding matrix of shape (n_texts, embedding_dim)
                       For Nomic model: (n_texts, 768)
                       Each row is a normalized embedding vector

        Example Usage:
        --------------
        embedder = MovieEmbedder()

        # Embed movies (corpus)
        movies = ["Action thriller", "Romantic comedy", "Sci-fi adventure"]
        doc_embeddings = embedder.embed(movies, task_type="search_document")
        # Shape: (3, 768)

        # Embed user query
        query = ["funny movie"]
        query_embedding = embedder.embed(query, task_type="search_query")
        # Shape: (1, 768)

        # Find similar movies (cosine similarity = dot product for normalized vectors)
        similarities = query_embedding @ doc_embeddings.T
        # Shape: (1, 3) - similarity scores

        Educational Note - Batch Processing:
        ------------------------------------
        The model processes texts in batches internally for efficiency:
        - Single text: 100ms
        - 32 texts (batch): 120ms (30× speedup!)
        - GPU utilization improves with larger batches
        - Always batch when possible (as we do in process_data.py)
        """
        # ====================================================================
        # STEP 1: ADD TASK PREFIX
        # ====================================================================
        # Nomic models use special prefixes to distinguish task types
        # This tells the model "I'm a query" vs "I'm a document to be found"
        prefix = (
            "search_query: " if task_type == "search_query" else "search_document: "
        )

        # Prepend prefix to each text
        # Example: "Action movie" → "search_document: Action movie"
        texts = [prefix + text for text in texts]

        # ====================================================================
        # STEP 2: GENERATE EMBEDDINGS
        # ====================================================================
        # The encode() method handles the entire pipeline:
        # Tokenization → Model Forward Pass → Pooling → Normalization
        embeddings = self.model.encode(
            texts,
            # convert_to_numpy=True: Return NumPy array (default is torch.Tensor)
            # We use NumPy for compatibility with sklearn, matplotlib, pandas
            convert_to_numpy=True,
            # normalize_embeddings=True: Scale vectors to unit length (||v|| = 1)
            # This makes cosine similarity equivalent to dot product:
            # cos(a, b) = (a · b) / (||a|| × ||b||) = a · b  (when normalized)
            # Dot product is faster to compute than cosine similarity!
            normalize_embeddings=True,
        )

        return embeddings


# ========================================================================
# SINGLETON PATTERN IMPLEMENTATION
# ========================================================================
# Global variable to hold the single embedder instance
# None = not yet created
# MovieEmbedder instance = created and cached
_embedder = None


def get_embedder():
    """
    Factory function to get the global embedder instance (singleton pattern).

    This ensures we only load the model ONCE across the entire application.

    Singleton Pattern:
    ------------------
    A design pattern that restricts instantiation of a class to a single object.

    Benefits:
    - Single source of truth
    - Shared resource management
    - Global access point

    Implementation:
    ---------------
    1. First call: Creates and caches instance
    2. Subsequent calls: Returns cached instance
    3. All callers share the same model

    Returns:
        MovieEmbedder: The global embedder instance

    Example Usage:
    --------------
    # In module A
    from utils.embedding import get_embedder
    embedder = get_embedder()  # Creates model
    emb1 = embedder.embed(["text 1"])

    # In module B
    from utils.embedding import get_embedder
    embedder = get_embedder()  # Reuses same model!
    emb2 = embedder.embed(["text 2"])

    # embedder in A and B are the SAME object (id(embedder_a) == id(embedder_b))

    Educational Note:
    -----------------
    In Python, we could also implement singletons using:
    - Class decorators
    - Metaclasses
    - Module-level instances

    This function-based approach is simple and clear for educational purposes.
    """
    global _embedder

    if _embedder is None:
        # First call - create the model
        print("Initializing embedder (first use)...")
        _embedder = MovieEmbedder()
    else:
        # Subsequent calls - reuse existing model
        print("Reusing existing embedder instance")

    return _embedder
