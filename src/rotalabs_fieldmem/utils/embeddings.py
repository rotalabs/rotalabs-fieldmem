"""
Embedding Manager for FTCS

Provides high-quality text embeddings using sentence-transformers
to improve memory retrieval relevance.
"""
import logging
from typing import List, Optional, Union, Dict, Any
import jax.numpy as jnp
import numpy as np  # Keep for sentence-transformers compatibility
from pathlib import Path
import json

# Handle optional sentence-transformers dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Using fallback embeddings.")


class EmbeddingManager:
    """
    Manages text embeddings for FTCS memory system.
    
    Supports multiple backends:
    1. Sentence-transformers (preferred)
    2. Simple hash-based fallback
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384,
                 cache_dir: Optional[str] = None,
                 use_gpu: bool = False):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of sentence-transformer model
            embedding_dim: Dimension of embeddings
            cache_dir: Directory for caching models
            use_gpu: Whether to use GPU for embeddings
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu
        
        self.logger = logging.getLogger("EmbeddingManager")
        
        # Initialize embedding model
        self._init_model()
        
        # Cache for computed embeddings
        self._embedding_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _init_model(self):
        """Initialize the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Popular lightweight models with their dimensions
                model_dims = {
                    "all-MiniLM-L6-v2": 384,
                    "all-MiniLM-L12-v2": 384,
                    "all-mpnet-base-v2": 768,
                    "multi-qa-MiniLM-L6-cos-v1": 384,
                    "paraphrase-MiniLM-L6-v2": 384,
                    "paraphrase-multilingual-MiniLM-L12-v2": 384
                }
                
                # Update embedding dimension based on model
                if self.model_name in model_dims:
                    self.embedding_dim = model_dims[self.model_name]
                
                # Load model
                device = "cuda" if self.use_gpu else "cpu"
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device=device
                )
                
                # Verify dimension
                test_embedding = self.model.encode("test", convert_to_numpy=True)
                actual_dim = len(test_embedding)
                
                if actual_dim != self.embedding_dim:
                    self.logger.warning(
                        f"Model {self.model_name} has dimension {actual_dim}, "
                        f"not {self.embedding_dim}. Updating."
                    )
                    self.embedding_dim = actual_dim
                
                self.logger.info(
                    f"Initialized {self.model_name} with dimension {self.embedding_dim}"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to load sentence-transformers: {e}")
                self.model = None
        else:
            self.model = None
            self.logger.info("Using fallback hash-based embeddings")
    
    def encode(self, 
               texts: Union[str, List[str]], 
               normalize: bool = True,
               use_cache: bool = True) -> jnp.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            use_cache: Whether to use caching
            
        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Check cache
        embeddings = []
        texts_to_encode = []
        indices_to_fill = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = hash(text)
                if cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                    self._cache_hits += 1
                else:
                    texts_to_encode.append(text)
                    indices_to_fill.append(i)
                    self._cache_misses += 1
        else:
            texts_to_encode = texts
            indices_to_fill = list(range(len(texts)))
        
        # Encode missing texts
        if texts_to_encode:
            if self.model is not None:
                # Use sentence-transformers
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize
                )
                # Convert numpy to JAX array
                new_embeddings = jnp.array(new_embeddings)
            else:
                # Fallback to simple embeddings
                new_embeddings = jnp.array([
                    self._simple_embedding(text, normalize)
                    for text in texts_to_encode
                ])
            
            # Update cache
            if use_cache:
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    cache_key = hash(text)
                    self._embedding_cache[cache_key] = embedding
        
        # Combine cached and new embeddings
        if use_cache and embeddings:
            # Reconstruct full embedding array
            full_embeddings = jnp.zeros((len(texts), self.embedding_dim))
            
            # Fill cached embeddings
            cache_idx = 0
            new_idx = 0
            
            for i in range(len(texts)):
                if i in indices_to_fill:
                    full_embeddings = full_embeddings.at[i].set(new_embeddings[new_idx])
                    new_idx += 1
                else:
                    full_embeddings = full_embeddings.at[i].set(embeddings[cache_idx])
                    cache_idx += 1
            
            result = full_embeddings
        else:
            result = new_embeddings if texts_to_encode else jnp.array(embeddings)
        
        # Return single embedding if input was single text
        if single_text:
            return result[0]
        
        return result
    
    def _simple_embedding(self, text: str, normalize: bool = True) -> jnp.ndarray:
        """
        Simple fallback embedding using word statistics.
        
        Args:
            text: Input text
            normalize: Whether to normalize
            
        Returns:
            Embedding vector
        """
        # Create embedding based on character n-grams
        embedding = jnp.zeros(self.embedding_dim)
        
        # Convert to lowercase
        text = text.lower()
        
        # Character trigrams
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            # Hash to embedding dimension
            idx = hash(trigram) % self.embedding_dim
            embedding = embedding.at[idx].add(1.0)
        
        # Word unigrams
        words = text.split()
        for word in words:
            idx = hash(word) % self.embedding_dim
            embedding = embedding.at[idx].add(2.0)  # Higher weight for whole words
        
        # Add some positional information
        for i, word in enumerate(words[:10]):  # First 10 words
            idx = (hash(word) + i) % self.embedding_dim
            embedding = embedding.at[idx].add(0.5)
        
        # Normalize if requested
        if normalize:
            norm = jnp.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def cosine_similarity(self, 
                         embedding1: jnp.ndarray, 
                         embedding2: jnp.ndarray) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Handle dimension mismatch
        min_dim = min(len(embedding1), len(embedding2))
        e1 = embedding1[:min_dim]
        e2 = embedding2[:min_dim]
        
        # Calculate norms
        norm1 = jnp.linalg.norm(e1)
        norm2 = jnp.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        return jnp.dot(e1, e2) / (norm1 * norm2)
    
    def batch_similarities(self,
                          query_embedding: jnp.ndarray,
                          embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate similarities between query and multiple embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: Array of embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(embeddings) == 0:
            return jnp.array([])
        
        # Ensure same dimensions
        min_dim = min(len(query_embedding), embeddings.shape[1])
        query = query_embedding[:min_dim]
        embeds = embeddings[:, :min_dim]
        
        # Normalize query
        query_norm = jnp.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Normalize embeddings
        norms = jnp.linalg.norm(embeds, axis=1)
        valid_mask = norms > 0
        embeds = embeds.at[valid_mask].set(embeds[valid_mask] / norms[valid_mask, jnp.newaxis])
        
        # Calculate similarities
        similarities = jnp.dot(embeds, query)
        similarities = similarities.at[~valid_mask].set(0.0)
        
        return similarities
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding manager statistics."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "using_sentence_transformers": self.model is not None,
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0.0
            )
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("Embedding cache cleared")
    
    def save_config(self, file_path: str):
        """Save embedding manager configuration."""
        config = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "cache_dir": self.cache_dir,
            "use_gpu": self.use_gpu
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def from_config(cls, file_path: str) -> "EmbeddingManager":
        """Load embedding manager from configuration."""
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        return cls(**config)


# Singleton instance for easy access
_embedding_manager = None


def get_embedding_manager(
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> EmbeddingManager:
    """
    Get or create the singleton embedding manager.
    
    Args:
        model_name: Model to use
        **kwargs: Additional arguments for EmbeddingManager
        
    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(model_name, **kwargs)
    
    return _embedding_manager