"""Utilities for FTMS.

Provides metrics, embeddings, persistence, and configuration utilities.
"""

from rotalabs_fieldmem.utils.schemas import FTCSConfig
from rotalabs_fieldmem.utils.config import ConfigManager, ConfigError, load_config
from rotalabs_fieldmem.utils.metrics import QualityMetrics, QualityScore
from rotalabs_fieldmem.utils.persistence import MemoryPersistence, PersistenceMetadata

# Optional: embeddings (requires sentence-transformers)
try:
    from rotalabs_fieldmem.utils.embeddings import (
        EmbeddingManager,
        get_embedding_manager,
    )
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

__all__ = [
    # Config
    "FTCSConfig",
    "ConfigManager",
    "ConfigError",
    "load_config",
    # Metrics
    "QualityMetrics",
    "QualityScore",
    # Persistence
    "MemoryPersistence",
    "PersistenceMetadata",
    # Embeddings (conditional)
    "HAS_EMBEDDINGS",
]

if HAS_EMBEDDINGS:
    __all__.extend(["EmbeddingManager", "get_embedding_manager"])
