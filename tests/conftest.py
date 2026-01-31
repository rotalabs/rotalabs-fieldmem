"""Pytest configuration and shared fixtures for FTMS tests.

Provides common test fixtures, configuration, and utilities used across
the test suite to ensure consistent and reliable testing.
"""

import os
from typing import Generator, Tuple

# Set JAX to CPU for consistent testing - MUST be before JAX import
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest
import jax
import jax.numpy as jnp
from jax import random

from rotalabs_fieldmem.fields import MemoryField, FieldConfig


@pytest.fixture(scope="session")
def jax_config() -> None:
    """Configure JAX for testing environment."""
    # Ensure reproducible results
    jax.config.update('jax_enable_x64', True)


@pytest.fixture
def rng_key() -> jax.random.PRNGKey:
    """Provide a fixed random key for reproducible tests."""
    return random.PRNGKey(42)


@pytest.fixture
def field_config() -> FieldConfig:
    """Provide a standard field configuration for testing."""
    return FieldConfig(
        shape=(32, 32),
        dt=0.1,
        diffusion_rate=0.01,
        temperature=0.05,
        boundary_conditions="neumann"
    )


@pytest.fixture
def memory_field(field_config: FieldConfig) -> MemoryField:
    """Provide a fresh memory field for each test."""
    return MemoryField(field_config)


@pytest.fixture
def sample_embedding(rng_key: jax.random.PRNGKey) -> jnp.ndarray:
    """Provide a sample embedding vector for testing."""
    return random.normal(rng_key, (64,))


@pytest.fixture
def sample_embeddings(rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, ...]:
    """Provide multiple sample embeddings for testing."""
    keys = random.split(rng_key, 5)
    return tuple(random.normal(key, (32,)) for key in keys)


@pytest.fixture
def populated_field(
    memory_field: MemoryField,
    sample_embeddings: Tuple[jnp.ndarray, ...]
) -> MemoryField:
    """Provide a memory field pre-populated with test memories."""
    positions = [(8, 8), (24, 24), (8, 24), (24, 8)]

    for i, (embedding, position) in enumerate(zip(sample_embeddings, positions)):
        memory_field.inject_memory(
            embedding,
            position=position,
            importance=1.0 - i * 0.1
        )

    # Evolve field a few steps to advance time
    for _ in range(3):
        memory_field.step()

    return memory_field


class TestConstants:
    """Test constants and tolerances."""

    # Numerical tolerances
    FLOAT_TOL = 1e-6
    ENERGY_TOL = 1e-5

    # Field parameters
    MIN_FIELD_SIZE = (8, 8)
    MAX_FIELD_SIZE = (256, 256)

    # Performance thresholds
    MAX_EVOLUTION_TIME_MS = 100  # per step
    MAX_INJECTION_TIME_MS = 10   # per injection
    MAX_QUERY_TIME_MS = 50       # per query
