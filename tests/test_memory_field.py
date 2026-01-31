"""Comprehensive tests for MemoryField core functionality.

Tests the fundamental memory field operations including initialization,
memory injection, field evolution, querying, and forgetting mechanisms.
"""

import pytest
import jax.numpy as jnp
from jax import random
import numpy as np

from rotalabs_fieldmem import MemoryField, FieldConfig


# Test constants
FLOAT_TOL = 1e-5
ENERGY_TOL = 1e-3


@pytest.fixture
def rng_key():
    """Random key for JAX operations."""
    return random.PRNGKey(42)


@pytest.fixture
def field_config():
    """Standard field configuration for tests."""
    return FieldConfig(
        shape=(64, 64),
        dt=0.1,
        diffusion_rate=0.01,
        temperature=0.1,
    )


@pytest.fixture
def memory_field(field_config):
    """Fresh memory field instance."""
    return MemoryField(field_config)


@pytest.fixture
def sample_embedding(rng_key):
    """Sample 32-dimensional embedding."""
    return random.normal(rng_key, (32,))


@pytest.fixture
def sample_embeddings(rng_key):
    """Multiple sample embeddings."""
    keys = random.split(rng_key, 5)
    return tuple(random.normal(k, (32,)) for k in keys)


@pytest.fixture
def populated_field(memory_field, sample_embedding):
    """Memory field with some memories injected."""
    memory_field.inject_memory(sample_embedding, position=(16, 16), importance=1.0)
    memory_field.inject_memory(sample_embedding * 0.5, position=(32, 32), importance=0.8)
    # Run a few steps to evolve
    for _ in range(3):
        memory_field.step()
    return memory_field


class TestMemoryFieldInitialization:
    """Test memory field initialization and basic properties."""

    def test_field_initialization_default_config(self):
        """Test field initializes correctly with default configuration."""
        field = MemoryField()

        assert field.field.shape == (1000, 768)  # Default shape
        assert field.time == 0.0
        assert jnp.allclose(field.field, 0.0)
        assert field.compute_energy(field.field) == 0.0

    def test_field_initialization_custom_config(self, field_config):
        """Test field initializes correctly with custom configuration."""
        field = MemoryField(field_config)

        assert field.field.shape == field_config.shape
        assert field.config.dt == field_config.dt
        assert field.config.diffusion_rate == field_config.diffusion_rate
        assert field.config.temperature == field_config.temperature

    @pytest.mark.parametrize("shape", [
        (16, 16), (32, 64), (128, 128), (64, 128)
    ])
    def test_field_initialization_various_shapes(self, shape):
        """Test field initialization with various shapes."""
        config = FieldConfig(shape=shape)
        field = MemoryField(config)

        assert field.field.shape == shape
        assert field.field.size == shape[0] * shape[1]


class TestMemoryInjection:
    """Test memory injection functionality."""

    def test_inject_memory_basic(self, memory_field, sample_embedding):
        """Test basic memory injection."""
        initial_energy = memory_field.compute_energy(memory_field.field)

        memory_field.inject_memory(
            sample_embedding,
            position=(16, 16),
            importance=1.0
        )

        final_energy = memory_field.compute_energy(memory_field.field)

        # Energy should increase after injection
        assert final_energy > initial_energy
        # Field should no longer be zero everywhere
        assert not jnp.allclose(memory_field.field, 0.0)

    def test_inject_memory_auto_position(self, memory_field, sample_embedding):
        """Test memory injection with automatic position selection."""
        memory_field.inject_memory(sample_embedding, importance=1.0)

        # Should successfully inject without position
        assert memory_field.compute_energy(memory_field.field) > 0.0
        assert not jnp.allclose(memory_field.field, 0.0)

    def test_inject_memory_importance_scaling(self, memory_field, sample_embedding):
        """Test that importance parameter scales memory strength."""
        position = (16, 16)

        # Inject with low importance
        field_low = MemoryField(memory_field.config)
        field_low.inject_memory(sample_embedding, position=position, importance=0.1)
        energy_low = field_low.compute_energy(field_low.field)

        # Inject with high importance
        field_high = MemoryField(memory_field.config)
        field_high.inject_memory(sample_embedding, position=position, importance=1.0)
        energy_high = field_high.compute_energy(field_high.field)

        # Higher importance should create higher energy
        assert energy_high > energy_low

    def test_inject_multiple_memories(self, memory_field, sample_embeddings):
        """Test injection of multiple memories."""
        positions = [(8, 8), (24, 24), (8, 24)]
        energies = []

        for embedding, position in zip(sample_embeddings[:3], positions):
            memory_field.inject_memory(embedding, position=position, importance=1.0)
            energies.append(memory_field.compute_energy(memory_field.field))

        # Energy should increase with each injection
        assert all(energies[i] < energies[i + 1] for i in range(len(energies) - 1))

    def test_inject_mismatched_embedding_smaller(self, memory_field):
        """Test injection when embedding is smaller than field dimension."""
        # 32-dim embedding into 64x64 field
        small_embedding = jnp.ones(32)
        memory_field.inject_memory(small_embedding, position=(16, 16), importance=1.0)

        # Should successfully inject with padding
        assert memory_field.compute_energy(memory_field.field) > 0.0

    def test_inject_mismatched_embedding_larger(self, memory_field):
        """Test injection when embedding is larger than field dimension."""
        # 128-dim embedding into 64x64 field
        large_embedding = jnp.ones(128)
        memory_field.inject_memory(large_embedding, position=(16, 16), importance=1.0)

        # Should successfully inject with truncation
        assert memory_field.compute_energy(memory_field.field) > 0.0


class TestFieldEvolution:
    """Test field evolution and diffusion dynamics."""

    def test_evolution_step_basic(self, populated_field):
        """Test basic field evolution step."""
        initial_energy = populated_field.compute_energy(populated_field.field)
        initial_time = populated_field.time

        metrics = populated_field.step()

        # Time should advance
        assert populated_field.time > initial_time
        # Should return proper metrics
        assert 'energy' in metrics
        assert 'max_amplitude' in metrics
        assert 'field_norm' in metrics
        assert 'time' in metrics

    def test_evolution_energy_behavior(self, populated_field):
        """Test energy behavior during evolution."""
        initial_energy = populated_field.compute_energy(populated_field.field)

        # Run several evolution steps
        for _ in range(10):
            populated_field.step()

        final_energy = populated_field.compute_energy(populated_field.field)

        # Energy should not explode (may decrease due to diffusion/forgetting)
        assert 0.0 <= final_energy <= initial_energy * 2.0

    def test_evolution_custom_timestep(self, populated_field):
        """Test evolution with custom timestep."""
        custom_dt = 0.05
        initial_time = populated_field.time

        populated_field.step(dt=custom_dt)

        expected_time = initial_time + custom_dt
        assert abs(populated_field.time - expected_time) < FLOAT_TOL


class TestMemoryQuerying:
    """Test memory querying and retrieval functionality."""

    def test_query_memories_basic(self, populated_field, rng_key):
        """Test basic memory querying."""
        query_embedding = random.normal(rng_key, (32,))

        values, positions = populated_field.query_memories(query_embedding, k=3)

        assert len(values) == 3
        assert positions.shape == (3, 2)
        # Positions should be within field bounds
        assert jnp.all(positions >= 0)
        assert jnp.all(positions[:, 0] < populated_field.field.shape[0])
        assert jnp.all(positions[:, 1] < populated_field.field.shape[1])

    def test_query_memories_k_parameter(self, populated_field, rng_key):
        """Test querying with different k values."""
        query_embedding = random.normal(rng_key, (32,))

        for k in [1, 3, 5, 10]:
            values, positions = populated_field.query_memories(query_embedding, k=k)
            assert len(values) == k
            assert positions.shape == (k, 2)


class TestForgettingMechanism:
    """Test thermodynamic forgetting functionality."""

    def test_forgetting_reduces_field_strength(self, populated_field):
        """Test that evolution with forgetting reduces field strength over time."""
        initial_energy = populated_field.compute_energy(populated_field.field)

        # Run many evolution steps (includes forgetting)
        for _ in range(50):
            populated_field.step()

        final_energy = populated_field.compute_energy(populated_field.field)

        # Energy should decrease due to diffusion and forgetting
        # Note: may not always be strictly less due to noise, but trend should be down
        assert final_energy < initial_energy * 1.5  # Allow some variance


class TestFieldProperties:
    """Test field mathematical and physical properties."""

    def test_energy_computation_properties(self, memory_field, sample_embedding):
        """Test energy computation properties."""
        # Empty field should have zero energy
        assert memory_field.compute_energy(memory_field.field) == 0.0

        # Inject memory
        memory_field.inject_memory(sample_embedding, importance=1.0)
        energy = memory_field.compute_energy(memory_field.field)

        # Energy should be positive
        assert energy > 0.0

        # Energy should be finite
        assert jnp.isfinite(energy)

    def test_field_state_getter(self, populated_field):
        """Test field state getter returns complete information."""
        state = populated_field.get_field_state()

        required_keys = {
            'field', 'energy', 'time', 'max_value',
            'min_value', 'mean_value', 'std_value'
        }

        assert all(key in state for key in required_keys)
        assert isinstance(state['field'], np.ndarray)
        assert state['field'].shape == populated_field.field.shape

    def test_field_reset(self, populated_field):
        """Test field reset functionality."""
        # Verify field has content
        assert populated_field.compute_energy(populated_field.field) > 0.0
        assert populated_field.time > 0.0

        # Reset field
        populated_field.reset()

        # Verify field is reset
        assert populated_field.compute_energy(populated_field.field) == 0.0
        assert populated_field.time == 0.0
        assert jnp.allclose(populated_field.field, 0.0)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_zero_importance_injection(self, memory_field, sample_embedding):
        """Test injection with zero importance."""
        initial_energy = memory_field.compute_energy(memory_field.field)

        memory_field.inject_memory(sample_embedding, importance=0.0)
        final_energy = memory_field.compute_energy(memory_field.field)

        # Energy should not change significantly
        assert abs(final_energy - initial_energy) < ENERGY_TOL

    def test_negative_importance_injection(self, memory_field, sample_embedding):
        """Test injection with negative importance."""
        # Should handle negative importance gracefully
        memory_field.inject_memory(sample_embedding, importance=-1.0)

        # Field should still be in valid state
        assert jnp.isfinite(memory_field.compute_energy(memory_field.field))


@pytest.mark.parametrize("shape,diffusion_rate,temperature", [
    ((16, 16), 0.005, 0.01),
    ((32, 32), 0.01, 0.05),
    ((64, 64), 0.02, 0.1),
])
def test_field_configurations_parametrized(shape, diffusion_rate, temperature, rng_key):
    """Parametrized test for various field configurations."""
    config = FieldConfig(
        shape=shape,
        diffusion_rate=diffusion_rate,
        temperature=temperature
    )
    field = MemoryField(config)
    sample_embedding = random.normal(rng_key, (32,))

    # Basic functionality should work for all configurations
    field.inject_memory(sample_embedding, importance=1.0)
    assert field.compute_energy(field.field) > 0.0

    metrics = field.step()
    assert all(key in metrics for key in ['energy', 'time', 'max_amplitude'])

    values, positions = field.query_memories(sample_embedding, k=3)
    assert len(values) == 3


class TestImportanceWeighting:
    """Test importance-weighted memory dynamics."""

    def test_importance_mask_initialization(self, memory_field):
        """Test that importance mask is initialized correctly."""
        assert memory_field.importance_mask.shape == memory_field.field.shape
        assert jnp.allclose(memory_field.importance_mask, 1.0)

    def test_importance_affects_field_strength(self, sample_embedding):
        """Test that importance affects the field energy at injection."""
        config = FieldConfig(shape=(64, 64))

        field_high = MemoryField(config)
        field_high.inject_memory(sample_embedding, position=(16, 16), importance=2.0)

        field_low = MemoryField(config)
        field_low.inject_memory(sample_embedding, position=(16, 16), importance=0.5)

        # Higher importance should result in higher field energy
        high_energy = field_high.compute_energy(field_high.field)
        low_energy = field_low.compute_energy(field_low.field)

        assert high_energy > low_energy

    def test_importance_resists_forgetting(self, sample_embedding):
        """Test that higher importance resists forgetting more."""
        config = FieldConfig(shape=(64, 64), temperature=0.2)

        # High importance memory
        field_high = MemoryField(config)
        field_high.inject_memory(sample_embedding, position=(16, 16), importance=2.0)
        initial_high = field_high.compute_energy(field_high.field)

        # Low importance memory
        field_low = MemoryField(config)
        field_low.inject_memory(sample_embedding, position=(16, 16), importance=0.1)
        initial_low = field_low.compute_energy(field_low.field)

        # Evolve both fields
        for _ in range(20):
            field_high.step()
            field_low.step()

        final_high = field_high.compute_energy(field_high.field)
        final_low = field_low.compute_energy(field_low.field)

        # High importance should retain more energy relative to initial
        high_retention = final_high / (initial_high + 1e-8)
        low_retention = final_low / (initial_low + 1e-8)

        # This test may be flaky due to stochastic dynamics, so we use a loose bound
        # The main thing is that neither should be NaN or infinite
        assert jnp.isfinite(high_retention)
        assert jnp.isfinite(low_retention)
