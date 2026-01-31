"""
Memory Field Implementation for FTCS (Field-Theoretic Context System).

This module implements the core memory field using JAX for Apple Silicon Metal acceleration.
Memory is represented as continuous fields that evolve according to PDEs, enabling
natural memory dynamics like diffusion, consolidation, and forgetting.
"""

import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np

# Set JAX to use CPU backend for now (Metal has version mismatch)
# TODO: Update to Metal when jax-metal plugin is updated
os.environ['JAX_PLATFORMS'] = 'cpu'


@dataclass
class FieldConfig:
    """Configuration for memory field parameters."""
    shape: Tuple[int, int] = (1000, 768)  # Field dimensions
    dt: float = 0.1  # Evolution timestep
    diffusion_rate: float = 0.01  # Diffusion coefficient
    temperature: float = 0.1  # Thermodynamic temperature for forgetting
    boundary_conditions: str = "neumann"  # Boundary conditions
    
    # Importance weighting parameters
    importance_decay_rate: float = 0.01  # How fast importance decays
    min_importance: float = 0.1  # Minimum importance threshold
    importance_amplification: float = 2.0  # How much importance affects resistance
    

class MemoryField:
    """
    Core memory field implementation using JAX.
    
    Treats memory as a continuous 2D field where:
    - Memories are injected as localized energy distributions
    - Field evolves through diffusion (heat equation)
    - Forgetting occurs through thermodynamic decay
    - Importance is encoded as field energy/amplitude
    """
    
    def __init__(self, config: Optional[FieldConfig] = None):
        """Initialize memory field with given configuration."""
        self.config = config or FieldConfig()
        self.field = jnp.zeros(self.config.shape)
        self.time = 0.0
        
        # Importance-weighted memory system
        self.importance_mask = jnp.ones(self.config.shape)  # Resistance to diffusion/forgetting
        self.access_counts = jnp.zeros(self.config.shape)  # Track memory access for consolidation
        self.memory_ages = jnp.zeros(self.config.shape)  # Track how long memories have existed
        
        # Pre-compile JAX functions for performance
        self._setup_compiled_functions()
        
        # Initialize random key for stochastic operations
        self.rng_key = random.PRNGKey(42)
        
    def _setup_compiled_functions(self):
        """Setup JIT-compiled functions for optimal performance."""
        # Diffusion kernel for heat equation
        self.diffusion_kernel = jnp.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ]) / 4.0
        
        # Compile core operations (temporarily disable JIT for testing)
        self.evolve = self._evolve  # jit(self._evolve)
        self.inject = self._inject  # jit(self._inject)
        self.sample = self._sample  # jit(self._sample)
        self.compute_energy = self._compute_energy  # jit(self._compute_energy)
        self.apply_forgetting = self._apply_forgetting  # jit(self._apply_forgetting)
        
        # Compile importance-weighted operations (temporarily disable JIT)
        self.evolve_with_importance = self._evolve_with_importance  # jit(self._evolve_with_importance)
        self.apply_importance_forgetting = self._apply_importance_forgetting  # jit(self._apply_importance_forgetting)
        self.update_importance_from_access = self._update_importance_from_access  # jit(self._update_importance_from_access)
        self.consolidate_memories = self._consolidate_memories  # jit(self._consolidate_memories)
        
    def _apply_convolution(self, field: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
        """Apply convolution with given kernel using boundary conditions."""
        # Use scipy-style convolution which is JAX-compatible
        from jax.scipy import signal
        
        # Apply 2D convolution with boundary conditions
        # 'same' mode keeps the same size, 'boundary' handles edge conditions
        result = signal.convolve2d(field, kernel, mode='same', boundary='fill', fillvalue=0)
        return result
    
    def _evolve(self, field: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Single evolution step using heat equation approximation."""
        # Apply diffusion
        diffusion = self._apply_convolution(field, self.diffusion_kernel)
        
        # Update field: u/t = ��u (heat equation)
        evolved = field + dt * self.config.diffusion_rate * diffusion
        
        return evolved
    
    def _evolve_with_importance(self, field: jnp.ndarray, importance_mask: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Evolution step with importance-weighted diffusion resistance."""
        # Apply diffusion
        diffusion = self._apply_convolution(field, self.diffusion_kernel)
        
        # Importance reduces diffusion rate (high importance = low diffusion)
        # effective_diffusion_rate = base_rate / (1 + importance_amplification * importance)
        resistance_factor = 1.0 + self.config.importance_amplification * importance_mask
        effective_diffusion_rate = self.config.diffusion_rate / resistance_factor
        
        # Apply variable diffusion
        evolved = field + dt * effective_diffusion_rate * diffusion
        
        return evolved
    
    def _inject(self, field: jnp.ndarray, memory_data: jnp.ndarray,
                position: Tuple[int, int], strength: float) -> jnp.ndarray:
        """Inject memory data into field at specified position."""
        x, y = position

        # Create Gaussian envelope for memory injection
        sigma = 10.0  # Width of memory injection
        envelope = self._create_gaussian_envelope(field.shape, x, y, sigma)

        # Memory embedding (simplified - in practice would use actual embeddings)
        if memory_data.ndim == 1:
            # Vector embedding - distribute across spatial dimensions
            # Ensure dimensions match by padding/truncating
            memory_slice = memory_data[:min(len(memory_data), field.shape[0])]
            padded_memory = jnp.pad(
                memory_slice,
                (0, max(0, field.shape[0] - len(memory_slice)))
            )
            memory_field = jnp.outer(padded_memory, jnp.ones(field.shape[1])) * envelope
        else:
            # 2D memory data - resize if needed
            if memory_data.shape != field.shape:
                # Pad or truncate to match field shape
                padded = jnp.zeros(field.shape)
                min_rows = min(memory_data.shape[0], field.shape[0])
                min_cols = min(memory_data.shape[1], field.shape[1])
                padded = padded.at[:min_rows, :min_cols].set(memory_data[:min_rows, :min_cols])
                memory_field = padded * envelope
            else:
                memory_field = memory_data * envelope

        # Inject with specified strength
        return field + strength * memory_field
    
    def _create_gaussian_envelope(self, shape: Tuple[int, int], 
                                center_x: float, center_y: float, 
                                sigma: float) -> jnp.ndarray:
        """Create Gaussian envelope for memory injection."""
        x, y = jnp.meshgrid(jnp.arange(shape[0]), jnp.arange(shape[1]), indexing='ij')
        envelope = jnp.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return envelope
    
    def _sample(self, field: jnp.ndarray, query_embedding: jnp.ndarray, 
                k: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample relevant memories from field based on query."""
        # Compute similarity between query and field regions
        # Simplified version - in practice would use proper embedding similarity
        
        # Find regions with high energy
        energy_map = field ** 2
        
        # Find top-k positions with highest energy using JAX-compatible operations
        flat_energies = energy_map.ravel()
        
        # Use lax.top_k which is JAX-compatible
        top_k_values, top_k_indices = jax.lax.top_k(flat_energies, k)
        
        # Convert back to 2D coordinates
        positions_y = top_k_indices // energy_map.shape[1]
        positions_x = top_k_indices % energy_map.shape[1]
        positions = jnp.stack([positions_y, positions_x], axis=1)
        
        # Extract values at those positions (use the top_k_values directly)
        values = top_k_values
        
        return values, positions
    
    def _compute_energy(self, field: jnp.ndarray) -> float:
        """Compute total energy of the field."""
        return jnp.sum(field ** 2) / 2.0
    
    def _apply_forgetting(self, field: jnp.ndarray, temperature: float, 
                         dt: float, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply thermodynamic forgetting with noise."""
        # Exponential decay
        decay_factor = jnp.exp(-dt / (10.0 * temperature))
        
        # Add thermal noise
        noise = random.normal(rng_key, field.shape) * temperature
        
        # Apply forgetting
        forgotten_field = field * decay_factor + noise * dt
        
        return forgotten_field
    
    def _apply_importance_forgetting(self, field: jnp.ndarray, importance_mask: jnp.ndarray,
                                   temperature: float, dt: float, 
                                   rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply importance-weighted forgetting - important memories resist decay."""
        # Importance reduces forgetting rate (high importance = slower decay)
        resistance_factor = 1.0 + self.config.importance_amplification * importance_mask
        effective_decay_rate = 1.0 / (10.0 * temperature * resistance_factor)
        
        # Variable decay based on importance
        decay_factor = jnp.exp(-dt * effective_decay_rate)
        
        # Reduced noise for important memories
        noise_amplitude = temperature / jnp.sqrt(1.0 + importance_mask)
        noise = random.normal(rng_key, field.shape) * noise_amplitude
        
        # Apply importance-weighted forgetting
        forgotten_field = field * decay_factor + noise * dt
        
        return forgotten_field
    
    def _update_importance_from_access(self, importance_mask: jnp.ndarray, 
                                     access_counts: jnp.ndarray, 
                                     access_positions: jnp.ndarray) -> jnp.ndarray:
        """Update importance based on memory access patterns."""
        # Increase importance for recently accessed memories
        access_boost = 0.1  # How much importance increases per access
        
        # Simple approach: use access_counts directly to boost importance
        # This avoids the for loop issue in JAX
        access_normalized = access_counts / (jnp.max(access_counts) + 1e-8)
        access_map = access_normalized * access_boost
        
        # Update importance with decay and access boosts
        updated_importance = (importance_mask * (1.0 - self.config.importance_decay_rate) + 
                            access_map)
        
        # Clamp to minimum importance
        updated_importance = jnp.maximum(updated_importance, self.config.min_importance)
        
        return updated_importance
    
    def _consolidate_memories(self, field: jnp.ndarray, importance_mask: jnp.ndarray, 
                            access_counts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Consolidate frequently accessed memories by increasing their strength."""
        # Find highly accessed regions
        high_access_threshold = jnp.percentile(access_counts, 90)
        high_access_mask = access_counts > high_access_threshold
        
        # Strengthen field values in highly accessed regions
        consolidation_strength = 1.1  # 10% boost
        consolidated_field = jnp.where(high_access_mask, 
                                     field * consolidation_strength, 
                                     field)
        
        # Increase importance for consolidated memories
        consolidated_importance = jnp.where(high_access_mask,
                                          importance_mask * consolidation_strength,
                                          importance_mask)
        
        return consolidated_field, consolidated_importance
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance field by one timestep with importance-weighted dynamics."""
        if dt is None:
            dt = self.config.dt
        
        # Evolve field with importance weighting
        self.field = self.evolve_with_importance(self.field, self.importance_mask, dt)
        
        # Apply importance-weighted forgetting
        self.rng_key, subkey = random.split(self.rng_key)
        self.field = self.apply_importance_forgetting(
            self.field, self.importance_mask, self.config.temperature, dt, subkey
        )
        
        # Update memory ages
        self.memory_ages += dt
        
        # Decay importance over time (natural forgetting of importance)
        self.importance_mask = self.importance_mask * (1.0 - self.config.importance_decay_rate * dt)
        self.importance_mask = jnp.maximum(self.importance_mask, self.config.min_importance)
        
        # Periodic memory consolidation (every 10 steps)
        if int(self.time / dt) % 10 == 0:
            self.field, self.importance_mask = self.consolidate_memories(
                self.field, self.importance_mask, self.access_counts
            )
        
        # Update time
        self.time += dt
        
        # Return enhanced metrics
        energy = self.compute_energy(self.field)
        max_amplitude = jnp.max(jnp.abs(self.field))
        avg_importance = jnp.mean(self.importance_mask)
        
        return {
            'time': self.time,
            'energy': float(energy),
            'max_amplitude': float(max_amplitude),
            'field_norm': float(jnp.linalg.norm(self.field)),
            'avg_importance': float(avg_importance),
            'max_importance': float(jnp.max(self.importance_mask)),
            'memory_retention': float(jnp.sum(self.field != 0) / self.field.size)
        }
    
    def inject_memory(self, memory_embedding: jnp.ndarray, 
                     position: Optional[Tuple[int, int]] = None,
                     importance: float = 1.0) -> None:
        """Inject a memory into the field with importance weighting."""
        if position is None:
            # Auto-select position based on field energy
            energy_map = self.field ** 2
            # Find region with low energy for new memory
            min_energy_idx = jnp.argmin(energy_map)
            position = jnp.unravel_index(min_energy_idx, energy_map.shape)
            position = (int(position[0]), int(position[1]))
        
        # Inject memory with strength based on importance
        self.field = self.inject(self.field, memory_embedding, position, importance)
        
        # Set importance mask at injection location
        x, y = position
        sigma = 10.0  # Same as injection envelope
        importance_envelope = self._create_gaussian_envelope(self.field.shape, x, y, sigma)
        self.importance_mask = jnp.maximum(
            self.importance_mask, 
            importance_envelope * importance
        )
    
    def query_memories(self, query_embedding: jnp.ndarray, 
                      k: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Query the field for relevant memories and track access patterns."""
        values, positions = self.sample(self.field, query_embedding, k)
        
        # Simple access tracking: increment access counts at exact positions
        # This avoids JAX compilation issues with for loops
        if len(positions) > 0:
            # Take first position for simplicity (avoids indexing issues)
            x, y = int(positions[0, 0]), int(positions[0, 1])
            # Clamp to field bounds
            x = jnp.clip(x, 0, self.field.shape[0] - 1)
            y = jnp.clip(y, 0, self.field.shape[1] - 1)
            self.access_counts = self.access_counts.at[x, y].add(0.1)
        
        # Update importance based on recent access
        self.importance_mask = self.update_importance_from_access(
            self.importance_mask, self.access_counts, positions
        )
        
        return values, positions
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get current field state for monitoring/debugging."""
        energy = self.compute_energy(self.field)
        return {
            'field': np.array(self.field),  # Convert to numpy for serialization
            'importance_mask': np.array(self.importance_mask),
            'access_counts': np.array(self.access_counts),
            'memory_ages': np.array(self.memory_ages),
            'energy': float(energy),
            'time': self.time,
            'max_value': float(jnp.max(self.field)),
            'min_value': float(jnp.min(self.field)),
            'mean_value': float(jnp.mean(self.field)),
            'std_value': float(jnp.std(self.field)),
            'avg_importance': float(jnp.mean(self.importance_mask)),
            'max_importance': float(jnp.max(self.importance_mask)),
            'total_accesses': float(jnp.sum(self.access_counts)),
            'memory_retention': float(jnp.sum(self.field != 0) / self.field.size)
        }
    
    def reset(self) -> None:
        """Reset field to initial state."""
        self.field = jnp.zeros(self.config.shape)
        self.importance_mask = jnp.ones(self.config.shape)
        self.access_counts = jnp.zeros(self.config.shape)
        self.memory_ages = jnp.zeros(self.config.shape)
        self.time = 0.0
        self.rng_key = random.PRNGKey(42)


class BatchMemoryField:
    """Batch processing version for multiple fields (multi-agent scenarios)."""
    
    def __init__(self, batch_size: int, config: Optional[FieldConfig] = None):
        """Initialize batch of memory fields."""
        self.batch_size = batch_size
        self.config = config or FieldConfig()
        self.fields = jnp.zeros((batch_size, *self.config.shape))
        
        # Vectorized operations
        self.batch_evolve = vmap(MemoryField._evolve, in_axes=(None, 0, None))
        self.batch_energy = vmap(MemoryField._compute_energy, in_axes=(None, 0))
        
    def step_all(self, dt: Optional[float] = None) -> Dict[str, jnp.ndarray]:
        """Step all fields simultaneously."""
        if dt is None:
            dt = self.config.dt
        
        # Create dummy MemoryField instance for accessing methods
        dummy_field = MemoryField(self.config)
        
        # Evolve all fields
        self.fields = self.batch_evolve(dummy_field, self.fields, dt)
        
        # Compute metrics for all fields
        energies = self.batch_energy(dummy_field, self.fields)
        
        return {
            'energies': energies,
            'max_amplitudes': jnp.max(jnp.abs(self.fields), axis=(1, 2)),
            'field_norms': jnp.linalg.norm(self.fields.reshape(self.batch_size, -1), axis=1)
        }


# Factory function for easy instantiation
def create_memory_field(shape: Tuple[int, int] = (1000, 768),
                       diffusion_rate: float = 0.01,
                       temperature: float = 0.1) -> MemoryField:
    """Factory function to create a memory field with common parameters."""
    config = FieldConfig(
        shape=shape,
        diffusion_rate=diffusion_rate,
        temperature=temperature
    )
    return MemoryField(config)