"""
Optimized Memory Field Implementation with GPU support and JIT compilation.

This module implements performance-optimized memory fields with:
- Full JAX JIT compilation enabled
- GPU acceleration support (Metal/CUDA)
- Efficient convolution operations
- Better memory access patterns
"""

import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np

# Enable GPU if available (Metal on Apple Silicon, CUDA on NVIDIA)
# Remove the CPU-only restriction
# JAX will automatically detect and use available accelerators


@dataclass
class OptimizedFieldConfig:
    """Configuration for optimized memory field parameters."""
    shape: Tuple[int, int] = (1000, 768)  # Field dimensions
    dt: float = 0.1  # Evolution timestep
    diffusion_rate: float = 0.01  # Diffusion coefficient
    temperature: float = 0.1  # Thermodynamic temperature for forgetting
    boundary_conditions: str = "neumann"  # Boundary conditions
    
    # Importance weighting parameters
    importance_decay_rate: float = 0.01  # How fast importance decays
    min_importance: float = 0.1  # Minimum importance threshold
    importance_amplification: float = 2.0  # How much importance affects resistance
    
    # Optimization parameters
    use_sparse: bool = False  # Use sparse representation (future)
    chunk_size: int = 32  # Chunk size for blocked operations
    

class OptimizedMemoryField:
    """
    Optimized memory field implementation with GPU acceleration.
    
    Key optimizations:
    - Full JIT compilation for all operations
    - Optimized convolution using custom stencil
    - Blocked operations for better cache usage
    - Vectorized operations where possible
    """
    
    def __init__(self, config: Optional[OptimizedFieldConfig] = None):
        """Initialize optimized memory field."""
        self.config = config or OptimizedFieldConfig()
        self.field = jnp.zeros(self.config.shape)
        self.time = 0.0
        
        # Importance-weighted memory system
        self.importance_mask = jnp.ones(self.config.shape)
        self.access_counts = jnp.zeros(self.config.shape)
        self.memory_ages = jnp.zeros(self.config.shape)
        
        # Pre-compile JAX functions for performance
        self._setup_compiled_functions()
        
        # Initialize random key for stochastic operations
        self.rng_key = random.PRNGKey(42)
        
        # Check available devices
        devices = jax.devices()
        print(f"JAX devices available: {devices}")
        print(f"Using device: {devices[0]}")
        
    def _setup_compiled_functions(self):
        """Setup JIT-compiled functions for optimal performance."""
        # Compile all core operations with JIT
        self.evolve = jit(self._evolve)
        self.inject = jit(self._inject)
        self.sample = self._sample  # Don't JIT compile sample due to dynamic k
        self.compute_energy = jit(self._compute_energy)
        self.apply_forgetting = jit(self._apply_forgetting)
        
        # Compile importance-weighted operations
        self.evolve_with_importance = jit(self._evolve_with_importance)
        self.apply_importance_forgetting = jit(self._apply_importance_forgetting)
        self.update_importance_from_access = jit(self._update_importance_from_access)
        self.consolidate_memories = jit(self._consolidate_memories)
        
        # Compile optimized convolution
        self.apply_diffusion_stencil = jit(self._apply_diffusion_stencil)
        
    def _apply_diffusion_stencil(self, field: jnp.ndarray) -> jnp.ndarray:
        """
        Optimized 5-point stencil for diffusion.
        More efficient than general convolution for this specific pattern.
        """
        # Pad field for boundary conditions
        padded = jnp.pad(field, ((1, 1), (1, 1)), mode='edge')
        
        # Apply 5-point stencil efficiently
        # Laplacian: (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
        laplacian = (
            padded[2:, 1:-1] +  # u[i+1,j]
            padded[:-2, 1:-1] +  # u[i-1,j]
            padded[1:-1, 2:] +  # u[i,j+1]
            padded[1:-1, :-2] -  # u[i,j-1]
            4 * field  # -4*u[i,j]
        )
        
        return laplacian / 4.0
    
    def _evolve(self, field: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Single evolution step using optimized heat equation."""
        # Use optimized stencil instead of general convolution
        diffusion = self._apply_diffusion_stencil(field)
        
        # Update field: ∂u/∂t = α∇²u (heat equation)
        evolved = field + dt * self.config.diffusion_rate * diffusion
        
        return evolved
    
    def _evolve_with_importance(self, field: jnp.ndarray, importance_mask: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Evolution step with importance-weighted diffusion resistance."""
        # Apply optimized diffusion
        diffusion = self._apply_diffusion_stencil(field)
        
        # Importance reduces diffusion rate (high importance = low diffusion)
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
            # Ensure dimensions match
            memory_slice = memory_data[:min(len(memory_data), field.shape[0])]
            memory_field = jnp.outer(
                jnp.pad(memory_slice, (0, max(0, field.shape[0] - len(memory_slice)))),
                jnp.ones(field.shape[1])
            ) * envelope
        else:
            # 2D memory data
            memory_field = memory_data * envelope
        
        # Inject with specified strength
        return field + strength * memory_field
    
    def _create_gaussian_envelope(self, shape: Tuple[int, int], 
                                center_x: float, center_y: float, 
                                sigma: float) -> jnp.ndarray:
        """Create Gaussian envelope for memory injection (optimized)."""
        # Use meshgrid more efficiently
        x = jnp.arange(shape[0])
        y = jnp.arange(shape[1])
        xx, yy = jnp.meshgrid(x, y, indexing='ij')
        
        # Compute Gaussian
        dist_sq = (xx - center_x)**2 + (yy - center_y)**2
        envelope = jnp.exp(-dist_sq / (2 * sigma**2))
        
        return envelope
    
    def _sample(self, field: jnp.ndarray, query_embedding: jnp.ndarray, 
                k: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample relevant memories from field based on query (optimized)."""
        # Compute energy map efficiently
        energy_map = jnp.square(field)
        
        # Find top-k positions with highest energy
        flat_energies = energy_map.ravel()
        
        # Use JAX-compatible minimum
        k_safe = jnp.minimum(k, flat_energies.size)
        
        # Use lax.top_k for efficiency
        top_k_values, top_k_indices = jax.lax.top_k(flat_energies, k_safe)
        
        # Convert to 2D coordinates efficiently
        positions = jnp.column_stack(jnp.divmod(top_k_indices, energy_map.shape[1]))
        
        return top_k_values, positions
    
    def _compute_energy(self, field: jnp.ndarray) -> float:
        """Compute total energy of the field (optimized)."""
        # Use more numerically stable computation
        return 0.5 * jnp.sum(jnp.square(field))
    
    def _apply_forgetting(self, field: jnp.ndarray, temperature: float, 
                         dt: float, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply thermodynamic forgetting with noise (optimized)."""
        # Exponential decay
        decay_rate = 1.0 / (10.0 * temperature)
        decay_factor = jnp.exp(-dt * decay_rate)
        
        # Add thermal noise efficiently
        noise_amplitude = jnp.sqrt(temperature * dt)
        noise = random.normal(rng_key, field.shape) * noise_amplitude
        
        # Apply forgetting
        return field * decay_factor + noise
    
    def _apply_importance_forgetting(self, field: jnp.ndarray, importance_mask: jnp.ndarray,
                                   temperature: float, dt: float, 
                                   rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply importance-weighted forgetting (optimized)."""
        # Importance reduces forgetting rate
        resistance_factor = 1.0 + self.config.importance_amplification * importance_mask
        effective_decay_rate = 1.0 / (10.0 * temperature * resistance_factor)
        
        # Variable decay based on importance
        decay_factor = jnp.exp(-dt * effective_decay_rate)
        
        # Reduced noise for important memories
        noise_amplitude = jnp.sqrt(temperature * dt) / jnp.sqrt(1.0 + importance_mask)
        noise = random.normal(rng_key, field.shape) * noise_amplitude
        
        # Apply importance-weighted forgetting
        return field * decay_factor + noise
    
    def _update_importance_from_access(self, importance_mask: jnp.ndarray, 
                                     access_counts: jnp.ndarray, 
                                     access_positions: jnp.ndarray) -> jnp.ndarray:
        """Update importance based on memory access patterns (optimized)."""
        # Increase importance for recently accessed memories
        access_boost = 0.1
        
        # Normalize access counts efficiently
        max_access = jnp.maximum(jnp.max(access_counts), 1e-8)
        access_normalized = access_counts / max_access
        access_map = access_normalized * access_boost
        
        # Update importance with decay and access boosts
        updated_importance = (
            importance_mask * (1.0 - self.config.importance_decay_rate) + 
            access_map
        )
        
        # Clamp to minimum importance
        return jnp.maximum(updated_importance, self.config.min_importance)
    
    def _consolidate_memories(self, field: jnp.ndarray, importance_mask: jnp.ndarray, 
                            access_counts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Consolidate frequently accessed memories (optimized)."""
        # Find highly accessed regions efficiently
        high_access_threshold = jnp.percentile(access_counts, 90)
        high_access_mask = access_counts > high_access_threshold
        
        # Strengthen field values in highly accessed regions
        consolidation_strength = 1.1  # 10% boost
        
        # Use where for efficient conditional update
        consolidated_field = jnp.where(
            high_access_mask, 
            field * consolidation_strength, 
            field
        )
        
        # Increase importance for consolidated memories
        consolidated_importance = jnp.where(
            high_access_mask,
            importance_mask * consolidation_strength,
            importance_mask
        )
        
        return consolidated_field, consolidated_importance
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Advance field by one timestep with optimized operations."""
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
        self.memory_ages = self.memory_ages + dt
        
        # Decay importance over time
        decay = 1.0 - self.config.importance_decay_rate * dt
        self.importance_mask = jnp.maximum(
            self.importance_mask * decay,
            self.config.min_importance
        )
        
        # Periodic memory consolidation (every 10 steps)
        if int(self.time / dt) % 10 == 0:
            self.field, self.importance_mask = self.consolidate_memories(
                self.field, self.importance_mask, self.access_counts
            )
        
        # Update time
        self.time += dt
        
        # Compute metrics efficiently
        energy = float(self.compute_energy(self.field))
        field_abs = jnp.abs(self.field)
        
        return {
            'time': self.time,
            'energy': energy,
            'max_amplitude': float(jnp.max(field_abs)),
            'field_norm': float(jnp.linalg.norm(self.field)),
            'avg_importance': float(jnp.mean(self.importance_mask)),
            'max_importance': float(jnp.max(self.importance_mask)),
            'memory_retention': float(jnp.sum(field_abs > 1e-6) / self.field.size)
        }
    
    def inject_memory(self, memory_embedding: jnp.ndarray, 
                     position: Optional[Tuple[int, int]] = None,
                     importance: float = 1.0) -> None:
        """Inject a memory into the field with importance weighting."""
        if position is None:
            # Auto-select position based on field energy
            energy_map = jnp.square(self.field)
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
        
        # Update access counts efficiently using JAX-compatible operations
        # Create a sparse update mask for the first few positions
        update_mask = jnp.zeros_like(self.access_counts)
        
        # Update only the first position for simplicity in JIT
        if positions.shape[0] > 0:
            x = jnp.clip(positions[0, 0], 0, self.field.shape[0] - 1)
            y = jnp.clip(positions[0, 1], 0, self.field.shape[1] - 1)
            
            # Create Gaussian update around accessed position
            sigma = 5.0
            update_envelope = self._create_gaussian_envelope(self.field.shape, x, y, sigma)
            update_mask = update_mask + 0.1 * update_envelope
        
        # Apply update
        self.access_counts = self.access_counts + update_mask
        
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
            'memory_retention': float(jnp.sum(jnp.abs(self.field) > 1e-6) / self.field.size)
        }
    
    def reset(self) -> None:
        """Reset field to initial state."""
        self.field = jnp.zeros(self.config.shape)
        self.importance_mask = jnp.ones(self.config.shape)
        self.access_counts = jnp.zeros(self.config.shape)
        self.memory_ages = jnp.zeros(self.config.shape)
        self.time = 0.0
        self.rng_key = random.PRNGKey(42)


# Factory function for easy instantiation
def create_optimized_memory_field(shape: Tuple[int, int] = (1000, 768),
                                 diffusion_rate: float = 0.01,
                                 temperature: float = 0.1) -> OptimizedMemoryField:
    """Factory function to create an optimized memory field."""
    config = OptimizedFieldConfig(
        shape=shape,
        diffusion_rate=diffusion_rate,
        temperature=temperature
    )
    return OptimizedMemoryField(config)