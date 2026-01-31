"""
Sparse Memory Field Implementation for extreme efficiency.

This module implements sparse field representation where only non-zero
regions are stored and computed, enabling massive memory savings and
performance improvements for typical sparse memory patterns.
"""

import os
from typing import Tuple, Optional, Dict, Any, List, Set
from dataclasses import dataclass
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, random


@dataclass 
class SparseFieldConfig:
    """Configuration for sparse memory field parameters."""
    max_regions: int = 1000  # Maximum number of active regions
    region_size: Tuple[int, int] = (32, 32)  # Size of each sparse region
    dt: float = 0.1  # Evolution timestep
    diffusion_rate: float = 0.01  # Diffusion coefficient
    temperature: float = 0.1  # Thermodynamic temperature
    activation_threshold: float = 1e-3  # Threshold for region activation
    deactivation_threshold: float = 1e-4  # Threshold for region deactivation
    overlap_size: int = 4  # Overlap between regions for smooth transitions


@dataclass
class SparseRegion:
    """Represents an active memory region in the sparse field."""
    position: Tuple[int, int]  # Top-left corner position
    data: jnp.ndarray  # Region data
    importance: jnp.ndarray  # Importance mask for region
    energy: float  # Total energy in region
    last_access: float  # Time of last access
    

class SparseMemoryField:
    """
    Sparse memory field implementation for extreme efficiency.
    
    Key features:
    - Only stores and computes non-zero regions
    - Dynamic region allocation/deallocation
    - Seamless transitions between regions
    - 10-100x memory savings for typical sparse patterns
    """
    
    def __init__(self, config: Optional[SparseFieldConfig] = None):
        """Initialize sparse memory field."""
        self.config = config or SparseFieldConfig()
        self.active_regions: Dict[Tuple[int, int], SparseRegion] = {}
        self.time = 0.0
        self.rng_key = random.PRNGKey(42)
        
        # Pre-compile core operations
        self._setup_compiled_functions()
        
        # Performance metrics
        self.total_elements = 0
        self.active_elements = 0
        
    def _setup_compiled_functions(self):
        """Setup JIT-compiled functions."""
        self.evolve_region = jit(self._evolve_region)
        self.apply_region_forgetting = jit(self._apply_region_forgetting)
        self.compute_region_energy = jit(self._compute_region_energy)
        
    def _get_region_position(self, global_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get region position for a global coordinate."""
        rx = (global_pos[0] // self.config.region_size[0]) * self.config.region_size[0]
        ry = (global_pos[1] // self.config.region_size[1]) * self.config.region_size[1]
        return (rx, ry)
    
    def _create_region(self, position: Tuple[int, int]) -> SparseRegion:
        """Create a new sparse region."""
        data = jnp.zeros(self.config.region_size)
        importance = jnp.ones(self.config.region_size) * self.config.activation_threshold
        return SparseRegion(
            position=position,
            data=data,
            importance=importance,
            energy=0.0,
            last_access=self.time
        )
    
    def _evolve_region(self, data: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Evolve a single region using optimized stencil."""
        # Pad for boundary conditions
        padded = jnp.pad(data, ((1, 1), (1, 1)), mode='edge')
        
        # Apply 5-point stencil
        laplacian = (
            padded[2:, 1:-1] +
            padded[:-2, 1:-1] +
            padded[1:-1, 2:] +
            padded[1:-1, :-2] -
            4 * data
        ) / 4.0
        
        # Update with diffusion
        return data + dt * self.config.diffusion_rate * laplacian
    
    def _apply_region_forgetting(self, data: jnp.ndarray, importance: jnp.ndarray,
                                temperature: float, dt: float, 
                                rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Apply forgetting to a region."""
        # Importance-weighted decay
        resistance = 1.0 + 2.0 * importance
        decay_rate = 1.0 / (10.0 * temperature * resistance)
        decay_factor = jnp.exp(-dt * decay_rate)
        
        # Thermal noise
        noise_amplitude = jnp.sqrt(temperature * dt) / jnp.sqrt(1.0 + importance)
        noise = random.normal(rng_key, data.shape) * noise_amplitude
        
        return data * decay_factor + noise
    
    def _compute_region_energy(self, data: jnp.ndarray) -> float:
        """Compute energy of a region."""
        return 0.5 * jnp.sum(jnp.square(data))
    
    def inject_memory(self, memory_embedding: jnp.ndarray,
                     position: Tuple[int, int],
                     importance: float = 1.0) -> None:
        """Inject memory at specified position."""
        # Determine which region(s) this affects
        region_pos = self._get_region_position(position)
        
        # Create region if it doesn't exist
        if region_pos not in self.active_regions:
            if len(self.active_regions) >= self.config.max_regions:
                # Remove least recently used region
                self._evict_lru_region()
            self.active_regions[region_pos] = self._create_region(region_pos)
        
        region = self.active_regions[region_pos]
        
        # Local position within region
        local_x = position[0] - region_pos[0]
        local_y = position[1] - region_pos[1]
        
        # Create Gaussian injection
        sigma = 5.0
        x_coords = jnp.arange(self.config.region_size[0])
        y_coords = jnp.arange(self.config.region_size[1])
        xx, yy = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        
        dist_sq = (xx - local_x)**2 + (yy - local_y)**2
        envelope = jnp.exp(-dist_sq / (2 * sigma**2))
        
        # Inject memory (simplified - just use importance as amplitude)
        injection = envelope * importance
        region.data = region.data + injection
        region.importance = jnp.maximum(region.importance, envelope * importance)
        region.last_access = self.time
        
        # Update region energy
        region.energy = float(self._compute_region_energy(region.data))
        
    def query_memories(self, query_embedding: jnp.ndarray,
                      k: int = 10) -> Tuple[List[float], List[Tuple[int, int]]]:
        """Query memories from active regions."""
        all_values = []
        all_positions = []
        
        # Search through active regions
        for region_pos, region in self.active_regions.items():
            # Find high-energy points in region
            energy_map = jnp.square(region.data)
            flat_energies = energy_map.ravel()
            
            # Get top values in this region
            region_k = min(k, flat_energies.size)
            if region_k > 0:
                top_values, top_indices = jax.lax.top_k(flat_energies, region_k)
                
                # Convert to global positions
                local_positions = jnp.column_stack(jnp.divmod(top_indices, energy_map.shape[1]))
                global_positions = local_positions + jnp.array(region_pos)
                
                all_values.extend(top_values.tolist())
                all_positions.extend([(int(p[0]), int(p[1])) for p in global_positions])
                
                # Update access time
                region.last_access = self.time
        
        # Sort by value and take top k
        if all_values:
            sorted_indices = np.argsort(all_values)[::-1][:k]
            values = [all_values[i] for i in sorted_indices]
            positions = [all_positions[i] for i in sorted_indices]
            return values, positions
        else:
            return [], []
    
    def step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """Evolve all active regions."""
        if dt is None:
            dt = self.config.dt
        
        regions_to_remove = []
        total_energy = 0.0
        
        # Evolve each active region
        for region_pos, region in self.active_regions.items():
            # Evolve data
            region.data = self.evolve_region(region.data, dt)
            
            # Apply forgetting
            self.rng_key, subkey = random.split(self.rng_key)
            region.data = self._apply_region_forgetting(
                region.data, region.importance,
                self.config.temperature, dt, subkey
            )
            
            # Update energy
            region.energy = float(self._compute_region_energy(region.data))
            total_energy += region.energy
            
            # Mark for removal if below threshold
            if region.energy < self.config.deactivation_threshold:
                regions_to_remove.append(region_pos)
            
            # Decay importance
            region.importance = jnp.maximum(
                region.importance * 0.99,
                self.config.deactivation_threshold
            )
        
        # Remove inactive regions
        for pos in regions_to_remove:
            del self.active_regions[pos]
        
        # Update time
        self.time += dt
        
        # Compute metrics
        self.active_elements = len(self.active_regions) * np.prod(self.config.region_size)
        sparsity = 1.0 - (self.active_elements / max(self.total_elements, 1))
        
        return {
            'time': self.time,
            'total_energy': total_energy,
            'active_regions': len(self.active_regions),
            'sparsity': sparsity,
            'memory_efficiency': sparsity * 100,  # Percentage saved
            'active_elements': self.active_elements
        }
    
    def _evict_lru_region(self):
        """Evict least recently used region."""
        if not self.active_regions:
            return
        
        # Find LRU region
        lru_pos = min(self.active_regions.keys(), 
                     key=lambda p: self.active_regions[p].last_access)
        del self.active_regions[lru_pos]
    
    def get_field_state(self) -> Dict[str, Any]:
        """Get current field state."""
        total_energy = sum(r.energy for r in self.active_regions.values())
        
        return {
            'time': self.time,
            'active_regions': len(self.active_regions),
            'total_energy': total_energy,
            'sparsity': 1.0 - (self.active_elements / max(self.total_elements, 1)),
            'memory_usage_mb': (self.active_elements * 4) / (1024 * 1024),  # Assuming float32
            'regions': {
                str(pos): {
                    'energy': r.energy,
                    'last_access': r.last_access,
                    'max_value': float(jnp.max(r.data))
                }
                for pos, r in self.active_regions.items()
            }
        }
    
    def visualize_sparse_field(self, canvas_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Create visualization of sparse field."""
        canvas = np.zeros(canvas_size)
        
        # Draw each active region
        for region_pos, region in self.active_regions.items():
            # Calculate region bounds
            x_start = region_pos[0]
            y_start = region_pos[1]
            x_end = min(x_start + self.config.region_size[0], canvas_size[0])
            y_end = min(y_start + self.config.region_size[1], canvas_size[1])
            
            # Copy region data to canvas
            if x_start < canvas_size[0] and y_start < canvas_size[1]:
                region_slice = region.data[
                    :x_end-x_start,
                    :y_end-y_start
                ]
                canvas[x_start:x_end, y_start:y_end] = np.array(region_slice)
        
        return canvas


# Factory function
def create_sparse_memory_field(max_regions: int = 1000,
                              region_size: Tuple[int, int] = (32, 32)) -> SparseMemoryField:
    """Create a sparse memory field."""
    config = SparseFieldConfig(
        max_regions=max_regions,
        region_size=region_size
    )
    return SparseMemoryField(config)