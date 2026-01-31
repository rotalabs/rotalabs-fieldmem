"""Field coupling mechanisms for multi-agent collective intelligence."""

from typing import Dict, List, Tuple, Optional, Set
import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from ..fields.memory_field import MemoryField


class CouplingTopology(Enum):
    """Types of agent coupling topologies."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    SMALL_WORLD = "small_world"
    FULLY_CONNECTED = "fully_connected"
    ADAPTIVE = "adaptive"


@dataclass
class CouplingConfig:
    """Configuration for field coupling."""
    topology: CouplingTopology = CouplingTopology.FULLY_CONNECTED
    coupling_strength: float = 0.1
    sync_interval: float = 1.0  # seconds
    importance_threshold: float = 0.5
    max_coupling_distance: int = 2  # for nearest neighbor
    small_world_probability: float = 0.1  # for small world
    adaptive_threshold: float = 0.7  # similarity threshold for adaptive


class FieldCoupler:
    """Manages field coupling between multiple agents."""
    
    def __init__(self, config: CouplingConfig = None):
        self.config = config or CouplingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Agent registry
        self.agents: Dict[str, MemoryField] = {}
        self.agent_positions: Dict[str, Tuple[int, int]] = {}  # For spatial topologies
        self.coupling_matrix: Optional[jnp.ndarray] = None
        
        # Coupling functions
        self._couple_fields = jit(self._couple_fields_impl)
        self._compute_coupling_term = jit(self._compute_coupling_term_impl)
        
    def register_agent(self, agent_id: str, field: MemoryField, 
                      position: Optional[Tuple[int, int]] = None):
        """Register an agent for field coupling."""
        self.agents[agent_id] = field
        if position:
            self.agent_positions[agent_id] = position
        
        # Rebuild coupling matrix
        self._build_coupling_matrix()
        
    def unregister_agent(self, agent_id: str):
        """Remove an agent from coupling."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.agent_positions:
                del self.agent_positions[agent_id]
            self._build_coupling_matrix()
            
    def _build_coupling_matrix(self):
        """Build coupling strength matrix based on topology."""
        n_agents = len(self.agents)
        if n_agents == 0:
            self.coupling_matrix = None
            return
            
        agent_ids = list(self.agents.keys())
        matrix = np.zeros((n_agents, n_agents))
        
        if self.config.topology == CouplingTopology.FULLY_CONNECTED:
            # All agents coupled with equal strength
            matrix = np.ones((n_agents, n_agents)) * self.config.coupling_strength
            np.fill_diagonal(matrix, 0)  # No self-coupling
            
        elif self.config.topology == CouplingTopology.NEAREST_NEIGHBOR:
            # Couple based on spatial distance
            for i, id1 in enumerate(agent_ids):
                for j, id2 in enumerate(agent_ids):
                    if i != j and id1 in self.agent_positions and id2 in self.agent_positions:
                        pos1 = self.agent_positions[id1]
                        pos2 = self.agent_positions[id2]
                        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                        if distance <= self.config.max_coupling_distance:
                            matrix[i, j] = self.config.coupling_strength / (1 + distance)
                            
        elif self.config.topology == CouplingTopology.SMALL_WORLD:
            # Start with nearest neighbor, add random long-range connections
            self._build_coupling_matrix()  # Start with nearest neighbor
            matrix = self.coupling_matrix.copy()
            
            # Add random long-range connections
            rng = np.random.default_rng(42)
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    if matrix[i, j] == 0 and rng.random() < self.config.small_world_probability:
                        strength = self.config.coupling_strength * 0.5  # Weaker long-range
                        matrix[i, j] = strength
                        matrix[j, i] = strength
                        
        elif self.config.topology == CouplingTopology.ADAPTIVE:
            # Coupling strength based on field similarity
            # Start with weak uniform coupling
            matrix = np.ones((n_agents, n_agents)) * self.config.coupling_strength * 0.1
            np.fill_diagonal(matrix, 0)
            
        self.coupling_matrix = jnp.array(matrix)
        
    def couple_fields(self, dt: float = 0.1) -> Dict[str, jnp.ndarray]:
        """Perform one step of field coupling between all agents."""
        if len(self.agents) < 2 or self.coupling_matrix is None:
            return {}
            
        agent_ids = list(self.agents.keys())
        fields = [self.agents[aid].field for aid in agent_ids]
        
        # Stack all fields
        stacked_fields = jnp.stack(fields)
        
        # Compute coupling updates
        coupled_fields = self._couple_fields(
            stacked_fields, 
            self.coupling_matrix,
            dt
        )
        
        # Update agent fields
        updates = {}
        for i, agent_id in enumerate(agent_ids):
            self.agents[agent_id].field = coupled_fields[i]
            updates[agent_id] = coupled_fields[i]
            
        # Update adaptive topology if needed
        if self.config.topology == CouplingTopology.ADAPTIVE:
            self._update_adaptive_coupling(stacked_fields)
            
        return updates
        
    def _couple_fields_impl(self, fields: jnp.ndarray, coupling_matrix: jnp.ndarray, 
                           dt: float) -> jnp.ndarray:
        """JAX implementation of field coupling."""
        n_agents = fields.shape[0]
        
        # Compute coupling terms for each agent
        coupling_terms = vmap(
            lambda i: self._compute_coupling_term_impl(
                fields[i], fields, coupling_matrix[i]
            )
        )(jnp.arange(n_agents))
        
        # Apply coupling
        return fields + dt * coupling_terms
        
    def _compute_coupling_term_impl(self, field_i: jnp.ndarray, all_fields: jnp.ndarray,
                                   coupling_strengths: jnp.ndarray) -> jnp.ndarray:
        """Compute coupling term for a single agent."""
        # Reshape coupling strengths for broadcasting
        strengths = coupling_strengths[:, None, None]
        
        # Compute differences weighted by coupling strength
        differences = all_fields - field_i[None, :, :]
        weighted_diff = differences * strengths
        
        # Sum over all agents
        return jnp.sum(weighted_diff, axis=0)
        
    def _update_adaptive_coupling(self, fields: jnp.ndarray):
        """Update coupling matrix based on field similarity."""
        n_agents = fields.shape[0]
        new_matrix = np.zeros((n_agents, n_agents))
        
        # Compute pairwise similarities
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Use normalized dot product as similarity
                field_i = fields[i].flatten()
                field_j = fields[j].flatten()
                
                norm_i = jnp.linalg.norm(field_i)
                norm_j = jnp.linalg.norm(field_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarity = jnp.dot(field_i, field_j) / (norm_i * norm_j)
                    
                    if similarity > self.config.adaptive_threshold:
                        strength = self.config.coupling_strength * float(similarity)
                        new_matrix[i, j] = strength
                        new_matrix[j, i] = strength
                        
        # Smooth transition: blend old and new matrices
        alpha = 0.1  # adaptation rate
        self.coupling_matrix = (1 - alpha) * self.coupling_matrix + alpha * jnp.array(new_matrix)
        
    def get_collective_field(self) -> Optional[jnp.ndarray]:
        """Compute the collective field (average of all agent fields)."""
        if not self.agents:
            return None
            
        fields = [agent.field for agent in self.agents.values()]
        return jnp.mean(jnp.stack(fields), axis=0)
        
    def get_coupling_statistics(self) -> Dict:
        """Get statistics about current coupling state."""
        if not self.agents or self.coupling_matrix is None:
            return {}
            
        collective_field = self.get_collective_field()
        
        # Compute field divergence (how different agents are)
        fields = jnp.stack([agent.field for agent in self.agents.values()])
        divergences = jnp.std(fields, axis=0).mean()
        
        # Coupling strength statistics
        non_zero_coupling = self.coupling_matrix[self.coupling_matrix > 0]
        
        return {
            "num_agents": len(self.agents),
            "topology": self.config.topology.value,
            "avg_coupling_strength": float(non_zero_coupling.mean()) if len(non_zero_coupling) > 0 else 0,
            "max_coupling_strength": float(non_zero_coupling.max()) if len(non_zero_coupling) > 0 else 0,
            "field_divergence": float(divergences),
            "collective_energy": float(jnp.sum(collective_field)),
            "connectivity": float(len(non_zero_coupling)) / (len(self.agents) * (len(self.agents) - 1))
        }
        
    def broadcast_memory(self, source_agent: str, memory: jnp.ndarray, 
                        importance: float) -> Set[str]:
        """Broadcast a high-importance memory from one agent to others."""
        if source_agent not in self.agents:
            return set()
            
        if importance < self.config.importance_threshold:
            return set()
            
        # Get coupling strengths for source agent
        source_idx = list(self.agents.keys()).index(source_agent)
        coupling_strengths = self.coupling_matrix[source_idx]
        
        # Inject memory into coupled agents
        injected_agents = set()
        for i, (agent_id, agent_field) in enumerate(self.agents.items()):
            if i != source_idx and coupling_strengths[i] > 0:
                # Scale injection by coupling strength
                injection_strength = coupling_strengths[i] * importance
                
                # Inject memory into agent's field
                # This is a simplified injection - in practice, would use MemoryField.inject_memory
                agent_field.field = agent_field.field.at[:].add(
                    memory * injection_strength
                )
                injected_agents.add(agent_id)
                
        return injected_agents