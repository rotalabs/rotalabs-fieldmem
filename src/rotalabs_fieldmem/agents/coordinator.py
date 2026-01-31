"""
Multi-Agent Memory Coordinator

Enables multiple FTCS agents to share memories through field coupling,
creating collective intelligence and shared understanding.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from .ftcs_agent import FTCSAgent, AgentConfig
from ..fields import FieldConfig


@dataclass
class CouplingConfig:
    """Configuration for agent field coupling."""
    coupling_strength: float = 0.1  # Strength of field interaction
    coupling_radius: int = 5  # Radius of influence between fields
    update_interval: float = 5.0  # Seconds between coupling updates
    symmetric: bool = True  # Whether coupling is bidirectional
    decay_with_distance: bool = True  # Whether coupling weakens with distance


@dataclass 
class AgentGroup:
    """Group of agents that share memories."""
    group_id: str
    agents: List[FTCSAgent]
    coupling_config: CouplingConfig
    shared_topics: Set[str]  # Topics this group focuses on


class MultiAgentCoordinator:
    """
    Coordinates memory sharing between multiple FTCS agents.
    
    Enables collective intelligence through field coupling where
    memories from one agent can influence others' fields.
    """
    
    def __init__(self,
                 coordinator_id: str = "coordinator",
                 max_agents: int = 10,
                 enable_async_updates: bool = True):
        """
        Initialize multi-agent coordinator.
        
        Args:
            coordinator_id: Unique ID for this coordinator
            max_agents: Maximum number of agents to coordinate
            enable_async_updates: Whether to update fields asynchronously
        """
        self.coordinator_id = coordinator_id
        self.max_agents = max_agents
        self.enable_async_updates = enable_async_updates
        
        # Agent management
        self.agents: Dict[str, FTCSAgent] = {}
        self.agent_groups: Dict[str, AgentGroup] = {}
        self.agent_positions: Dict[str, Tuple[int, int]] = {}  # Virtual positions
        
        # Coupling configuration
        self.global_coupling_config = CouplingConfig()
        
        # Threading for async updates
        self.update_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4) if enable_async_updates else None
        self.active_updates: List[Future] = []
        
        # Statistics
        self.total_couplings = 0
        self.last_coupling_time = time.time()
        
        self.logger = logging.getLogger(f"MultiAgentCoordinator-{coordinator_id}")
    
    def add_agent(self, 
                  agent: FTCSAgent,
                  position: Optional[Tuple[int, int]] = None,
                  group_id: Optional[str] = None) -> bool:
        """
        Add an agent to the coordinator.
        
        Args:
            agent: FTCS agent to add
            position: Virtual position for spatial coupling
            group_id: Optional group to add agent to
            
        Returns:
            True if successfully added
        """
        if len(self.agents) >= self.max_agents:
            self.logger.warning(f"Maximum agents ({self.max_agents}) reached")
            return False
        
        if agent.agent_id in self.agents:
            self.logger.warning(f"Agent {agent.agent_id} already exists")
            return False
        
        with self.update_lock:
            self.agents[agent.agent_id] = agent
            
            # Assign virtual position if not provided
            if position is None:
                # Arrange agents in a grid
                grid_size = int(np.ceil(np.sqrt(self.max_agents)))
                idx = len(self.agents) - 1
                position = (idx % grid_size, idx // grid_size)
            
            self.agent_positions[agent.agent_id] = position
            
            # Add to group if specified
            if group_id:
                self._add_to_group(agent, group_id)
        
        self.logger.info(f"Added agent {agent.agent_id} at position {position}")
        return True
    
    def _add_to_group(self, agent: FTCSAgent, group_id: str):
        """Add agent to a group."""
        if group_id not in self.agent_groups:
            self.agent_groups[group_id] = AgentGroup(
                group_id=group_id,
                agents=[],
                coupling_config=CouplingConfig(),
                shared_topics=set()
            )
        
        if agent not in self.agent_groups[group_id].agents:
            self.agent_groups[group_id].agents.append(agent)
    
    def create_group(self,
                    group_id: str,
                    agents: List[FTCSAgent],
                    coupling_config: Optional[CouplingConfig] = None,
                    shared_topics: Optional[Set[str]] = None) -> bool:
        """
        Create a group of agents with shared coupling configuration.
        
        Args:
            group_id: Unique group identifier
            agents: List of agents in the group
            coupling_config: Coupling configuration for this group
            shared_topics: Topics this group focuses on
            
        Returns:
            True if group created successfully
        """
        if group_id in self.agent_groups:
            self.logger.warning(f"Group {group_id} already exists")
            return False
        
        # Ensure all agents are registered
        for agent in agents:
            if agent.agent_id not in self.agents:
                self.add_agent(agent)
        
        self.agent_groups[group_id] = AgentGroup(
            group_id=group_id,
            agents=agents,
            coupling_config=coupling_config or CouplingConfig(),
            shared_topics=shared_topics or set()
        )
        
        self.logger.info(f"Created group {group_id} with {len(agents)} agents")
        return True
    
    def couple_agent_fields(self,
                           agent1_id: str,
                           agent2_id: str,
                           coupling_strength: Optional[float] = None) -> bool:
        """
        Couple two agents' memory fields.
        
        Args:
            agent1_id: First agent ID
            agent2_id: Second agent ID  
            coupling_strength: Override default coupling strength
            
        Returns:
            True if coupling successful
        """
        if agent1_id not in self.agents or agent2_id not in self.agents:
            self.logger.error("One or both agents not found")
            return False
        
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]
        
        if coupling_strength is None:
            coupling_strength = self.global_coupling_config.coupling_strength
        
        try:
            # Perform field coupling
            self._couple_fields(agent1, agent2, coupling_strength)
            self.total_couplings += 1
            
            self.logger.debug(f"Coupled {agent1_id} <-> {agent2_id} with strength {coupling_strength}")
            return True
            
        except Exception as e:
            self.logger.error(f"Coupling failed: {e}")
            return False
    
    def _couple_fields(self, 
                      agent1: FTCSAgent,
                      agent2: FTCSAgent,
                      coupling_strength: float):
        """
        Perform actual field coupling between agents.
        
        This allows memories from one agent to influence another's field.
        """
        # Get field dimensions
        field1 = agent1.memory_field.field
        field2 = agent2.memory_field.field
        
        # Ensure compatible dimensions
        if field1.shape != field2.shape:
            # Resize to smaller dimension
            min_shape = (
                min(field1.shape[0], field2.shape[0]),
                min(field1.shape[1], field2.shape[1])
            )
            field1 = field1[:min_shape[0], :min_shape[1]]
            field2 = field2[:min_shape[0], :min_shape[1]]
        
        # Calculate field influence
        # Agent 1 influences Agent 2
        influence_1_to_2 = field1 * coupling_strength
        # Agent 2 influences Agent 1  
        influence_2_to_1 = field2 * coupling_strength
        
        # Apply influences with distance decay if configured
        if self.global_coupling_config.decay_with_distance:
            pos1 = self.agent_positions.get(agent1.agent_id, (0, 0))
            pos2 = self.agent_positions.get(agent2.agent_id, (0, 0))
            
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            decay_factor = np.exp(-distance / self.global_coupling_config.coupling_radius)
            
            influence_1_to_2 *= decay_factor
            influence_2_to_1 *= decay_factor
        
        # Update fields
        # Create new field arrays to avoid modifying during iteration
        new_field1 = agent1.memory_field.field + influence_2_to_1[:agent1.memory_field.field.shape[0], 
                                                                   :agent1.memory_field.field.shape[1]]
        new_field2 = agent2.memory_field.field + influence_1_to_2[:agent2.memory_field.field.shape[0],
                                                                   :agent2.memory_field.field.shape[1]]
        
        # Apply updates
        agent1.memory_field.field = new_field1
        agent2.memory_field.field = new_field2
    
    def broadcast_memory(self,
                        source_agent_id: str,
                        content: str,
                        importance: float = 1.0,
                        target_group: Optional[str] = None) -> Dict[str, bool]:
        """
        Broadcast a memory from one agent to others.
        
        Args:
            source_agent_id: Agent broadcasting the memory
            content: Memory content
            importance: Memory importance
            target_group: Optional group to broadcast to
            
        Returns:
            Dict of agent_id -> success status
        """
        if source_agent_id not in self.agents:
            self.logger.error(f"Source agent {source_agent_id} not found")
            return {}
        
        source_agent = self.agents[source_agent_id]
        
        # Determine target agents
        if target_group and target_group in self.agent_groups:
            target_agents = [
                a for a in self.agent_groups[target_group].agents
                if a.agent_id != source_agent_id
            ]
        else:
            target_agents = [
                a for aid, a in self.agents.items()
                if aid != source_agent_id
            ]
        
        results = {}
        
        # Store memory in source agent first
        source_memory_id = source_agent.store_memory(
            content, 
            importance=importance,
            memory_type="broadcast"
        )
        
        # Broadcast to targets with distance-based importance decay
        source_pos = self.agent_positions.get(source_agent_id, (0, 0))
        
        for target_agent in target_agents:
            try:
                target_pos = self.agent_positions.get(target_agent.agent_id, (0, 0))
                distance = np.sqrt(
                    (source_pos[0] - target_pos[0])**2 + 
                    (source_pos[1] - target_pos[1])**2
                )
                
                # Decay importance with distance
                decayed_importance = importance * np.exp(-distance / 10.0)
                
                # Store in target with metadata about source
                target_agent.store_memory(
                    f"[From {source_agent_id}] {content}",
                    importance=decayed_importance,
                    memory_type="received_broadcast"
                )
                
                results[target_agent.agent_id] = True
                
            except Exception as e:
                self.logger.error(f"Failed to broadcast to {target_agent.agent_id}: {e}")
                results[target_agent.agent_id] = False
        
        self.logger.info(
            f"Broadcast from {source_agent_id} to {len(results)} agents "
            f"({sum(results.values())} successful)"
        )
        
        return results
    
    def synchronize_group_memories(self, group_id: str) -> bool:
        """
        Synchronize memories within a group through field coupling.
        
        Args:
            group_id: Group to synchronize
            
        Returns:
            True if synchronization successful
        """
        if group_id not in self.agent_groups:
            self.logger.error(f"Group {group_id} not found")
            return False
        
        group = self.agent_groups[group_id]
        agents = group.agents
        
        if len(agents) < 2:
            return True  # Nothing to synchronize
        
        try:
            # Couple all pairs within the group
            coupling_strength = group.coupling_config.coupling_strength
            
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    self._couple_fields(agents[i], agents[j], coupling_strength)
            
            self.logger.info(f"Synchronized {len(agents)} agents in group {group_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Group synchronization failed: {e}")
            return False
    
    def run_coupling_update(self, async_update: bool = None):
        """
        Run a coupling update across all configured agent pairs.
        
        Args:
            async_update: Override async setting for this update
        """
        if async_update is None:
            async_update = self.enable_async_updates
        
        if async_update and self.executor:
            # Submit async task
            future = self.executor.submit(self._perform_coupling_update)
            self.active_updates.append(future)
            
            # Clean up completed futures
            self.active_updates = [f for f in self.active_updates if not f.done()]
        else:
            # Synchronous update
            self._perform_coupling_update()
    
    def _perform_coupling_update(self):
        """Perform the actual coupling update."""
        start_time = time.time()
        
        with self.update_lock:
            # Update all groups
            for group_id, group in self.agent_groups.items():
                self.synchronize_group_memories(group_id)
            
            # Update global couplings based on distance
            if len(self.agents) > 1:
                agents_list = list(self.agents.values())
                
                for i in range(len(agents_list)):
                    for j in range(i + 1, len(agents_list)):
                        agent1 = agents_list[i]
                        agent2 = agents_list[j]
                        
                        # Check if within coupling radius
                        pos1 = self.agent_positions.get(agent1.agent_id, (0, 0))
                        pos2 = self.agent_positions.get(agent2.agent_id, (0, 0))
                        
                        distance = np.sqrt(
                            (pos1[0] - pos2[0])**2 + 
                            (pos1[1] - pos2[1])**2
                        )
                        
                        if distance <= self.global_coupling_config.coupling_radius:
                            self._couple_fields(
                                agent1, agent2,
                                self.global_coupling_config.coupling_strength
                            )
        
        self.last_coupling_time = time.time()
        update_time = time.time() - start_time
        
        self.logger.debug(f"Coupling update completed in {update_time:.3f}s")
    
    def get_collective_memory(self, 
                            query: str,
                            max_per_agent: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories from all agents for a query.
        
        Args:
            query: Query string
            max_per_agent: Max memories per agent
            
        Returns:
            Aggregated memory results with agent attribution
        """
        all_memories = []
        
        for agent_id, agent in self.agents.items():
            try:
                memories = agent.retrieve_memories(query, max_memories=max_per_agent)
                
                # Add agent attribution
                for memory in memories:
                    memory["source_agent"] = agent_id
                    all_memories.append(memory)
                    
            except Exception as e:
                self.logger.error(f"Failed to retrieve from {agent_id}: {e}")
        
        # Sort by relevance score if available
        all_memories.sort(
            key=lambda m: m.get("combined_score", m.get("importance", 0)),
            reverse=True
        )
        
        return all_memories
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            agent_stats[agent_id] = agent.get_statistics()
        
        return {
            "coordinator_id": self.coordinator_id,
            "num_agents": len(self.agents),
            "num_groups": len(self.agent_groups),
            "total_couplings": self.total_couplings,
            "time_since_last_coupling": time.time() - self.last_coupling_time,
            "active_async_updates": len(self.active_updates) if self.executor else 0,
            "agent_statistics": agent_stats,
            "group_sizes": {
                gid: len(group.agents) 
                for gid, group in self.agent_groups.items()
            }
        }
    
    def shutdown(self):
        """Shutdown the coordinator and cleanup resources."""
        if self.executor:
            # Wait for active updates
            for future in self.active_updates:
                future.result(timeout=5.0)
            
            self.executor.shutdown(wait=True)
        
        self.logger.info("Multi-agent coordinator shutdown complete")