"""
Memory Persistence for FTCS

Provides save/load functionality for memory fields, agent states, and multi-agent systems.
Supports efficient serialization of JAX arrays and full system state management.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import jax.numpy as jnp

from ..fields import MemoryField, FieldConfig
from ..agents.ftcs_agent import FTCSAgent, AgentConfig
from ..agents.coordinator import MultiAgentCoordinator


@dataclass
class PersistenceMetadata:
    """Metadata for saved states."""
    version: str = "1.0"
    timestamp: str = ""
    description: str = ""
    save_type: str = ""  # "field", "agent", "coordinator"
    field_shape: Optional[tuple] = None
    num_memories: Optional[int] = None
    field_energy: Optional[float] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MemoryPersistence:
    """
    Core persistence manager for FTCS memory system.
    
    Handles serialization and deserialization of:
    - Memory fields (JAX arrays)
    - Agent states (full configuration and memories)
    - Multi-agent coordinator states
    - Metadata and versioning
    """
    
    def __init__(self, base_dir: str = "results/saved_states"):
        """
        Initialize persistence manager.
        
        Args:
            base_dir: Base directory for saved states
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("MemoryPersistence")
        
        # Create subdirectories
        (self.base_dir / "fields").mkdir(exist_ok=True)
        (self.base_dir / "agents").mkdir(exist_ok=True)
        (self.base_dir / "coordinators").mkdir(exist_ok=True)
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
    
    def save_field(self, 
                   field: MemoryField, 
                   name: str, 
                   description: str = "") -> str:
        """
        Save a memory field to disk.
        
        Args:
            field: MemoryField to save
            name: Name for the saved field
            description: Optional description
            
        Returns:
            Path to saved file
        """
        save_path = self.base_dir / "fields" / f"{name}.npz"
        
        # Prepare metadata
        metadata = PersistenceMetadata(
            description=description,
            save_type="field",
            field_shape=field.config.shape,
            field_energy=float(jnp.sum(jnp.abs(field.field)))
        )
        
        # Convert JAX arrays to numpy for saving
        field_data = np.array(field.field)
        
        # Save field data and metadata
        np.savez_compressed(
            save_path,
            field=field_data,
            config=asdict(field.config),
            metadata=asdict(metadata),
            time=field.time,
            evolution_count=getattr(field, 'evolution_count', 0)
        )
        
        self.logger.info(f"Saved field '{name}' to {save_path}")
        return str(save_path)
    
    def load_field(self, name: str) -> MemoryField:
        """
        Load a memory field from disk.
        
        Args:
            name: Name of the saved field
            
        Returns:
            Restored MemoryField
        """
        load_path = self.base_dir / "fields" / f"{name}.npz"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Field '{name}' not found at {load_path}")
        
        # Load data
        data = np.load(load_path, allow_pickle=True)
        
        # Reconstruct config
        config_dict = data['config'].item()
        config = FieldConfig(**config_dict)
        
        # Create field
        field = MemoryField(config)
        
        # Restore state
        field.field = jnp.array(data['field'])
        field.time = float(data['time'])
        if 'evolution_count' in data:
            field.evolution_count = int(data['evolution_count'])
        
        # Load metadata for logging
        metadata = data['metadata'].item()
        self.logger.info(f"Loaded field '{name}' from {load_path}")
        self.logger.info(f"  Shape: {field.config.shape}, Energy: {metadata.get('field_energy', 'unknown')}")
        
        return field
    
    def save_agent(self, 
                   agent: FTCSAgent, 
                   name: str, 
                   description: str = "",
                   include_embeddings: bool = True) -> str:
        """
        Save a complete agent state to disk.
        
        Args:
            agent: FTCSAgent to save
            name: Name for the saved agent
            description: Optional description
            include_embeddings: Whether to save embedding manager state
            
        Returns:
            Path to saved directory
        """
        save_dir = self.base_dir / "agents" / name
        save_dir.mkdir(exist_ok=True)
        
        # Save memory field
        field_path = self.save_field(agent.memory_field, f"{name}_field", "Agent memory field")
        
        # Prepare agent metadata
        metadata = PersistenceMetadata(
            description=description,
            save_type="agent",
            field_shape=agent.memory_field.config.shape,
            num_memories=len(agent.memory_entries),
            field_energy=float(jnp.sum(jnp.abs(agent.memory_field.field)))
        )
        
        # Prepare agent state
        agent_state = {
            "agent_id": agent.agent_id,
            "config": asdict(agent.config),
            "memory_entries": {
                mem_id: {
                    "content": mem.content,
                    "embedding": mem.embedding.tolist(),  # Convert to list for JSON
                    "timestamp": mem.timestamp,
                    "importance": mem.importance,
                    "memory_type": mem.memory_type,
                    "context": mem.context
                } for mem_id, mem in agent.memory_entries.items()
            },
            "memory_positions": agent.memory_positions,
            "conversation_context": agent.conversation_context,
            "field_path": str(field_path),
            "statistics": agent.get_statistics(),
            "metadata": asdict(metadata)
        }
        
        # Save agent state
        state_path = save_dir / "agent_state.json"
        with open(state_path, 'w') as f:
            json.dump(agent_state, f, indent=2, default=str)
        
        # Save embedding manager if requested
        if include_embeddings and agent.embedding_manager:
            embedding_path = save_dir / "embedding_manager.pkl"
            with open(embedding_path, 'wb') as f:
                pickle.dump(agent.embedding_manager, f)
            self.logger.info(f"  Saved embedding manager to {embedding_path}")
        
        self.logger.info(f"Saved agent '{name}' to {save_dir}")
        self.logger.info(f"  Memories: {len(agent.memory_entries)}, Field energy: {metadata.field_energy:.2f}")
        
        return str(save_dir)
    
    def load_agent(self, name: str, restore_embeddings: bool = True) -> FTCSAgent:
        """
        Load a complete agent state from disk.
        
        Args:
            name: Name of the saved agent
            restore_embeddings: Whether to restore embedding manager
            
        Returns:
            Restored FTCSAgent
        """
        load_dir = self.base_dir / "agents" / name
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Agent '{name}' not found at {load_dir}")
        
        # Load agent state
        state_path = load_dir / "agent_state.json"
        with open(state_path, 'r') as f:
            agent_state = json.load(f)
        
        # Reconstruct config
        config = AgentConfig(**agent_state['config'])
        
        # Create agent
        agent = FTCSAgent(
            agent_id=agent_state['agent_id'],
            config=config
        )
        
        # Load memory field
        field_name = f"{name}_field"
        agent.memory_field = self.load_field(field_name)
        
        # Restore memory entries
        from ..agents.ftcs_agent import MemoryEntry
        agent.memory_entries = {}
        for mem_id, mem_dict in agent_state.get('memory_entries', {}).items():
            # Reconstruct memory entry
            memory = MemoryEntry(
                content=mem_dict['content'],
                embedding=jnp.array(mem_dict['embedding']),
                timestamp=mem_dict['timestamp'],
                importance=mem_dict['importance'],
                memory_type=mem_dict.get('memory_type', 'episodic'),
                context=mem_dict.get('context')
            )
            agent.memory_entries[mem_id] = memory
        
        # Restore memory positions
        agent.memory_positions = agent_state.get('memory_positions', {})
        
        # Restore conversation context
        agent.conversation_context = agent_state.get('conversation_context', [])
        
        # Restore embedding manager if requested
        if restore_embeddings:
            embedding_path = load_dir / "embedding_manager.pkl"
            if embedding_path.exists():
                with open(embedding_path, 'rb') as f:
                    agent.embedding_manager = pickle.load(f)
                self.logger.info(f"  Restored embedding manager from {embedding_path}")
        
        metadata = agent_state.get('metadata', {})
        self.logger.info(f"Loaded agent '{name}' from {load_dir}")
        self.logger.info(f"  Memories: {len(agent.memory_entries)}, Field energy: {metadata.get('field_energy', 'unknown')}")
        
        return agent
    
    def save_coordinator(self, 
                        coordinator: MultiAgentCoordinator, 
                        name: str, 
                        description: str = "") -> str:
        """
        Save a multi-agent coordinator state to disk.
        
        Args:
            coordinator: MultiAgentCoordinator to save
            name: Name for the saved coordinator
            description: Optional description
            
        Returns:
            Path to saved directory
        """
        save_dir = self.base_dir / "coordinators" / name
        save_dir.mkdir(exist_ok=True)
        
        # Save all agents
        agent_paths = {}
        for agent_id, agent in coordinator.agents.items():
            agent_path = self.save_agent(agent, f"{name}_{agent_id}", f"Agent from coordinator {name}")
            agent_paths[agent_id] = agent_path
        
        # Prepare coordinator metadata
        metadata = PersistenceMetadata(
            description=description,
            save_type="coordinator",
            num_memories=sum(len(agent.memory_entries) for agent in coordinator.agents.values())
        )
        
        # Prepare coordinator state
        coordinator_state = {
            "coordinator_id": coordinator.coordinator_id,
            "max_agents": coordinator.max_agents,
            "enable_async_updates": coordinator.enable_async_updates,
            "agent_paths": agent_paths,
            "agent_positions": coordinator.agent_positions,
            "agent_groups": {
                group_id: {
                    "group_id": group.group_id,
                    "agent_ids": [agent.agent_id for agent in group.agents],
                    "coupling_config": asdict(group.coupling_config),
                    "shared_topics": list(group.shared_topics)
                }
                for group_id, group in coordinator.agent_groups.items()
            },
            "global_coupling_config": asdict(coordinator.global_coupling_config),
            "statistics": coordinator.get_statistics(),
            "metadata": asdict(metadata)
        }
        
        # Save coordinator state
        state_path = save_dir / "coordinator_state.json"
        with open(state_path, 'w') as f:
            json.dump(coordinator_state, f, indent=2, default=str)
        
        self.logger.info(f"Saved coordinator '{name}' to {save_dir}")
        self.logger.info(f"  Agents: {len(coordinator.agents)}, Groups: {len(coordinator.agent_groups)}")
        
        return str(save_dir)
    
    def load_coordinator(self, name: str) -> MultiAgentCoordinator:
        """
        Load a multi-agent coordinator state from disk.
        
        Args:
            name: Name of the saved coordinator
            
        Returns:
            Restored MultiAgentCoordinator
        """
        load_dir = self.base_dir / "coordinators" / name
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Coordinator '{name}' not found at {load_dir}")
        
        # Load coordinator state
        state_path = load_dir / "coordinator_state.json"
        with open(state_path, 'r') as f:
            coordinator_state = json.load(f)
        
        # Create coordinator
        coordinator = MultiAgentCoordinator(
            coordinator_id=coordinator_state['coordinator_id'],
            max_agents=coordinator_state['max_agents'],
            enable_async_updates=coordinator_state['enable_async_updates']
        )
        
        # Load all agents
        for agent_id, agent_path in coordinator_state['agent_paths'].items():
            agent_name = f"{name}_{agent_id}"
            agent = self.load_agent(agent_name)
            
            # Add to coordinator with original position
            position = tuple(coordinator_state['agent_positions'][agent_id])
            coordinator.add_agent(agent, position=position)
        
        # Restore groups
        from ..agents.multi_agent_coordinator import AgentGroup, CouplingConfig
        for group_id, group_data in coordinator_state['agent_groups'].items():
            # Get agent objects
            agents = [coordinator.agents[aid] for aid in group_data['agent_ids']]
            
            # Restore coupling config
            coupling_config = CouplingConfig(**group_data['coupling_config'])
            
            # Create group
            coordinator.create_group(
                group_id=group_id,
                agents=agents,
                coupling_config=coupling_config,
                shared_topics=set(group_data['shared_topics'])
            )
        
        # Restore global coupling config
        from ..agents.multi_agent_coordinator import CouplingConfig
        coordinator.global_coupling_config = CouplingConfig(**coordinator_state['global_coupling_config'])
        
        metadata = coordinator_state.get('metadata', {})
        self.logger.info(f"Loaded coordinator '{name}' from {load_dir}")
        self.logger.info(f"  Agents: {len(coordinator.agents)}, Groups: {len(coordinator.agent_groups)}")
        
        return coordinator
    
    def list_saved_states(self) -> Dict[str, List[str]]:
        """
        List all saved states.
        
        Returns:
            Dictionary mapping state types to lists of saved names
        """
        saved_states = {
            "fields": [],
            "agents": [],
            "coordinators": []
        }
        
        # List fields
        fields_dir = self.base_dir / "fields"
        if fields_dir.exists():
            saved_states["fields"] = [
                f.stem for f in fields_dir.glob("*.npz")
            ]
        
        # List agents
        agents_dir = self.base_dir / "agents"
        if agents_dir.exists():
            saved_states["agents"] = [
                d.name for d in agents_dir.iterdir() if d.is_dir()
            ]
        
        # List coordinators
        coordinators_dir = self.base_dir / "coordinators"
        if coordinators_dir.exists():
            saved_states["coordinators"] = [
                d.name for d in coordinators_dir.iterdir() if d.is_dir()
            ]
        
        return saved_states
    
    def create_checkpoint(self, 
                         coordinator: MultiAgentCoordinator, 
                         checkpoint_name: str,
                         description: str = "") -> str:
        """
        Create a timestamped checkpoint of the entire system.
        
        Args:
            coordinator: MultiAgentCoordinator to checkpoint
            checkpoint_name: Base name for the checkpoint
            description: Optional description
            
        Returns:
            Path to checkpoint directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{checkpoint_name}_{timestamp}"
        
        checkpoint_path = self.save_coordinator(
            coordinator, 
            checkpoint_name, 
            f"Checkpoint: {description}"
        )
        
        # Also save to checkpoints directory
        checkpoint_link = self.base_dir / "checkpoints" / f"{checkpoint_name}.txt"
        with open(checkpoint_link, 'w') as f:
            f.write(f"Checkpoint path: {checkpoint_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Description: {description}\n")
        
        self.logger.info(f"Created checkpoint '{checkpoint_name}'")
        return checkpoint_path
    
    def cleanup_old_saves(self, keep_count: int = 10):
        """
        Clean up old saved states, keeping only the most recent.
        
        Args:
            keep_count: Number of saves to keep for each type
        """
        for save_type in ["fields", "agents", "coordinators"]:
            save_dir = self.base_dir / save_type
            if not save_dir.exists():
                continue
            
            # Get all saves sorted by modification time
            if save_type == "fields":
                saves = list(save_dir.glob("*.npz"))
            else:
                saves = [d for d in save_dir.iterdir() if d.is_dir()]
            
            saves.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old saves
            to_remove = saves[keep_count:]
            for save_path in to_remove:
                if save_path.is_dir():
                    import shutil
                    shutil.rmtree(save_path)
                else:
                    save_path.unlink()
                
                self.logger.info(f"Cleaned up old save: {save_path.name}")