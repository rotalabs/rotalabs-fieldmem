"""Multi-agent system with collective intelligence."""

from typing import Dict, List, Optional, Any, Callable
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, Future

from .ftcs_agent import FTCSAgent, AgentConfig
from .coupling import FieldCoupler, CouplingConfig, CouplingTopology
from .collective import CollectiveMemoryPool
from ..fields.memory_field import FieldConfig


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system."""
    num_agents: int = 4
    coupling_config: CouplingConfig = None
    field_config: FieldConfig = None
    collective_pool_size: int = 10000
    sync_interval: float = 1.0  # seconds
    enable_specialization: bool = True
    consensus_threshold: float = 0.6
    max_workers: int = 4  # for parallel processing


class MultiAgentSystem:
    """Orchestrates multiple FTCS agents with collective intelligence."""
    
    def __init__(self, config: MultiAgentConfig = None):
        self.config = config or MultiAgentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.agents: Dict[str, FTCSAgent] = {}
        self.field_coupler = FieldCoupler(self.config.coupling_config)
        self.collective_pool = CollectiveMemoryPool(
            max_memories=self.config.collective_pool_size,
            consensus_threshold=self.config.consensus_threshold
        )
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.sync_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Agent specializations
        self.agent_specializations: Dict[str, str] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "collective_hits": 0,
            "sync_cycles": 0,
            "avg_response_time": 0
        }
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Create and register initial agents."""
        
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            
            # Create agent with unique characteristics
            agent_config = AgentConfig(
                memory_field_shape=(100, 100),  # Smaller field for multi-agent
                diffusion_rate=0.005 * (1 + 0.1 * np.random.randn()),
                temperature=0.08 * (1 + 0.1 * np.random.randn()),
                importance_decay_rate=0.1,
                use_proper_embeddings=False,  # Faster for demos
                embedding_dim=100  # Match field dimension to avoid mismatch
            )
            
            agent = FTCSAgent(
                agent_id=agent_id,
                config=agent_config
            )
            
            self.add_agent(agent_id, agent)
            
    def add_agent(self, agent_id: str, agent: FTCSAgent, 
                  position: Optional[tuple] = None,
                  specialization: Optional[str] = None):
        """Add an agent to the system."""
        with self.lock:
            self.agents[agent_id] = agent
            
            # Register with field coupler
            self.field_coupler.register_agent(agent_id, agent.memory_field, position)
            
            # Set specialization if provided
            if specialization:
                self.agent_specializations[agent_id] = specialization
                
            self.logger.info(f"Added agent {agent_id} to system")
            
    def remove_agent(self, agent_id: str):
        """Remove an agent from the system."""
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                self.field_coupler.unregister_agent(agent_id)
                
                if agent_id in self.agent_specializations:
                    del self.agent_specializations[agent_id]
                    
                self.logger.info(f"Removed agent {agent_id} from system")
                
    def start(self):
        """Start the multi-agent system."""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            
            self.logger.info("Multi-agent system started")
            
    def stop(self):
        """Stop the multi-agent system."""
        with self.lock:
            self.running = False
            
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
            
        self.executor.shutdown(wait=True)
        self.logger.info("Multi-agent system stopped")
        
    def _sync_loop(self):
        """Background synchronization loop."""
        while self.running:
            try:
                # Perform field coupling
                self.field_coupler.couple_fields(dt=0.1)
                
                # Share high-importance memories
                self._share_important_memories()
                
                # Update specializations if enabled
                if self.config.enable_specialization:
                    self._update_specializations()
                    
                self.performance_stats["sync_cycles"] += 1
                
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                
            time.sleep(self.config.sync_interval)
            
    def _share_important_memories(self):
        """Share high-importance memories to collective pool."""
        with self.lock:
            for agent_id, agent in self.agents.items():
                # Get recent high-importance memories
                recent_memories = agent.get_recent_memories(
                    time_window=self.config.sync_interval * 2,
                    importance_threshold=0.7
                )
                
                # Propose to collective pool
                for memory in recent_memories:
                    self.collective_pool.propose_memory(
                        agent_id=agent_id,
                        content=memory['content'],
                        embedding=memory['embedding'],
                        importance=memory['importance']
                    )
                    
    def _update_specializations(self):
        """Update agent specializations based on their memory patterns."""
        # Analyze each agent's memory distribution
        specialization_scores = {}
        
        with self.lock:
            for agent_id, agent in self.agents.items():
                # Get memory statistics
                stats = agent.get_memory_statistics()
                
                # Simple specialization detection based on memory clusters
                if 'dominant_topics' in stats:
                    specialization_scores[agent_id] = stats['dominant_topics']
                    
        # Update specializations
        # This is simplified - in practice would use more sophisticated clustering
        for agent_id, topics in specialization_scores.items():
            if topics:
                self.agent_specializations[agent_id] = topics[0]  # Most dominant topic
                
    def collective_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a query using collective intelligence."""
        start_time = time.time()
        
        with self.lock:
            self.performance_stats["total_queries"] += 1
            
            # Get query embedding (simplified - would use real embeddings)
            query_embedding = self._get_embedding(query)
            
            # First, check collective pool
            collective_memories = self.collective_pool.retrieve_memories(
                query_embedding, 
                k=5,
                consensus_only=True
            )
            
            if collective_memories:
                self.performance_stats["collective_hits"] += 1
                
            # Query all agents in parallel
            futures = {}
            for agent_id, agent in self.agents.items():
                # Route to specialized agents if applicable
                if agent_id in self.agent_specializations:
                    specialization = self.agent_specializations[agent_id]
                    # Simple keyword matching - would be more sophisticated
                    if specialization.lower() not in query.lower():
                        continue
                        
                future = self.executor.submit(
                    self._query_agent,
                    agent, 
                    query,
                    context,
                    collective_memories
                )
                futures[agent_id] = future
                
            # Collect responses
            responses = {}
            for agent_id, future in futures.items():
                try:
                    responses[agent_id] = future.result(timeout=5)
                except Exception as e:
                    self.logger.error(f"Error querying agent {agent_id}: {e}")
                    
            # Build consensus response
            consensus_response = self._build_consensus(responses, collective_memories)
            
            # Update performance stats
            response_time = time.time() - start_time
            self.performance_stats["avg_response_time"] = (
                (self.performance_stats["avg_response_time"] * 
                 (self.performance_stats["total_queries"] - 1) + response_time) /
                self.performance_stats["total_queries"]
            )
            
            return {
                "response": consensus_response,
                "agent_responses": responses,
                "collective_memories": [(m_id, m.content) for m_id, m in collective_memories],
                "response_time": response_time,
                "participating_agents": list(responses.keys())
            }
            
    def _query_agent(self, agent: FTCSAgent, query: str, context: Optional[Dict],
                    collective_memories: List[tuple]) -> Dict[str, Any]:
        """Query a single agent."""
        # Inject collective memories into agent's context
        enriched_context = context or {}
        enriched_context['collective_memories'] = [
            {"content": m.content, "importance": m.importance}
            for _, m in collective_memories
        ]
        
        # Process query
        try:
            if hasattr(agent, 'process_query'):
                response = agent.process_query(query)
            else:
                # Fallback for basic agent
                response = agent.query(query, enriched_context)
                
            return {
                "response": response,
                "confidence": 0.8,  # Would be computed based on memory quality
                "sources": len(collective_memories)
            }
        except Exception as e:
            self.logger.error(f"Error in agent query: {e}")
            return {
                "response": "",
                "confidence": 0.0,
                "error": str(e)
            }
            
    def _build_consensus(self, agent_responses: Dict[str, Dict],
                        collective_memories: List[tuple]) -> str:
        """Build consensus response from multiple agents."""
        if not agent_responses:
            return "No agents available to process query."
            
        # Filter valid responses
        valid_responses = [
            (agent_id, resp) for agent_id, resp in agent_responses.items()
            if resp.get("confidence", 0) > 0.5 and "error" not in resp
        ]
        
        if not valid_responses:
            return "Unable to generate confident response."
            
        # Simple consensus: combine responses with confidence weighting
        # In practice, would use more sophisticated consensus building
        if len(valid_responses) == 1:
            return valid_responses[0][1]["response"]
            
        # Combine multiple responses
        response_parts = []
        total_confidence = 0
        
        for agent_id, resp in valid_responses:
            confidence = resp.get("confidence", 0.5)
            response_parts.append(f"[{agent_id}]: {resp['response']}")
            total_confidence += confidence
            
        # Build combined response
        consensus = "Based on collective intelligence:\n\n"
        consensus += "\n\n".join(response_parts)
        
        if collective_memories:
            consensus += f"\n\nSupported by {len(collective_memories)} collective memories."
            
        return consensus
        
    def _get_embedding(self, text: str) -> jnp.ndarray:
        """Get embedding for text (simplified)."""
        # In practice, would use real embedding model
        # For now, return random embedding
        return jnp.array(np.random.randn(768))
        
    def demonstrate_collective_problem_solving(self, problem: str) -> Dict[str, Any]:
        """Demonstrate collective problem-solving capabilities."""
        self.logger.info(f"Collective problem solving: {problem}")
        
        # Phase 1: Individual exploration
        exploration_results = {}
        with self.lock:
            for agent_id, agent in self.agents.items():
                # Each agent explores the problem independently
                agent.store_memory(
                    f"Exploring problem: {problem}",
                    {"problem": problem, "phase": "exploration"}
                )
                exploration_results[agent_id] = f"Agent {agent_id} exploring: {problem}"
                
        # Let fields evolve and couple
        time.sleep(self.config.sync_interval * 2)
        
        # Phase 2: Share insights through coupling
        self._share_important_memories()
        
        # Phase 3: Collective synthesis
        collective_response = self.collective_query(
            f"Based on our collective exploration, how should we solve: {problem}",
            {"phase": "synthesis"}
        )
        
        # Analyze emergent behavior
        coupling_stats = self.field_coupler.get_coupling_statistics()
        pool_stats = self.collective_pool.get_statistics()
        
        return {
            "problem": problem,
            "exploration_results": exploration_results,
            "collective_solution": collective_response,
            "emergent_metrics": {
                "field_convergence": 1.0 - coupling_stats.get("field_divergence", 1.0),
                "consensus_rate": pool_stats.get("consensus_rate", 0),
                "collective_memories": pool_stats.get("total_memories", 0),
                "agent_collaboration": pool_stats.get("avg_contributors", 1)
            }
        }
        
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self.lock:
            coupling_stats = self.field_coupler.get_coupling_statistics()
            pool_stats = self.collective_pool.get_statistics()
            
            # Agent statistics
            agent_stats = {}
            for agent_id, agent in self.agents.items():
                contribution = self.collective_pool.get_agent_contribution(agent_id)
                agent_stats[agent_id] = {
                    "specialization": self.agent_specializations.get(agent_id, "generalist"),
                    "contribution": contribution,
                    "memory_count": agent.get_memory_count()
                }
                
            return {
                "system": {
                    "num_agents": len(self.agents),
                    "topology": coupling_stats.get("topology", "unknown"),
                    "sync_cycles": self.performance_stats["sync_cycles"],
                    "running": self.running
                },
                "performance": self.performance_stats.copy(),
                "coupling": coupling_stats,
                "collective_pool": pool_stats,
                "agents": agent_stats
            }