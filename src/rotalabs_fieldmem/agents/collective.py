"""Collective memory pool for multi-agent systems."""

from typing import Dict, List, Optional, Tuple, Any
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import threading
import logging
from collections import defaultdict

from ..fields.memory_field import MemoryField


@dataclass
class CollectiveMemory:
    """A memory shared across multiple agents."""
    content: str
    embedding: jnp.ndarray
    importance: float
    contributors: List[str] = field(default_factory=list)
    consensus_score: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def add_contributor(self, agent_id: str, importance_vote: float):
        """Add an agent as contributor with their importance vote."""
        if agent_id not in self.contributors:
            self.contributors.append(agent_id)
        
        # Update consensus score (simple average for now)
        self.consensus_score = (self.consensus_score * (len(self.contributors) - 1) + importance_vote) / len(self.contributors)
        
    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CollectiveMemoryPool:
    """Manages shared memories across multiple agents."""
    
    def __init__(self, max_memories: int = 10000, consensus_threshold: float = 0.6):
        self.max_memories = max_memories
        self.consensus_threshold = consensus_threshold
        self.logger = logging.getLogger(__name__)
        
        # Memory storage
        self.memories: Dict[str, CollectiveMemory] = {}
        self.memory_index: Dict[str, List[str]] = defaultdict(list)  # agent_id -> memory_ids
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_memories": 0,
            "consensus_memories": 0,
            "avg_contributors": 0,
            "memory_conflicts": 0
        }
        
    def propose_memory(self, agent_id: str, content: str, embedding: jnp.ndarray,
                      importance: float, memory_id: Optional[str] = None) -> str:
        """Propose a memory to the collective pool."""
        with self.lock:
            # Generate memory ID if not provided
            if memory_id is None:
                memory_id = f"{agent_id}_{datetime.now().timestamp()}"
            
            # Check if similar memory exists
            similar_memory_id = self._find_similar_memory(embedding)
            
            if similar_memory_id:
                # Update existing memory
                existing = self.memories[similar_memory_id]
                existing.add_contributor(agent_id, importance)
                
                # Update importance as weighted average
                n = len(existing.contributors)
                existing.importance = ((n - 1) * existing.importance + importance) / n
                
                # Update embedding as weighted average
                existing.embedding = ((n - 1) * existing.embedding + embedding) / n
                
                # Track in agent's index
                if agent_id not in self.memory_index:
                    self.memory_index[agent_id] = []
                if similar_memory_id not in self.memory_index[agent_id]:
                    self.memory_index[agent_id].append(similar_memory_id)
                
                self._update_stats()
                return similar_memory_id
            else:
                # Create new memory
                new_memory = CollectiveMemory(
                    content=content,
                    embedding=embedding,
                    importance=importance
                )
                new_memory.add_contributor(agent_id, importance)
                
                self.memories[memory_id] = new_memory
                self.memory_index[agent_id].append(memory_id)
                
                # Evict if necessary
                if len(self.memories) > self.max_memories:
                    self._evict_memory()
                
                self._update_stats()
                return memory_id
    
    def _find_similar_memory(self, embedding: jnp.ndarray, threshold: float = 0.9) -> Optional[str]:
        """Find memory with similar embedding."""
        best_similarity = 0.0
        best_id = None
        
        for mem_id, memory in self.memories.items():
            similarity = self._compute_similarity(embedding, memory.embedding)
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_id = mem_id
                
        return best_id
    
    def _compute_similarity(self, emb1: jnp.ndarray, emb2: jnp.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = jnp.linalg.norm(emb1)
        norm2 = jnp.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(jnp.dot(emb1, emb2) / (norm1 * norm2))
    
    def retrieve_memories(self, query_embedding: jnp.ndarray, k: int = 5,
                         agent_id: Optional[str] = None,
                         consensus_only: bool = False) -> List[Tuple[str, CollectiveMemory]]:
        """Retrieve k most relevant memories."""
        with self.lock:
            # Filter memories
            candidates = []
            for mem_id, memory in self.memories.items():
                # Apply consensus filter if requested
                if consensus_only and memory.consensus_score < self.consensus_threshold:
                    continue
                    
                # Apply agent filter if requested
                if agent_id and agent_id not in memory.contributors:
                    continue
                    
                # Compute relevance score
                similarity = self._compute_similarity(query_embedding, memory.embedding)
                relevance = similarity * memory.importance * (1 + memory.consensus_score)
                
                candidates.append((mem_id, memory, relevance))
                
            # Sort by relevance and return top k
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Update access counts
            results = []
            for mem_id, memory, _ in candidates[:k]:
                memory.access()
                results.append((mem_id, memory))
                
            return results
    
    def get_agent_contribution(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics about an agent's contributions."""
        with self.lock:
            agent_memories = self.memory_index.get(agent_id, [])
            
            if not agent_memories:
                return {
                    "total_contributions": 0,
                    "avg_importance": 0,
                    "consensus_rate": 0,
                    "unique_contributions": 0
                }
            
            # Gather statistics
            importances = []
            consensus_count = 0
            unique_count = 0
            
            for mem_id in agent_memories:
                if mem_id not in self.memories:
                    continue
                    
                memory = self.memories[mem_id]
                importances.append(memory.importance)
                
                if memory.consensus_score >= self.consensus_threshold:
                    consensus_count += 1
                    
                if len(memory.contributors) == 1:
                    unique_count += 1
                    
            return {
                "total_contributions": len(agent_memories),
                "avg_importance": float(np.mean(importances)) if importances else 0,
                "consensus_rate": consensus_count / len(agent_memories) if agent_memories else 0,
                "unique_contributions": unique_count,
                "collaborative_contributions": len(agent_memories) - unique_count
            }
    
    def resolve_conflicts(self, memory_ids: List[str]) -> Optional[str]:
        """Resolve conflicts between multiple similar memories."""
        with self.lock:
            if not memory_ids:
                return None
                
            # Get all memories
            memories = []
            for mem_id in memory_ids:
                if mem_id in self.memories:
                    memories.append((mem_id, self.memories[mem_id]))
                    
            if not memories:
                return None
                
            # Find memory with highest consensus
            best_id = max(memories, key=lambda x: x[1].consensus_score)[0]
            
            # Merge other memories into the best one
            best_memory = self.memories[best_id]
            for mem_id, memory in memories:
                if mem_id != best_id:
                    # Transfer contributors
                    for contributor in memory.contributors:
                        if contributor not in best_memory.contributors:
                            best_memory.add_contributor(contributor, memory.importance)
                    
                    # Remove merged memory
                    del self.memories[mem_id]
                    
                    # Update indices
                    for agent_id in self.memory_index:
                        if mem_id in self.memory_index[agent_id]:
                            self.memory_index[agent_id].remove(mem_id)
                            if best_id not in self.memory_index[agent_id]:
                                self.memory_index[agent_id].append(best_id)
                                
            self.stats["memory_conflicts"] += len(memories) - 1
            self._update_stats()
            return best_id
    
    def _evict_memory(self):
        """Evict least valuable memory."""
        if not self.memories:
            return
            
        # Score memories by value (importance * consensus * recency)
        memory_scores = []
        now = datetime.now()
        
        for mem_id, memory in self.memories.items():
            age_factor = 1.0 / (1 + (now - memory.last_accessed).total_seconds() / 3600)  # Hours
            access_factor = 1.0 + np.log1p(memory.access_count)
            score = memory.importance * memory.consensus_score * age_factor * access_factor
            memory_scores.append((mem_id, score))
            
        # Evict lowest scoring memory
        evict_id = min(memory_scores, key=lambda x: x[1])[0]
        
        # Remove from all indices
        for agent_id in self.memory_index:
            if evict_id in self.memory_index[agent_id]:
                self.memory_index[agent_id].remove(evict_id)
                
        del self.memories[evict_id]
        
    def _update_stats(self):
        """Update pool statistics."""
        if not self.memories:
            self.stats = {
                "total_memories": 0,
                "consensus_memories": 0,
                "avg_contributors": 0,
                "memory_conflicts": self.stats.get("memory_conflicts", 0)
            }
            return
            
        consensus_count = sum(1 for m in self.memories.values() 
                             if m.consensus_score >= self.consensus_threshold)
        avg_contributors = np.mean([len(m.contributors) for m in self.memories.values()])
        
        self.stats.update({
            "total_memories": len(self.memories),
            "consensus_memories": consensus_count,
            "avg_contributors": float(avg_contributors),
            "consensus_rate": consensus_count / len(self.memories) if self.memories else 0
        })
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        with self.lock:
            return self.stats.copy()