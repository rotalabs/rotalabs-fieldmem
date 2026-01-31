"""FTCS Agent with field-theoretic memory capabilities.

This module implements an AI agent that uses the Field-Theoretic Context System
for memory storage, retrieval, and natural forgetting dynamics.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

import jax.numpy as jnp
from jax import random
import numpy as np

from ..fields import MemoryField, FieldConfig
from ..memory.importance import SemanticImportanceAnalyzer, QuickImportanceScorer

# Import embedding manager
try:
    from ..utils.embeddings import get_embedding_manager
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class MemoryEntry:
    """Represents a single memory entry with metadata."""
    content: str
    embedding: jnp.ndarray
    timestamp: float
    importance: float
    memory_type: str = "episodic"  # episodic, semantic, procedural
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Configuration for FTCS Agent behavior with coherence optimizations."""
    memory_field_shape: Tuple[int, int] = (128, 128)
    diffusion_rate: float = 0.003      # Reduced from 0.005 for better coherence
    temperature: float = 0.05          # Reduced from 0.08 to reduce noise
    max_memories_per_query: int = 5
    memory_evolution_interval: float = 120.0  # Increased from 60s to reduce evolution frequency
    importance_decay_rate: float = 0.05       # Reduced from 0.1
    embedding_dim: int = 64
    use_proper_embeddings: bool = True  # Use sentence-transformers
    embedding_model: str = "all-MiniLM-L6-v2"
    # Coherence optimization parameters
    semantic_similarity_weight: float = 0.7    # Increased from 0.4
    field_strength_weight: float = 0.1         # Decreased from 0.3
    importance_weight: float = 0.1             # Decreased from 0.2
    recency_weight: float = 0.1                # Same
    use_semantic_clustering: bool = True       # New: cluster related memories


class FTCSAgent:
    """
    AI Agent with Field-Theoretic Context System memory.
    
    Provides natural memory storage, retrieval, and forgetting through
    continuous field dynamics rather than traditional key-value storage.
    """
    
    def __init__(self, 
                 agent_id: str,
                 config: Optional[AgentConfig] = None):
        """Initialize FTCS Agent with memory field."""
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        
        # Initialize memory field
        field_config = FieldConfig(
            shape=self.config.memory_field_shape,
            diffusion_rate=self.config.diffusion_rate,
            temperature=self.config.temperature
        )
        self.memory_field = MemoryField(field_config)
        
        # Memory tracking
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.memory_positions: Dict[str, Tuple[int, int]] = {}
        self.last_evolution_time = time.time()
        
        # Initialize importance analyzers
        self.importance_analyzer = SemanticImportanceAnalyzer()
        self.quick_scorer = QuickImportanceScorer()
        
        # Initialize random key for embeddings
        self.rng_key = random.PRNGKey(hash(agent_id) % 2**32)
        
        # Agent state
        self.conversation_context: List[str] = []
        self.active_memories: List[str] = []
        
        # Semantic clustering for coherence
        self.semantic_clusters: Dict[str, List[str]] = {}  # cluster_id -> memory_ids
        
        # Initialize embedding manager if available
        if self.config.use_proper_embeddings and EMBEDDINGS_AVAILABLE:
            self.embedding_manager = get_embedding_manager(
                model_name=self.config.embedding_model,
                embedding_dim=self.config.embedding_dim
            )
            # Update embedding dimension based on model
            self.config.embedding_dim = self.embedding_manager.embedding_dim
        else:
            self.embedding_manager = None
        
    def _generate_embedding(self, text: str) -> jnp.ndarray:
        """Generate embedding for text."""
        if self.embedding_manager is not None:
            # Use proper embeddings
            embedding = self.embedding_manager.encode(text, normalize=True)
            return jnp.array(embedding)
        else:
            # Fallback to hash-based pseudo-embeddings
            text_hash = hashlib.md5(text.encode()).hexdigest()
            hash_int = int(text_hash, 16)
            
            # Create deterministic but varied embedding
            self.rng_key = random.PRNGKey(hash_int % 2**32)
            embedding = random.normal(self.rng_key, (self.config.embedding_dim,))
            
            # Add text-specific features
            text_features = jnp.array([
                len(text) / 100.0,  # Length feature
                text.count(' ') / 50.0,  # Word count feature
                text.count('?') + text.count('!'),  # Emotional markers
                1.0 if any(word in text.lower() for word in ['important', 'remember', 'crucial']) else 0.0
            ])
            
            # Combine random embedding with text features
            return jnp.concatenate([embedding, text_features])
    
    def _find_semantic_cluster(self, embedding: jnp.ndarray, threshold: float = 0.7) -> Optional[str]:
        """Find existing semantic cluster for an embedding."""
        if not self.config.use_semantic_clustering:
            return None
            
        best_cluster = None
        best_similarity = threshold
        
        for cluster_id, memory_ids in self.semantic_clusters.items():
            if not memory_ids:
                continue
                
            # Get representative embedding (centroid)
            cluster_embeddings = []
            for memory_id in memory_ids:
                if memory_id in self.memory_entries:
                    cluster_embeddings.append(self.memory_entries[memory_id].embedding)
            
            if cluster_embeddings:
                centroid = jnp.mean(jnp.stack(cluster_embeddings), axis=0)
                similarity = float(jnp.dot(embedding, centroid) / 
                                 (jnp.linalg.norm(embedding) * jnp.linalg.norm(centroid)))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id
        
        return best_cluster
    
    def _find_position_for_memory(self, embedding: jnp.ndarray, cluster_id: Optional[str] = None) -> Tuple[int, int]:
        """Find optimal position for memory with semantic clustering."""
        if cluster_id and cluster_id in self.semantic_clusters:
            # Place near existing cluster members
            cluster_positions = []
            for memory_id in self.semantic_clusters[cluster_id]:
                if memory_id in self.memory_positions:
                    cluster_positions.append(self.memory_positions[memory_id])
            
            if cluster_positions:
                # Find centroid of cluster
                centroid_x = int(np.mean([pos[0] for pos in cluster_positions]))
                centroid_y = int(np.mean([pos[1] for pos in cluster_positions]))
                
                # Look for nearby low-energy position
                for radius in [5, 10, 20]:
                    for dx in range(-radius, radius + 1):
                        for dy in range(-radius, radius + 1):
                            x = max(0, min(self.memory_field.field.shape[0] - 1, centroid_x + dx))
                            y = max(0, min(self.memory_field.field.shape[1] - 1, centroid_y + dy))
                            
                            if abs(self.memory_field.field[x, y]) < 0.1:  # Low energy
                                return (x, y)
        
        # Fallback: find global low-energy position
        energy_map = self.memory_field.field ** 2
        min_energy_idx = jnp.argmin(energy_map)
        position = jnp.unravel_index(min_energy_idx, energy_map.shape)
        return (int(position[0]), int(position[1]))
    
    def _compute_importance(self, content: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Compute memory importance score using semantic analysis."""
        # Use full semantic analysis for more accurate importance
        try:
            scores = self.importance_analyzer.analyze(content, context)
            base_importance = scores.total
        except Exception as e:
            # Fallback to quick scorer if analysis fails
            base_importance = self.quick_scorer.compute_importance(content)
        
        # Apply context-based boosts (if provided)
        if context:
            # User emphasis signals
            if context.get('user_emphasis', False):
                base_importance += 0.2
            
            # Correction importance (mistakes to remember)
            if context.get('correction', False):
                base_importance += 0.3
            
            # Repeated mentions indicate importance
            if context.get('repeated_mention', 0) > 1:
                base_importance += 0.1 * min(context.get('repeated_mention', 0), 3)
            
            # Task-specific importance
            if context.get('task_critical', False):
                base_importance += 0.25
        
        return min(base_importance, 1.0)
    
    def _evolve_memory_field(self, force: bool = False) -> None:
        """Evolve memory field if enough time has passed."""
        current_time = time.time()
        time_since_evolution = current_time - self.last_evolution_time
        
        if force or time_since_evolution >= self.config.memory_evolution_interval:
            # Number of evolution steps based on elapsed time
            num_steps = max(1, int(time_since_evolution / 10.0))
            
            for _ in range(num_steps):
                self.memory_field.step()
            
            self.last_evolution_time = current_time
    
    def store_memory(self, 
                    content: str,
                    memory_type: str = "episodic",
                    importance: Optional[float] = None,
                    context: Optional[Dict[str, Any]] = None,
                    position: Optional[Tuple[int, int]] = None) -> str:
        """
        Store a memory in the field-theoretic memory system.
        
        Args:
            content: Text content of the memory
            memory_type: Type of memory (episodic, semantic, procedural)
            importance: Optional importance override
            context: Additional context for importance computation
            position: Optional specific position in field
            
        Returns:
            Memory ID for later reference
        """
        # Generate memory ID
        memory_id = hashlib.md5(f"{content}_{time.time()}_{self.agent_id}".encode()).hexdigest()[:16]
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Compute importance
        if importance is None:
            importance = self._compute_importance(content, context)
        
        # Find semantic cluster
        cluster_id = self._find_semantic_cluster(embedding)
        if cluster_id is None and self.config.use_semantic_clustering:
            # Create new cluster
            cluster_id = f"cluster_{len(self.semantic_clusters)}"
            self.semantic_clusters[cluster_id] = []
            
        # Log importance breakdown for transparency (in debug mode)
        if hasattr(self, 'logger') and self.logger.isEnabledFor(10):  # DEBUG level
            try:
                scores = self.importance_analyzer.analyze(content, context)
                self.logger.debug(f"Importance breakdown for '{content[:50]}...':")
                self.logger.debug(f"  Entities: {scores.entities:.2f}, Causal: {scores.causality:.2f}")
                self.logger.debug(f"  Temporal: {scores.temporal:.2f}, Instructional: {scores.instructional:.2f}")
                self.logger.debug(f"  Total importance: {importance:.2f}")
            except:
                pass
        
        # Create memory entry
        memory_entry = MemoryEntry(
            content=content,
            embedding=embedding,
            timestamp=time.time(),
            importance=importance,
            memory_type=memory_type,
            context=context
        )
        
        # Store in field and get actual position
        if position is None:
            # Use clustering-aware position finding
            position = self._find_position_for_memory(embedding, cluster_id)
        
        self.memory_field.inject_memory(
            embedding, 
            position=position,
            importance=importance
        )
        
        # Track memory with position
        self.memory_entries[memory_id] = memory_entry
        self.memory_positions[memory_id] = position
        
        # Add to cluster
        if cluster_id:
            self.semantic_clusters[cluster_id].append(memory_id)
        
        # Add to conversation context if episodic
        if memory_type == "episodic":
            self.conversation_context.append(content)
            # Keep context manageable
            if len(self.conversation_context) > 10:
                self.conversation_context = self.conversation_context[-10:]
        
        return memory_id
    
    def retrieve_memories(self, 
                         query: str,
                         memory_type: Optional[str] = None,
                         max_memories: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Query text for memory retrieval
            memory_type: Optional filter by memory type
            max_memories: Maximum number of memories to return
            
        Returns:
            List of relevant memories with metadata
        """
        if max_memories is None:
            max_memories = self.config.max_memories_per_query
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Evolve field before querying
        self._evolve_memory_field()
        
        # Compute similarity scores for all stored memories
        memory_scores = []
        
        for memory_id, memory_entry in self.memory_entries.items():
            # Filter by type if specified
            if memory_type and memory_entry.memory_type != memory_type:
                continue
            
            # Compute embedding similarity
            memory_embedding = memory_entry.embedding
            
            if self.embedding_manager is not None:
                # Use embedding manager's optimized similarity calculation
                similarity = self.embedding_manager.cosine_similarity(
                    query_embedding,
                    memory_embedding
                )
            else:
                # Fallback to manual calculation
                # Handle dimension mismatch if needed
                min_dim = min(len(query_embedding), len(memory_embedding))
                query_vec = query_embedding[:min_dim]
                memory_vec = memory_embedding[:min_dim]
                
                # Cosine similarity
                query_norm = jnp.linalg.norm(query_vec)
                memory_norm = jnp.linalg.norm(memory_vec)
                
                if query_norm > 0 and memory_norm > 0:
                    similarity = jnp.dot(query_vec, memory_vec) / (query_norm * memory_norm)
                else:
                    similarity = 0.0
            
            # Get current field strength at memory position
            if memory_id in self.memory_positions:
                pos = self.memory_positions[memory_id]
                field_strength = float(jnp.abs(self.memory_field.field[pos[0], pos[1]]))
            else:
                field_strength = 0.0
            
            # Combined score: similarity + field strength + importance + recency
            age_hours = (time.time() - memory_entry.timestamp) / 3600.0
            recency_score = max(0, 1.0 - age_hours / 24.0)  # Decay over 24 hours
            
            # ENHANCED COHERENCE FIX: Use configurable weights for better coherence
            combined_score = (
                float(similarity) * self.config.semantic_similarity_weight +      # 0.7 (was 0.4)
                field_strength * self.config.field_strength_weight +             # 0.1 (was 0.3)
                memory_entry.importance * self.config.importance_weight +        # 0.1 (was 0.2)
                recency_score * self.config.recency_weight                       # 0.1 (same)
            )
            
            memory_scores.append({
                'memory_id': memory_id,
                'score': combined_score,
                'similarity': float(similarity),
                'field_strength': field_strength,
                'memory_entry': memory_entry
            })
        
        # Sort by combined score
        memory_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Build retrieved memories list
        retrieved_memories = []
        for i, scored_memory in enumerate(memory_scores[:max_memories]):
            memory_entry = scored_memory['memory_entry']
            memory_id = scored_memory['memory_id']
            
            retrieved_memories.append({
                'memory_id': memory_id,
                'content': memory_entry.content,
                'importance': memory_entry.importance,
                'memory_type': memory_entry.memory_type,
                'timestamp': memory_entry.timestamp,
                'context': memory_entry.context,
                'similarity': scored_memory['similarity'],
                'field_strength': scored_memory['field_strength'],
                'combined_score': scored_memory['score'],
                'position': self.memory_positions.get(memory_id, (0, 0)),
                'age_hours': (time.time() - memory_entry.timestamp) / 3600.0
            })
        
        return retrieved_memories[:max_memories]
    
    def process_conversation_turn(self, 
                                 user_input: str,
                                 agent_response: str,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a conversation turn, storing both input and response.
        
        Args:
            user_input: User's input text
            agent_response: Agent's response text  
            context: Additional context for the conversation
            
        Returns:
            Processing results and retrieved memories
        """
        # Store user input as memory
        user_memory_id = self.store_memory(
            f"User: {user_input}",
            memory_type="episodic",
            context=context
        )
        
        # Store agent response
        agent_memory_id = self.store_memory(
            f"Agent: {agent_response}",
            memory_type="episodic", 
            importance=0.3  # Lower importance for agent's own responses
        )
        
        # Retrieve relevant memories for this conversation
        relevant_memories = self.retrieve_memories(user_input)
        
        # Update active memories
        self.active_memories = [m['memory_id'] for m in relevant_memories]
        
        return {
            'user_memory_id': user_memory_id,
            'agent_memory_id': agent_memory_id,
            'relevant_memories': relevant_memories,
            'field_energy': self.memory_field.compute_energy(),
            'total_memories': len(self.memory_entries)
        }
    
    def get_memory_field_state(self) -> Dict[str, Any]:
        """Get current memory field state for analysis."""
        return {
            'field_state': self.memory_field.get_field_state(),
            'total_memories': len(self.memory_entries),
            'conversation_length': len(self.conversation_context),
            'active_memories': len(self.active_memories),
            'last_evolution': self.last_evolution_time,
            'agent_id': self.agent_id
        }
    
    def forget_old_memories(self, age_threshold_hours: float = 24.0) -> List[str]:
        """
        Remove very old or weak memories from tracking.
        
        Args:
            age_threshold_hours: Age threshold for memory removal
            
        Returns:
            List of forgotten memory IDs
        """
        current_time = time.time()
        forgotten_memories = []
        
        for memory_id, memory_entry in list(self.memory_entries.items()):
            age_hours = (current_time - memory_entry.timestamp) / 3600.0
            
            # Forget if too old and low importance
            if age_hours > age_threshold_hours and memory_entry.importance < 0.3:
                forgotten_memories.append(memory_id)
                del self.memory_entries[memory_id]
                if memory_id in self.memory_positions:
                    del self.memory_positions[memory_id]
        
        return forgotten_memories
    
    def reset_memory(self) -> None:
        """Reset all memory state."""
        self.memory_field.reset()
        self.memory_entries.clear()
        self.memory_positions.clear()
        self.semantic_clusters.clear()
        self.conversation_context.clear()
        self.active_memories.clear()
        self.last_evolution_time = time.time()
    
    def clear_memory(self):
        """Clear all memories and reset the memory field."""
        self.reset_memory()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_energy = float(jnp.sum(self.memory_field.field))
        
        return {
            "agent_id": self.agent_id,
            "total_memories": len(self.memory_entries),
            "field_energy": total_energy,
            "conversation_length": len(self.conversation_context),
            "active_memories": len(self.active_memories),
            "field_shape": self.config.memory_field_shape,
            "last_evolution": self.last_evolution_time,
            "total_retrievals": 0,  # Would need to track this
            "total_stored": len(self.memory_entries)
        }
    
    def process_query(self, query: str) -> str:
        """
        Process a query and generate a response.
        
        This method provides compatibility with the context retention tester
        and other evaluation frameworks that expect query processing capability.
        
        Args:
            query: The input query to process
            
        Returns:
            Generated response based on retrieved memories and context
        """
        # Retrieve relevant memories
        relevant_memories = self.retrieve_memories(query, max_memories=5)
        
        # Build context from memories
        memory_context = []
        for memory in relevant_memories:
            memory_context.append(f"Memory: {memory['content']}")
        
        # Create a simple response based on context
        if memory_context:
            context_str = "\n".join(memory_context[:3])  # Use top 3 memories
            response = f"Based on my memories:\n{context_str}\n\nRegarding '{query}': I have processed this information with field-based memory retrieval."
        else:
            response = f"I understand your query '{query}' but don't have specific relevant memories to draw from."
        
        # Store this interaction as a memory
        self.store_memory(
            content=f"Query: {query} | Response: {response}",
            importance=0.7,
            memory_type="episodic"
        )
        
        return response
    
    def respond(self, query: str) -> str:
        """Alias for process_query for compatibility."""
        return self.process_query(query)
    
    def generate_response(self, query: str) -> str:
        """Alias for process_query for compatibility."""
        return self.process_query(query)
    
    def get_recent_memories(self, time_window: float = 60.0, 
                           importance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get recent high-importance memories.
        
        Args:
            time_window: Time window in seconds
            importance_threshold: Minimum importance threshold
            
        Returns:
            List of recent memory dictionaries
        """
        current_time = time.time()
        recent_memories = []
        
        for memory_id, memory_entry in self.memory_entries.items():
            # Check time window
            if current_time - memory_entry.timestamp > time_window:
                continue
                
            # Check importance threshold
            if memory_entry.importance < importance_threshold:
                continue
                
            recent_memories.append({
                'memory_id': memory_id,
                'content': memory_entry.content,
                'embedding': memory_entry.embedding,
                'importance': memory_entry.importance,
                'timestamp': memory_entry.timestamp,
                'context': memory_entry.context
            })
            
        return recent_memories
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get detailed memory statistics including topic analysis."""
        stats = self.get_statistics()
        
        # Analyze memory topics (simplified)
        topics = {}
        for memory_entry in self.memory_entries.values():
            # Simple keyword-based topic detection
            content_lower = memory_entry.content.lower()
            for topic in ['technical', 'creative', 'analytical', 'conversational']:
                if topic in content_lower:
                    topics[topic] = topics.get(topic, 0) + 1
                    
        # Sort topics by frequency
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        dominant_topics = [topic for topic, _ in sorted_topics[:3]]
        
        stats['dominant_topics'] = dominant_topics
        stats['topic_distribution'] = topics
        
        return stats
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return len(self.memory_entries)
    
    def query(self, query: str, context: Optional[Dict] = None) -> str:
        """Process query with optional context (for multi-agent compatibility)."""
        # Incorporate collective memories from context if available
        if context and 'collective_memories' in context:
            collective_info = "\n\nCollective knowledge:\n"
            for mem in context['collective_memories']:
                collective_info += f"- {mem['content']}\n"
            
            # Enhance query with collective context
            enhanced_query = query + collective_info
            return self.process_query(enhanced_query)
        
        return self.process_query(query)
    
    def process_conversation_turn(self, 
                                 user_input: str, 
                                 agent_response: str,
                                 importance: float = 1.0) -> Dict[str, Any]:
        """
        Process a conversation turn and update memory.
        
        Args:
            user_input: User's message
            agent_response: Agent's response
            importance: Importance of this turn
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Store user input
        user_memory_id = self.store_memory(
            f"User: {user_input}",
            importance=importance,
            memory_type="episodic"
        )
        
        # Store agent response
        agent_memory_id = self.store_memory(
            f"Agent: {agent_response}",
            importance=importance,
            memory_type="episodic"
        )
        
        processing_time = time.time() - start_time
        
        return {
            "user_memory_id": user_memory_id,
            "agent_memory_id": agent_memory_id,
            "processing_time_ms": processing_time * 1000,
            "total_memories": len(self.memory_entries)
        }