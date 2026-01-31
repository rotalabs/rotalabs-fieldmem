"""
Quality Metrics for FTCS vs Traditional Memory Comparison

This module provides scientific metrics to evaluate conversation quality,
context retention, and relevance improvements in field-theoretic memory systems.
"""
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import re
from collections import Counter


@dataclass
class QualityScore:
    """Individual quality assessment result."""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: float


@dataclass
class ConversationScenario:
    """Test scenario for quality evaluation."""
    scenario_id: str
    description: str
    conversation_turns: List[Tuple[str, str]]  # (user_input, expected_context)
    quality_targets: Dict[str, float]  # Expected quality thresholds
    metadata: Dict[str, Any]


class QualityMetrics:
    """Scientific quality metrics for memory system evaluation."""
    
    def __init__(self):
        self.logger = logging.getLogger("QualityMetrics")
    
    def coherence_score(self, 
                       conversation_history: List[Dict[str, Any]], 
                       retrieved_context: List[str]) -> QualityScore:
        """
        Measure conversation coherence based on topic consistency.
        
        Args:
            conversation_history: Full conversation history
            retrieved_context: Context retrieved for current query
            
        Returns:
            Coherence quality score (0.0 to 1.0)
        """
        if not conversation_history or not retrieved_context:
            return QualityScore(
                metric_name="coherence",
                score=0.0,
                max_score=1.0,
                details={"reason": "insufficient_data"},
                timestamp=time.time()
            )
        
        try:
            # Extract topics from conversation history
            conversation_topics = self._extract_topics(
                [turn.get("user_input", "") + " " + turn.get("agent_response", "") 
                 for turn in conversation_history[-5:]]  # Last 5 turns
            )
            
            # Extract topics from retrieved context
            context_topics = self._extract_topics(retrieved_context)
            
            # Calculate topic overlap
            if not conversation_topics or not context_topics:
                overlap_score = 0.0
            else:
                common_topics = set(conversation_topics) & set(context_topics)
                overlap_score = len(common_topics) / max(len(conversation_topics), len(context_topics))
            
            # Calculate semantic consistency
            semantic_score = self._semantic_consistency(conversation_history, retrieved_context)
            
            # Combined coherence score
            coherence = (overlap_score * 0.6 + semantic_score * 0.4)
            
            return QualityScore(
                metric_name="coherence",
                score=coherence,
                max_score=1.0,
                details={
                    "topic_overlap": overlap_score,
                    "semantic_consistency": semantic_score,
                    "conversation_topics": list(conversation_topics),
                    "context_topics": list(context_topics),
                    "common_topics": list(set(conversation_topics) & set(context_topics))
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Coherence scoring failed: {e}")
            return QualityScore(
                metric_name="coherence",
                score=0.0,
                max_score=1.0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def relevance_score(self, 
                       query: str, 
                       retrieved_memories: List[str],
                       ground_truth_context: Optional[List[str]] = None) -> QualityScore:
        """
        Measure relevance of retrieved memories to current query.
        
        Args:
            query: Current user query
            retrieved_memories: List of retrieved memory contents
            ground_truth_context: Optional expected relevant context
            
        Returns:
            Relevance quality score (0.0 to 1.0)
        """
        if not query or not retrieved_memories:
            return QualityScore(
                metric_name="relevance",
                score=0.0,
                max_score=1.0,
                details={"reason": "insufficient_data"},
                timestamp=time.time()
            )
        
        try:
            # Extract key terms from query
            query_terms = self._extract_key_terms(query)
            
            # Calculate term overlap for each retrieved memory
            memory_scores = []
            for memory in retrieved_memories:
                memory_terms = self._extract_key_terms(memory)
                if memory_terms:
                    overlap = len(set(query_terms) & set(memory_terms)) / len(set(query_terms) | set(memory_terms))
                    memory_scores.append(overlap)
                else:
                    memory_scores.append(0.0)
            
            # Average relevance score
            avg_relevance = np.mean(memory_scores) if memory_scores else 0.0
            
            # If ground truth provided, calculate precision
            precision = 1.0  # Default if no ground truth
            if ground_truth_context:
                precision = self._calculate_precision(retrieved_memories, ground_truth_context)
            
            # Combined relevance score
            relevance = (avg_relevance * 0.7 + precision * 0.3)
            
            return QualityScore(
                metric_name="relevance",
                score=relevance,
                max_score=1.0,
                details={
                    "query_terms": query_terms,
                    "memory_scores": memory_scores,
                    "avg_relevance": avg_relevance,
                    "precision": precision,
                    "num_retrieved": len(retrieved_memories)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Relevance scoring failed: {e}")
            return QualityScore(
                metric_name="relevance",
                score=0.0,
                max_score=1.0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def context_retention_score(self, 
                              agent_memory_system,
                              test_scenarios: List[ConversationScenario]) -> QualityScore:
        """
        Measure how well the memory system retains important context over time.
        
        Args:
            agent_memory_system: Agent with memory to test
            test_scenarios: Standardized test scenarios
            
        Returns:
            Context retention quality score (0.0 to 1.0)
        """
        if not test_scenarios:
            return QualityScore(
                metric_name="context_retention",
                score=0.0,
                max_score=1.0,
                details={"reason": "no_test_scenarios"},
                timestamp=time.time()
            )
        
        try:
            scenario_scores = []
            
            for scenario in test_scenarios:
                # Clear agent memory
                if hasattr(agent_memory_system, 'clear_memory'):
                    agent_memory_system.clear_memory()
                
                # Process conversation turns
                for i, (user_input, expected_context) in enumerate(scenario.conversation_turns):
                    # Store the conversation turn
                    agent_response = f"Response to: {user_input}"
                    agent_memory_system.process_conversation_turn(user_input, agent_response)
                    
                    # Test retrieval after several turns
                    if i >= 3:  # Start testing after 3rd turn
                        # Handle different agent APIs
                        if hasattr(agent_memory_system, 'memory_field'):  # FTCS Agent
                            retrieved = agent_memory_system.retrieve_memories(user_input, max_memories=3)
                            retrieved_content = [mem["content"] for mem in retrieved]
                        else:  # Baseline Agent
                            retrieved = agent_memory_system.retrieve_memories(user_input, k=3)
                            retrieved_content = [mem[0].content for mem in retrieved]
                        
                        # Check if expected context is retrievable
                        retention_score = self._check_context_retention(
                            retrieved_content, expected_context
                        )
                        scenario_scores.append(retention_score)
            
            # Average retention across scenarios
            avg_retention = np.mean(scenario_scores) if scenario_scores else 0.0
            
            return QualityScore(
                metric_name="context_retention",
                score=avg_retention,
                max_score=1.0,
                details={
                    "scenario_scores": scenario_scores,
                    "num_scenarios": len(test_scenarios),
                    "avg_retention": avg_retention
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Context retention scoring failed: {e}")
            return QualityScore(
                metric_name="context_retention",
                score=0.0,
                max_score=1.0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def response_quality_score(self, 
                             query: str,
                             agent_response: str,
                             retrieved_context: List[str]) -> QualityScore:
        """
        Measure quality of agent response given available context.
        
        Args:
            query: User query
            agent_response: Agent's response
            retrieved_context: Context used for response
            
        Returns:
            Response quality score (0.0 to 1.0)
        """
        if not query or not agent_response:
            return QualityScore(
                metric_name="response_quality",
                score=0.0,
                max_score=1.0,
                details={"reason": "insufficient_data"},
                timestamp=time.time()
            )
        
        try:
            # Context utilization score
            context_usage = self._context_utilization(agent_response, retrieved_context)
            
            # Response completeness
            completeness = self._response_completeness(query, agent_response)
            
            # Response coherence
            response_coherence = self._response_coherence(agent_response)
            
            # Combined response quality
            quality = (context_usage * 0.4 + completeness * 0.4 + response_coherence * 0.2)
            
            return QualityScore(
                metric_name="response_quality",
                score=quality,
                max_score=1.0,
                details={
                    "context_usage": context_usage,
                    "completeness": completeness,
                    "response_coherence": response_coherence,
                    "response_length": len(agent_response.split()),
                    "context_items": len(retrieved_context)
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Response quality scoring failed: {e}")
            return QualityScore(
                metric_name="response_quality",
                score=0.0,
                max_score=1.0,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _extract_topics(self, texts: List[str]) -> List[str]:
        """Extract key topics from text using simple keyword extraction."""
        all_words = []
        for text in texts:
            # Clean and tokenize
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        # Get most common words as topics (excluding stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        word_counts = Counter(w for w in all_words if w not in stopwords and len(w) > 3)
        return [word for word, count in word_counts.most_common(10)]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        return [w for w in words if w not in stopwords and len(w) > 3]
    
    def _semantic_consistency(self, conversation_history: List[Dict], retrieved_context: List[str]) -> float:
        """Calculate semantic consistency between conversation and context."""
        # Simple implementation: check for shared keywords
        conv_text = " ".join([
            turn.get("user_input", "") + " " + turn.get("agent_response", "")
            for turn in conversation_history[-3:]  # Last 3 turns
        ])
        
        context_text = " ".join(retrieved_context)
        
        conv_terms = set(self._extract_key_terms(conv_text))
        context_terms = set(self._extract_key_terms(context_text))
        
        if not conv_terms or not context_terms:
            return 0.0
        
        return len(conv_terms & context_terms) / len(conv_terms | context_terms)
    
    def _calculate_precision(self, retrieved: List[str], ground_truth: List[str]) -> float:
        """Calculate precision of retrieval against ground truth."""
        if not retrieved or not ground_truth:
            return 0.0
        
        # Simple string matching for precision
        relevant_count = 0
        for ret_item in retrieved:
            for gt_item in ground_truth:
                if self._text_similarity(ret_item, gt_item) > 0.5:
                    relevant_count += 1
                    break
        
        return relevant_count / len(retrieved)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap."""
        words1 = set(self._extract_key_terms(text1))
        words2 = set(self._extract_key_terms(text2))
        
        if not words1 or not words2:
            return 0.0
        
        return len(words1 & words2) / len(words1 | words2)
    
    def _check_context_retention(self, retrieved_content: List[str], expected_context: str) -> float:
        """Check if expected context is retained in retrieved memories."""
        if not retrieved_content or not expected_context:
            return 0.0
        
        expected_terms = set(self._extract_key_terms(expected_context))
        retrieved_terms = set()
        
        for content in retrieved_content:
            retrieved_terms.update(self._extract_key_terms(content))
        
        if not expected_terms:
            return 1.0  # No specific context expected
        
        return len(expected_terms & retrieved_terms) / len(expected_terms)
    
    def _context_utilization(self, response: str, context: List[str]) -> float:
        """Measure how well the response utilizes available context."""
        if not context or not response:
            return 0.0
        
        response_terms = set(self._extract_key_terms(response))
        context_terms = set()
        
        for ctx in context:
            context_terms.update(self._extract_key_terms(ctx))
        
        if not context_terms:
            return 1.0  # No context to utilize
        
        # Calculate how much context was used
        utilized = len(response_terms & context_terms)
        return min(1.0, utilized / len(context_terms))
    
    def _response_completeness(self, query: str, response: str) -> float:
        """Measure completeness of response relative to query."""
        query_terms = set(self._extract_key_terms(query))
        response_terms = set(self._extract_key_terms(response))
        
        if not query_terms:
            return 1.0
        
        # Check if response addresses query terms
        addressed = len(query_terms & response_terms)
        return min(1.0, addressed / len(query_terms))
    
    def _response_coherence(self, response: str) -> float:
        """Measure internal coherence of response."""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence: consistent vocabulary across sentences
        sentence_terms = [set(self._extract_key_terms(s)) for s in sentences if s.strip()]
        
        if len(sentence_terms) < 2:
            return 1.0
        
        # Calculate term overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentence_terms) - 1):
            if sentence_terms[i] and sentence_terms[i + 1]:
                overlap = len(sentence_terms[i] & sentence_terms[i + 1])
                union = len(sentence_terms[i] | sentence_terms[i + 1])
                if union > 0:
                    overlaps.append(overlap / union)
        
        return np.mean(overlaps) if overlaps else 0.5


class StandardTestScenarios:
    """Pre-defined test scenarios for comparative evaluation."""
    
    @staticmethod
    def get_context_switching_scenario() -> ConversationScenario:
        """Scenario testing memory across context switches."""
        return ConversationScenario(
            scenario_id="context_switching",
            description="Test memory retention across topic changes",
            conversation_turns=[
                ("What's the weather like?", "weather discussion"),
                ("It's sunny today", "current weather"),
                ("Let's talk about programming", "programming topic"),
                ("I'm learning Python", "python programming"),
                ("Do you remember what we said about weather?", "weather discussion")
            ],
            quality_targets={"context_retention": 0.7, "relevance": 0.6},
            metadata={"difficulty": "medium", "context_switches": 2}
        )
    
    @staticmethod
    def get_long_conversation_scenario() -> ConversationScenario:
        """Scenario testing memory in extended conversations."""
        return ConversationScenario(
            scenario_id="long_conversation",
            description="Test memory retention in extended conversation",
            conversation_turns=[
                ("I'm planning a trip to Japan", "travel planning"),
                ("I want to visit Tokyo and Kyoto", "japan cities"),
                ("What's the best time to visit?", "travel timing"),
                ("I'm interested in temples and food", "japan interests"),
                ("How much should I budget?", "travel budget"),
                ("What about transportation?", "japan transport"),
                ("Can you recommend hotels in Tokyo?", "tokyo hotels"),
                ("Remember I mentioned my budget earlier?", "travel budget")
            ],
            quality_targets={"context_retention": 0.8, "coherence": 0.7},
            metadata={"difficulty": "high", "conversation_length": "long"}
        )
    
    @staticmethod
    def get_technical_discussion_scenario() -> ConversationScenario:
        """Scenario testing technical context retention."""
        return ConversationScenario(
            scenario_id="technical_discussion",
            description="Test memory in technical discussions",
            conversation_turns=[
                ("I'm building a machine learning model", "ml development"),
                ("I'm using neural networks for classification", "neural networks"),
                ("The model has overfitting issues", "overfitting problem"),
                ("I tried dropout and regularization", "overfitting solutions"),
                ("Now I'm working on feature engineering", "feature engineering"),
                ("What overfitting solutions did we discuss?", "overfitting solutions")
            ],
            quality_targets={"context_retention": 0.9, "relevance": 0.8},
            metadata={"difficulty": "high", "domain": "technical"}
        )
    
    @staticmethod
    def get_all_scenarios() -> List[ConversationScenario]:
        """Get all standard test scenarios."""
        return [
            StandardTestScenarios.get_context_switching_scenario(),
            StandardTestScenarios.get_long_conversation_scenario(),
            StandardTestScenarios.get_technical_discussion_scenario()
        ]