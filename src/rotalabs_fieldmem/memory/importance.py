"""
Smart Importance Assignment System for FTCS Memory.

This module provides sophisticated importance scoring beyond simple keyword matching.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ImportanceScores:
    """Breakdown of importance scores by factor."""
    semantic: float = 0.0
    entities: float = 0.0
    causality: float = 0.0
    temporal: float = 0.0
    instructional: float = 0.0
    emotional: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'semantic': self.semantic,
            'entities': self.entities,
            'causality': self.causality,
            'temporal': self.temporal,
            'instructional': self.instructional,
            'emotional': self.emotional,
            'total': self.total
        }


class SemanticImportanceAnalyzer:
    """Analyzes semantic content to determine memory importance."""
    
    def __init__(self):
        # Causal relationship markers
        self.causal_markers = {
            'cause': ['because', 'due to', 'caused by', 'results from', 'stems from'],
            'effect': ['therefore', 'consequently', 'results in', 'leads to', 'causes'],
            'conditional': ['if', 'when', 'unless', 'provided that', 'assuming'],
            'dependency': ['depends on', 'requires', 'needs', 'relies on']
        }
        
        # Temporal markers
        self.temporal_markers = {
            'future': ['will', 'tomorrow', 'next', 'upcoming', 'scheduled', 'deadline'],
            'urgent': ['immediately', 'urgent', 'asap', 'right now', 'today', 'soon'],
            'recurring': ['always', 'every', 'usually', 'often', 'regularly']
        }
        
        # Instructional markers
        self.instruction_markers = [
            'how to', 'step', 'first', 'then', 'finally', 'procedure',
            'method', 'approach', 'technique', 'guide', 'tutorial'
        ]
        
        # Emotional/personal markers
        self.emotional_markers = [
            'love', 'hate', 'important to me', 'personal', 'favorite',
            'believe', 'feel', 'think', 'opinion', 'prefer'
        ]
    
    def analyze(self, content: str, context: Optional[Dict] = None) -> ImportanceScores:
        """
        Analyze content and return importance scores.
        
        Args:
            content: The memory content to analyze
            context: Optional context information
            
        Returns:
            ImportanceScores with breakdown by factor
        """
        scores = ImportanceScores()
        
        # Analyze different aspects
        scores.entities = self._score_entities(content)
        scores.causality = self._score_causality(content)
        scores.temporal = self._score_temporal(content)
        scores.instructional = self._score_instructions(content)
        scores.emotional = self._score_emotional(content)
        
        # Combine for semantic score
        scores.semantic = np.average([
            scores.entities,
            scores.causality,
            scores.temporal,
            scores.instructional,
            scores.emotional
        ], weights=[0.2, 0.3, 0.2, 0.2, 0.1])
        
        # Calculate total with base importance
        base_importance = 0.3
        scores.total = min(base_importance + scores.semantic, 1.0)
        
        return scores
    
    def _score_entities(self, content: str) -> float:
        """Score based on named entities, numbers, and specific information."""
        score = 0.0
        
        # Check for capitalized names (simple approach)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        names = re.findall(name_pattern, content)
        score += min(len(names) * 0.1, 0.3)
        
        # Check for dates
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b',  # Date formats
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}',
        ]
        for pattern in date_patterns:
            if re.search(pattern, content, re.I):
                score += 0.15
                break
        
        # Check for specific numbers/quantities
        if re.search(r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|dollars?|hours?|minutes?|days?|weeks?|months?|years?))\b', content, re.I):
            score += 0.1
        
        # Check for locations (simple approach - capitalized words after location markers)
        location_markers = ['in', 'at', 'from', 'to', 'near']
        for marker in location_markers:
            if re.search(rf'\b{marker}\s+[A-Z][a-z]+', content):
                score += 0.05
                break
        
        return min(score, 0.5)
    
    def _score_causality(self, content: str) -> float:
        """Score based on causal relationships and logical connections."""
        score = 0.0
        content_lower = content.lower()
        
        # Check for causal markers
        for category, markers in self.causal_markers.items():
            for marker in markers:
                if marker in content_lower:
                    if category == 'cause' or category == 'effect':
                        score += 0.25
                    else:
                        score += 0.15
                    break
            if score > 0:
                break
        
        # Check for reasoning patterns
        reasoning_patterns = [
            r'\bbecause\s+.*?,\s*\w+',  # "because X, Y"
            r'\bif\s+.*?\bthen\b',      # "if X then Y"
            r'\b\w+\s+leads?\s+to\s+\w+',  # "X leads to Y"
            r'\b\w+\s+results?\s+in\s+\w+'  # "X results in Y"
        ]
        
        for pattern in reasoning_patterns:
            if re.search(pattern, content_lower):
                score += 0.2
                break
        
        return min(score, 0.6)
    
    def _score_temporal(self, content: str) -> float:
        """Score based on time-sensitive information."""
        score = 0.0
        content_lower = content.lower()
        
        # Check urgent markers
        for marker in self.temporal_markers['urgent']:
            if marker in content_lower:
                score += 0.3
                break
        
        # Check future events
        for marker in self.temporal_markers['future']:
            if marker in content_lower:
                score += 0.2
                break
        
        # Check recurring patterns
        for marker in self.temporal_markers['recurring']:
            if marker in content_lower:
                score += 0.15
                break
        
        # Check for specific times
        time_pattern = r'\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b'
        if re.search(time_pattern, content_lower):
            score += 0.15
        
        return min(score, 0.5)
    
    def _score_instructions(self, content: str) -> float:
        """Score based on instructional or procedural content."""
        score = 0.0
        content_lower = content.lower()
        
        # Check instruction markers
        for marker in self.instruction_markers:
            if marker in content_lower:
                score += 0.2
                break
        
        # Check for numbered steps
        if re.search(r'\b(?:step\s+)?\d+[.)]|\bfirst\b|\bsecond\b|\bthird\b|\bthen\b|\bfinally\b', content_lower):
            score += 0.25
        
        # Check for imperative mood (simple heuristic)
        imperative_patterns = [
            r'^[A-Z][a-z]+\s+(?:the|a|an|your|this)',  # "Do the..."
            r'^(?:Don\'t|Do not|Never|Always)\s+',     # Direct commands
            r'^(?:Make|Create|Build|Write|Read|Check|Verify)\s+'  # Action verbs
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, content):
                score += 0.15
                break
        
        return min(score, 0.5)
    
    def _score_emotional(self, content: str) -> float:
        """Score based on emotional or personal significance."""
        score = 0.0
        content_lower = content.lower()
        
        # Check emotional markers
        for marker in self.emotional_markers:
            if marker in content_lower:
                score += 0.15
        
        # Check for personal pronouns in important contexts
        personal_patterns = [
            r'\b(?:my|our)\s+(?:goal|objective|priority|concern)',
            r'\b(?:I|we)\s+(?:need|want|must|should)',
            r'important\s+(?:to|for)\s+(?:me|us)'
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, content_lower):
                score += 0.2
                break
        
        # Check for exclamation marks (emphasis)
        if '!' in content:
            score += 0.1
        
        return min(score, 0.4)


class QuickImportanceScorer:
    """Quick implementation for immediate improvement."""
    
    @staticmethod
    def compute_importance(content: str) -> float:
        """
        Quick semantic importance calculation.
        This is a simplified version that can be used immediately.
        """
        importance = 0.3  # Base importance
        
        # Named entities (simple regex)
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', content):  # Names
            importance += 0.2
        
        # Dates and times
        if re.search(r'\b\d{4}\b|\b\d{1,2}[/-]\d{1,2}\b|\b\d{1,2}:\d{2}', content):
            importance += 0.15
        
        # Causal markers
        causal_words = ['because', 'therefore', 'results in', 'leads to', 'causes', 'due to']
        if any(word in content.lower() for word in causal_words):
            importance += 0.25
        
        # Instructions and procedures
        if re.search(r'\b(how to|step \d+|first|then|finally)\b', content, re.I):
            importance += 0.2
        
        # Urgency markers
        urgent_words = ['urgent', 'immediately', 'asap', 'deadline', 'important']
        if any(word in content.lower() for word in urgent_words):
            importance += 0.2
        
        # Questions (often important to remember)
        if '?' in content:
            importance += 0.1
        
        return min(importance, 1.0)


# Example usage and testing
if __name__ == "__main__":
    analyzer = SemanticImportanceAnalyzer()
    quick_scorer = QuickImportanceScorer()
    
    test_cases = [
        "Remember to call John Smith tomorrow at 3:00 PM",
        "The meeting is scheduled for next Tuesday",
        "The project deadline is urgent - due by Friday!",
        "How to fix the memory leak: first check the logs, then analyze heap dumps",
        "Because the temperature rises, the field evolution accelerates",
        "I love this approach to memory management",
        "The cat sat on the mat",  # Low importance baseline
        "System performance depends on proper configuration"
    ]
    
    print("Semantic Importance Analysis Test Results:")
    print("=" * 60)
    
    for content in test_cases:
        detailed_scores = analyzer.analyze(content)
        quick_score = quick_scorer.compute_importance(content)
        
        print(f"\nContent: {content}")
        print(f"Quick Score: {quick_score:.2f}")
        print(f"Detailed Scores: {detailed_scores.to_dict()}")
        print(f"  - Entities: {detailed_scores.entities:.2f}")
        print(f"  - Causality: {detailed_scores.causality:.2f}")
        print(f"  - Temporal: {detailed_scores.temporal:.2f}")
        print(f"  - Instructional: {detailed_scores.instructional:.2f}")
        print(f"  - Total: {detailed_scores.total:.2f}")