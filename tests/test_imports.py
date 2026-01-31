"""Test basic imports and version."""

import pytest


def test_version():
    """Test version is accessible."""
    from rotalabs_fieldmem import __version__
    assert __version__ == "0.1.0"


def test_core_imports():
    """Test core imports work."""
    from rotalabs_fieldmem import MemoryField, FieldConfig
    assert MemoryField is not None
    assert FieldConfig is not None


def test_agent_imports():
    """Test agent imports work."""
    from rotalabs_fieldmem import FTCSAgent, AgentConfig, MemoryEntry
    assert FTCSAgent is not None
    assert AgentConfig is not None
    assert MemoryEntry is not None


def test_memory_imports():
    """Test memory processing imports work."""
    from rotalabs_fieldmem import (
        SemanticImportanceAnalyzer,
        QuickImportanceScorer,
        ImportanceScores,
    )
    assert SemanticImportanceAnalyzer is not None
    assert QuickImportanceScorer is not None
    assert ImportanceScores is not None


def test_field_config():
    """Test FieldConfig creation."""
    from rotalabs_fieldmem import FieldConfig

    config = FieldConfig(
        shape=(64, 64),
        diffusion_rate=0.01,
        temperature=0.05,
    )
    assert config.shape == (64, 64)
    assert config.diffusion_rate == 0.01
    assert config.temperature == 0.05


def test_memory_field_creation():
    """Test MemoryField creation."""
    from rotalabs_fieldmem import MemoryField, FieldConfig

    config = FieldConfig(shape=(32, 32))
    field = MemoryField(config)

    assert field.field.shape == (32, 32)
    assert field.time == 0.0


def test_agent_creation():
    """Test FTCSAgent creation."""
    from rotalabs_fieldmem import FTCSAgent, AgentConfig

    config = AgentConfig(
        memory_field_shape=(64, 64),
        use_proper_embeddings=False,
    )
    agent = FTCSAgent(agent_id="test_agent", config=config)

    assert agent.agent_id == "test_agent"


def test_importance_analyzer():
    """Test importance analyzer."""
    from rotalabs_fieldmem import SemanticImportanceAnalyzer

    analyzer = SemanticImportanceAnalyzer()
    scores = analyzer.analyze("Important meeting tomorrow at 3 PM!")

    assert scores.total > 0
    assert scores.temporal > 0  # Contains time reference


def test_quick_scorer():
    """Test quick importance scorer."""
    from rotalabs_fieldmem import QuickImportanceScorer

    scorer = QuickImportanceScorer()
    importance = scorer.compute_importance("URGENT: Deadline today!")

    assert importance >= 0.5  # Should be high importance
