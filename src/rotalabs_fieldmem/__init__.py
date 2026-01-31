"""rotalabs-ftms - Fuzzy Temporal Memory System.

Field-theoretic memory for AI agents with natural decay and consolidation.

Based on the physics of heat diffusion:
    ∂u/∂t = α∇²u - γu + η(t)

Where:
    - u: Memory field state
    - α: Diffusion rate (memory spreading)
    - γ: Decay rate (natural forgetting)
    - η(t): Thermal noise (stochastic fluctuations)

https://rotalabs.ai
"""

from rotalabs_fieldmem._version import __version__

# Core field implementations
from rotalabs_fieldmem.fields import MemoryField, FieldConfig

# Core agent
from rotalabs_fieldmem.agents import (
    FTCSAgent,
    AgentConfig,
    MemoryEntry,
)

# Memory processing
from rotalabs_fieldmem.memory import (
    SemanticImportanceAnalyzer,
    QuickImportanceScorer,
    ImportanceScores,
)

__all__ = [
    # Version
    "__version__",
    # Fields
    "MemoryField",
    "FieldConfig",
    # Agent
    "FTCSAgent",
    "AgentConfig",
    "MemoryEntry",
    # Memory
    "SemanticImportanceAnalyzer",
    "QuickImportanceScorer",
    "ImportanceScores",
]

# Optional: sparse field (may require additional dependencies)
try:
    from rotalabs_fieldmem.fields import SparseMemoryField
    __all__.append("SparseMemoryField")
except ImportError:
    pass

# Optional: multi-agent system
try:
    from rotalabs_fieldmem.agents import (
        MultiAgentSystem,
        MultiAgentConfig,
        MultiAgentCoordinator,
        FieldCoupler,
        CouplingConfig,
        CouplingTopology,
        CollectiveMemoryPool,
    )
    __all__.extend([
        "MultiAgentSystem",
        "MultiAgentConfig",
        "MultiAgentCoordinator",
        "FieldCoupler",
        "CouplingConfig",
        "CouplingTopology",
        "CollectiveMemoryPool",
    ])
except ImportError:
    pass
