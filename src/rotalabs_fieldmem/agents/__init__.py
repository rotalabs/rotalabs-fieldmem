"""FTMS-enabled agents.

Provides agents with field-theoretic memory capabilities.
"""

from rotalabs_fieldmem.agents.ftcs_agent import (
    FTCSAgent,
    AgentConfig,
    MemoryEntry,
)
from rotalabs_fieldmem.agents.coordinator import (
    MultiAgentCoordinator,
    CouplingConfig as CoordinatorCouplingConfig,
    AgentGroup,
)
from rotalabs_fieldmem.agents.coupling import (
    FieldCoupler,
    CouplingConfig,
    CouplingTopology,
)
from rotalabs_fieldmem.agents.collective import (
    CollectiveMemory,
    CollectiveMemoryPool,
)
from rotalabs_fieldmem.agents.multi_agent import (
    MultiAgentSystem,
    MultiAgentConfig,
)

__all__ = [
    # Core agent
    "FTCSAgent",
    "AgentConfig",
    "MemoryEntry",
    # Coordination
    "MultiAgentCoordinator",
    "CoordinatorCouplingConfig",
    "AgentGroup",
    # Coupling
    "FieldCoupler",
    "CouplingConfig",
    "CouplingTopology",
    # Collective memory
    "CollectiveMemory",
    "CollectiveMemoryPool",
    # Multi-agent system
    "MultiAgentSystem",
    "MultiAgentConfig",
]
