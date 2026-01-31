# Getting Started

## Installation

### Basic Installation

```bash
pip install rotalabs-ftms
```

### With Embeddings Support

```bash
pip install rotalabs-ftms[embeddings]
```

### Development Installation

```bash
pip install rotalabs-ftms[dev]
```

## Core Concepts

### Memory Fields

Memory in FTMS is represented as a continuous field that evolves according to the heat equation. Key parameters:

- **grid_size**: Resolution of the memory field (higher = more capacity)
- **diffusion_rate (α)**: How quickly memories spread to nearby regions
- **decay_rate (γ)**: How quickly memories fade without reinforcement
- **noise_level**: Amount of stochastic fluctuation

### Importance Weighting

Not all memories are equal. FTMS uses importance scoring to determine how strongly memories resist decay:

- **Entity importance**: Named entities, facts
- **Causal importance**: Cause-effect relationships
- **Temporal importance**: Time-sensitive information
- **Instructional importance**: Commands, directives

### Multi-Agent Systems

Multiple agents can share memory through:

- **Field coupling**: Coupled PDEs for memory synchronization
- **Collective memory pools**: Shared memory with consensus voting
- **Topologies**: Ring, star, mesh, hierarchical

## Basic Usage

### Creating an Agent

```python
from rotalabs_ftms import FTCSAgent, AgentConfig, FieldConfig

config = AgentConfig(
    name="my_agent",
    field_config=FieldConfig(
        grid_size=64,
        diffusion_rate=0.1,
        decay_rate=0.01,
    ),
)

agent = FTCSAgent(config=config)
```

### Storing Memories

```python
# Store with automatic importance scoring
agent.store("Important fact to remember")

# Store with explicit importance
agent.store("Critical instruction", importance=0.95)
```

### Querying Memories

```python
# Semantic query
results = agent.query("relevant topic")

# Get most active memories
active = agent.get_active_memories(top_k=10)
```

### Time Evolution

```python
# Evolve the field forward in time
agent.step(dt=0.1)

# Run multiple steps
for _ in range(100):
    agent.step(dt=0.01)
```
