# rotalabs-ftms

Fuzzy Temporal Memory System - Field-theoretic memory for AI agents with natural decay and consolidation.

## Overview

FTMS models agent memory as a continuous field governed by the heat equation:

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u - \gamma u + \eta(t)$$

Where:

- **u**: Memory field state (activation levels across the field)
- **α**: Diffusion rate (how memories spread to nearby locations)
- **γ**: Decay rate (natural forgetting over time)
- **η(t)**: Thermal noise (stochastic fluctuations)

## Features

- **Field-theoretic Memory**: Continuous memory fields with diffusion and decay
- **Importance-weighted Dynamics**: Preserve important memories while forgetting noise
- **Multi-agent Coordination**: Collective memory pools and field coupling
- **JAX Backend**: GPU-accelerated PDE solvers

## Installation

```bash
pip install rotalabs-ftms
```

With optional embeddings support:

```bash
pip install rotalabs-ftms[embeddings]
```

## Quick Example

```python
from rotalabs_ftms import FTCSAgent, AgentConfig, FieldConfig

# Configure the memory field
field_config = FieldConfig(
    grid_size=64,
    diffusion_rate=0.1,
    decay_rate=0.01,
)

# Create an agent with fuzzy temporal memory
agent_config = AgentConfig(
    name="memory_agent",
    field_config=field_config,
)

agent = FTCSAgent(config=agent_config)

# Store a memory
agent.store("The capital of France is Paris", importance=0.8)

# Memories naturally decay and diffuse over time
agent.step(dt=0.1)

# Query the memory field
results = agent.query("France capital")
```
