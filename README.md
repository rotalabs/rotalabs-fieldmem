# rotalabs-fieldmem

Field-Theoretic Memory System - Field-theoretic memory for AI agents with natural decay and consolidation.

[![PyPI version](https://badge.fury.io/py/rotalabs-fieldmem.svg)](https://badge.fury.io/py/rotalabs-fieldmem)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FieldMem implements a physics-inspired approach to AI memory, treating memories as continuous fields that evolve according to partial differential equations. This enables:

- **Natural forgetting**: Memories decay over time following thermodynamic principles
- **Semantic diffusion**: Related memories influence each other spatially
- **Importance-weighted retention**: Important memories resist decay
- **Collective intelligence**: Multi-agent memory sharing through field coupling

Based on the heat equation:

```
∂u/∂t = α∇²u - γu + η(t)
```

Where:
- `u`: Memory field state
- `α`: Diffusion rate (memory spreading)
- `γ`: Decay rate (natural forgetting)
- `η(t)`: Thermal noise (stochastic fluctuations)

## Installation

```bash
pip install rotalabs-fieldmem
```

### Optional dependencies

```bash
# For sentence-transformers embeddings
pip install rotalabs-fieldmem[embeddings]

# For visualization
pip install rotalabs-fieldmem[viz]

# For Google Cloud/Vertex AI integration
pip install rotalabs-fieldmem[gcp]

# All optional dependencies
pip install rotalabs-fieldmem[all]
```

## Quick Start

### Basic Memory Field

```python
from rotalabs_fieldmem import MemoryField, FieldConfig

# Create a memory field
config = FieldConfig(
    shape=(128, 128),
    diffusion_rate=0.01,
    temperature=0.05,
)
field = MemoryField(config)

# Inject a memory
import jax.numpy as jnp
embedding = jnp.ones(64)  # Example embedding
field.inject_memory(embedding, position=(64, 64), importance=0.8)

# Evolve the field (memories diffuse and decay)
field.step()

# Query memories
values, positions = field.query_memories(embedding, k=5)
```

### FTCS Agent

```python
from rotalabs_fieldmem import FTCSAgent, AgentConfig

# Create an agent with field-theoretic memory
config = AgentConfig(
    memory_field_shape=(128, 128),
    diffusion_rate=0.003,
    temperature=0.05,
)
agent = FTCSAgent(agent_id="agent_1", config=config)

# Store memories
agent.store_memory("The capital of France is Paris.", importance=0.9)
agent.store_memory("Python is a programming language.", importance=0.7)

# Retrieve relevant memories
results = agent.retrieve_memories("What European capitals do you know?")
for memory in results:
    print(f"- {memory['content']} (score: {memory['combined_score']:.2f})")
```

### Importance Scoring

```python
from rotalabs_fieldmem import SemanticImportanceAnalyzer, QuickImportanceScorer

# Detailed semantic analysis
analyzer = SemanticImportanceAnalyzer()
scores = analyzer.analyze("URGENT: Meeting tomorrow at 3 PM!")
print(f"Total importance: {scores.total:.2f}")
print(f"Temporal score: {scores.temporal:.2f}")

# Quick scoring for real-time use
scorer = QuickImportanceScorer()
importance = scorer.compute_importance("Remember to call John Smith.")
```

## Key Features

### 1. Field-Theoretic Memory

Memories are stored as energy distributions in a continuous field that evolves over time:

- **Diffusion**: Memories spread to nearby regions, creating semantic neighborhoods
- **Decay**: Low-importance memories fade naturally
- **Noise**: Thermal fluctuations add stochastic forgetting

### 2. Importance-Weighted Dynamics

The `SemanticImportanceAnalyzer` scores memories based on:

- **Entity detection**: Names, dates, locations
- **Causal relationships**: "because", "therefore", "leads to"
- **Temporal markers**: Deadlines, urgency, recurring events
- **Instructional content**: How-to, step-by-step procedures
- **Emotional significance**: Personal preferences, strong opinions

### 3. Multi-Agent Collective Intelligence

```python
from rotalabs_fieldmem import MultiAgentSystem, MultiAgentConfig

# Create a multi-agent system
config = MultiAgentConfig(
    num_agents=4,
    consensus_threshold=0.6,
)
system = MultiAgentSystem(config)

# Agents share knowledge through field coupling
response = system.collective_query("What do we know about machine learning?")
```

### 4. Field Coupling Topologies

- **Fully Connected**: All agents share equally
- **Nearest Neighbor**: Spatial locality
- **Small World**: Local + random long-range connections
- **Adaptive**: Coupling strength based on semantic similarity

## Architecture

```
src/rotalabs_fieldmem/
├── fields/           # Core field implementations
│   ├── memory_field.py   # PDE-based memory field
│   ├── sparse.py         # Sparse optimization (42x memory reduction)
│   └── operations.py     # Field operations
├── agents/           # FieldMem-enabled agents
│   ├── ftcs_agent.py     # Single agent
│   ├── multi_agent.py    # Multi-agent system
│   ├── coordinator.py    # Agent coordination
│   ├── collective.py     # Collective memory pool
│   └── coupling.py       # Field coupling
├── memory/           # Memory processing
│   ├── importance.py     # Semantic importance scoring
│   ├── consolidation.py  # Memory consolidation
│   └── forgetting.py     # Thermodynamic forgetting
├── utils/            # Utilities
│   ├── metrics.py        # Quality metrics
│   ├── embeddings.py     # Embedding management
│   └── persistence.py    # State persistence
└── viz/              # Visualization (optional)
    └── plotting.py       # Field visualization
```

## Benchmark Results

Evaluated on standard long-context conversation benchmarks:

### LongMemEval (ICLR 2025)

| Memory Ability | FieldMem | Baseline | Improvement | Significance |
|----------------|----------|----------|-------------|--------------|
| Multi-session | 0.042 F1 | 0.020 F1 | **+116%** | p<0.01, d=3.06 |
| Temporal reasoning | 0.120 F1 | 0.084 F1 | **+43.8%** | p<0.001, d=9.21 |
| Preference tracking | 0.268 F1 | 0.168 F1 | **+59.1%** | p<0.001, d=8.96 |
| Knowledge update | 0.147 recall | 0.115 recall | **+27.8%** | p<0.001 |

### LoCoMo (ACL 2024)

| Category | FieldMem | Baseline | Improvement |
|----------|----------|----------|-------------|
| Category 3 (F1) | 0.017 | 0.016 | +9.5%* (p<0.05) |
| Category 4 (EM) | 0.271 | 0.256 | +6.1% |

### Computational Performance

| Metric | FieldMem | Traditional |
|--------|----------|-------------|
| Memory efficiency | 42x reduction | baseline |
| JAX JIT speedup | 518x | baseline |
| Multi-agent sharing | Native | Manual |

## Scientific Background

FieldMem builds on the Field-Theoretic Context System (FTCS) published in [Technical Disclosure Commons](https://www.tdcommons.org/dpubs_series/8022/). The full methodology and benchmark analysis are available in our arXiv paper.

## Links

- **Paper**: [arXiv:cs.ET](https://arxiv.org/) (FieldMem: Field-Theoretic Memory for AI Agents)
- **Prior Work**: [FTCS Technical Disclosure](https://www.tdcommons.org/dpubs_series/8022/)
- Website: https://rotalabs.ai
- GitHub: https://github.com/rotalabs/rotalabs-fieldmem
- Documentation: https://rotalabs.github.io/rotalabs-fieldmem/
- Contact: research@rotalabs.ai

## License

MIT License - see LICENSE for details.
