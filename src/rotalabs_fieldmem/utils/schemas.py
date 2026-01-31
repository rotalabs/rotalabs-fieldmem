"""
Configuration schemas with validation for FTCS.

Defines Pydantic models for all FTCS configuration options with
validation, type checking, and documentation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class BoundaryCondition(str, Enum):
    """Boundary condition types for field evolution."""
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"
    PERIODIC = "periodic"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    MINI_LM_L6 = "all-MiniLM-L6-v2"
    MINI_LM_L12 = "all-MiniLM-L12-v2"
    DISTILBERT = "all-distilroberta-v1"
    MPNET = "all-mpnet-base-v2"
    SENTENCE_T5 = "sentence-t5-base"
    # Google Cloud models
    TEXTEMBEDDING_GECKO_003 = "textembedding-gecko@003"
    TEXTEMBEDDING_GECKO_MULTILINGUAL = "textembedding-gecko-multilingual@001"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConfigFieldConfig(BaseModel):
    """Field configuration with validation."""
    
    shape: Tuple[int, int] = Field(
        default=(128, 128),
        description="Memory field dimensions (height, width)"
    )
    diffusion_rate: float = Field(
        default=0.005,
        ge=0.0001,
        le=0.1,
        description="Rate of memory diffusion across field"
    )
    temperature: float = Field(
        default=0.08,
        ge=0.01,
        le=1.0,
        description="Temperature for forgetting dynamics"
    )
    evolution_dt: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Time step for field evolution"
    )
    boundary_condition: BoundaryCondition = Field(
        default=BoundaryCondition.NEUMANN,
        description="Boundary conditions for field evolution"
    )
    max_iterations: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum evolution iterations"
    )
    convergence_threshold: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-3,
        description="Convergence threshold for evolution"
    )
    
    @field_validator('shape')
    @classmethod
    def validate_shape(cls, v):
        """Validate field shape."""
        if len(v) != 2:
            raise ValueError("Shape must be a 2-tuple (height, width)")
        if v[0] < 8 or v[1] < 8:
            raise ValueError("Field dimensions must be at least 8x8")
        if v[0] > 2048 or v[1] > 2048:
            raise ValueError("Field dimensions must not exceed 2048x2048")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration with validation."""
    
    model_name: EmbeddingModel = Field(
        default=EmbeddingModel.MINI_LM_L6,
        description="Embedding model to use"
    )
    embedding_dim: Optional[int] = Field(
        default=None,
        ge=32,
        le=1536,
        description="Embedding dimension (auto-detected if None)"
    )
    use_proper_embeddings: bool = Field(
        default=True,
        description="Use sentence-transformers (vs simple embeddings)"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Normalize embeddings to unit length"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Cache embeddings for repeated text"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding computation"
    )
    device: Optional[str] = Field(
        default=None,
        description="Device for embedding computation (auto if None)"
    )


class ConfigAgentConfig(BaseModel):
    """Agent configuration with validation."""
    
    memory_field_shape: Tuple[int, int] = Field(
        default=(128, 128),
        description="Shape of agent's memory field"
    )
    max_memories_per_query: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum memories returned per query"
    )
    memory_evolution_interval: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Seconds between memory field evolution"
    )
    importance_decay_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Rate of importance decay over time"
    )
    forgetting_threshold_hours: float = Field(
        default=24.0,
        ge=1.0,
        le=8760.0,  # 1 year
        description="Hours after which low-importance memories are forgotten"
    )
    conversation_context_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recent conversation turns to track"
    )
    enable_auto_forgetting: bool = Field(
        default=True,
        description="Enable automatic forgetting of old memories"
    )


class MultiAgentConfig(BaseModel):
    """Multi-agent system configuration."""
    
    max_agents: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of agents in coordinator"
    )
    enable_async_updates: bool = Field(
        default=True,
        description="Enable asynchronous field updates"
    )
    coupling_strength: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Default coupling strength between agents"
    )
    update_interval: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Seconds between multi-agent field updates"
    )
    enable_field_sharing: bool = Field(
        default=True,
        description="Enable memory field sharing between agents"
    )
    group_memory_threshold: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Threshold for sharing memories between grouped agents"
    )


class PersistenceConfig(BaseModel):
    """Persistence configuration."""
    
    base_dir: str = Field(
        default="results/saved_states",
        description="Base directory for saved states"
    )
    enable_compression: bool = Field(
        default=True,
        description="Enable compression for saved files"
    )
    auto_checkpoint_interval: Optional[float] = Field(
        default=None,
        ge=60.0,
        le=86400.0,  # 1 day
        description="Seconds between automatic checkpoints (None to disable)"
    )
    max_saved_states: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of saved states to keep"
    )
    backup_on_save: bool = Field(
        default=False,
        description="Create backup before overwriting saves"
    )
    include_embeddings_by_default: bool = Field(
        default=True,
        description="Include embedding models in saves by default"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance metrics collection"
    )
    enable_memory_profiling: bool = Field(
        default=False,
        description="Enable memory usage profiling"
    )
    metrics_export_interval: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description="Seconds between metrics exports"
    )
    enable_field_visualization: bool = Field(
        default=False,
        description="Enable real-time field visualization"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for console only)"
    )
    enable_distributed_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing (OpenTelemetry)"
    )


class FTCSConfig(BaseModel):
    """Complete FTCS configuration."""
    
    # Core components
    field: ConfigFieldConfig = Field(
        default_factory=ConfigFieldConfig,
        description="Memory field configuration"
    )
    agent: ConfigAgentConfig = Field(
        default_factory=ConfigAgentConfig,
        description="Agent behavior configuration"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration"
    )
    multi_agent: MultiAgentConfig = Field(
        default_factory=MultiAgentConfig,
        description="Multi-agent system configuration"
    )
    persistence: PersistenceConfig = Field(
        default_factory=PersistenceConfig,
        description="Persistence and storage configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring and logging configuration"
    )
    
    # Global settings
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    @model_validator(mode='before')
    @classmethod
    def validate_config_consistency(cls, values):
        """Validate configuration consistency across components."""
        if not isinstance(values, dict):
            return values
            
        field_config = values.get('field', {})
        agent_config = values.get('agent', {})
        
        # Ensure agent field shape matches field config
        if 'shape' in field_config:
            # Always use field config shape to ensure consistency
            shape = field_config['shape']
            # Convert list to tuple if needed
            if isinstance(shape, list):
                shape = tuple(shape)
            if 'agent' not in values:
                values['agent'] = {}
            values['agent']['memory_field_shape'] = shape
        
        # Validate environment-specific settings
        environment = values.get('environment', 'development')
        monitoring = values.get('monitoring', {})
        
        if environment == 'production':
            # Production should have monitoring enabled
            if 'enable_performance_monitoring' not in monitoring:
                monitoring['enable_performance_monitoring'] = True
            
            # Production should not have debug visualization
            if monitoring.get('enable_field_visualization', False):
                monitoring['enable_field_visualization'] = False
        
        values['agent'] = agent_config
        values['monitoring'] = monitoring
        
        return values
    
    @field_validator('environment')
    @classmethod
    def set_environment_defaults(cls, v):
        """Set appropriate defaults based on environment."""
        # Note: In Pydantic v2, we handle environment-specific defaults
        # in the model_validator above rather than field_validator
        return v
    
    model_config = {
        "extra": "forbid",  # Forbid extra fields
        "validate_assignment": True,  # Validate on assignment
        "use_enum_values": True  # Use enum values in serialization
    }


# Environment variable mappings
ENV_MAPPING = {
    'FTCS_ENVIRONMENT': 'environment',
    'FTCS_DEBUG': 'debug',
    'FTCS_SEED': 'seed',
    'FTCS_LOG_LEVEL': 'monitoring.log_level',
    'FTCS_LOG_FILE': 'monitoring.log_file',
    'FTCS_FIELD_SHAPE_HEIGHT': 'field.shape[0]',
    'FTCS_FIELD_SHAPE_WIDTH': 'field.shape[1]',
    'FTCS_DIFFUSION_RATE': 'field.diffusion_rate',
    'FTCS_TEMPERATURE': 'field.temperature',
    'FTCS_EMBEDDING_MODEL': 'embedding.model_name',
    'FTCS_EMBEDDING_DIM': 'embedding.embedding_dim',
    'FTCS_USE_PROPER_EMBEDDINGS': 'embedding.use_proper_embeddings',
    'FTCS_MAX_AGENTS': 'multi_agent.max_agents',
    'FTCS_COUPLING_STRENGTH': 'multi_agent.coupling_strength',
    'FTCS_PERSISTENCE_DIR': 'persistence.base_dir',
    'FTCS_AUTO_CHECKPOINT': 'persistence.auto_checkpoint_interval',
    'FTCS_ENABLE_MONITORING': 'monitoring.enable_performance_monitoring',
    'FTCS_METRICS_INTERVAL': 'monitoring.metrics_export_interval'
}


def get_config_from_env() -> Dict[str, Any]:
    """Extract FTCS configuration from environment variables."""
    config_dict = {}
    
    for env_var, config_path in ENV_MAPPING.items():
        value = os.getenv(env_var)
        if value is not None:
            # Handle special cases
            if config_path.endswith('[0]') or config_path.endswith('[1]'):
                # Handle tuple elements
                base_path = config_path.split('[')[0]
                index = int(config_path.split('[')[1].split(']')[0])
                
                if base_path not in config_dict:
                    config_dict[base_path] = [None, None]
                
                # Convert to appropriate type
                try:
                    config_dict[base_path][index] = int(value)
                except ValueError:
                    config_dict[base_path][index] = value
                    
                # Convert list to tuple for shape when both values are set
                if base_path == 'field.shape' and index == 1:  # After setting width
                    if all(v is not None for v in config_dict[base_path]):
                        config_dict[base_path] = tuple(config_dict[base_path])
            else:
                # Handle nested paths
                parts = config_path.split('.')
                current = config_dict
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Convert to appropriate type
                final_key = parts[-1]
                try:
                    # Try int first
                    current[final_key] = int(value)
                except ValueError:
                    try:
                        # Try float
                        current[final_key] = float(value)
                    except ValueError:
                        # Try bool
                        if value.lower() in ('true', 'false'):
                            current[final_key] = value.lower() == 'true'
                        else:
                            # Keep as string
                            current[final_key] = value
    
    # Copy all items from config_dict to prepare for nested conversion
    final_dict = config_dict.copy()
    
    # Convert dotted keys to nested structure
    nested_config = {}
    for key, value in final_dict.items():
        if '.' in key:
            parts = key.split('.')
            current = nested_config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            nested_config[key] = value
    
    return nested_config