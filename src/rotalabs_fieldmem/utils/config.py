"""
Configuration Manager for FTCS.

Handles loading, validation, and management of FTCS configurations
from YAML files, environment variables, and programmatic sources.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml

from .schemas import FTCSConfig, get_config_from_env


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class ConfigManager:
    """
    Manages FTCS configuration loading and validation.
    
    Supports loading from:
    - YAML/JSON configuration files
    - Environment variables
    - Programmatic configuration
    - Default values
    
    Provides validation, merging, and environment-specific overrides.
    """
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 environment: Optional[str] = None,
                 enable_env_override: bool = True):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment name (overrides config file)
            enable_env_override: Allow environment variables to override config
        """
        self.config_path = Path(config_path) if config_path else None
        self.environment = environment or os.getenv('FTCS_ENVIRONMENT', 'development')
        self.enable_env_override = enable_env_override
        self.logger = logging.getLogger('ConfigManager')
        
        # Loaded configuration
        self._config: Optional[FTCSConfig] = None
        self._config_dict: Dict[str, Any] = {}
        
    def load_config(self) -> FTCSConfig:
        """
        Load and validate configuration.
        
        Returns:
            Validated FTCS configuration
            
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            # Start with default configuration
            config_dict = {}
            
            # Load from file if specified
            if self.config_path and self.config_path.exists():
                config_dict.update(self._load_from_file(self.config_path))
                self.logger.info(f"Loaded configuration from {self.config_path}")
            
            # Apply environment-specific overrides
            config_dict.update(self._load_environment_overrides(config_dict))
            
            # Apply environment variable overrides
            if self.enable_env_override:
                env_config = get_config_from_env()
                if env_config:
                    self.logger.info(f"Environment config: {env_config}")
                    config_dict = self._deep_merge(config_dict, env_config)
                    self.logger.info(f"Merged config: {config_dict}")
                    self.logger.info(f"Applied {len(env_config)} environment variable overrides")
            
            # Set environment in config
            config_dict['environment'] = self.environment
            
            # Validate and create configuration
            self._config = FTCSConfig(**config_dict)
            self._config_dict = config_dict
            
            self.logger.info(f"Configuration loaded successfully for environment: {self.environment}")
            return self._config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}") from e
    
    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ConfigError(f"Unsupported config file format: {path.suffix}")
            
            if not isinstance(config, dict):
                raise ConfigError(f"Configuration file must contain a dictionary, got {type(config)}")
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML syntax in {path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON syntax in {path}: {e}")
        except IOError as e:
            raise ConfigError(f"Cannot read configuration file {path}: {e}")
    
    def _load_environment_overrides(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load environment-specific configuration overrides."""
        config_dir = self.config_path.parent if self.config_path else Path("configs")
        env_config_path = config_dir / f"{self.environment}.yaml"
        
        if env_config_path.exists():
            env_overrides = self._load_from_file(env_config_path)
            self.logger.info(f"Loaded environment overrides from {env_config_path}")
            return self._deep_merge(base_config, env_overrides)
        
        return base_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config(self) -> FTCSConfig:
        """Get the loaded configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> FTCSConfig:
        """Reload configuration from sources."""
        self._config = None
        self._config_dict = {}
        return self.load_config()
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Output path (defaults to original config path)
        """
        if self._config is None:
            raise ConfigError("No configuration loaded to save")
        
        output_path = Path(path) if path else self.config_path
        if output_path is None:
            raise ConfigError("No output path specified")
        
        # Convert to dictionary for serialization
        config_dict = self._config.model_dump()
        
        # Convert tuples and enums for YAML compatibility
        def convert_for_yaml(obj):
            if isinstance(obj, dict):
                return {k: convert_for_yaml(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_for_yaml(item) for item in obj]
            elif hasattr(obj, 'value'):  # Enum
                return obj.value
            return obj
        
        config_dict = convert_for_yaml(config_dict)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif output_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ConfigError(f"Unsupported output format: {output_path.suffix}")
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except IOError as e:
            raise ConfigError(f"Cannot write configuration to {output_path}: {e}")
    
    def validate_config(self, config_dict: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate configuration and return any errors.
        
        Args:
            config_dict: Configuration to validate (uses loaded config if None)
            
        Returns:
            List of validation error messages
        """
        if config_dict is None:
            if self._config is None:
                return ["No configuration loaded"]
            config_dict = self._config_dict
        
        errors = []
        
        try:
            FTCSConfig(**config_dict)
        except Exception as e:
            if hasattr(e, 'errors'):
                # Pydantic validation errors
                for error in e.errors():
                    field_path = " -> ".join(str(loc) for loc in error['loc'])
                    errors.append(f"{field_path}: {error['msg']}")
            else:
                errors.append(str(e))
        
        return errors
    
    def get_effective_config_dict(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary."""
        if self._config is None:
            raise ConfigError("No configuration loaded")
        return self._config.model_dump()
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate after updating
        """
        if self._config is None:
            raise ConfigError("No configuration loaded")
        
        # Deep merge updates
        updated_dict = self._deep_merge(self._config_dict, updates)
        
        if validate:
            errors = self.validate_config(updated_dict)
            if errors:
                raise ConfigError(f"Configuration validation failed: {errors}")
        
        # Update configuration
        self._config = FTCSConfig(**updated_dict)
        self._config_dict = updated_dict
        
        self.logger.info("Configuration updated successfully")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        if self._config is None:
            return {"status": "No configuration loaded"}
        
        config = self._config
        return {
            "environment": config.environment,
            "debug": config.debug,
            "field_shape": config.field.shape,
            "diffusion_rate": config.field.diffusion_rate,
            "temperature": config.field.temperature,
            "embedding_model": config.embedding.model_name,
            "use_proper_embeddings": config.embedding.use_proper_embeddings,
            "max_agents": config.multi_agent.max_agents,
            "coupling_strength": config.multi_agent.coupling_strength,
            "persistence_dir": config.persistence.base_dir,
            "auto_checkpoint": config.persistence.auto_checkpoint_interval,
            "log_level": config.monitoring.log_level,
            "performance_monitoring": config.monitoring.enable_performance_monitoring,
            "field_visualization": config.monitoring.enable_field_visualization,
            "loaded_from": str(self.config_path) if self.config_path else "defaults",
            "env_overrides_enabled": self.enable_env_override
        }


def load_config(config_path: Optional[Union[str, Path]] = None,
                environment: Optional[str] = None,
                enable_env_override: bool = True) -> FTCSConfig:
    """
    Convenience function to load FTCS configuration.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        enable_env_override: Allow environment variable overrides
        
    Returns:
        Validated FTCS configuration
    """
    manager = ConfigManager(
        config_path=config_path,
        environment=environment,
        enable_env_override=enable_env_override
    )
    return manager.load_config()


def create_default_config(output_path: Union[str, Path]) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Where to save the default configuration
    """
    default_config = FTCSConfig()
    output_path = Path(output_path)
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    with open(output_path, 'w') as f:
        yaml.dump(
            default_config.model_dump(), 
            f, 
            default_flow_style=False, 
            indent=2,
            sort_keys=False
        )
    
    print(f"Default configuration created at {output_path}")


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        manager = ConfigManager(config_path=config_path)
        manager.load_config()
        print(f"✓ Configuration file {config_path} is valid")
        return True
    except ConfigError as e:
        print(f"✗ Configuration file {config_path} is invalid: {e}")
        return False