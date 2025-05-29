"""
Configuration Loader Module

Provides utilities for loading, validating, and managing strategy configurations.
This module supports generic configuration loading for any strategy type.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""


class StrategyConfigLoader:
    """
    Generic strategy configuration loader and validator.

    This class provides functionality to:
    - Load strategy configurations from JSON files
    - Validate configuration parameters
    - Merge configurations with defaults
    - Handle configuration templates
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Optional custom configuration directory
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to production directory
            self.config_dir = (
                Path(__file__).parent / "strategy_params" / "production"
            )

        # Ensure directory exists
        if not self.config_dir.exists():
            # Fall back to strategy_params if production doesn't exist
            self.config_dir = Path(__file__).parent / "strategy_params"
            if not self.config_dir.exists():
                raise ConfigurationError(
                    f"Configuration directory not found: {self.config_dir}"
                )

        logger.info(f"Initialized StrategyConfigLoader with directory: {self.config_dir}")

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a strategy configuration from a JSON file.

        Args:
            config_name: Name of the configuration file (with or without .json extension)

        Returns:
            Dictionary containing the configuration

        Raises:
            ConfigurationError: If the configuration file is not found or invalid
        """
        # Ensure .json extension
        if not config_name.endswith(".json"):
            config_name += ".json"

        config_path = self.config_dir / config_name

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            logger.info(f"Loaded configuration from {config_path}")
            return config

        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file {config_path}: {e}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration file {config_path}: {e}"
            )

    def save_config(self, config: Dict[str, Any], config_name: str) -> str:
        """
        Save a strategy configuration to a JSON file.

        Args:
            config: Configuration dictionary to save
            config_name: Name of the configuration file (with or without .json extension)

        Returns:
            Path to the saved configuration file
        """
        # Ensure .json extension
        if not config_name.endswith(".json"):
            config_name += ".json"

        config_path = self.config_dir / config_name

        # Add metadata
        if "metadata" not in config:
            config["metadata"] = {}

        config["metadata"]["last_modified"] = datetime.now().isoformat()

        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

            logger.info(f"Saved configuration to {config_path}")
            return str(config_path)

        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration file {config_path}: {e}"
            )

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a strategy configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required top-level fields
        required_fields = ["strategy_name", "strategy_class", "description"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Validate technical parameters
        if "technical_parameters" in config:
            tech_params = config["technical_parameters"]
            if not isinstance(tech_params, dict):
                errors.append("technical_parameters must be a dictionary")

        # Validate risk management parameters
        if "risk_management" in config:
            risk_params = config["risk_management"]
            if not isinstance(risk_params, dict):
                errors.append("risk_management must be a dictionary")
            else:
                # Check for positive values where required
                positive_fields = [
                    "stop_loss_value",
                    "take_profit_value",
                    "max_position_size",
                    "risk_per_trade",
                ]
                for field in positive_fields:
                    if field in risk_params and risk_params[field] <= 0:
                        errors.append(f"risk_management.{field} must be positive")

        # Validate portfolio parameters
        if "portfolio_parameters" in config:
            portfolio_params = config["portfolio_parameters"]
            if not isinstance(portfolio_params, dict):
                errors.append("portfolio_parameters must be a dictionary")
            else:
                # Check for positive values
                positive_fields = ["init_cash", "fees", "slippage"]
                for field in positive_fields:
                    if field in portfolio_params and portfolio_params[field] < 0:
                        errors.append(
                            f"portfolio_parameters.{field} must be non-negative"
                        )

        # Validate optimization ranges
        if "optimization_ranges" in config:
            opt_ranges = config["optimization_ranges"]
            if not isinstance(opt_ranges, dict):
                errors.append("optimization_ranges must be a dictionary")
            else:
                for param, values in opt_ranges.items():
                    if not isinstance(values, list) or len(values) == 0:
                        errors.append(
                            f"optimization_ranges.{param} must be a non-empty list"
                        )

        # Validate parameter constraints if present
        if (
            "validation_rules" in config
            and "parameter_constraints" in config["validation_rules"]
        ):
            constraints = config["validation_rules"]["parameter_constraints"]
            errors.extend(self._validate_constraints(config, constraints))

        return errors

    def _validate_constraints(
        self, config: Dict[str, Any], constraints: Dict[str, bool]
    ) -> List[str]:
        """
        Validate parameter constraints.

        Args:
            config: Configuration dictionary
            constraints: Dictionary of constraint expressions

        Returns:
            List of constraint violation error messages
        """
        errors = []

        # Flatten config for easier access
        flat_config = self._flatten_config(config)

        for constraint, should_be_true in constraints.items():
            if not should_be_true:
                continue

            try:
                # Simple constraint evaluation
                # This is a basic implementation - could be enhanced with a proper expression parser
                if self._evaluate_constraint(constraint, flat_config) != should_be_true:
                    errors.append(f"Constraint violation: {constraint}")
            except Exception as e:
                errors.append(f"Error evaluating constraint '{constraint}': {e}")

        return errors

    def _flatten_config(
        self, config: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary.

        Args:
            config: Configuration dictionary to flatten
            prefix: Prefix for keys

        Returns:
            Flattened dictionary
        """
        flat = {}

        for key, value in config.items():
            if key.startswith("_"):  # Skip comment fields
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value

        return flat

    def _evaluate_constraint(
        self, constraint: str, flat_config: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a simple constraint expression.

        Args:
            constraint: Constraint expression (e.g., "param1 > param2")
            flat_config: Flattened configuration dictionary

        Returns:
            True if constraint is satisfied, False otherwise
        """
        # Basic constraint evaluation - could be enhanced
        # Supports: >, <, >=, <=, ==, !=

        operators = [">=", "<=", "==", "!=", ">", "<"]

        for op in operators:
            if op in constraint:
                left, right = constraint.split(op, 1)
                left = left.strip()
                right = right.strip()

                # Get values
                left_val = self._get_config_value(left, flat_config)
                right_val = self._get_config_value(right, flat_config)

                # Evaluate
                if op == ">":
                    return left_val > right_val
                elif op == "<":
                    return left_val < right_val
                elif op == ">=":
                    return left_val >= right_val
                elif op == "<=":
                    return left_val <= right_val
                elif op == "==":
                    return left_val == right_val
                elif op == "!=":
                    return left_val != right_val

        return True  # Default to True if no operator found

    def _get_config_value(self, key: str, flat_config: Dict[str, Any]) -> Any:
        """
        Get a configuration value, handling both direct keys and nested keys.

        Args:
            key: Configuration key (may be nested like "technical_parameters.period")
            flat_config: Flattened configuration dictionary

        Returns:
            Configuration value
        """
        # Try direct key first
        if key in flat_config:
            value = flat_config[key]
            return self._convert_value_type(value)

        # Try with technical_parameters prefix (common case)
        tech_key = f"technical_parameters.{key}"
        if tech_key in flat_config:
            value = flat_config[tech_key]
            return self._convert_value_type(value)

        # Try with risk_management prefix
        risk_key = f"risk_management.{key}"
        if risk_key in flat_config:
            value = flat_config[risk_key]
            return self._convert_value_type(value)

        # Try with other common prefixes
        for prefix in [
            "entry_conditions",
            "exit_conditions",
            "signal_processing",
            "portfolio_parameters",
        ]:
            prefixed_key = f"{prefix}.{key}"
            if prefixed_key in flat_config:
                value = flat_config[prefixed_key]
                return self._convert_value_type(value)

        # Try as numeric literal
        try:
            return float(key)
        except ValueError:
            pass

        # Try as string literal
        if key.startswith('"') and key.endswith('"'):
            return key[1:-1]

        raise ValueError(f"Configuration key not found: {key}")

    def _convert_value_type(self, value: Any) -> Any:
        """
        Convert configuration values to appropriate types.
        
        Args:
            value: Raw configuration value
            
        Returns:
            Converted value with appropriate type
        """
        # If already a number, return as-is
        if isinstance(value, (int, float)):
            return value
            
        # If it's a string that looks like a number, convert it
        if isinstance(value, str):
            # Try to convert to int first
            try:
                # Check if it's a whole number
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                else:
                    return float_val
            except (ValueError, TypeError):
                # Not a number, return as string
                return value
        
        # For other types (bool, list, dict), return as-is
        return value

    def list_configs(self) -> List[str]:
        """
        List all available configuration files.

        Returns:
            List of configuration file names (without .json extension)
        """
        configs = []

        for file_path in self.config_dir.glob("*.json"):
            if file_path.name != "strategy_template.json":  # Exclude template
                configs.append(file_path.stem)

        return sorted(configs)

    def create_from_template(
        self, strategy_name: str, template_name: str = "strategy_template"
    ) -> Dict[str, Any]:
        """
        Create a new configuration from a template.

        Args:
            strategy_name: Name for the new strategy configuration
            template_name: Name of the template to use (default: "strategy_template")

        Returns:
            New configuration dictionary based on template
        """
        template_config = self.load_config(template_name)

        # Update basic fields
        template_config["strategy_name"] = strategy_name
        template_config["strategy_class"] = f"{strategy_name.replace(' ', '')}Strategy"

        # Update metadata
        if "metadata" not in template_config:
            template_config["metadata"] = {}

        template_config["metadata"]["created_date"] = datetime.now().isoformat()
        template_config["metadata"]["last_modified"] = datetime.now().isoformat()

        return template_config

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configurations, with override_config taking precedence.

        Args:
            base_config: Base configuration dictionary
            override_config: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged


# Convenience functions for common operations
def load_strategy_config(
    config_name: str, config_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a strategy configuration.

    Args:
        config_name: Name of the configuration file
        config_dir: Optional custom configuration directory

    Returns:
        Configuration dictionary
    """
    loader = StrategyConfigLoader(config_dir)
    return loader.load_config(config_name)


def validate_strategy_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a strategy configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    loader = StrategyConfigLoader()
    return loader.validate_config(config)


def list_available_configs(config_dir: Optional[str] = None) -> List[str]:
    """
    List all available strategy configurations.

    Args:
        config_dir: Optional custom configuration directory

    Returns:
        List of configuration names
    """
    loader = StrategyConfigLoader(config_dir)
    return loader.list_configs()


class EnhancedConfigLoader(StrategyConfigLoader):
    """
    Enhanced configuration loader with VectorBTPro support and additional features.
    
    This class extends the base loader with:
    - VectorBTPro parameter conversion
    - Environment variable substitution
    - Optimal parameter integration
    - Strategy-specific validation
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the enhanced configuration loader."""
        super().__init__(config_dir)
        
    def load_config_with_env(self, config_name: str) -> Dict[str, Any]:
        """
        Load config with environment variable substitution.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Configuration with environment variables substituted
        """
        import os
        import re
        
        config = self.load_config(config_name)
        return self._substitute_env_vars(config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in config.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        import os
        import re
        
        if isinstance(obj, str):
            # Pattern for ${VAR} or ${VAR:default}
            pattern = r'\$\{([^}:]+)(?::([^}]+))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2)
                return os.environ.get(var_name, default_value or match.group(0))
            
            return re.sub(pattern, replacer, obj)
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        else:
            return obj
    
    def to_vbt_params(self, optimization_ranges: Dict) -> Dict[str, Any]:
        """
        Convert optimization ranges to VBT Param objects.
        
        Args:
            optimization_ranges: Dictionary of parameter ranges
            
        Returns:
            Dictionary with vbt.Param objects
        """
        import vectorbtpro as vbt
        
        vbt_params = {}
        
        for param, values in optimization_ranges.items():
            if isinstance(values, list):
                # Simple list of values
                vbt_params[param] = vbt.Param(values)
            elif isinstance(values, dict):
                # Advanced specification
                if values.get('type') == 'range':
                    # Create range
                    start = values.get('start', 0)
                    stop = values.get('stop', 100)
                    step = values.get('step', 1)
                    vbt_params[param] = vbt.Param(
                        range(start, stop, step)
                    )
                elif values.get('type') == 'linspace':
                    # Create linspace
                    start = values.get('start', 0)
                    stop = values.get('stop', 1)
                    num = values.get('num', 10)
                    import numpy as np
                    vbt_params[param] = vbt.Param(
                        np.linspace(start, stop, num)
                    )
                elif values.get('type') == 'list':
                    # Explicit list
                    vbt_params[param] = vbt.Param(values.get('values', []))
            else:
                # Single value, not for optimization
                vbt_params[param] = values
        
        return vbt_params
    
    def merge_with_optimal(
        self, 
        config: Dict[str, Any], 
        symbol: str, 
        timeframe: str,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge config with optimal parameters if available.
        
        Args:
            config: Base configuration
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Optional strategy name for lookup
            
        Returns:
            Merged configuration
        """
        try:
            from .optimal_parameters_db import OptimalParametersDB
            
            param_db = OptimalParametersDB()
            
            # Use provided strategy name or extract from config
            if not strategy_name:
                strategy_name = config.get('strategy_class', '').replace('Strategy', '')
                if not strategy_name:
                    strategy_name = "DMAATRTrendStrategy"  # Default fallback
            
            # Get optimal parameters from database
            optimal_params = param_db.get_optimal_params(symbol, timeframe, strategy_name)
            
            if optimal_params:
                # Deep merge optimal parameters with type conversion
                merged_config = config.copy()
                if 'technical_parameters' not in merged_config:
                    merged_config['technical_parameters'] = {}
                
                # Apply type conversion to optimal parameters before merging
                converted_optimal = {}
                for key, value in optimal_params.items():
                    converted_optimal[key] = self._convert_value_type(value)
                
                merged_config['technical_parameters'].update(converted_optimal)
                
                # Add metadata about optimization
                if 'metadata' not in merged_config:
                    merged_config['metadata'] = {}
                merged_config['metadata']['optimal_params_loaded'] = True
                merged_config['metadata']['optimal_params_source'] = 'database'
                merged_config['metadata']['strategy_name'] = strategy_name
                
                logger.info(f"Merged optimal parameters for {symbol} {timeframe} ({strategy_name})")
                return merged_config
                
        except ImportError:
            logger.debug("OptimalParametersDB not available")
        except Exception as e:
            logger.warning(f"Failed to merge optimal parameters: {e}")
        
        return config
    
    def validate_strategy_config(
        self, 
        config: Dict[str, Any], 
        strategy_class: type
    ) -> List[str]:
        """
        Validate config against strategy requirements.
        
        Args:
            config: Configuration to validate
            strategy_class: Strategy class to validate against
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check strategy class compatibility
        expected_class = config.get('strategy_class')
        actual_class = strategy_class.__name__
        
        if expected_class and expected_class != actual_class:
            errors.append(
                f"Config is for {expected_class}, not {actual_class}"
            )
        
        # Check required parameters if defined
        if hasattr(strategy_class, 'REQUIRED_PARAMS'):
            required_params = strategy_class.REQUIRED_PARAMS
            tech_params = config.get('technical_parameters', {})
            
            for param in required_params:
                if param not in tech_params:
                    errors.append(f"Missing required parameter: {param}")
        
        # Run base validation
        base_errors = self.validate_config(config)
        errors.extend(base_errors)
        
        return errors
    
    def create_multi_symbol_config(
        self,
        base_config: Dict[str, Any],
        symbols: List[str],
        symbol_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create multi-symbol configuration with symbol-specific overrides.
        
        Args:
            base_config: Base configuration
            symbols: List of symbols
            symbol_overrides: Optional symbol-specific parameter overrides
            
        Returns:
            Multi-symbol configuration
        """
        config = base_config.copy()
        
        # Add multi-symbol section
        config['multi_symbol_config'] = {
            'symbols': symbols,
            'symbol_specific_params': symbol_overrides or {},
            'base_params': config.get('technical_parameters', {})
        }
        
        # Update data parameters
        if 'data_parameters' not in config:
            config['data_parameters'] = {}
        config['data_parameters']['symbols'] = symbols
        
        return config
    
    def create_mtf_config(
        self,
        base_config: Dict[str, Any],
        timeframes: List[str],
        base_timeframe: str,
        timeframe_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create multi-timeframe configuration.
        
        Args:
            base_config: Base configuration
            timeframes: List of timeframes
            base_timeframe: Primary timeframe
            timeframe_weights: Optional weights for each timeframe
            
        Returns:
            MTF configuration
        """
        config = base_config.copy()
        
        # Add MTF section
        config['mtf_config'] = {
            'base_timeframe': base_timeframe,
            'confirmation_timeframes': [tf for tf in timeframes if tf != base_timeframe],
            'timeframe_weights': timeframe_weights or {tf: 1.0 for tf in timeframes},
            'use_mtf_confirmation': True
        }
        
        return config


# Update convenience functions to use enhanced loader
def load_strategy_config_enhanced(
    config_name: str, 
    config_dir: Optional[str] = None,
    with_env: bool = True
) -> Dict[str, Any]:
    """
    Load a strategy configuration with enhanced features.
    
    Args:
        config_name: Name of the configuration file
        config_dir: Optional custom configuration directory
        with_env: Whether to substitute environment variables
        
    Returns:
        Configuration dictionary
    """
    loader = EnhancedConfigLoader(config_dir)
    
    if with_env:
        return loader.load_config_with_env(config_name)
    else:
        return loader.load_config(config_name)
