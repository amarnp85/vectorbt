"""
Configuration System Module

This module provides a comprehensive configuration management system for the backtesting framework.

Key Features:
- Strategy parameter configuration with validation
- VectorBTPro parameter conversion for optimization
- Multi-symbol and multi-timeframe configuration support
- Database-based parameter storage and retrieval
- Environment variable substitution
- Configuration templates for all strategy types

Components:
- ConfigManager: High-level interface for configuration management
- StrategyConfigLoader: Base configuration loader with validation
- EnhancedConfigLoader: Extended loader with VBT support
- OptimalParametersDB: Database-based storage and retrieval of optimization results
"""

from .config_loader import (
    StrategyConfigLoader,
    EnhancedConfigLoader,
    load_strategy_config,
    load_strategy_config_enhanced,
    validate_strategy_config,
    list_available_configs
)

from .config_manager import ConfigManager

from .optimal_parameters_db import OptimalParametersDB

__all__ = [
    # Main interface
    'ConfigManager',
    
    # Loaders
    'StrategyConfigLoader',
    'EnhancedConfigLoader',
    
    # Convenience functions
    'load_strategy_config',
    'load_strategy_config_enhanced',
    'validate_strategy_config',
    'list_available_configs',
    
    # Database-based parameter storage
    'OptimalParametersDB'
] 