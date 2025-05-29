"""
Configuration Manager Module

This module provides a high-level interface for managing strategy configurations.
It wraps the StrategyConfigLoader functionality with additional features.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import vectorbtpro as vbt

from .config_loader import EnhancedConfigLoader


class ConfigManager:
    """
    High-level configuration manager for the backtesting framework.
    
    This class provides convenient methods for loading and managing
    strategy configurations, including parameter extraction and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a specific configuration file
        """
        self.loader = EnhancedConfigLoader()
        self._config = None
        self._config_path = config_path
        
        # Load config if path provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str, with_env: bool = True) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_path: Path to the configuration file
            with_env: Whether to substitute environment variables
            
        Returns:
            Configuration dictionary
        """
        # Extract filename from path
        config_name = Path(config_path).name
        
        if with_env:
            self._config = self.loader.load_config_with_env(config_name)
        else:
            self._config = self.loader.load_config(config_name)
            
        self._config_path = config_path
        return self._config
    
    def load_with_optimal_params(
        self, 
        config_path: str, 
        symbol: str, 
        timeframe: str,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration and merge with optimal parameters if available.
        
        Args:
            config_path: Path to the configuration file
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Optional strategy name for lookup
            
        Returns:
            Configuration with optimal parameters merged
        """
        config = self.load_config(config_path)
        
        # Extract strategy name from config if not provided
        if not strategy_name:
            strategy_name = config.get('strategy_class', '').replace('Strategy', '')
            if not strategy_name:
                strategy_name = "DMAATRTrendStrategy"  # Default fallback
        
        # Merge with optimal parameters using database
        self._config = self.loader.merge_with_optimal(
            config, symbol, timeframe, strategy_name
        )
        
        return self._config
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Extract strategy parameters from the loaded configuration.
        
        Returns:
            Dictionary of strategy parameters
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        params = {}
        
        # Technical parameters
        if "technical_parameters" in self._config:
            params.update(self._config["technical_parameters"])
            
        # Risk management parameters
        if "risk_management" in self._config:
            params.update(self._config["risk_management"])
            
        # Trend confirmation parameters
        if "trend_confirmation" in self._config:
            params.update(self._config["trend_confirmation"])
            
        # Signal processing parameters
        if "signal_processing" in self._config:
            params.update(self._config["signal_processing"])
            
        # MTF config parameters
        if "mtf_config" in self._config:
            params.update(self._config["mtf_config"])
            
        # Multi-symbol config parameters
        if "multi_symbol_config" in self._config:
            # Don't merge all multi-symbol config, just relevant params
            ms_config = self._config["multi_symbol_config"]
            if "correlation_threshold" in ms_config:
                params["correlation_threshold"] = ms_config["correlation_threshold"]
            if "max_positions" in ms_config:
                params["max_active_symbols"] = ms_config["max_positions"]
            
        return params
    
    def get_vbt_params(self) -> Dict[str, Any]:
        """
        Get parameters formatted for VectorBTPro optimization.
        
        Returns:
            Dictionary with vbt.Param objects for optimization
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        # Check for VBT optimization section first
        if "vbt_optimization" in self._config:
            return self.loader.to_vbt_params(self._config["vbt_optimization"])
        
        # Fall back to regular optimization ranges
        if "optimization_ranges" in self._config:
            return self.loader.to_vbt_params(self._config["optimization_ranges"])
            
        # No optimization parameters
        return {}
    
    def get_portfolio_params(self) -> Dict[str, Any]:
        """
        Extract portfolio parameters from the loaded configuration.
        
        Returns:
            Dictionary of portfolio parameters
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        return self._config.get("portfolio_parameters", {
            "init_cash": 100000,
            "fees": 0.001,
            "slippage": 0.0005
        })
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        Extract optimization parameters from the loaded configuration.
        
        Returns:
            Dictionary of optimization parameters
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        return self._config.get("optimization_ranges", {})
    
    def get_data_params(self) -> Dict[str, Any]:
        """
        Extract data parameters from the loaded configuration.
        
        Returns:
            Dictionary of data parameters
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        return self._config.get("data_parameters", {})
    
    def validate_config(self, strategy_class: Optional[type] = None) -> List[str]:
        """
        Validate the loaded configuration.
        
        Args:
            strategy_class: Optional strategy class to validate against
            
        Returns:
            List of validation errors (empty if valid)
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        if strategy_class:
            return self.loader.validate_strategy_config(self._config, strategy_class)
        else:
            return self.loader.validate_config(self._config)
    
    def create_multi_symbol_config(
        self,
        symbols: List[str],
        symbol_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create multi-symbol configuration based on current config.
        
        Args:
            symbols: List of symbols
            symbol_overrides: Optional symbol-specific parameter overrides
            
        Returns:
            Multi-symbol configuration
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        return self.loader.create_multi_symbol_config(
            self._config, symbols, symbol_overrides
        )
    
    def create_mtf_config(
        self,
        timeframes: List[str],
        base_timeframe: str,
        timeframe_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create multi-timeframe configuration based on current config.
        
        Args:
            timeframes: List of timeframes
            base_timeframe: Primary timeframe
            timeframe_weights: Optional weights for each timeframe
            
        Returns:
            MTF configuration
        """
        if not self._config:
            raise ValueError("No configuration loaded")
            
        return self.loader.create_mtf_config(
            self._config, timeframes, base_timeframe, timeframe_weights
        )
    
    def save_config(self, config_path: Optional[str] = None) -> str:
        """
        Save the current configuration.
        
        Args:
            config_path: Optional path to save to (uses loaded path if not provided)
            
        Returns:
            Path where configuration was saved
        """
        if not self._config:
            raise ValueError("No configuration to save")
            
        save_path = config_path or self._config_path
        if not save_path:
            raise ValueError("No save path specified")
            
        config_name = Path(save_path).name
        return self.loader.save_config(self._config, config_name)
    
    @property
    def config(self) -> Optional[Dict[str, Any]]:
        """Get the loaded configuration."""
        return self._config
    
    def __repr__(self) -> str:
        """String representation."""
        if self._config:
            return f"ConfigManager(config='{self._config.get('strategy_name', 'Unknown')}')"
        return "ConfigManager(no config loaded)" 