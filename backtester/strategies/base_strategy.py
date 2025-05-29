"""
Base Strategy Module

Provides an abstract base class for all trading strategies in the backtesting system.
This class defines the standard interface and workflow that all strategies should follow.
"""

from abc import ABC, abstractmethod
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Union
import json
import os

# Use the new structured logging system
from ..utilities.structured_logging import get_logger

logger = get_logger("strategy")


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    This class defines the common interface and functionality that all strategies should implement.
    Strategies should focus ONLY on:
    1. Calculating indicators
    2. Generating signals
    3. Providing strategy metadata
    
    Portfolio simulation, performance analysis, and plotting should be handled by their respective modules.
    """

    def __init__(
        self,
        data: vbt.Data,  # Always use vbt.Data for consistency
        params: Optional[Union[Dict[str, Any], str]] = None,
    ):
        """Initialize the strategy with data and parameters.

        Args:
            data: VectorBTPro Data object with OHLCV data
            params: Dictionary of strategy parameters OR path to config file
        """
        # Ensure we have vbt.Data
        if not isinstance(data, vbt.Data):
            raise TypeError("Strategy requires vbt.Data object. Use vbt.Data.from_data() to convert DataFrames.")
            
        self.data = data
        self.config = None
        
        # Handle config file loading
        if isinstance(params, str):
            # Load from config file
            from ..config.config_loader import load_strategy_config
            self.config = load_strategy_config(params)
            
            # Extract parameters from config
            self.params = {}
            # Merge technical parameters
            if 'technical_parameters' in self.config:
                self.params.update(self.config['technical_parameters'])
            # Merge risk management parameters
            if 'risk_management' in self.config:
                self.params.update(self.config['risk_management'])
            # Merge any other relevant sections
            if 'signal_processing' in self.config:
                self.params.update(self.config['signal_processing'])
            if 'trend_confirmation' in self.config:
                self.params.update(self.config['trend_confirmation'])
                
            logger.debug(f"Loaded configuration from {params}")
        else:
            self.params = params or {}
            
        self.indicators = {}
        self.signals = {}
        
        # Store data metadata
        self.symbols = data.symbols if hasattr(data, 'symbols') else None
        self.is_multi_symbol = self.symbols is not None and len(self.symbols) > 1

        # Only log initialization in non-quiet mode (respects structured logging level)
        logger.debug(f"Initialized {self.__class__.__name__} strategy")

    @abstractmethod
    def init_indicators(self) -> Dict[str, Any]:
        """Calculate and store all necessary indicators.

        This method should be implemented by each strategy to calculate
        the specific indicators required by the strategy.

        Returns:
            Dictionary of calculated indicators
        """

    # Alias for consistency with tests
    def calculate_indicators(self) -> Dict[str, Any]:
        """Alias for init_indicators for backward compatibility."""
        return self.init_indicators()

    @abstractmethod
    def generate_signals(self) -> Dict[str, pd.Series]:
        """Generate entry/exit signals based on indicators and parameters.

        This method should be implemented by each strategy to generate
        the specific trading signals based on the calculated indicators.

        Returns:
            Dictionary with keys:
            - 'long_entries': Boolean Series for long entry signals
            - 'long_exits': Boolean Series for long exit signals
            - 'short_entries': Boolean Series for short entry signals
            - 'short_exits': Boolean Series for short exit signals
            - 'sl_levels': Numeric Series for stop-loss levels (optional)
            - 'tp_levels': Numeric Series for take-profit levels (optional)
        """

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """

    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate strategy parameters.
        
        Returns:
            True if parameters are valid
        """

    @abstractmethod
    def get_strategy_description(self) -> str:
        """Get a human-readable description of the strategy.
        
        Returns:
            String description of the strategy
        """

    def run_signal_generation(self) -> Dict[str, Any]:
        """Run the signal generation workflow.

        This is the main method that should be called to generate signals.
        Portfolio simulation should be handled separately by the portfolio module.

        Returns:
            Dictionary containing indicators and signals
        """
        logger.debug(f"Starting signal generation for {self.__class__.__name__}")

        # Calculate indicators
        self.init_indicators()
        
        # Generate signals
        self.generate_signals()

        logger.debug("Signal generation completed successfully")

        return {
            "indicators": self.indicators,
            "signals": self.signals,
            "params": self.params,
            "metadata": self.get_metadata()
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "strategy_name": self.__class__.__name__,
            "is_multi_symbol": self.is_multi_symbol,
            "symbols": self.symbols,
            "data_shape": self.data.shape if hasattr(self.data, "shape") else None,
            "indicators_calculated": list(self.indicators.keys()),
            "signals_generated": list(self.signals.keys()),
        }

    def validate_signals(self) -> bool:
        """Validate generated signals using VectorBTPro utilities.
        
        Returns:
            True if signals are valid
        """
        if not self.signals:
            logger.warning("No signals to validate")
            return False
            
        # Use centralized signal validation
        from ..signals.signal_utils import validate_signals
        
        result = validate_signals(self.signals)
        
        if not result.is_valid:
            logger.error(f"Signal validation failed: {result.errors}")
            return False
            
        # Log any warnings
        for warning in result.warnings:
            logger.warning(warning)
            
        return True

    def clean_signals(self) -> Dict[str, pd.Series]:
        """Clean signals using VectorBTPro's signal utilities.
        
        Returns:
            Dictionary of cleaned signals
        """
        if not self.signals:
            raise ValueError("No signals to clean. Generate signals first.")
            
        # Use VectorBTPro's signal cleaning
        cleaned_signals = {}
        
        # Clean opposing signals (long vs short)
        if 'long_entries' in self.signals and 'short_entries' in self.signals:
            long_entries = self.signals['long_entries']
            short_entries = self.signals['short_entries']
            
            # Use VBT's signal cleaning - it automatically handles conflicts
            cleaned_long, cleaned_short = long_entries.vbt.signals.clean(short_entries)
            cleaned_signals['long_entries'] = cleaned_long
            cleaned_signals['short_entries'] = cleaned_short
        else:
            cleaned_signals['long_entries'] = self.signals.get('long_entries')
            cleaned_signals['short_entries'] = self.signals.get('short_entries')
            
        # Clean entry/exit pairs
        if 'long_entries' in cleaned_signals and 'long_exits' in self.signals:
            entries = cleaned_signals['long_entries']
            exits = self.signals['long_exits']
            
            # Clean to ensure proper entry/exit sequence
            cleaned_entries, cleaned_exits = entries.vbt.signals.clean(exits)
            cleaned_signals['long_entries'] = cleaned_entries
            cleaned_signals['long_exits'] = cleaned_exits
        else:
            cleaned_signals['long_exits'] = self.signals.get('long_exits')
            
        # Same for short signals
        if 'short_entries' in cleaned_signals and 'short_exits' in self.signals:
            entries = cleaned_signals['short_entries']
            exits = self.signals['short_exits']
            
            cleaned_entries, cleaned_exits = entries.vbt.signals.clean(exits)
            cleaned_signals['short_entries'] = cleaned_entries
            cleaned_signals['short_exits'] = cleaned_exits
        else:
            cleaned_signals['short_exits'] = self.signals.get('short_exits')
            
        # Copy over stop levels
        cleaned_signals['sl_levels'] = self.signals.get('sl_levels')
        cleaned_signals['tp_levels'] = self.signals.get('tp_levels')
        
        return cleaned_signals

    def save_params(self, filename: str) -> str:
        """Save strategy parameters to a JSON file.

        Args:
            filename: Output JSON filename

        Returns:
            Path to saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Convert any non-serializable values to strings
        serializable_params = {}
        for key, value in self.params.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_params[key] = value
            except (TypeError, ValueError):
                serializable_params[key] = str(value)

        # Save parameters to JSON
        with open(filename, "w") as f:
            json.dump(serializable_params, f, indent=4)

        logger.debug(f"Saved parameters to {filename}")
        return filename

    @classmethod
    def load_params(cls, filename: str) -> Dict[str, Any]:
        """Load strategy parameters from a JSON file.

        Args:
            filename: Input JSON filename

        Returns:
            Dictionary of parameters
        """
        with open(filename, "r") as f:
            params = json.load(f)

        logger.debug(f"Loaded parameters from {filename}")
        return params

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the strategy.

        Returns:
            Dictionary containing strategy summary
        """
        return {
            "strategy_name": self.__class__.__name__,
            "parameters": self.params,
            "data_shape": self.data.shape if hasattr(self.data, "shape") else "Unknown",
            "indicators_calculated": list(self.indicators.keys()),
            "signals_generated": list(self.signals.keys()),
            "is_multi_symbol": self.is_multi_symbol,
            "symbols": self.symbols,
        }

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(params={self.params})"

    # Deprecated method for backward compatibility
    def run_backtest(self, portfolio_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """DEPRECATED: Use run_signal_generation() instead.
        
        Portfolio simulation should be handled by the portfolio module.
        """
        logger.warning(
            "run_backtest() is deprecated. Use run_signal_generation() and "
            "pass signals to PortfolioSimulator for backtesting."
        )
        return self.run_signal_generation()
