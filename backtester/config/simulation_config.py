"""
Unified Simulation Configuration System

This module provides standardized configuration for optimization and backtesting
to ensure consistent, realistic results across all operations.

Key Features:
- Single source of truth for simulation settings
- Production-ready default configurations
- Environment-specific overrides (development, testing, production)
- Configuration validation and versioning
- Easy switching between realistic and fast optimization modes
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal
from enum import Enum
import json
from pathlib import Path

from ..portfolio.simulation_engine import SimulationConfig


class SimulationMode(Enum):
    """Simulation modes with different realism vs speed tradeoffs."""
    FAST = "fast"              # Fast optimization, minimal costs
    REALISTIC = "realistic"    # Production-ready settings
    PRODUCTION = "production"  # Full production settings with all costs


class ConfigEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"


@dataclass
class StandardSimulationConfig:
    """
    Standardized simulation configuration used across optimization and backtesting.
    
    This ensures optimization and backtest use identical realistic settings.
    """
    
    # Core settings
    mode: SimulationMode = SimulationMode.REALISTIC
    environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT
    
    # Portfolio settings
    init_cash: float = 100000.0
    
    # Transaction costs (realistic defaults)
    fees: float = 0.001        # 0.1% trading fees (typical for crypto)
    slippage: float = 0.0005   # 0.05% slippage (realistic for liquid pairs)
    fixed_fees: float = 0.0    # Fixed per-trade fees
    
    # Position sizing
    position_size_value: float = 0.95  # 95% of available cash per position
    position_size_type: str = "percent"
    max_position_size: Optional[float] = None
    
    # Risk management
    max_leverage: float = 1.0
    cash_sharing: bool = True
    
    # Advanced features
    use_stops: bool = True              # Include SL/TP in simulation
    stop_validation: bool = True        # Validate stop levels
    allow_partial_fills: bool = True
    
    # Frequency (auto-detected from data)
    freq: Optional[str] = None
    
    # Metadata
    version: str = "1.0"
    description: str = ""
    
    # Additional parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_simulation_config(self) -> SimulationConfig:
        """Convert to portfolio.SimulationConfig for actual simulation."""
        return SimulationConfig(
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            fixed_fees=self.fixed_fees,
            freq=self.freq,
            position_size_value=self.position_size_value,
            max_leverage=self.max_leverage,
            cash_sharing=self.cash_sharing,
            additional_params=self.additional_params.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            'mode': self.mode.value,
            'environment': self.environment.value,
            'init_cash': self.init_cash,
            'fees': self.fees,
            'slippage': self.slippage,
            'fixed_fees': self.fixed_fees,
            'position_size_value': self.position_size_value,
            'position_size_type': self.position_size_type,
            'max_position_size': self.max_position_size,
            'max_leverage': self.max_leverage,
            'cash_sharing': self.cash_sharing,
            'use_stops': self.use_stops,
            'stop_validation': self.stop_validation,
            'allow_partial_fills': self.allow_partial_fills,
            'freq': self.freq,
            'version': self.version,
            'description': self.description,
            'additional_params': self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardSimulationConfig':
        """Create from dictionary."""
        # Convert enum strings back to enums
        if 'mode' in data:
            data['mode'] = SimulationMode(data['mode'])
        if 'environment' in data:
            data['environment'] = ConfigEnvironment(data['environment'])
        
        return cls(**data)


class SimulationConfigManager:
    """
    Manager for simulation configurations with predefined presets.
    """
    
    def __init__(self):
        self._presets = self._create_presets()
    
    def _create_presets(self) -> Dict[str, StandardSimulationConfig]:
        """Create predefined configuration presets."""
        return {
            # Fast optimization - minimal costs for parameter exploration
            'fast_optimization': StandardSimulationConfig(
                mode=SimulationMode.FAST,
                environment=ConfigEnvironment.DEVELOPMENT,
                fees=0.0005,     # Reduced fees for faster optimization
                slippage=0.0,    # No slippage for speed
                description="Fast optimization with minimal transaction costs"
            ),
            
            # Realistic trading - production-like settings for validation
            'realistic_trading': StandardSimulationConfig(
                mode=SimulationMode.REALISTIC,
                environment=ConfigEnvironment.TESTING,
                fees=0.001,      # Realistic trading fees
                slippage=0.0005, # Realistic slippage
                description="Realistic trading simulation with production-like costs"
            ),
            
            # Production trading - full production settings
            'production_trading': StandardSimulationConfig(
                mode=SimulationMode.PRODUCTION,
                environment=ConfigEnvironment.PRODUCTION,
                fees=0.0015,     # Higher fees for conservative estimates
                slippage=0.001,  # Higher slippage for conservative estimates
                fixed_fees=0.0,  # Could add fixed fees per trade
                description="Conservative production settings with full transaction costs"
            ),
            
            # Development testing - balanced for development work
            'development': StandardSimulationConfig(
                mode=SimulationMode.REALISTIC,
                environment=ConfigEnvironment.DEVELOPMENT,
                fees=0.001,
                slippage=0.0005,
                description="Development configuration balancing realism and speed"
            )
        }
    
    def get_config(self, preset_name: str = 'realistic_trading') -> StandardSimulationConfig:
        """Get a configuration preset."""
        if preset_name not in self._presets:
            available = list(self._presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        return self._presets[preset_name]
    
    def get_available_presets(self) -> Dict[str, str]:
        """Get available presets with descriptions."""
        return {
            name: config.description 
            for name, config in self._presets.items()
        }
    
    def create_custom_config(
        self,
        base_preset: str = 'realistic_trading',
        **overrides
    ) -> StandardSimulationConfig:
        """Create custom configuration based on a preset with overrides."""
        base_config = self.get_config(base_preset)
        
        # Apply overrides
        config_dict = base_config.to_dict()
        config_dict.update(overrides)
        
        return StandardSimulationConfig.from_dict(config_dict)
    
    def save_config(self, config: StandardSimulationConfig, name: str, path: Optional[str] = None):
        """Save configuration to file."""
        if path is None:
            config_dir = Path(__file__).parent / "simulation_configs"
            config_dir.mkdir(exist_ok=True)
            path = config_dir / f"{name}.json"
        
        with open(path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def load_config(self, path: str) -> StandardSimulationConfig:
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return StandardSimulationConfig.from_dict(data)


# Global configuration manager instance
config_manager = SimulationConfigManager()


def get_simulation_config(
    preset: str = 'realistic_trading',
    **overrides
) -> StandardSimulationConfig:
    """
    Convenience function to get simulation configuration.
    
    Args:
        preset: Preset name ('fast_optimization', 'realistic_trading', 'production_trading', 'development')
        **overrides: Override specific configuration values
    
    Returns:
        StandardSimulationConfig ready for use in optimization and backtesting
    
    Example:
        # Get realistic trading config
        config = get_simulation_config('realistic_trading')
        
        # Get production config with custom fees
        config = get_simulation_config('production_trading', fees=0.002)
        
        # Get fast optimization config
        config = get_simulation_config('fast_optimization')
    """
    return config_manager.create_custom_config(preset, **overrides)


def get_optimization_config(**overrides) -> StandardSimulationConfig:
    """Get configuration optimized for parameter optimization (fast but realistic)."""
    return get_simulation_config('realistic_trading', **overrides)


def get_backtest_config(**overrides) -> StandardSimulationConfig:
    """Get configuration for backtesting (realistic trading simulation)."""
    return get_simulation_config('realistic_trading', **overrides)


def get_production_config(**overrides) -> StandardSimulationConfig:
    """Get conservative production configuration."""
    return get_simulation_config('production_trading', **overrides)


def validate_config_consistency(
    optimization_config: StandardSimulationConfig,
    backtest_config: StandardSimulationConfig
) -> bool:
    """
    Validate that optimization and backtest configurations are consistent.
    
    Returns True if configurations are compatible for comparison.
    """
    # Key fields that should match for valid comparison
    critical_fields = ['fees', 'slippage', 'init_cash', 'position_size_value', 'use_stops']
    
    for field in critical_fields:
        opt_val = getattr(optimization_config, field)
        bt_val = getattr(backtest_config, field)
        
        if opt_val != bt_val:
            print(f"WARNING: Configuration mismatch in '{field}': optimization={opt_val}, backtest={bt_val}")
            return False
    
    return True


def print_config_summary(config: StandardSimulationConfig):
    """Print a human-readable summary of the configuration."""
    print(f"Simulation Configuration ({config.mode.value.title()}):")
    print(f"  Environment: {config.environment.value}")
    print(f"  Initial Cash: ${config.init_cash:,.0f}")
    print(f"  Trading Fees: {config.fees:.4f} ({config.fees*100:.2f}%)")
    print(f"  Slippage: {config.slippage:.4f} ({config.slippage*100:.3f}%)")
    print(f"  Position Size: {config.position_size_value:.1%} of available cash")
    print(f"  Use Stops: {config.use_stops}")
    print(f"  Total Cost per Trade: {(config.fees + config.slippage)*100:.3f}%")
    if config.description:
        print(f"  Description: {config.description}")