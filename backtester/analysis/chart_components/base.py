"""
Base Classes and Interfaces for Chart Components

This module provides the foundation for all chart components in the modular architecture.
It defines abstract base classes and interfaces that ensure consistency and extensibility
across different chart types and rendering strategies.

Architecture Overview:
====================
The chart system is built on several key abstractions:

1. IChartComponent: Base interface for all chart components
2. IDataProcessor: Interface for data extraction and preparation
3. IRenderer: Interface for rendering visual elements
4. IChartBuilder: Interface for constructing chart layouts
5. IManager: Interface for managing chart state and configuration

Component Relationships:
=======================
┌─────────────────────┐
│ TradingChartsEngine │ (Orchestrator in trading_charts.py)
└──────┬──────────────┘
       │ uses
       ├─► IDataProcessor (processors.py)
       │     └─► DataProcessor: Extracts OHLCV data from VBT objects
       │     └─► IndicatorProcessor: Categorizes and processes indicators
       │
       ├─► IChartBuilder (builders.py)
       │     └─► ChartBuilder: Creates subplot structure and layout
       │
       ├─► IRenderer (renderers.py)
       │     └─► IndicatorRenderer: Renders indicators on charts
       │     └─► SignalRenderer: Renders trading signals
       │
       └─► IManager (managers.py)
             └─► LegendManager: Manages legend configuration
             └─► LayoutManager: Manages chart layout (future)

Usage Example:
=============
    from backtester.analysis.chart_components.base import IDataProcessor
    from backtester.analysis.chart_components.processors import DataProcessor
    
    # All components implement consistent interfaces
    processor: IDataProcessor = DataProcessor(portfolio, data)
    ohlcv_data = processor.get_ohlcv_data()

Key Design Principles:
=====================
1. Interface Segregation: Each interface has a single, clear purpose
2. Dependency Inversion: High-level modules depend on abstractions, not concrete classes
3. Open/Closed: Easy to extend with new implementations without modifying existing code
4. Strategy Pattern: Different rendering and processing strategies can be swapped
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from dataclasses import dataclass
from enum import Enum


class IChartComponent(ABC):
    """
    Base interface for all chart components.
    
    This interface ensures all components have consistent initialization
    and configuration patterns.
    
    Implemented by: All chart components
    Used by: TradingChartsEngine and factory classes
    """
    
    @abstractmethod
    def __init__(self, config: Optional[Any] = None):
        """Initialize component with optional configuration."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate component state and configuration.
        
        Returns:
            bool: True if component is properly configured and ready to use
        """
        pass


class IDataProcessor(IChartComponent):
    """
    Interface for data extraction and processing components.
    
    Data processors are responsible for extracting and preparing data
    from VectorBT objects for chart rendering. They handle data validation,
    cleaning, and transformation.
    
    Implemented by: DataProcessor, IndicatorProcessor (processors.py)
    Used by: Chart builders and renderers that need processed data
    
    Workflow:
    1. Extract raw data from VBT objects
    2. Clean and validate data (remove NaN, inf, outliers)
    3. Transform data for chart compatibility
    4. Provide clean data to renderers
    """
    
    @abstractmethod
    def process(self) -> Any:
        """
        Process raw data into chart-ready format.
        
        Returns:
            Processed data ready for chart rendering
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the processed data.
        
        Returns:
            Dictionary containing data statistics and metadata
        """
        pass


class IRenderer(IChartComponent):
    """
    Interface for chart rendering components.
    
    Renderers are responsible for adding visual elements to Plotly figures.
    Each renderer handles a specific type of visual element (indicators,
    signals, volume bars, etc.).
    
    Implemented by: IndicatorRenderer, SignalRenderer (renderers.py)
    Used by: Chart builders to add visual elements
    
    Rendering Flow:
    1. Receive processed data and figure reference
    2. Apply visual styling based on configuration
    3. Add traces to appropriate subplot
    4. Manage legend entries to avoid duplication
    """
    
    @abstractmethod
    def render(self, fig: go.Figure, data: Any, row: int = 1, col: int = 1) -> None:
        """
        Render visual elements on the provided figure.
        
        Args:
            fig: Plotly figure to render on
            data: Data to render (format depends on renderer type)
            row: Subplot row to render on
            col: Subplot column to render on
        """
        pass
    
    @abstractmethod
    def get_required_data_type(self) -> type:
        """
        Get the type of data this renderer requires.
        
        Returns:
            Type of data object required for rendering
        """
        pass


class IChartBuilder(IChartComponent):
    """
    Interface for chart structure builders.
    
    Chart builders create the overall structure and layout of charts,
    including subplots, axes configuration, and general appearance.
    
    Implemented by: ChartBuilder (builders.py)
    Used by: TradingChartsEngine to create chart structure
    
    Building Process:
    1. Determine required subplots based on data and configuration
    2. Calculate optimal subplot heights
    3. Create figure with appropriate layout
    4. Configure axes and interactivity
    5. Return configured figure ready for rendering
    """
    
    @abstractmethod
    def build_structure(self, **kwargs) -> go.Figure:
        """
        Build the chart structure with subplots.
        
        Args:
            **kwargs: Configuration parameters for structure
            
        Returns:
            Configured Plotly figure with subplot structure
        """
        pass
    
    @abstractmethod
    def apply_layout(self, fig: go.Figure, config: Any) -> None:
        """
        Apply layout configuration to the figure.
        
        Args:
            fig: Plotly figure to configure
            config: Layout configuration
        """
        pass


class IManager(IChartComponent):
    """
    Interface for chart state and configuration managers.
    
    Managers handle stateful operations and maintain consistency
    across chart components. Examples include legend deduplication,
    theme management, and layout coordination.
    
    Implemented by: LegendManager (managers.py)
    Used by: Renderers and builders for state management
    
    Management Tasks:
    1. Track rendered elements to avoid duplication
    2. Coordinate configuration across components
    3. Maintain chart state during rendering
    4. Provide consistent styling and behavior
    """
    
    @abstractmethod
    def reset(self) -> None:
        """Reset manager state to initial conditions."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current manager state.
        
        Returns:
            Dictionary containing current state information
        """
        pass


# Concrete base classes that provide common functionality

class BaseDataProcessor(IDataProcessor):
    """
    Base class for data processors with common functionality.
    
    This class provides shared functionality for all data processors,
    including data validation, error handling, and logging.
    
    Subclasses should override:
    - _extract_data(): Extract raw data from source
    - _validate_data(): Validate extracted data
    - _transform_data(): Transform data for charts
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize with optional configuration."""
        self.config = config
        self._processed_data = None
        self._data_info = {}
    
    def validate(self) -> bool:
        """Validate processor configuration."""
        return True
    
    def process(self) -> Any:
        """
        Template method for data processing.
        
        Follows the template pattern:
        1. Extract raw data
        2. Validate data
        3. Transform data
        4. Cache results
        """
        if self._processed_data is None:
            raw_data = self._extract_data()
            
            if not self._validate_data(raw_data):
                raise ValueError("Data validation failed")
            
            self._processed_data = self._transform_data(raw_data)
            self._data_info = self._calculate_data_info(self._processed_data)
        
        return self._processed_data
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about processed data."""
        if self._processed_data is None:
            self.process()
        return self._data_info
    
    @abstractmethod
    def _extract_data(self) -> Any:
        """Extract raw data from source."""
        pass
    
    @abstractmethod
    def _validate_data(self, data: Any) -> bool:
        """Validate extracted data."""
        pass
    
    @abstractmethod
    def _transform_data(self, data: Any) -> Any:
        """Transform data for chart compatibility."""
        pass
    
    def _calculate_data_info(self, data: Any) -> Dict[str, Any]:
        """Calculate data statistics and metadata."""
        return {"processed": True}


class BaseRenderer(IRenderer):
    """
    Base class for renderers with common functionality.
    
    Provides shared functionality for all renderers including
    style management, legend handling, and error handling.
    
    Subclasses should override:
    - _prepare_render_data(): Prepare data for rendering
    - _create_traces(): Create Plotly traces
    - _apply_styling(): Apply visual styling
    """
    
    def __init__(self, config: Optional[Any] = None, legend_manager: Optional[Any] = None):
        """Initialize with configuration and optional legend manager."""
        self.config = config
        self.legend_manager = legend_manager
    
    def validate(self) -> bool:
        """Validate renderer configuration."""
        return self.config is not None
    
    def render(self, fig: go.Figure, data: Any, row: int = 1, col: int = 1) -> None:
        """
        Template method for rendering.
        
        Follows the template pattern:
        1. Prepare data for rendering
        2. Create traces
        3. Apply styling
        4. Add to figure
        """
        render_data = self._prepare_render_data(data)
        traces = self._create_traces(render_data)
        
        for trace in traces:
            self._apply_styling(trace)
            
            # Handle legend deduplication if manager provided
            if self.legend_manager and hasattr(trace, 'name'):
                trace.showlegend = self.legend_manager.should_show_legend(trace.name)
            
            fig.add_trace(trace, row=row, col=col)
    
    @abstractmethod
    def _prepare_render_data(self, data: Any) -> Any:
        """Prepare data for rendering."""
        pass
    
    @abstractmethod
    def _create_traces(self, data: Any) -> List[go.Scatter]:
        """Create Plotly traces from data."""
        pass
    
    def _apply_styling(self, trace: go.Scatter) -> None:
        """Apply visual styling to trace."""
        # Default implementation - can be overridden
        pass
    
    def get_required_data_type(self) -> type:
        """Get required data type - default to Any."""
        return Any


# Configuration classes that work with the interfaces

@dataclass
class ComponentConfig:
    """Base configuration for all components."""
    name: str = "default"
    enabled: bool = True
    debug: bool = False


@dataclass
class ProcessorConfig(ComponentConfig):
    """Configuration for data processors."""
    clean_data: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 10.0  # IQR multiplier


@dataclass
class RendererConfig(ComponentConfig):
    """Configuration for renderers."""
    show_legend: bool = True
    opacity: float = 0.8
    line_width: int = 2


# Enums for consistent type definitions

class ChartType(Enum):
    """Types of charts available in the system."""
    MAIN = "main"           # Full trading chart with all components
    SIMPLE = "simple"       # Simple candlestick chart
    PERFORMANCE = "perf"    # Performance analysis chart
    SIGNAL = "signal"       # Signal-focused chart


class RenderPriority(Enum):
    """Rendering priority for layering visual elements."""
    BACKGROUND = 0   # Volume bars, grid lines
    DATA = 10        # OHLCV candles
    INDICATORS = 20  # Technical indicators
    SIGNALS = 30     # Trading signals
    OVERLAYS = 40    # Special overlays


# Module exports - these are the public interfaces
__all__ = [
    # Interfaces
    'IChartComponent',
    'IDataProcessor',
    'IRenderer', 
    'IChartBuilder',
    'IManager',
    # Base classes
    'BaseDataProcessor',
    'BaseRenderer',
    # Configuration
    'ComponentConfig',
    'ProcessorConfig',
    'RendererConfig',
    # Enums
    'ChartType',
    'RenderPriority'
]