"""
Chart Components Module

This module provides modular, reusable components for building trading charts.
Components are organized by responsibility and implement well-defined interfaces
for maximum flexibility and testability.

Module Structure:
================
- base.py: Interfaces and base classes
- processors.py: Data extraction and processing
- managers.py: State and configuration management
- renderers.py: Visual element rendering (coming soon)
- builders.py: Chart structure and layout (coming soon)

Quick Start:
===========
    from backtester.analysis.chart_components import (
        DataProcessor,
        IndicatorProcessor,
        LegendManager
    )
    
    # Process data
    data_processor = DataProcessor(portfolio, data)
    ohlcv = data_processor.get_ohlcv_data()
    
    # Categorize indicators
    indicator_processor = IndicatorProcessor(indicators, ohlcv)
    categorized = indicator_processor.get_categorized_indicators()
    
    # Manage legends
    legend_mgr = LegendManager()
    if legend_mgr.should_show_legend("Long Entry"):
        # Show in legend

For more details, see the documentation in each module.
"""

# Core interfaces and base classes
from .base import (
    IChartComponent,
    IDataProcessor,
    IRenderer,
    IChartBuilder,
    IManager,
    BaseDataProcessor,
    BaseRenderer,
    ComponentConfig,
    ProcessorConfig,
    RendererConfig,
    ChartType,
    RenderPriority
)

# Data processors
from .processors import (
    DataProcessor,
    IndicatorProcessor,
    IndicatorType
)

# Managers
from .managers import (
    LegendManager,
    LegendConfig,
    LayoutManager,
    LayoutConfig,
    ThemeManager,
    ThemeConfig
)

# Builders
from .builders import ChartBuilder

# Renderers
from .renderers import (
    IndicatorRenderer,
    CandlestickRenderer,
    VolumeRenderer,
    EquityRenderer,
    SignalRenderer
)

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
    # Processors
    'DataProcessor',
    'IndicatorProcessor',
    'IndicatorType',
    # Managers
    'LegendManager',
    'LegendConfig',
    'LayoutManager',
    'LayoutConfig',
    'ThemeManager',
    'ThemeConfig',
    # Builders
    'ChartBuilder',
    # Renderers
    'IndicatorRenderer',
    'CandlestickRenderer',
    'VolumeRenderer',
    'EquityRenderer',
    'SignalRenderer',
    # Configuration
    'ComponentConfig',
    'ProcessorConfig',
    'RendererConfig',
    # Enums
    'ChartType',
    'RenderPriority'
]