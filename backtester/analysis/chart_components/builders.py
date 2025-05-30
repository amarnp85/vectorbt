"""
Chart Structure Builders

This module contains components responsible for creating chart structures,
layouts, and subplot arrangements. Builders implement the IChartBuilder
interface defined in base.py.

Module Structure:
================
1. ChartBuilder: Main builder for creating subplot structures and layouts

Integration with Chart System:
=============================
Builders are used by the TradingChartsEngine to create the initial chart
structure before renderers add visual elements. They handle:
- Subplot creation and arrangement
- Height calculations for optimal display
- Axes configuration
- Layout and styling

Chart Building Flow:
===================
TradingChartsEngine → ChartBuilder → Create subplot structure
                                  ↓
                                  Configure axes and layout
                                  ↓
                                  Apply smart ranges
                                  ↓
                                  Return configured figure

Related Modules:
===============
- base.py: Defines IChartBuilder interface
- managers.py: LayoutManager helps calculate subplot heights
- renderers.py: Add visual elements to the structure
- trading_charts.py: Orchestrates the building process
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

from backtester.analysis.chart_components.base import IChartBuilder, ComponentConfig
from backtester.analysis.chart_components.managers import LayoutManager, LayoutConfig
from backtester.analysis.chart_components.processors import IndicatorType
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class ChartBuilder(IChartBuilder):
    """
    Builds chart structure and layout with subplots.
    
    This builder creates the foundational chart structure including:
    - Main price subplot (always present)
    - Optional volume subplot
    - Optional equity curve subplot
    - Indicator subplots as needed
    
    It uses the LayoutManager to calculate optimal heights and spacing
    for all subplots to ensure readability.
    
    Usage:
        config = ChartConfig(title="My Chart", height=1000)
        builder = ChartBuilder(config)
        
        fig = builder.create_subplot_structure(
            n_indicator_subplots=3,
            has_volume=True,
            has_equity=True
        )
        
        builder.apply_smart_ranges(fig, ohlcv_data, ...)
    
    Integration Points:
    - Used by: TradingChartsEngine
    - Uses: LayoutManager for height calculations
    - Provides: Configured Plotly figure ready for rendering
    """
    
    def __init__(self, config: Any, layout_manager: Optional[LayoutManager] = None):
        """
        Initialize the chart builder.
        
        Args:
            config: Chart configuration (ChartConfig from main module)
            layout_manager: Optional layout manager for height calculations
        """
        self.config = config
        self.layout_manager = layout_manager or LayoutManager()
    
    def validate(self) -> bool:
        """Validate builder configuration."""
        return hasattr(self.config, 'title') and hasattr(self.config, 'height')
    
    def build_structure(self, **kwargs) -> go.Figure:
        """
        Build the chart structure (interface method).
        
        This is a wrapper for create_subplot_structure to match interface.
        """
        return self.create_subplot_structure(**kwargs)
    
    def create_subplot_structure(
        self, 
        n_indicator_subplots: int, 
        has_volume: bool, 
        has_equity: bool
    ) -> go.Figure:
        """
        Create the subplot structure for the chart.
        
        Creates a vertical stack of subplots with shared x-axes:
        1. Main price chart (always)
        2. Volume subplot (optional)
        3. Equity curve subplot (optional)
        4. Indicator subplots (0 or more)
        
        Args:
            n_indicator_subplots: Number of separate indicator subplots needed
            has_volume: Whether to include volume subplot
            has_equity: Whether to include equity curve subplot
            
        Returns:
            Configured Plotly figure with subplot structure
        """
        # Calculate total subplots
        n_subplots = 1  # Main price chart
        if has_volume:
            n_subplots += 1
        if has_equity:
            n_subplots += 1
        n_subplots += n_indicator_subplots
        
        # Create subplot titles
        subplot_titles = self._create_subplot_titles(
            has_volume, has_equity, n_indicator_subplots
        )
        
        # Calculate row heights using layout manager
        row_heights = self.layout_manager.calculate_subplot_heights(
            has_volume, has_equity, n_indicator_subplots
        )
        
        # Calculate vertical spacing
        vertical_spacing = self.layout_manager.calculate_vertical_spacing(n_subplots)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,  # All subplots share x-axis
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            specs=[[{"secondary_y": False}] for _ in range(n_subplots)]
        )
        
        # Apply initial layout configuration
        self._configure_layout(fig, n_subplots)
        
        logger.debug(f"Created chart structure with {n_subplots} subplots")
        return fig
    
    def _create_subplot_titles(
        self, 
        has_volume: bool, 
        has_equity: bool, 
        n_indicators: int
    ) -> List[str]:
        """Create titles for each subplot."""
        titles = ["Price & Indicators"]
        
        if has_volume:
            titles.append("Volume")
        if has_equity:
            titles.append("Equity Curve")
            
        # Indicator subplots get generic titles (will be updated by renderers)
        titles.extend([f"Indicator {i+1}" for i in range(n_indicators)])
        
        return titles
    
    def _configure_layout(self, fig: go.Figure, n_subplots: int):
        """
        Configure chart layout and interactivity.
        
        Sets up:
        - Title and general appearance
        - Legend configuration
        - Margins and spacing
        - Interactive features (zoom, pan, hover)
        - Axes configuration
        """
        # Get layout margins from manager
        margins = self.layout_manager.get_margin_dict()
        
        fig.update_layout(
            title=dict(
                text=self.config.title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=self.config.height,
            showlegend=True,
            template=self.config.theme,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)"
            ),
            margin=margins,
            dragmode='zoom',        # Default to zoom mode
            hovermode='x unified',  # Unified hover across subplots
            autosize=True,
            font=dict(size=12)
        )
        
        # Configure all axes
        for i in range(1, n_subplots + 1):
            # X-axis configuration
            fig.update_xaxes(
                rangeslider_visible=False,  # No range slider
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                type='date',
                tickformat='%Y-%m-%d',
                autorange=False,   # Will be set by apply_smart_ranges
                fixedrange=False,  # Allow user zooming
                row=i, col=1
            )
            
            # Y-axis configuration
            fig.update_yaxes(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                fixedrange=False,
                autorange=False,   # Will be set by apply_smart_ranges
                row=i, col=1
            )
        
        # Label the last x-axis
        fig.update_xaxes(title_text="Date", row=n_subplots, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
    
    def apply_layout(self, fig: go.Figure, config: Any) -> None:
        """
        Apply layout configuration to figure (interface method).
        
        Args:
            fig: Plotly figure to configure
            config: Layout configuration
        """
        # This is handled in _configure_layout during creation
        pass
    
    def apply_smart_ranges(
        self, 
        fig: go.Figure, 
        ohlcv_data: pd.DataFrame,
        indicators: Dict[str, Any],
        portfolio_equity: pd.Series,
        n_subplots: int
    ):
        """
        Apply smart range settings to prevent chart scaling issues.
        
        This method sets appropriate ranges for all axes based on the data
        to prevent common issues like:
        - Chart zooming to 0,0
        - Indicators with different scales overlapping
        - Volume bars dominating the chart
        
        Args:
            fig: Plotly figure to update
            ohlcv_data: OHLCV data for price ranges
            indicators: Processed indicators (from IndicatorProcessor)
            portfolio_equity: Portfolio equity series
            n_subplots: Total number of subplots
        """
        # 1. Set X-axis range (dates) for all subplots
        if len(ohlcv_data) > 0:
            date_range = self._calculate_date_range(ohlcv_data)
            for i in range(1, n_subplots + 1):
                fig.update_xaxes(range=date_range, row=i, col=1)
        
        # 2. Set Y-axis ranges for each subplot
        current_row = 1
        
        # Price chart range
        if len(ohlcv_data) > 0:
            price_range = self._calculate_price_range(ohlcv_data)
            fig.update_yaxes(range=price_range, row=current_row, col=1)
            current_row += 1
        
        # Volume chart range
        if 'Volume' in ohlcv_data.columns:
            volume_range = self._calculate_volume_range(ohlcv_data['Volume'])
            if volume_range:
                fig.update_yaxes(range=volume_range, row=current_row, col=1)
                current_row += 1
        
        # Equity chart range
        if len(portfolio_equity) > 0:
            equity_range = self._calculate_series_range(portfolio_equity, padding_pct=0.05)
            if equity_range:
                fig.update_yaxes(range=equity_range, row=current_row, col=1)
                current_row += 1
        
        # Indicator subplot ranges
        if hasattr(indicators, 'processed'):
            subplot_indicators = indicators.processed[IndicatorType.SEPARATE_SUBPLOT]
            for name, indicator in subplot_indicators.items():
                if len(indicator) > 0:
                    indicator_range = self._calculate_series_range(indicator, padding_pct=0.1)
                    if indicator_range:
                        fig.update_yaxes(range=indicator_range, row=current_row, col=1)
                        current_row += 1
        
        logger.info("Applied smart ranges to prevent chart scaling issues")
    
    def _calculate_date_range(self, data: pd.DataFrame) -> List:
        """Calculate x-axis date range with padding."""
        date_range = [data.index[0], data.index[-1]]
        
        # Add 2% padding
        date_span = (date_range[1] - date_range[0]).total_seconds()
        padding = pd.Timedelta(seconds=date_span * 0.02)
        
        return [date_range[0] - padding, date_range[1] + padding]
    
    def _calculate_price_range(self, data: pd.DataFrame) -> List[float]:
        """Calculate price y-axis range."""
        price_min = min(data['Low'].min(), data['Close'].min())
        price_max = max(data['High'].max(), data['Close'].max())
        
        # Add 5% padding
        price_range = price_max - price_min
        padding = price_range * 0.05
        
        return [price_min - padding, price_max + padding]
    
    def _calculate_volume_range(self, volume: pd.Series) -> Optional[List[float]]:
        """Calculate volume y-axis range."""
        if len(volume) == 0 or volume.max() == 0:
            return None
        
        volume_max = volume.max()
        volume_min = max(0, volume.min())
        
        # Add 10% padding at top only
        padding = (volume_max - volume_min) * 0.1
        
        return [volume_min, volume_max + padding]
    
    def _calculate_series_range(self, series: pd.Series, padding_pct: float = 0.05) -> Optional[List[float]]:
        """Calculate y-axis range for a generic series."""
        if len(series) == 0:
            return None
        
        series_min = series.min()
        series_max = series.max()
        
        if series_max <= series_min:
            return None
        
        # Add specified padding
        series_range = series_max - series_min
        padding = series_range * padding_pct
        
        return [series_min - padding, series_max + padding]


# Module exports
__all__ = ['ChartBuilder']