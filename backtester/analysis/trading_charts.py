"""
Trading Charts Engine - Modular Architecture

This is the refactored, simplified version of the trading charts engine that uses
the modular components from the chart_components package. The main TradingChartsEngine
class now acts as an orchestrator, delegating specific responsibilities to
specialized components.

Architecture Overview:
====================
┌─────────────────────┐
│ TradingChartsEngine │ (This file - Orchestrator)
└──────┬──────────────┘
       │ coordinates
       ├─► DataProcessor (processors.py) - Extracts and cleans data
       ├─► IndicatorProcessor (processors.py) - Categorizes indicators
       ├─► SignalProcessor (trading_signals.py) - Processes signals
       ├─► ChartBuilder (builders.py) - Creates chart structure
       ├─► Various Renderers (renderers.py) - Add visual elements
       └─► Managers (managers.py) - Handle state and configuration

Benefits of Modular Architecture:
================================
1. Single Responsibility: Each component has one clear purpose
2. Easy Testing: Components can be tested in isolation
3. Extensibility: New chart types can be added easily
4. Maintainability: Smaller files are easier to understand
5. Reusability: Components can be used in different contexts

Usage Example:
=============
    from backtester.analysis.trading_charts import TradingChartsEngine, ChartConfig
    
    # Create engine with all data
    engine = TradingChartsEngine(portfolio, data, indicators, signals)
    
    # Generate main chart
    fig = engine.create_main_chart()
    
    # Save to file
    engine.save_chart(fig, "analysis.html")

Related Modules:
===============
- chart_components/: All modular components
- trading_signals.py: Signal processing (to be refactored next)
- ChartConfig: Configuration dataclass (kept here for compatibility)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from plotly.subplots import make_subplots

# Import all modular components
from backtester.analysis.chart_components import (
    # Processors
    DataProcessor,
    IndicatorProcessor,
    IndicatorType,
    # Builders
    ChartBuilder,
    # Renderers
    IndicatorRenderer,
    CandlestickRenderer,
    VolumeRenderer,
    EquityRenderer,
    SignalRenderer,
    # Managers
    LegendManager,
    LayoutManager,
    ThemeManager,
)

# Import signal processing (now using simplified version)
from backtester.analysis.trading_signals import SignalProcessor, SignalConfig
# SignalRenderer is now part of chart_components (imported above)

from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChartConfig:
    """
    Configuration for chart appearance and behavior.
    
    This is kept in the main file for backward compatibility.
    In future versions, this could be moved to a configs module.
    """
    title: str = "Trading Strategy Analysis"
    height: int = 1000
    show_volume: bool = True
    show_equity: bool = True
    theme: str = "plotly_white"
    date_range: Optional[Tuple[str, str]] = None
    
    # Color scheme for non-signal elements
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'indicators': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            }


class TradingChartsEngine:
    """
    Simplified trading charts engine using modular components.
    
    This class orchestrates the chart creation process by coordinating
    various specialized components. Each component handles a specific
    aspect of chart creation:
    
    - Data processing and cleaning
    - Indicator categorization
    - Signal extraction and processing
    - Chart structure building
    - Visual element rendering
    - State management
    
    The engine itself contains minimal logic, delegating all specific
    tasks to the appropriate components. This makes the code much more
    maintainable and testable.
    """
    
    def __init__(
        self,
        portfolio: vbt.Portfolio,
        data: vbt.Data,
        indicators: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
        chart_config: Optional[ChartConfig] = None,
        signal_config: Optional[SignalConfig] = None
    ):
        """
        Initialize the trading charts engine with modular components.
        
        Args:
            portfolio: VectorBT Portfolio object
            data: VectorBT Data object
            indicators: Dictionary of indicators
            signals: Dictionary of strategy signals
            chart_config: Chart configuration
            signal_config: Signal processing configuration
        """
        self.portfolio = portfolio
        self.data = data
        self.chart_config = chart_config or ChartConfig()
        self.signal_config = signal_config or SignalConfig()
        
        # Initialize managers
        self.legend_manager = LegendManager()
        self.layout_manager = LayoutManager()
        self.theme_manager = ThemeManager()
        
        # Initialize processors
        logger.info("Initializing data processors...")
        self.data_processor = DataProcessor(portfolio, data)
        self.indicator_processor = IndicatorProcessor(
            indicators or {}, 
            self.data_processor.get_ohlcv_data()
        )
        
        # Initialize signal processing
        logger.info("Initializing signal processor...")
        self.signal_processor = SignalProcessor(
            portfolio, 
            self.data_processor, 
            signals, 
            self.signal_config
        )
        
        # Initialize builders
        self.chart_builder = ChartBuilder(self.chart_config, self.layout_manager)
        
        # Initialize renderers
        self.indicator_renderer = IndicatorRenderer(
            self.chart_config, 
            self.legend_manager,
            self.theme_manager
        )
        self.candlestick_renderer = CandlestickRenderer(
            self.chart_config,
            self.theme_manager
        )
        self.volume_renderer = VolumeRenderer(
            self.chart_config,
            self.theme_manager
        )
        self.equity_renderer = EquityRenderer(
            self.chart_config,
            self.legend_manager
        )
        self.signal_renderer = SignalRenderer(
            self.signal_config,
            self.legend_manager
        )
        
        logger.info(f"TradingChartsEngine initialized with {len(indicators or {})} indicators")
        logger.info(f"Signal timing mode: {self.signal_config.signal_timing_mode}")
    
    def create_main_chart(
        self,
        title: Optional[str] = None,
        show_volume: Optional[bool] = None,
        show_equity: Optional[bool] = None,
        show_signals: Optional[bool] = None,
        date_range: Optional[Tuple[str, str]] = None,
        height: Optional[int] = None
    ) -> go.Figure:
        """
        Create the main trading chart with all components.
        
        This method orchestrates the entire chart creation process:
        1. Process and clean data
        2. Build chart structure
        3. Render all visual elements
        4. Apply smart ranges and final configuration
        
        Args:
            title: Chart title
            show_volume: Whether to show volume
            show_equity: Whether to show equity curve
            show_signals: Whether to show signals
            date_range: Optional date range filter
            height: Chart height
            
        Returns:
            Complete Plotly figure ready for display
        """
        # Update configuration with provided parameters
        if title is not None:
            self.chart_config.title = title
        if show_volume is not None:
            self.chart_config.show_volume = show_volume
        if show_equity is not None:
            self.chart_config.show_equity = show_equity
        if show_signals is not None:
            self.signal_config.show_signals = show_signals
        if date_range is not None:
            self.chart_config.date_range = date_range
        if height is not None:
            self.chart_config.height = height
        
        # Reset managers for new chart
        self.legend_manager.reset()
        self.theme_manager.reset()
        
        # Step 1: Get processed data
        logger.info("Processing data...")
        ohlcv = self._apply_date_filter(self.data_processor.get_ohlcv_data())
        
        if len(ohlcv) == 0:
            raise ValueError("No valid OHLCV data available for charting")
        
        # Step 2: Get categorized indicators
        categorized_indicators = self.indicator_processor.get_categorized_indicators()
        n_indicator_subplots = len(categorized_indicators[IndicatorType.SEPARATE_SUBPLOT])
        
        # Step 3: Determine chart structure
        has_volume = self.chart_config.show_volume and 'Volume' in ohlcv.columns
        has_equity = self.chart_config.show_equity
        
        # Step 4: Build chart structure
        logger.info("Building chart structure...")
        fig = self.chart_builder.create_subplot_structure(
            n_indicator_subplots=n_indicator_subplots,
            has_volume=has_volume,
            has_equity=has_equity
        )
        
        # Step 5: Render all components
        logger.info("Rendering chart components...")
        
        # 5a. Render main price chart
        self.candlestick_renderer.render(fig, ohlcv)
        
        # 5b. Render price overlay indicators
        self.indicator_renderer.add_price_overlays(
            fig, 
            categorized_indicators[IndicatorType.PRICE_OVERLAY],
            ohlcv
        )
        
        # 5c. Render signals if enabled
        if self.signal_config.show_signals:
            signals_dict = self.signal_processor.get_signals_dict()
            self.signal_renderer.add_signals_to_chart(fig, signals_dict, ohlcv)
        
        # 5d. Render volume if enabled
        current_row = 2
        if has_volume:
            self.volume_renderer.render(fig, ohlcv, row=current_row)
            current_row += 1
        
        # 5e. Render equity curve if enabled
        portfolio_equity = pd.Series(dtype=float)
        if has_equity:
            portfolio_equity = self.data_processor.get_portfolio_equity()
            if len(portfolio_equity) > 0:
                self.equity_renderer.render(fig, portfolio_equity, row=current_row)
                current_row += 1
        
        # 5f. Render indicator subplots
        self.indicator_renderer.add_subplot_indicators(
            fig,
            categorized_indicators[IndicatorType.SEPARATE_SUBPLOT],
            ohlcv,
            start_row=current_row
        )
        
        # Step 6: Apply smart ranges to prevent scaling issues
        logger.info("Applying smart ranges...")
        total_subplots = 1 + (1 if has_volume else 0) + (1 if has_equity else 0) + n_indicator_subplots
        
        self.chart_builder.apply_smart_ranges(
            fig, ohlcv, self.indicator_processor, portfolio_equity, total_subplots
        )
        
        # Step 7: Finalize chart
        self._finalize_chart_layout(fig)
        
        logger.info("Chart creation complete")
        return fig
    
    def create_strategy_analysis_chart(
        self,
        title: Optional[str] = None,
        date_range: Optional[Tuple[str, str]] = None,
        height: Optional[int] = None
    ) -> go.Figure:
        """
        Create a comprehensive strategy analysis chart with performance metrics.
        
        This chart includes multiple subplots showing:
        - Trade distribution over time
        - Win rate analysis
        - Drawdown visualization
        - Trade duration distribution
        - Returns distribution
        - Monthly performance heatmap
        
        Args:
            title: Chart title
            date_range: Optional date range filter
            height: Chart height (default: 1200)
            
        Returns:
            Plotly figure with strategy analysis
        """
        # Update configuration
        if title is not None:
            self.chart_config.title = title
        else:
            self.chart_config.title = "Strategy Performance Analysis"
        if date_range is not None:
            self.chart_config.date_range = date_range
        if height is not None:
            self.chart_config.height = height
        else:
            self.chart_config.height = 1200
        
        # Get portfolio data
        logger.info("Preparing strategy analysis data...")
        
        # Get trades data
        if not hasattr(self.portfolio, 'trades') or len(self.portfolio.trades.records) == 0:
            logger.warning("No trades found for analysis")
            return self._create_empty_analysis_chart()
        
        trades = self.portfolio.trades.records_readable
        portfolio_value = self.portfolio.value
        
        # Apply date filter if specified
        if self.chart_config.date_range:
            start_date, end_date = self.chart_config.date_range
            mask = (portfolio_value.index >= start_date) & (portfolio_value.index <= end_date)
            portfolio_value = portfolio_value[mask]
            # Filter trades as well
            trades = trades[(trades['Entry Index'] >= start_date) & (trades['Exit Index'] <= end_date)]
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            row_heights=[0.3, 0.25, 0.25, 0.2],
            column_widths=[0.5, 0.5],
            subplot_titles=(
                'Portfolio Equity Curve', 'Trade Distribution & Win Rate',
                'Drawdown Analysis', 'Trade Returns Distribution',
                'Trade Duration Distribution', 'Monthly Returns Heatmap'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "heatmap", "rowspan": 2}],
                [{"secondary_y": False}, None]
            ]
        )
        
        # 1. Portfolio Equity Curve (top left)
        self._add_equity_curve(fig, portfolio_value, row=1, col=1)
        
        # 2. Trade Distribution & Win Rate (top right)
        self._add_trade_distribution(fig, trades, row=1, col=2)
        
        # 3. Drawdown Analysis (middle left)
        self._add_drawdown_analysis(fig, portfolio_value, row=2, col=1)
        
        # 4. Trade Returns Distribution (middle right)
        self._add_returns_distribution(fig, trades, row=2, col=2)
        
        # 5. Trade Duration Distribution (bottom left)
        self._add_duration_distribution(fig, trades, row=3, col=1)
        
        # 6. Monthly Returns Heatmap (bottom right, spans 2 rows)
        self._add_monthly_heatmap(fig, portfolio_value, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=self.chart_config.title,
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            height=self.chart_config.height,
            showlegend=True,
            template=self.chart_config.theme,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        logger.info("Strategy analysis chart creation complete")
        return fig
    
    def save_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html"
    ) -> str:
        """
        Save chart to file with enhanced configuration.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            format: Output format (html, png, jpg, pdf, svg)
            
        Returns:
            Path to saved file
        """
        filepath = Path(filename)
        
        if format.lower() == "html":
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',
                'modeBarButtonsToAdd': ['resetScale2d'],
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': filepath.stem,
                    'height': 1080,
                    'width': 1920,
                    'scale': 2
                }
            }
            
            fig.write_html(
                filepath.with_suffix('.html'),
                config=config,
                include_plotlyjs='cdn',
                div_id="chart",
                full_html=True
            )
        elif format.lower() in ["png", "jpg", "jpeg", "pdf", "svg"]:
            fig.write_image(
                filepath.with_suffix(f'.{format.lower()}'),
                width=1920,
                height=1080,
                scale=2,
                engine="kaleido"
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Chart saved to {filepath}")
        return str(filepath)
    
    def get_chart_info(self) -> Dict[str, Any]:
        """Get information about the chart data for debugging."""
        signals_dict = self.signal_processor.get_signals_dict()
        
        return {
            'ohlcv_shape': self.data_processor.get_ohlcv_data().shape,
            'indicators_count': len(self.indicator_processor.indicators),
            'signals_extracted': {
                'long_entries': signals_dict.get('long_entries', pd.Series()).sum(),
                'short_entries': signals_dict.get('short_entries', pd.Series()).sum(),
                'exits': signals_dict.get('exits', signals_dict.get('long_exits', pd.Series())).sum()
            },
            'portfolio_has_trades': hasattr(self.portfolio, 'trades') and len(getattr(self.portfolio.trades, 'records', [])) > 0,
            'signal_timing_mode': self.signal_config.signal_timing_mode,
            'execution_delay': self.signal_config.execution_delay
        }
    
    def _apply_date_filter(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Apply date range filter if specified."""
        if self.chart_config.date_range:
            start_date, end_date = self.chart_config.date_range
            mask = (ohlcv.index >= start_date) & (ohlcv.index <= end_date)
            return ohlcv[mask]
        return ohlcv
    
    def _finalize_chart_layout(self, fig: go.Figure):
        """Finalize chart layout with timing information."""
        # Add timing suffix to title if signals are shown
        timing_suffix = ""
        if self.signal_config.show_signals:
            timing_suffix = f" ({self.signal_config.signal_timing_mode.title()} Timing)"
        
        full_title = f"{self.chart_config.title}{timing_suffix}"
        
        fig.update_layout(
            title=dict(
                text=full_title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            )
        )
    
    # Helper methods for strategy analysis chart
    
    def _create_empty_analysis_chart(self) -> go.Figure:
        """Create an empty analysis chart when no trades are available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No trades available for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title="Strategy Analysis - No Data",
            height=600,
            template=self.chart_config.theme
        )
        return fig
    
    def _add_equity_curve(self, fig: go.Figure, portfolio_value: pd.Series, row: int, col: int):
        """Add equity curve to the analysis chart."""
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add starting value reference line
        fig.add_hline(
            y=portfolio_value.iloc[0],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${portfolio_value.iloc[0]:,.0f}",
            annotation_position="left",
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Portfolio Value ($)", row=row, col=col)
    
    def _add_trade_distribution(self, fig: go.Figure, trades: pd.DataFrame, row: int, col: int):
        """Add trade distribution and win rate over time."""
        # Calculate cumulative trades and win rate over time
        trades_sorted = trades.sort_values('Exit Index')
        trades_sorted['Trade_Num'] = range(1, len(trades_sorted) + 1)
        trades_sorted['Is_Win'] = trades_sorted['Return'] > 0
        trades_sorted['Cumulative_Wins'] = trades_sorted['Is_Win'].cumsum()
        trades_sorted['Rolling_WinRate'] = trades_sorted['Cumulative_Wins'] / trades_sorted['Trade_Num'] * 100
        
        # Add cumulative trades (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=trades_sorted['Exit Index'],
                y=trades_sorted['Trade_Num'],
                mode='lines',
                name='Cumulative Trades',
                line=dict(color='purple', width=2),
                yaxis='y'
            ),
            row=row, col=col
        )
        
        # Add rolling win rate (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=trades_sorted['Exit Index'],
                y=trades_sorted['Rolling_WinRate'],
                mode='lines',
                name='Win Rate %',
                line=dict(color='green', width=2),
                yaxis='y2'
            ),
            row=row, col=col, secondary_y=True
        )
        
        # Add 50% reference line for win rate
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="gray",
            annotation_text="50%",
            annotation_position="right",
            row=row, col=col,
            secondary_y=True
        )
        
        fig.update_yaxes(title_text="Cumulative Trades", row=row, col=col, secondary_y=False)
        fig.update_yaxes(title_text="Win Rate %", row=row, col=col, secondary_y=True)
    
    def _add_drawdown_analysis(self, fig: go.Figure, portfolio_value: pd.Series, row: int, col: int):
        """Add drawdown analysis visualization."""
        # Calculate drawdown
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Highlight maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd,
            text=f"Max DD: {max_dd:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=20,
            ay=-30,
            row=row, col=col
        )
        
        fig.update_yaxes(title_text="Drawdown %", row=row, col=col)
    
    def _add_returns_distribution(self, fig: go.Figure, trades: pd.DataFrame, row: int, col: int):
        """Add trade returns distribution histogram."""
        returns = trades['Return'].values * 100  # Convert to percentage
        
        # Color wins and losses differently
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Trade Returns',
                nbinsx=30,
                marker_color=colors,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add mean line
        mean_return = returns.mean()
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Mean: {mean_return:.1f}%",
            annotation_position="top",
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Return %", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_duration_distribution(self, fig: go.Figure, trades: pd.DataFrame, row: int, col: int):
        """Add trade duration distribution."""
        # Calculate trade duration in hours or days
        durations = (pd.to_datetime(trades['Exit Index']) - pd.to_datetime(trades['Entry Index'])).dt.total_seconds() / 3600
        
        # Group by duration ranges
        duration_bins = [0, 4, 24, 72, 168, float('inf')]
        duration_labels = ['<4h', '4h-1d', '1-3d', '3-7d', '>7d']
        duration_counts = pd.cut(durations, bins=duration_bins, labels=duration_labels).value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=duration_counts.index,
                y=duration_counts.values,
                name='Trade Duration',
                marker_color='orange',
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Duration", row=row, col=col)
        fig.update_yaxes(title_text="Number of Trades", row=row, col=col)
    
    def _add_monthly_heatmap(self, fig: go.Figure, portfolio_value: pd.Series, row: int, col: int):
        """Add monthly returns heatmap."""
        # Calculate monthly returns
        monthly_returns = portfolio_value.resample('M').last().pct_change() * 100
        monthly_returns = monthly_returns.dropna()
        
        if len(monthly_returns) > 0:
            # Reshape for heatmap (months x years)
            returns_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            # Create pivot table
            heatmap_data = returns_pivot.pivot_table(
                values='Return',
                index='Month',
                columns='Year',
                aggfunc='mean'
            )
            
            # Month names for y-axis
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns,
                    y=[month_names[i-1] for i in heatmap_data.index],
                    colorscale='RdYlGn',
                    zmid=0,
                    text=np.round(heatmap_data.values, 1),
                    texttemplate='%{text}%',
                    textfont={"size": 10},
                    showscale=True,
                    colorbar=dict(title="Return %")
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Month", row=row, col=col)


# Convenience functions for backward compatibility

def create_strategy_analysis(
    portfolio: vbt.Portfolio,
    data: vbt.Data,
    indicators: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    title: str = "Strategy Performance Analysis"
) -> go.Figure:
    """Create a comprehensive strategy analysis chart."""
    charts = TradingChartsEngine(portfolio, data, indicators, signals)
    return charts.create_strategy_analysis_chart(title=title)


def create_trading_analysis(
    portfolio: vbt.Portfolio,
    data: vbt.Data,
    indicators: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    title: str = "Trading Analysis",
    signal_timing_mode: str = "execution"
) -> go.Figure:
    """Create a comprehensive trading analysis chart."""
    signal_config = SignalConfig(signal_timing_mode=signal_timing_mode)
    chart_config = ChartConfig(title=title)
    
    charts = TradingChartsEngine(
        portfolio, data, indicators, signals,
        chart_config=chart_config, signal_config=signal_config
    )
    return charts.create_main_chart()


# Module exports
__all__ = [
    'TradingChartsEngine',
    'ChartConfig',
    'create_strategy_analysis',
    'create_trading_analysis'
]