"""
Chart Visual Element Renderers

This module contains components responsible for rendering visual elements
on trading charts. Renderers implement the IRenderer interface defined
in base.py.

Module Structure:
================
1. IndicatorRenderer: Renders technical indicators on appropriate subplots
2. CandlestickRenderer: Renders OHLC candlestick charts
3. VolumeRenderer: Renders volume bars
4. EquityRenderer: Renders portfolio equity curves

Integration with Chart System:
=============================
Renderers receive processed data from processors and add visual elements
to the chart structure created by builders. Each renderer handles a
specific type of visual element.

Rendering Flow:
==============
Processed Data → Renderer → Create Plotly traces
                         ↓
                         Apply styling
                         ↓
                         Manage legends
                         ↓
                         Add to figure

Related Modules:
===============
- base.py: Defines IRenderer interface and BaseRenderer
- processors.py: Provides processed data for rendering
- managers.py: LegendManager prevents duplicate legends
- builders.py: Creates the chart structure to render on
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List

from backtester.analysis.chart_components.base import BaseRenderer, RendererConfig
from backtester.analysis.chart_components.processors import IndicatorType
from backtester.analysis.chart_components.managers import ThemeManager
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class IndicatorRenderer(BaseRenderer):
    """
    Renders technical indicators on trading charts.
    
    This renderer handles all types of indicators:
    - Price overlays (MAs, Bollinger Bands)
    - Volume overlays (OBV, Volume MAs)
    - Separate subplot indicators (RSI, MACD)
    
    It uses the categorization from IndicatorProcessor to place
    indicators in the correct location and applies appropriate styling.
    
    Usage:
        renderer = IndicatorRenderer(config, legend_manager)
        
        # Render price overlays
        renderer.add_price_overlays(fig, price_indicators, ohlcv_data)
        
        # Render subplot indicators
        renderer.add_subplot_indicators(fig, subplot_indicators, ohlcv_data, start_row=3)
    
    Integration Points:
    - Used by: TradingChartsEngine
    - Requires: Categorized indicators from IndicatorProcessor
    - Uses: LegendManager for deduplication, ThemeManager for colors
    """
    
    def __init__(self, config: Any, legend_manager: Any, theme_manager: Optional[ThemeManager] = None):
        """
        Initialize the indicator renderer.
        
        Args:
            config: Chart configuration
            legend_manager: Legend manager for deduplication
            theme_manager: Optional theme manager for colors
        """
        super().__init__(config, legend_manager)
        self.theme_manager = theme_manager or ThemeManager()
        
        # Important indicators to prioritize in legend
        self.important_indicators = [
            'fast_ma', 'slow_ma', 'sma_fast', 'sma_slow', 
            'ema_fast', 'ema_slow', 'signal', 'macd'
        ]
    
    def add_price_overlays(
        self, 
        fig: go.Figure, 
        indicators: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ):
        """
        Add price overlay indicators to the main chart.
        
        These indicators overlay directly on the price chart and use
        the same scale as price data.
        
        Args:
            fig: Plotly figure
            indicators: Dictionary of indicator name -> Series
            ohlcv_data: OHLCV data for alignment
            row: Subplot row (default 1 for main chart)
            col: Subplot column
        """
        if not indicators:
            return
        
        for i, (name, indicator) in enumerate(indicators.items()):
            # Align indicator with OHLCV index
            aligned_indicator = indicator.reindex(ohlcv_data.index).dropna()
            
            if len(aligned_indicator) == 0:
                continue
            
            # Get color from theme manager
            color = self.theme_manager.get_indicator_color(i)
            
            # Check if this is an important indicator for legend
            is_important = any(imp in name.lower() for imp in self.important_indicators)
            show_legend = is_important and self.legend_manager.should_show_legend(name)
            
            # Create trace
            fig.add_trace(
                go.Scatter(
                    x=aligned_indicator.index,
                    y=aligned_indicator.values,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    opacity=0.8,
                    showlegend=show_legend,
                    hovertemplate=f'{name}: %{{y:.4f}}<br>Date: %{{x}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            logger.debug(f"Added price overlay: {name}")
    
    def add_subplot_indicators(
        self,
        fig: go.Figure,
        indicators: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame,
        start_row: int
    ):
        """
        Add indicators that need separate subplots.
        
        Each indicator gets its own subplot starting from start_row.
        Reference lines are added for specific indicators (RSI, MACD).
        
        Args:
            fig: Plotly figure
            indicators: Dictionary of indicator name -> Series
            ohlcv_data: OHLCV data for alignment
            start_row: First row to use for indicators
        """
        if not indicators:
            return
        
        for i, (name, indicator) in enumerate(indicators.items()):
            # Align indicator with OHLCV index
            aligned_indicator = indicator.reindex(ohlcv_data.index).dropna()
            
            if len(aligned_indicator) == 0:
                continue
            
            row = start_row + i
            
            # Get color from theme manager
            color = self.theme_manager.get_indicator_color()
            
            # Add main indicator trace
            fig.add_trace(
                go.Scatter(
                    x=aligned_indicator.index,
                    y=aligned_indicator.values,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    showlegend=False,  # Don't clutter main legend
                    connectgaps=True,
                    hovertemplate=f'{name}: %{{y:.4f}}<br>Date: %{{x}}<extra></extra>'
                ),
                row=row, col=1
            )
            
            # Update subplot title
            fig.update_yaxes(title_text=name, row=row, col=1)
            
            # Add reference lines for specific indicators
            self._add_reference_lines(fig, name, row)
            
            logger.debug(f"Added subplot indicator: {name} at row {row}")
    
    def _add_reference_lines(self, fig: go.Figure, indicator_name: str, row: int):
        """
        Add reference lines for specific indicators.
        
        For example:
        - RSI: 30, 50, 70 levels
        - MACD: Zero line
        - Stochastic: 20, 80 levels
        """
        name_lower = indicator_name.lower()
        
        if 'rsi' in name_lower:
            # RSI reference levels
            for level, color in [(30, 'green'), (70, 'red'), (50, 'gray')]:
                fig.add_hline(
                    y=level, 
                    line_dash='dash',
                    line_color=color,
                    opacity=0.5,
                    row=row, col=1
                )
        
        elif 'macd' in name_lower:
            # MACD zero line
            fig.add_hline(
                y=0,
                line_dash='solid',
                line_color='gray',
                opacity=0.5,
                row=row, col=1
            )
        
        elif 'stoch' in name_lower:
            # Stochastic levels
            for level, color in [(20, 'green'), (80, 'red')]:
                fig.add_hline(
                    y=level,
                    line_dash='dash',
                    line_color=color,
                    opacity=0.5,
                    row=row, col=1
                )
    
    def _prepare_render_data(self, data: Any) -> Any:
        """Prepare data for rendering (BaseRenderer interface)."""
        return data
    
    def _create_traces(self, data: Any) -> List[go.Scatter]:
        """Create traces (BaseRenderer interface)."""
        # This renderer uses specific methods instead
        return []
    
    def get_required_data_type(self) -> type:
        """Get required data type."""
        return dict


class CandlestickRenderer(BaseRenderer):
    """
    Renders OHLC candlestick charts.
    
    Creates candlestick visualization for price data with proper
    coloring for bullish/bearish candles.
    
    Usage:
        renderer = CandlestickRenderer(config)
        renderer.render(fig, ohlcv_data)
    """
    
    def __init__(self, config: Any, theme_manager: Optional[ThemeManager] = None):
        """Initialize candlestick renderer."""
        super().__init__(config)
        self.theme_manager = theme_manager or ThemeManager()
    
    def render(self, fig: go.Figure, data: pd.DataFrame, row: int = 1, col: int = 1):
        """
        Render candlestick chart.
        
        Args:
            fig: Plotly figure
            data: OHLCV DataFrame
            row: Subplot row
            col: Subplot column
        """
        if data.empty:
            return
        
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC',
                increasing_line_color=self.theme_manager.config.bullish_color,
                decreasing_line_color=self.theme_manager.config.bearish_color,
                increasing_fillcolor=self.theme_manager.config.bullish_color,
                decreasing_fillcolor=self.theme_manager.config.bearish_color,
                line=dict(width=1),
                showlegend=False  # Don't show in legend
            ),
            row=row, col=col
        )
    
    def _prepare_render_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data."""
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        return data
    
    def _create_traces(self, data: pd.DataFrame) -> List[go.Candlestick]:
        """Create candlestick trace."""
        return []  # Handled in render method
    
    def get_required_data_type(self) -> type:
        """Get required data type."""
        return pd.DataFrame


class VolumeRenderer(BaseRenderer):
    """
    Renders volume bar charts.
    
    Creates volume bars colored by price movement (green for up, red for down).
    
    Usage:
        renderer = VolumeRenderer(config, theme_manager)
        renderer.render(fig, ohlcv_data, row=2)
    """
    
    def __init__(self, config: Any, theme_manager: Optional[ThemeManager] = None):
        """Initialize volume renderer."""
        super().__init__(config)
        self.theme_manager = theme_manager or ThemeManager()
    
    def render(self, fig: go.Figure, data: pd.DataFrame, row: int, col: int = 1):
        """
        Render volume bars.
        
        Args:
            fig: Plotly figure
            data: OHLCV DataFrame with Volume column
            row: Subplot row for volume
            col: Subplot column
        """
        if 'Volume' not in data.columns or data.empty:
            return
        
        # Determine bar colors based on price movement
        is_bullish = data['Close'] >= data['Open']
        volume_colors = self.theme_manager.get_volume_colors(is_bullish)
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                showlegend=False,
                hovertemplate='Volume: %{y:,.0f}<br>Date: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update y-axis title
        fig.update_yaxes(title_text="Volume", row=row, col=col)
    
    def _prepare_render_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate volume data."""
        if 'Volume' not in data.columns:
            raise ValueError("Volume column not found in data")
        return data
    
    def _create_traces(self, data: pd.DataFrame) -> List[go.Bar]:
        """Create volume bar trace."""
        return []  # Handled in render method
    
    def get_required_data_type(self) -> type:
        """Get required data type."""
        return pd.DataFrame


class EquityRenderer(BaseRenderer):
    """
    Renders portfolio equity curves with drawdown.
    
    Creates a dual-axis plot showing:
    - Portfolio value over time (primary axis)
    - Drawdown percentage (secondary axis)
    
    Usage:
        renderer = EquityRenderer(config, legend_manager)
        renderer.render(fig, equity_series, row=3)
    """
    
    def __init__(self, config: Any, legend_manager: Any):
        """Initialize equity renderer."""
        super().__init__(config, legend_manager)
    
    def render(self, fig: go.Figure, data: pd.Series, row: int, col: int = 1):
        """
        Render equity curve with drawdown.
        
        Args:
            fig: Plotly figure
            data: Portfolio equity Series
            row: Subplot row for equity
            col: Subplot column
        """
        if data.empty:
            return
        
        # Main equity curve
        show_legend = self.legend_manager.should_show_legend('Portfolio Value')
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                showlegend=show_legend,
                connectgaps=True,
                hovertemplate='Portfolio: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Calculate and add drawdown
        running_max = data.expanding().max()
        drawdown = (data - running_max) / running_max * 100
        
        if drawdown.min() < -0.01:  # Only show if meaningful drawdown
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    showlegend=False,
                    connectgaps=True,
                    hovertemplate='Drawdown: %{y:.2f}%<br>Date: %{x}<extra></extra>',
                    yaxis='y2'  # Secondary axis
                ),
                row=row, col=col
            )
            
            # Configure secondary y-axis for drawdown
            # Note: This needs special handling in the main chart engine
            
        # Update y-axis title
        fig.update_yaxes(title_text="Portfolio Value", row=row, col=col)
    
    def _prepare_render_data(self, data: pd.Series) -> pd.Series:
        """Validate equity data."""
        if data.empty:
            raise ValueError("Equity data is empty")
        return data
    
    def _create_traces(self, data: pd.Series) -> List[go.Scatter]:
        """Create equity traces."""
        return []  # Handled in render method
    
    def get_required_data_type(self) -> type:
        """Get required data type."""
        return pd.Series


class SignalRenderer(BaseRenderer):
    """
    Renders trading signals on charts with proper styling and legend management.
    
    This renderer handles the visualization of trading signals including:
    - Entry signals (long/short)
    - Exit signals
    - Stop loss and take profit levels
    - Proper timing display based on configuration
    
    The renderer integrates with the timing system to display signals at
    the correct time (signal generation vs execution time) and provides
    comprehensive stop level visualization.
    
    Usage:
        renderer = SignalRenderer(signal_config, legend_manager)
        renderer.add_signals_to_chart(fig, signals_dict, ohlcv_data)
    """
    
    def __init__(self, config: Any, legend_manager: Any):
        """
        Initialize signal renderer.
        
        Args:
            config: Signal configuration (SignalConfig)
            legend_manager: Legend manager for deduplication
        """
        super().__init__(config, legend_manager)
        
        # Define signal styles
        self.signal_styles = {
            'long_entry': {
                'symbol': 'triangle-up',
                'size': 18,
                'color': config.colors['long_entry'],
                'line_color': 'darkgreen',
                'name': 'Long Entry'
            },
            'short_entry': {
                'symbol': 'triangle-down',
                'size': 18,
                'color': config.colors['short_entry'],
                'line_color': 'darkred',
                'name': 'Short Entry'
            },
            'exit': {
                'symbol': 'square',
                'size': 12,
                'color': config.colors['exit'],
                'line_color': 'darkviolet',
                'name': 'Exit'
            }
        }
    
    def add_signals_to_chart(
        self,
        fig: go.Figure,
        signals: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ):
        """
        Add all signal types to the chart.
        
        This is the main entry point for signal rendering.
        
        Args:
            fig: Plotly figure
            signals: Dictionary of signal series
            ohlcv_data: OHLCV data for alignment
            row: Subplot row
            col: Subplot column
        """
        if not self.config.show_signals:
            logger.debug("Signal rendering disabled in configuration")
            return
        
        # Log signal summary
        logger.info("=== SIGNAL RENDERING SUMMARY ===")
        logger.info(f"Long entries: {signals.get('long_entries', pd.Series()).sum()}")
        logger.info(f"Short entries: {signals.get('short_entries', pd.Series()).sum()}")
        logger.info(f"Exits: {signals.get('exits', pd.Series()).sum()}")
        
        # Analyze stop level availability
        sl_price_count = (~signals.get('sl_price_levels', pd.Series()).isna()).sum() if 'sl_price_levels' in signals else 0
        tp_price_count = (~signals.get('tp_price_levels', pd.Series()).isna()).sum() if 'tp_price_levels' in signals else 0
        sl_pct_count = (~signals.get('sl_levels', pd.Series()).isna()).sum() if 'sl_levels' in signals else 0
        tp_pct_count = (~signals.get('tp_levels', pd.Series()).isna()).sum() if 'tp_levels' in signals else 0
        
        logger.info(f"Stop levels - SL prices: {sl_price_count}, TP prices: {tp_price_count}")
        logger.info(f"Stop levels - SL %: {sl_pct_count}, TP %: {tp_pct_count}")
        
        # Validate signals before rendering
        self._validate_rendering_signals(signals, ohlcv_data)
        
        # Signal mappings for rendering
        signal_mappings = [
            ('long_entries', 'entry_prices', 'long_entry'),
            ('short_entries', 'entry_prices', 'short_entry'),
            ('exits', 'exit_prices', 'exit')
        ]
        
        # Add execution signals
        for signal_key, price_key, style_key in signal_mappings:
            self._add_signal_type(fig, signals, ohlcv_data, signal_key, price_key, style_key, row, col)
        
        # Add stop level indicators
        if self.config.show_stop_levels:
            if sl_price_count > 0 or tp_price_count > 0 or sl_pct_count > 0 or tp_pct_count > 0:
                logger.info("✅ SL/TP data detected - adding stop level symbols")
                self._add_stop_levels_at_entries(fig, signals, ohlcv_data, row, col)
            else:
                logger.warning("⚠️ No SL/TP data available for rendering")
    
    def _validate_rendering_signals(self, signals: Dict[str, pd.Series], ohlcv_data: pd.DataFrame):
        """Validate signals before rendering."""
        # Check for overlapping entry signals
        if 'long_entries' in signals and 'short_entries' in signals:
            overlaps = signals['long_entries'] & signals['short_entries']
            if overlaps.any():
                overlap_times = ohlcv_data.index[overlaps]
                logger.error(f"CRITICAL: Found overlapping entry signals at {len(overlap_times)} timestamps")
        
        # Check for entries without prices
        if 'long_entries' in signals and 'entry_prices' in signals:
            long_no_price = signals['long_entries'] & signals['entry_prices'].isna()
            if long_no_price.any():
                logger.warning(f"Long entries without prices: {long_no_price.sum()}")
        
        if 'short_entries' in signals and 'entry_prices' in signals:
            short_no_price = signals['short_entries'] & signals['entry_prices'].isna()
            if short_no_price.any():
                logger.warning(f"Short entries without prices: {short_no_price.sum()}")
    
    def _add_signal_type(
        self,
        fig: go.Figure,
        signals: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame,
        signal_key: str,
        price_key: str,
        style_key: str,
        row: int,
        col: int
    ):
        """Add a specific signal type to the chart."""
        if signal_key not in signals:
            return
        
        signal_mask = signals[signal_key].reindex(ohlcv_data.index, fill_value=False)
        prices = signals.get(price_key, pd.Series(np.nan, index=ohlcv_data.index)).reindex(ohlcv_data.index, fill_value=np.nan)
        
        valid_signals = signal_mask & ~prices.isna()
        if not valid_signals.any():
            logger.debug(f"No valid {signal_key} signals to render")
            return
        
        signal_times = ohlcv_data.index[valid_signals]
        signal_prices = prices[valid_signals]
        
        style = self.signal_styles[style_key]
        show_legend = self.legend_manager.should_show_legend(style['name'])
        
        logger.info(f"Rendering {len(signal_times)} {style['name']} signals")
        
        fig.add_trace(
            go.Scatter(
                x=signal_times,
                y=signal_prices,
                mode='markers',
                name=style['name'],
                marker=dict(
                    symbol=style['symbol'],
                    size=style['size'],
                    color=style['color'],
                    line=dict(color=style['line_color'], width=2)
                ),
                showlegend=show_legend,
                hovertemplate=f'<b>{style["name"]}</b><br>Date: %{{x}}<br>Price: $%{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_stop_levels_at_entries(
        self,
        fig: go.Figure,
        signals: Dict[str, pd.Series],
        ohlcv_data: pd.DataFrame,
        row: int,
        col: int
    ):
        """Add stop loss and take profit levels at entry points."""
        try:
            # Get entry signals and reindex to match OHLCV data
            long_entries = signals.get('long_entries', pd.Series(dtype=bool))
            short_entries = signals.get('short_entries', pd.Series(dtype=bool))
            
            # Reindex to match the filtered OHLCV data
            if not long_entries.empty:
                long_entries = long_entries.reindex(ohlcv_data.index, fill_value=False)
            else:
                long_entries = pd.Series(False, index=ohlcv_data.index)
                
            if not short_entries.empty:
                short_entries = short_entries.reindex(ohlcv_data.index, fill_value=False)
            else:
                short_entries = pd.Series(False, index=ohlcv_data.index)
            
            # Build lists of stop levels
            sl_times, sl_prices = [], []
            tp_times, tp_prices = [], []
            
            # Process long entries
            if not long_entries.empty and long_entries.any():
                long_entry_indices = ohlcv_data.index[long_entries]
                for entry_idx in long_entry_indices:
                    entry_price = signals.get('entry_prices', pd.Series(index=ohlcv_data.index)).get(entry_idx, np.nan)
                    
                    if pd.isna(entry_price):
                        entry_price = ohlcv_data.loc[entry_idx, 'Close']
                    
                    # Get stop loss level
                    sl_level = self._get_stop_level(signals, entry_idx, entry_price, True, ohlcv_data)
                    if sl_level is not None and not pd.isna(sl_level):
                        sl_times.append(entry_idx)
                        sl_prices.append(sl_level)
                    
                    # Get take profit level
                    tp_level = self._get_profit_level(signals, entry_idx, entry_price, True, ohlcv_data)
                    if tp_level is not None and not pd.isna(tp_level):
                        tp_times.append(entry_idx)
                        tp_prices.append(tp_level)
            
            # Process short entries
            if not short_entries.empty and short_entries.any():
                short_entry_indices = ohlcv_data.index[short_entries]
                for entry_idx in short_entry_indices:
                    entry_price = signals.get('entry_prices', pd.Series(index=ohlcv_data.index)).get(entry_idx, np.nan)
                    
                    if pd.isna(entry_price):
                        entry_price = ohlcv_data.loc[entry_idx, 'Close']
                    
                    # Get stop loss level (for shorts, SL is above entry)
                    sl_level = self._get_stop_level(signals, entry_idx, entry_price, False, ohlcv_data)
                    if sl_level is not None and not pd.isna(sl_level):
                        sl_times.append(entry_idx)
                        sl_prices.append(sl_level)
                    
                    # Get take profit level (for shorts, TP is below entry)
                    tp_level = self._get_profit_level(signals, entry_idx, entry_price, False, ohlcv_data)
                    if tp_level is not None and not pd.isna(tp_level):
                        tp_times.append(entry_idx)
                        tp_prices.append(tp_level)
            
            # Add stop loss symbols
            if sl_times:
                show_legend = self.legend_manager.should_show_legend('Stop Loss')
                fig.add_trace(
                    go.Scatter(
                        x=sl_times,
                        y=sl_prices,
                        mode='markers',
                        name='Stop Loss',
                        marker=dict(
                            symbol='line-ew',
                            size=12,
                            color=self.config.colors['stop_level'],
                            line=dict(color=self.config.colors['stop_level'], width=2)
                        ),
                        showlegend=show_legend,
                        hovertemplate='<b>Stop Loss</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                logger.debug(f"Added {len(sl_times)} stop loss symbols")
            
            # Add take profit symbols
            if tp_times:
                show_legend = self.legend_manager.should_show_legend('Take Profit')
                fig.add_trace(
                    go.Scatter(
                        x=tp_times,
                        y=tp_prices,
                        mode='markers',
                        name='Take Profit',
                        marker=dict(
                            symbol='line-ew',
                            size=12,
                            color=self.config.colors['profit_level'],
                            line=dict(color=self.config.colors['profit_level'], width=2)
                        ),
                        showlegend=show_legend,
                        hovertemplate='<b>Take Profit</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                logger.debug(f"Added {len(tp_times)} take profit symbols")
            
            logger.info(f"Stop level rendering complete: {len(sl_times)} SL, {len(tp_times)} TP symbols")
            
        except Exception as e:
            logger.error(f"Failed to add stop levels: {e}")
    
    def _get_stop_level(self, signals: Dict[str, pd.Series], entry_idx, entry_price: float, is_long: bool, ohlcv_data: pd.DataFrame) -> Optional[float]:
        """Get stop loss level for an entry point."""
        # For execution timing, we need to check the previous bar for stop levels
        # since they were generated at signal time (T) but shown at execution time (T+1)
        check_indices = [entry_idx]
        
        # If we're in execution timing mode, also check the previous bar
        if self.config.signal_timing_mode == 'execution' and entry_idx in ohlcv_data.index:
            idx_pos = ohlcv_data.index.get_loc(entry_idx)
            if idx_pos > 0:
                prev_idx = ohlcv_data.index[idx_pos - 1]
                check_indices.insert(0, prev_idx)  # Check previous bar first
        
        # Try price levels first
        if 'sl_price_levels' in signals:
            for idx in check_indices:
                sl_price = signals['sl_price_levels'].get(idx)
                if sl_price is not None and not pd.isna(sl_price):
                    return float(sl_price)
        
        # Try percentage levels
        if 'sl_levels' in signals and not pd.isna(entry_price):
            for idx in check_indices:
                sl_pct = signals['sl_levels'].get(idx)
                if sl_pct is not None and not pd.isna(sl_pct):
                    if is_long:
                        return float(entry_price * (1 - abs(sl_pct) / 100))
                    else:
                        return float(entry_price * (1 + abs(sl_pct) / 100))
        
        return None
    
    def _get_profit_level(self, signals: Dict[str, pd.Series], entry_idx, entry_price: float, is_long: bool, ohlcv_data: pd.DataFrame) -> Optional[float]:
        """Get take profit level for an entry point."""
        # For execution timing, we need to check the previous bar for stop levels
        # since they were generated at signal time (T) but shown at execution time (T+1)
        check_indices = [entry_idx]
        
        # If we're in execution timing mode, also check the previous bar
        if self.config.signal_timing_mode == 'execution' and entry_idx in ohlcv_data.index:
            idx_pos = ohlcv_data.index.get_loc(entry_idx)
            if idx_pos > 0:
                prev_idx = ohlcv_data.index[idx_pos - 1]
                check_indices.insert(0, prev_idx)  # Check previous bar first
        
        # Try price levels first
        if 'tp_price_levels' in signals:
            for idx in check_indices:
                tp_price = signals['tp_price_levels'].get(idx)
                if tp_price is not None and not pd.isna(tp_price):
                    return float(tp_price)
        
        # Try percentage levels
        if 'tp_levels' in signals and not pd.isna(entry_price):
            for idx in check_indices:
                tp_pct = signals['tp_levels'].get(idx)
                if tp_pct is not None and not pd.isna(tp_pct):
                    if is_long:
                        return float(entry_price * (1 + abs(tp_pct) / 100))
                    else:
                        return float(entry_price * (1 - abs(tp_pct) / 100))
        
        return None
    
    def _prepare_render_data(self, data: Any) -> Any:
        """Prepare data for rendering (BaseRenderer interface)."""
        return data
    
    def _create_traces(self, data: Any) -> List[go.Scatter]:
        """Create traces (BaseRenderer interface)."""
        return []  # Handled by specific methods
    
    def get_required_data_type(self) -> type:
        """Get required data type."""
        return dict


# Module exports
__all__ = [
    'IndicatorRenderer',
    'CandlestickRenderer',
    'VolumeRenderer',
    'EquityRenderer',
    'SignalRenderer'
]