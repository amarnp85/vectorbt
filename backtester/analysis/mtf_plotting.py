"""
Multi-Timeframe Plotting Module

Provides specialized plotting functionality for multi-timeframe analysis,
including aligned indicators, trend comparisons, and signal validation.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class MTFPlottingEngine:
    """
    Specialized plotting engine for multi-timeframe analysis.
    
    This class provides various visualization methods for MTF strategies,
    including aligned indicators, cross-timeframe comparisons, and
    signal validation visualizations.
    """
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize MTF plotting engine.
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self._setup_plotting_theme()
        
        # Color palette for different timeframes
        self.tf_colors = {
            '1m': '#FF6B6B',
            '5m': '#4ECDC4',
            '15m': '#45B7D1',
            '30m': '#96CEB4',
            '1h': '#FECA57',
            '4h': '#FF9FF3',
            '1d': '#54A0FF',
            '1w': '#48DBFB',
        }
        
        logger.info("MTFPlottingEngine initialized")
    
    def _setup_plotting_theme(self):
        """Set up plotting theme."""
        try:
            vbt.settings.plotting.layout.template = self.theme
            vbt.settings.plotting.layout.width = 1200
            vbt.settings.plotting.layout.height = 800
        except Exception as e:
            logger.debug(f"Could not set plotting theme: {e}")
    
    def plot_mtf_price_overview(
        self,
        mtf_data: Dict[str, vbt.Data],
        symbol: Optional[str] = None,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Plot price data across multiple timeframes.
        
        Args:
            mtf_data: Dictionary of timeframe to vbt.Data
            symbol: Specific symbol to plot
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly figure
        """
        try:
            timeframes = list(mtf_data.keys())
            n_timeframes = len(timeframes)
            
            # Calculate subplot heights
            if show_volume:
                row_heights = [0.6] + [0.4/n_timeframes] * n_timeframes
                rows = n_timeframes + 1
            else:
                row_heights = [1.0/n_timeframes] * n_timeframes
                rows = n_timeframes
            
            # Create subplots
            subplot_titles = [f"{tf} Chart" for tf in timeframes]
            if show_volume:
                subplot_titles.insert(0, "Price Overview")
            
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=subplot_titles,
                row_heights=row_heights
            )
            
            # Main price chart (lowest timeframe with the most detail)
            base_tf = min(timeframes, key=lambda x: self._tf_to_minutes(x))
            base_data = mtf_data[base_tf]
            
            if show_volume:
                self._add_candlestick(fig, base_data, 1, 1, f"{base_tf} Price", symbol)
                row_offset = 2
            else:
                row_offset = 1
            
            # Add each timeframe
            for i, tf in enumerate(timeframes):
                data = mtf_data[tf]
                color = self.tf_colors.get(tf, '#636EFA')
                
                # Plot close price for each timeframe
                close = self._get_series(data.close, symbol)
                
                fig.add_trace(
                    go.Scatter(
                        x=close.index,
                        y=close.values,
                        name=f"{tf} Close",
                        line=dict(color=color, width=2),
                        showlegend=True
                    ),
                    row=i + row_offset,
                    col=1
                )
                
                # Add SMA for context
                if len(close) > 20:
                    sma = close.rolling(20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=sma.index,
                            y=sma.values,
                            name=f"{tf} SMA(20)",
                            line=dict(color=color, width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=i + row_offset,
                        col=1
                    )
            
            # Update layout
            fig.update_layout(
                title="Multi-Timeframe Price Analysis",
                height=200 * rows,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(title_text="Date", row=rows, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create MTF price overview: {e}")
            raise
    
    def plot_aligned_indicators(
        self,
        indicators: Dict[str, Dict[str, pd.Series]],
        indicator_name: str,
        base_timeframe: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot aligned indicators from multiple timeframes.
        
        Args:
            indicators: MTF indicators dict {timeframe: {name: series}}
            indicator_name: Name of indicator to plot
            base_timeframe: Base timeframe for alignment
            title: Plot title
            
        Returns:
            Plotly figure
        """
        try:
            if title is None:
                title = f"Multi-Timeframe {indicator_name}"
            
            fig = go.Figure()
            
            # Get base index for x-axis
            base_index = None
            if base_timeframe in indicators and indicator_name in indicators[base_timeframe]:
                base_index = indicators[base_timeframe][indicator_name].index
            
            # Plot each timeframe's indicator
            for tf, tf_indicators in indicators.items():
                if indicator_name not in tf_indicators:
                    continue
                
                indicator = tf_indicators[indicator_name]
                color = self.tf_colors.get(tf, '#636EFA')
                
                # Determine line style based on timeframe hierarchy
                if tf == base_timeframe:
                    line_style = dict(color=color, width=3)
                else:
                    minutes = self._tf_to_minutes(tf)
                    base_minutes = self._tf_to_minutes(base_timeframe)
                    if minutes > base_minutes:
                        # Higher timeframe - dashed line
                        line_style = dict(color=color, width=2, dash='dash')
                    else:
                        # Lower timeframe - dotted line
                        line_style = dict(color=color, width=2, dash='dot')
                
                fig.add_trace(
                    go.Scatter(
                        x=indicator.index,
                        y=indicator.values,
                        name=f"{tf} {indicator_name}",
                        line=line_style,
                        mode='lines'
                    )
                )
            
            # Add layout
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=indicator_name,
                height=600,
                showlegend=True,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create aligned indicators plot: {e}")
            raise
    
    def plot_trend_alignment(
        self,
        mtf_data: Dict[str, vbt.Data],
        window: int = 20,
        symbol: Optional[str] = None
    ) -> go.Figure:
        """
        Plot trend alignment across timeframes.
        
        Args:
            mtf_data: Dictionary of timeframe to data
            window: SMA window for trend calculation
            symbol: Specific symbol to analyze
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[
                    "Price with Multi-Timeframe SMAs",
                    "Trend Direction by Timeframe",
                    "Trend Alignment Score"
                ],
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Calculate trends for each timeframe
            trends = {}
            base_tf = min(mtf_data.keys(), key=lambda x: self._tf_to_minutes(x))
            base_close = self._get_series(mtf_data[base_tf].close, symbol)
            
            # Plot 1: Price with MTF SMAs
            fig.add_trace(
                go.Scatter(
                    x=base_close.index,
                    y=base_close.values,
                    name="Price",
                    line=dict(color='black', width=1),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            for tf, data in mtf_data.items():
                close = self._get_series(data.close, symbol)
                sma = close.rolling(window).mean()
                
                # Store trend direction
                trends[tf] = (close > sma).astype(int)
                
                # Plot SMA
                color = self.tf_colors.get(tf, '#636EFA')
                fig.add_trace(
                    go.Scatter(
                        x=sma.index,
                        y=sma.values,
                        name=f"{tf} SMA({window})",
                        line=dict(color=color, width=2),
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Trend direction heatmap
            trend_df = pd.DataFrame(trends)
            
            # Create heatmap data
            z_data = []
            y_labels = []
            for tf in sorted(mtf_data.keys(), key=lambda x: self._tf_to_minutes(x)):
                if tf in trend_df.columns:
                    z_data.append(trend_df[tf].values)
                    y_labels.append(tf)
            
            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=trend_df.index,
                    y=y_labels,
                    colorscale=[[0, 'red'], [1, 'green']],
                    showscale=False,
                    hovertemplate='%{y}<br>%{x}<br>Trend: %{z}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Plot 3: Alignment score
            alignment_score = trend_df.mean(axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=alignment_score.index,
                    y=alignment_score.values,
                    name="Alignment Score",
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title="Multi-Timeframe Trend Alignment Analysis",
                height=900,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Timeframe", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=3, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create trend alignment plot: {e}")
            raise
    
    def plot_mtf_signals(
        self,
        signals: Dict[str, pd.Series],
        mtf_data: Dict[str, vbt.Data],
        symbol: Optional[str] = None
    ) -> go.Figure:
        """
        Plot trading signals with MTF context.
        
        Args:
            signals: Dictionary of signal series
            mtf_data: MTF data for context
            symbol: Specific symbol
            
        Returns:
            Plotly figure
        """
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=["Signals with MTF Context", "Signal Confirmation"],
                row_heights=[0.7, 0.3]
            )
            
            # Get base data
            base_tf = min(mtf_data.keys(), key=lambda x: self._tf_to_minutes(x))
            base_close = self._get_series(mtf_data[base_tf].close, symbol)
            
            # Plot price
            fig.add_trace(
                go.Scatter(
                    x=base_close.index,
                    y=base_close.values,
                    name="Price",
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )
            
            # Plot entry signals
            if 'entries' in signals:
                entry_points = base_close[signals['entries']]
                fig.add_trace(
                    go.Scatter(
                        x=entry_points.index,
                        y=entry_points.values,
                        mode='markers',
                        name='Long Entries',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green'
                        )
                    ),
                    row=1, col=1
                )
            
            # Plot exit signals
            if 'exits' in signals:
                exit_points = base_close[signals['exits']]
                fig.add_trace(
                    go.Scatter(
                        x=exit_points.index,
                        y=exit_points.values,
                        mode='markers',
                        name='Long Exits',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red'
                        )
                    ),
                    row=1, col=1
                )
            
            # Add MTF context lines
            for tf, data in mtf_data.items():
                if tf == base_tf:
                    continue
                
                close = self._get_series(data.close, symbol)
                color = self.tf_colors.get(tf, '#636EFA')
                
                fig.add_trace(
                    go.Scatter(
                        x=close.index,
                        y=close.values,
                        name=f"{tf} Close",
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.5
                    ),
                    row=1, col=1
                )
            
            # Plot signal strength/confirmation
            if 'entries' in signals and 'exits' in signals:
                # Simple signal strength: 1 for long, -1 for short, 0 for neutral
                signal_strength = pd.Series(0, index=base_close.index)
                signal_strength[signals['entries']] = 1
                signal_strength[signals['exits']] = -1
                signal_strength = signal_strength.cumsum()
                
                fig.add_trace(
                    go.Scatter(
                        x=signal_strength.index,
                        y=signal_strength.values,
                        name="Signal Strength",
                        fill='tozeroy',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="Multi-Timeframe Signal Analysis",
                height=900,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Strength", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create MTF signals plot: {e}")
            raise
    
    def _get_series(
        self,
        data: Union[pd.Series, pd.DataFrame],
        symbol: Optional[str] = None
    ) -> pd.Series:
        """Extract series from data."""
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            if symbol and symbol in data.columns:
                return data[symbol]
            elif data.shape[1] == 1:
                return data.iloc[:, 0]
            else:
                return data.iloc[:, 0]  # Default to first column
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _tf_to_minutes(self, tf: str) -> int:
        """Convert timeframe to minutes."""
        mappings = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
        }
        return mappings.get(tf, 60)
    
    def _add_candlestick(
        self,
        fig: go.Figure,
        data: vbt.Data,
        row: int,
        col: int,
        name: str,
        symbol: Optional[str] = None
    ):
        """Add candlestick chart to figure."""
        ohlc_data = {}
        for field in ['open', 'high', 'low', 'close']:
            if hasattr(data, field):
                series = getattr(data, field)
                ohlc_data[field] = self._get_series(series, symbol)
        
        if len(ohlc_data) == 4:
            fig.add_trace(
                go.Candlestick(
                    x=ohlc_data['open'].index,
                    open=ohlc_data['open'].values,
                    high=ohlc_data['high'].values,
                    low=ohlc_data['low'].values,
                    close=ohlc_data['close'].values,
                    name=name
                ),
                row=row, col=col
            ) 