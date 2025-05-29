"""
Trading Charts Engine - Refactored for Clean Architecture

Modern, modular plotting library for single backtest analysis using VectorBT Pro data objects.
Signal processing has been moved to the trading_signals module for better separation of concerns.

IMPORTANT TIMING NOTE:
======================
This module works with the trading_signals module which properly handles VectorBT's execution timing model:
- Signal Generation: Happens at bar close (time T) based on available information  
- Order Execution: Happens at next bar open (time T+1) to prevent lookahead bias
- Chart Display: Configurable timing modes via SignalConfig

Features:
- Clean modular architecture with separated concerns
- Professional chart building and layout management
- Smart indicator categorization and placement
- Extensible design for any strategy type
- Professional export capabilities
- Comprehensive data processing capabilities

Key Classes:
- TradingChartsEngine: Main coordinator (cleaned up)
- ChartBuilder: Subplot creation and layout
- IndicatorRenderer: Indicator placement
- LegendManager: Legend configuration
- DataProcessor: Data extraction and preparation
- IndicatorProcessor: Indicator categorization

Usage:
    from backtester.analysis.trading_charts_refactored import TradingChartsEngine, ChartConfig
    from backtester.analysis.trading_signals import SignalConfig
    
    # Create with custom signal timing
    signal_config = SignalConfig(signal_timing_mode="execution", execution_delay=1)
    chart_config = ChartConfig(title="My Strategy Analysis")
    
    charts = TradingChartsEngine(portfolio, data, indicators, signals, 
                               chart_config=chart_config, signal_config=signal_config)
    fig = charts.create_main_chart()
    charts.save_chart(fig, "analysis.html")
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from backtester.utilities.structured_logging import get_logger
from backtester.analysis.trading_signals import SignalProcessor, SignalRenderer, SignalConfig

logger = get_logger(__name__)


class IndicatorType(Enum):
    """Enumeration for indicator types."""
    PRICE_OVERLAY = "price_overlay"
    VOLUME_OVERLAY = "volume_overlay"
    SEPARATE_SUBPLOT = "separate_subplot"


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior (non-signal related)."""
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


class DataProcessor:
    """Handles data extraction and preparation from VectorBT objects."""
    
    def __init__(self, portfolio: vbt.Portfolio, data: vbt.Data):
        self.portfolio = portfolio
        self.data = data
        self.ohlcv_data = self._extract_ohlcv_data()
    
    def _extract_ohlcv_data(self) -> pd.DataFrame:
        """Extract OHLCV data from VectorBT Data object with comprehensive validation."""
        try:
            ohlcv = pd.DataFrame({
                'Open': self._extract_series(self.data.get('open')),
                'High': self._extract_series(self.data.get('high')),
                'Low': self._extract_series(self.data.get('low')),
                'Close': self._extract_series(self.data.get('close')),
            })
            
            # Add volume if available
            try:
                volume = self.data.get('volume')
                if volume is not None:
                    ohlcv['Volume'] = self._extract_series(volume)
            except (AttributeError, KeyError):
                logger.debug("Volume data not available")
            
            # CRITICAL: Clean invalid data that causes chart scaling issues
            ohlcv = self._clean_and_validate_data(ohlcv)
            
            logger.debug(f"Extracted OHLCV data: {ohlcv.shape[0]} rows")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Failed to extract OHLCV data: {e}")
            raise
    
    def _clean_and_validate_data(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data to prevent chart scaling issues."""
        original_count = len(ohlcv)
        
        # Remove rows with any NaN/inf values in OHLC
        ohlcv = ohlcv.replace([np.inf, -np.inf], np.nan)
        ohlcv = ohlcv.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(ohlcv) == 0:
            raise ValueError("No valid OHLCV data after cleaning")
        
        # Remove rows with zero or negative prices (causes 0,0 scaling issues)
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in ohlcv.columns:
                ohlcv = ohlcv[ohlcv[col] > 0]
        
        # Validate OHLC relationships
        valid_ohlc = (
            (ohlcv['High'] >= ohlcv['Open']) &
            (ohlcv['High'] >= ohlcv['Close']) &
            (ohlcv['Low'] <= ohlcv['Open']) &
            (ohlcv['Low'] <= ohlcv['Close']) &
            (ohlcv['High'] >= ohlcv['Low'])
        )
        ohlcv = ohlcv[valid_ohlc]
        
        # Remove extreme outliers that could skew chart scaling
        for col in price_cols:
            if col in ohlcv.columns:
                Q1 = ohlcv[col].quantile(0.01)
                Q3 = ohlcv[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 10 * IQR  # Very generous bounds
                upper_bound = Q3 + 10 * IQR
                ohlcv = ohlcv[(ohlcv[col] >= lower_bound) & (ohlcv[col] <= upper_bound)]
        
        # Clean volume data
        if 'Volume' in ohlcv.columns:
            ohlcv = ohlcv[ohlcv['Volume'] >= 0]  # Remove negative volume
            # Replace zero volume with small positive value to avoid log scale issues
            zero_volume_mask = ohlcv['Volume'] == 0
            if zero_volume_mask.any():
                min_positive_volume = ohlcv['Volume'][ohlcv['Volume'] > 0].min()
                replacement_value = float(min_positive_volume / 10)  # Explicit float conversion
                ohlcv.loc[zero_volume_mask, 'Volume'] = replacement_value
        
        cleaned_count = len(ohlcv)
        if cleaned_count < original_count:
            logger.info(f"Data cleaning: removed {original_count - cleaned_count} invalid rows, "
                       f"kept {cleaned_count} valid rows")
        
        if len(ohlcv) == 0:
            raise ValueError("No valid data remaining after cleaning")
            
        return ohlcv
    
    def _extract_series(self, data: Any) -> pd.Series:
        """Extract pandas Series from various VectorBT data formats with validation."""
        if data is None:
            return pd.Series(dtype=float)
        
        if isinstance(data, pd.Series):
            series = data
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                series = data.iloc[:, 0]
            else:
                logger.debug(f"Multi-column data detected, using first column")
                series = data.iloc[:, 0]
        elif hasattr(data, 'values') and hasattr(data, 'index'):
            series = pd.Series(data.values, index=data.index)
        else:
            logger.debug(f"Converting unknown data type to Series: {type(data)}")
            series = pd.Series(data)
        
        # Clean the series to prevent chart issues
        if len(series) > 0:
            series = series.replace([np.inf, -np.inf], np.nan)
            series = series.dropna()
        
        return series
    
    def get_portfolio_equity(self) -> pd.Series:
        """Get portfolio equity curve with validation."""
        try:
            portfolio_value = self.portfolio.value
            if hasattr(portfolio_value, 'values'):
                equity_values = portfolio_value.values.flatten()
                portfolio_index = portfolio_value.index
            else:
                equity_values = portfolio_value
                portfolio_index = self.portfolio.wrapper.index
            
            # Create equity series with portfolio's original index
            equity_series = pd.Series(equity_values, index=portfolio_index)
            
            # Align with cleaned OHLCV data index (this handles the length mismatch)
            equity_series = equity_series.reindex(self.ohlcv_data.index, method='ffill')
            
            # Clean equity data
            equity_series = equity_series.replace([np.inf, -np.inf], np.nan)
            equity_series = equity_series.dropna()
            
            # Ensure positive values
            if len(equity_series) > 0 and equity_series.min() <= 0:
                logger.warning("Portfolio equity contains non-positive values, filtering...")
                equity_series = equity_series[equity_series > 0]
            
            return equity_series
            
        except Exception as e:
            logger.error(f"Failed to extract portfolio equity: {e}")
            return pd.Series([], index=self.ohlcv_data.index)
    
    def get_portfolio_returns(self) -> pd.Series:
        """Get portfolio returns with validation."""
        try:
            returns = self.portfolio.returns
            if hasattr(returns, 'values'):
                returns_series = pd.Series(returns.values.flatten(), index=self.ohlcv_data.index)
            else:
                returns_series = returns
            
            # Clean returns data
            returns_series = returns_series.replace([np.inf, -np.inf], np.nan)
            returns_series = returns_series.dropna()
            
            return returns_series
            
        except Exception as e:
            logger.error(f"Failed to extract portfolio returns: {e}")
            return pd.Series([], index=self.ohlcv_data.index)


class IndicatorProcessor:
    """Processes and categorizes indicators for proper chart placement."""
    
    def __init__(self, indicators: Dict[str, Any], ohlcv_data: pd.DataFrame):
        self.indicators = indicators
        self.ohlcv_data = ohlcv_data
        self.processed = self._process_indicators()
    
    def _process_indicators(self) -> Dict[IndicatorType, Dict[str, pd.Series]]:
        """Process and categorize indicators."""
        processed = {
            IndicatorType.PRICE_OVERLAY: {},
            IndicatorType.VOLUME_OVERLAY: {},
            IndicatorType.SEPARATE_SUBPLOT: {}
        }
        
        for name, indicator in self.indicators.items():
            try:
                series = self._extract_series(indicator)
                indicator_type = self._categorize_indicator(name, series)
                processed[indicator_type][name] = series
            except Exception as e:
                logger.warning(f"Could not process indicator {name}: {e}")
        
        logger.debug(f"Processed indicators: "
                    f"{len(processed[IndicatorType.PRICE_OVERLAY])} price overlay, "
                    f"{len(processed[IndicatorType.SEPARATE_SUBPLOT])} subplots, "
                    f"{len(processed[IndicatorType.VOLUME_OVERLAY])} volume")
        
        return processed
    
    def _extract_series(self, indicator: Any) -> pd.Series:
        """Extract pandas Series from indicator."""
        if isinstance(indicator, pd.Series):
            return indicator
        elif isinstance(indicator, pd.DataFrame):
            return indicator.iloc[:, 0]
        elif hasattr(indicator, 'values') and hasattr(indicator, 'index'):
            return pd.Series(indicator.values, index=indicator.index)
        else:
            return pd.Series(indicator)
    
    def _categorize_indicator(self, name: str, series: pd.Series) -> IndicatorType:
        """Categorize indicator based on name and values."""
        name_lower = name.lower()
        
        # Volume indicators first
        if self._is_volume_indicator(name_lower):
            return IndicatorType.VOLUME_OVERLAY
        
        # Price overlay indicators
        if self._is_price_overlay_indicator(name_lower, series):
            return IndicatorType.PRICE_OVERLAY
        
        # Everything else goes to separate subplot
        return IndicatorType.SEPARATE_SUBPLOT
    
    def _is_volume_indicator(self, name: str) -> bool:
        """Check if indicator is volume-related."""
        volume_keywords = ['volume', 'vol_', 'obv', 'ad', 'cmf', 'mfi', 'vwap', 'pvt']
        return any(keyword in name for keyword in volume_keywords)
    
    def _is_price_overlay_indicator(self, name: str, series: pd.Series) -> bool:
        """Check if indicator should overlay on price chart."""
        overlay_names = ['ma', 'sma', 'ema', 'bollinger', 'bb', 'keltner', 'vwap']
        
        if any(overlay_name in name for overlay_name in overlay_names):
            return True
        
        # Check value range similarity to price
        if len(series.dropna()) > 0 and len(self.ohlcv_data) > 0:
            try:
                price_range = self.ohlcv_data['Close'].max() - self.ohlcv_data['Close'].min()
                indicator_range = series.max() - series.min()
                
                if price_range > 0 and 0.1 <= (indicator_range / price_range) <= 10:
                    return True
            except (ZeroDivisionError, ValueError):
                pass
        
        return False


class LegendManager:
    """Manages legend configuration and prevents duplicates."""
    
    def __init__(self):
        self.legend_items = set()
        self.legend_config = {
            'orientation': "h",
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "right",
            'x': 1,
            'bgcolor': "rgba(255,255,255,0.8)"
        }
    
    def should_show_legend(self, trace_name: str) -> bool:
        """Determine if trace should show in legend (avoid duplicates)."""
        if trace_name in self.legend_items:
            return False
        self.legend_items.add(trace_name)
        return True
    
    def reset(self):
        """Reset legend items."""
        self.legend_items.clear()


class IndicatorRenderer:
    """Renders indicators on charts."""
    
    def __init__(self, config: ChartConfig, legend_manager: LegendManager):
        self.config = config
        self.legend_manager = legend_manager
    
    def add_price_overlays(self, fig: go.Figure, indicators: Dict[str, pd.Series], 
                          ohlcv_data: pd.DataFrame, row: int = 1, col: int = 1):
        """Add price overlay indicators to the main chart."""
        colors = self.config.colors['indicators']
        
        # Prioritize important indicators for legend display
        important_indicators = ['fast_ma', 'slow_ma', 'sma_fast', 'sma_slow', 'ema_fast', 'ema_slow']
        
        for i, (name, indicator) in enumerate(indicators.items()):
            aligned_indicator = indicator.reindex(ohlcv_data.index).dropna()
            
            if len(aligned_indicator) > 0:
                color = colors[i % len(colors)]
                
                # Only show important indicators in legend to reduce clutter
                is_important = any(important in name.lower() for important in important_indicators)
                show_legend = is_important and self.legend_manager.should_show_legend(name)
                
                fig.add_trace(
                    go.Scatter(
                        x=aligned_indicator.index,
                        y=aligned_indicator.values,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                        opacity=0.8,
                        showlegend=show_legend
                    ),
                    row=row, col=col
                )
    
    def add_subplot_indicators(self, fig: go.Figure, indicators: Dict[str, pd.Series], 
                             ohlcv_data: pd.DataFrame, start_row: int):
        """Add indicators that need separate subplots."""
        for i, (name, indicator) in enumerate(indicators.items()):
            aligned_indicator = indicator.reindex(ohlcv_data.index).dropna()
            
            if len(aligned_indicator) > 0:
                row = start_row + i
                
                # Use different colors for different indicators
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                color = colors[i % len(colors)]
                
                # Don't show subplot indicators in main legend to reduce clutter
                fig.add_trace(
                    go.Scatter(
                        x=aligned_indicator.index,
                        y=aligned_indicator.values,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                        showlegend=False,  # Don't clutter main legend
                        # Ensure line connects all points for full area usage
                        connectgaps=True,
                        hovertemplate=f'{name}: %{{y:.4f}}<br>Date: %{{x}}<extra></extra>'
                    ),
                    row=row, col=1
                )
                
                # Add subplot title for clarity
                fig.update_yaxes(title_text=name, row=row, col=1)
                
                # Add reference lines for specific indicators
                self._add_reference_lines(fig, name, row)
    
    def _add_reference_lines(self, fig: go.Figure, indicator_name: str, row: int):
        """Add reference lines for specific indicators."""
        name_lower = indicator_name.lower()
        
        if 'rsi' in name_lower:
            for level, color in [(30, 'green'), (70, 'red'), (50, 'gray')]:
                fig.add_hline(
                    y=level, line_dash='dash', line_color=color,
                    opacity=0.5, row=row, col=1
                )
        elif 'macd' in name_lower:
            fig.add_hline(
                y=0, line_dash='solid', line_color='gray',
                opacity=0.5, row=row, col=1
            )


class ChartBuilder:
    """Builds chart structure and layout."""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def create_subplot_structure(self, n_indicator_subplots: int, has_volume: bool, has_equity: bool) -> go.Figure:
        """Create the subplot structure for the chart."""
        n_subplots = 1  # Main price chart
        
        if has_volume:
            n_subplots += 1
        if has_equity:
            n_subplots += 1
        n_subplots += n_indicator_subplots
        
        # Create subplot titles
        subplot_titles = ["Price & Indicators"]
        if has_volume:
            subplot_titles.append("Volume")
        if has_equity:
            subplot_titles.append("Equity Curve")
        subplot_titles.extend([f"Indicator {i+1}" for i in range(n_indicator_subplots)])
        
        # Calculate row heights
        row_heights = self._calculate_row_heights(n_subplots, has_volume, has_equity, n_indicator_subplots)
        
        # Improved vertical spacing based on number of subplots
        if n_subplots <= 2:
            vertical_spacing = 0.05
        elif n_subplots <= 4:
            vertical_spacing = 0.03
        else:
            vertical_spacing = 0.02
        
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            # Improved subplot configuration
            specs=[[{"secondary_y": False}] for _ in range(n_subplots)]
        )
        
        self._configure_layout(fig, n_subplots)
        return fig
    
    def apply_smart_ranges(self, fig: go.Figure, ohlcv_data: pd.DataFrame, 
                          indicators: Dict[str, Any], portfolio_equity: pd.Series,
                          n_subplots: int):
        """Apply smart range settings to prevent chart scaling issues."""
        
        # 1. Set X-axis range (dates)
        if len(ohlcv_data) > 0:
            date_range = [ohlcv_data.index[0], ohlcv_data.index[-1]]
            # Add 2% padding to date range
            date_span = (date_range[1] - date_range[0]).total_seconds()
            padding = pd.Timedelta(seconds=date_span * 0.02)
            date_range = [date_range[0] - padding, date_range[1] + padding]
            
            # Apply to all subplots
            for i in range(1, n_subplots + 1):
                fig.update_xaxes(range=date_range, row=i, col=1)
        
        # 2. Set price chart Y-axis range
        if len(ohlcv_data) > 0:
            price_min = min(ohlcv_data['Low'].min(), ohlcv_data['Close'].min())
            price_max = max(ohlcv_data['High'].max(), ohlcv_data['Close'].max())
            
            # Add 5% padding to price range
            price_range = price_max - price_min
            padding = price_range * 0.05
            price_y_range = [price_min - padding, price_max + padding]
            
            fig.update_yaxes(range=price_y_range, row=1, col=1)
            logger.debug(f"Set price range: {price_y_range}")
        
        # 3. Set volume chart range
        current_row = 2
        if 'Volume' in ohlcv_data.columns:
            volume = ohlcv_data['Volume']
            if len(volume) > 0 and volume.max() > 0:
                volume_max = volume.max()
                volume_min = max(0, volume.min())  # Ensure non-negative
                
                # Add 10% padding to volume range
                volume_range = volume_max - volume_min
                padding = volume_range * 0.1
                volume_y_range = [volume_min, volume_max + padding]
                
                fig.update_yaxes(range=volume_y_range, row=current_row, col=1)
                logger.debug(f"Set volume range: {volume_y_range}")
            current_row += 1
        
        # 4. Set equity chart range
        if len(portfolio_equity) > 0:
            equity_min = portfolio_equity.min()
            equity_max = portfolio_equity.max()
            
            if equity_max > equity_min:
                # Add 5% padding to equity range
                equity_range = equity_max - equity_min
                padding = equity_range * 0.05
                equity_y_range = [equity_min - padding, equity_max + padding]
                
                fig.update_yaxes(range=equity_y_range, row=current_row, col=1)
                logger.debug(f"Set equity range: {equity_y_range}")
            current_row += 1
        
        # 5. Set indicator subplot ranges
        if hasattr(indicators, 'processed'):
            subplot_indicators = indicators.processed[IndicatorType.SEPARATE_SUBPLOT]
            for i, (name, indicator) in enumerate(subplot_indicators.items()):
                if len(indicator) > 0:
                    indicator_min = indicator.min()
                    indicator_max = indicator.max()
                    
                    if indicator_max > indicator_min:
                        # Add 10% padding to indicator range
                        indicator_range = indicator_max - indicator_min
                        padding = indicator_range * 0.1
                        indicator_y_range = [indicator_min - padding, indicator_max + padding]
                        
                        fig.update_yaxes(range=indicator_y_range, row=current_row + i, col=1)
                        logger.debug(f"Set {name} range: {indicator_y_range}")
        
        logger.info("Applied smart ranges to prevent chart scaling issues")
    
    def _calculate_row_heights(self, n_subplots: int, has_volume: bool, has_equity: bool, n_indicators: int) -> List[float]:
        """Calculate optimal row heights for subplots."""
        if n_subplots == 1:
            return [1.0]
        
        # Improved height distribution to give more space to indicators
        if n_indicators == 0:
            # No indicators - distribute between main, volume, equity
            main_height = 0.7
            equity_height = 0.2 if has_equity else 0.0
            volume_height = 0.1 if has_volume else 0.0
        else:
            # With indicators - allocate more space to them
            main_height = 0.4  # Reduced from 0.5
            volume_height = 0.08 if has_volume else 0.0  # Reduced from 0.1
            equity_height = 0.15 if has_equity else 0.0  # Reduced from 0.25
            
            # Give remaining space to indicators with minimum height
            remaining_height = 1.0 - main_height - equity_height - volume_height
            min_indicator_height = 0.12  # Minimum 12% per indicator
            indicator_height = max(min_indicator_height, remaining_height / n_indicators)
            
            # If indicators need more space, reduce other components proportionally
            total_indicator_space = indicator_height * n_indicators
            if total_indicator_space > remaining_height:
                scale_factor = (1.0 - total_indicator_space) / (main_height + equity_height + volume_height)
                main_height *= scale_factor
                equity_height *= scale_factor
                volume_height *= scale_factor
        
        row_heights = [main_height]
        if has_volume:
            row_heights.append(volume_height)
        if has_equity:
            row_heights.append(equity_height)
        
        if n_indicators > 0:
            row_heights.extend([indicator_height] * n_indicators)
        
        return row_heights
    
    def _configure_layout(self, fig: go.Figure, n_subplots: int):
        """Configure chart layout and interactivity with explicit range settings."""
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
            margin=dict(l=80, r=80, t=120, b=80),  # Increased margins for better spacing
            dragmode='zoom',
            hovermode='x unified',
            autosize=True,
            # Improved responsiveness
            font=dict(size=12)
        )
        
        # Configure axes with explicit range settings to prevent scaling issues
        for i in range(1, n_subplots + 1):
            fig.update_xaxes(
                rangeslider_visible=False,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                # Improved x-axis formatting
                type='date',
                tickformat='%Y-%m-%d',
                # CRITICAL: Use explicit range control instead of autorange
                autorange=False,  # Disable autorange to prevent scaling issues
                fixedrange=False,  # Allow user zooming
                row=i, col=1
            )
            
            fig.update_yaxes(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                fixedrange=False,
                # CRITICAL: Use explicit range control
                autorange=False,  # Disable autorange to prevent scaling issues
                # Remove rangemode that could cause issues
                row=i, col=1
            )
        
        # Label the last x-axis
        fig.update_xaxes(title_text="Date", row=n_subplots, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)


class TradingChartsEngine:
    """
    Refactored trading charts engine with clean modular architecture.
    
    This engine coordinates chart building while delegating signal processing
    to the dedicated trading_signals module for better separation of concerns.
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
        Initialize the trading charts engine.
        
        Args:
            portfolio: VectorBT Portfolio object
            data: VectorBT Data object
            indicators: Dictionary of indicators
            signals: Dictionary of strategy signals
            chart_config: Chart configuration (non-signal related)
            signal_config: Signal processing configuration
        """
        self.portfolio = portfolio
        self.data = data
        self.chart_config = chart_config or ChartConfig()
        self.signal_config = signal_config or SignalConfig()
        
        # Initialize core processors
        self.data_processor = DataProcessor(portfolio, data)
        self.indicator_processor = IndicatorProcessor(indicators or {}, self.data_processor.ohlcv_data)
        
        # Initialize signal processing (from trading_signals module)
        self.signal_processor = SignalProcessor(portfolio, self.data_processor, signals, self.signal_config)
        
        # Initialize renderers
        self.legend_manager = LegendManager()
        self.signal_renderer = SignalRenderer(self.signal_config, self.legend_manager)
        self.indicator_renderer = IndicatorRenderer(self.chart_config, self.legend_manager)
        self.chart_builder = ChartBuilder(self.chart_config)
        
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
        
        Args:
            title: Chart title (uses config default if None)
            show_volume: Whether to show volume (uses config default if None)
            show_equity: Whether to show equity curve (uses config default if None)
            show_signals: Whether to show signals (uses config default if None)
            date_range: Optional date range filter
            height: Chart height (uses config default if None)
            
        Returns:
            Plotly figure with complete trading analysis
        """
        # Update configs with provided parameters
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
        
        # Reset legend manager
        self.legend_manager.reset()
        
        # Filter data by date range
        ohlcv = self._apply_date_filter(self.data_processor.ohlcv_data)
        
        # Validate we have valid data
        if len(ohlcv) == 0:
            raise ValueError("No valid OHLCV data available for charting")
        
        # Create chart structure
        n_indicator_subplots = len(self.indicator_processor.processed[IndicatorType.SEPARATE_SUBPLOT])
        has_volume = self.chart_config.show_volume and 'Volume' in ohlcv.columns
        has_equity = self.chart_config.show_equity
        
        fig = self.chart_builder.create_subplot_structure(n_indicator_subplots, has_volume, has_equity)
        
        # Add main price chart
        self._add_price_chart(fig, ohlcv)
        
        # Add indicators
        self._add_indicators(fig, ohlcv)
        
        # Add signals (delegated to signal renderer)
        if self.signal_config.show_signals:
            # Get signals in dictionary format for compatibility with renderer
            signals_dict = self.signal_processor.get_signals_dict()
            self.signal_renderer.add_signals_to_chart(
                fig, signals_dict, ohlcv
            )
        
        # Add volume
        current_row = 2
        if has_volume:
            self._add_volume_chart(fig, ohlcv, current_row)
            current_row += 1
        
        # Add equity curve
        portfolio_equity = pd.Series(dtype=float)
        if has_equity:
            portfolio_equity = self._add_equity_chart(fig, ohlcv, current_row)
            current_row += 1
        
        # CRITICAL: Apply smart ranges to fix chart scaling issues
        try:
            # Calculate total number of subplots
            total_subplots = 1  # Main price chart
            if has_volume:
                total_subplots += 1
            if has_equity:
                total_subplots += 1
            total_subplots += n_indicator_subplots
            
            self.chart_builder.apply_smart_ranges(
                fig, ohlcv, self.indicator_processor, portfolio_equity, total_subplots
            )
        except Exception as e:
            logger.warning(f"Could not apply smart ranges: {e}, using fallback autorange")
            # Fallback to autorange if smart ranges fail
            total_subplots = 1 + (1 if has_volume else 0) + (1 if has_equity else 0) + n_indicator_subplots
            for i in range(1, total_subplots + 1):
                fig.update_xaxes(autorange=True, row=i, col=1)
                fig.update_yaxes(autorange=True, row=i, col=1)
        
        # Update layout with timing-aware title and annotations
        self._finalize_chart_layout(fig, ohlcv)
        
        return fig
    
    def _apply_date_filter(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Apply date range filter if specified."""
        if self.chart_config.date_range:
            start_date, end_date = self.chart_config.date_range
            mask = (ohlcv.index >= start_date) & (ohlcv.index <= end_date)
            return ohlcv[mask]
        return ohlcv
    
    def _add_price_chart(self, fig: go.Figure, ohlcv: pd.DataFrame):
        """Add main candlestick price chart."""
        fig.add_trace(
            go.Candlestick(
                x=ohlcv.index,
                open=ohlcv['Open'],
                high=ohlcv['High'],
                low=ohlcv['Low'],
                close=ohlcv['Close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350',
                line=dict(width=1),
                showlegend=False  # Don't show OHLC in legend to prioritize signals
            ),
            row=1, col=1
        )
    
    def _add_indicators(self, fig: go.Figure, ohlcv: pd.DataFrame):
        """Add all indicators to appropriate chart locations."""
        # Price overlays
        price_overlays = self.indicator_processor.processed[IndicatorType.PRICE_OVERLAY]
        self.indicator_renderer.add_price_overlays(fig, price_overlays, ohlcv)
        
        # Separate subplots
        subplot_indicators = self.indicator_processor.processed[IndicatorType.SEPARATE_SUBPLOT]
        start_row = 2
        if self.chart_config.show_volume and 'Volume' in ohlcv.columns:
            start_row += 1
        if self.chart_config.show_equity:
            start_row += 1
        
        self.indicator_renderer.add_subplot_indicators(fig, subplot_indicators, ohlcv, start_row)
    
    def _add_volume_chart(self, fig: go.Figure, ohlcv: pd.DataFrame, row: int):
        """Add volume chart."""
        if 'Volume' not in ohlcv.columns:
            return
        
        # Color volume bars based on price movement
        volume_colors = [
            'rgba(38, 166, 154, 0.7)' if close >= open else 'rgba(239, 83, 80, 0.7)'
            for close, open in zip(ohlcv['Close'], ohlcv['Open'])
        ]
        
        fig.add_trace(
            go.Bar(
                x=ohlcv.index,
                y=ohlcv['Volume'],
                name='Volume',
                marker_color=volume_colors,
                showlegend=False  # Don't show volume in legend to prioritize signals
            ),
            row=row, col=1
        )
        
        fig.update_yaxes(title_text="Volume", row=row, col=1)
    
    def _add_equity_chart(self, fig: go.Figure, ohlcv: pd.DataFrame, row: int) -> pd.Series:
        """Add equity curve chart with drawdown and return equity series for range calculation."""
        try:
            equity = self.data_processor.get_portfolio_equity()
            equity_aligned = equity.reindex(ohlcv.index).dropna()
            
            if len(equity_aligned) > 0:
                # Equity curve - only show in legend if important
                show_equity_legend = self.legend_manager.should_show_legend('Portfolio Value')
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_aligned.index,
                        y=equity_aligned.values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#1f77b4', width=2),
                        showlegend=show_equity_legend,
                        # Ensure line connects all points for full area usage
                        connectgaps=True,
                        hovertemplate='Portfolio Value: %{y:.2f}<br>Date: %{x}<extra></extra>'
                    ),
                    row=row, col=1
                )
                
                # Drawdown as filled area - don't show in main legend
                running_max = equity_aligned.expanding().max()
                drawdown = (equity_aligned - running_max) / running_max * 100
                
                # Only add drawdown if there are actual drawdowns
                if drawdown.min() < -0.01:  # At least 0.01% drawdown
                    fig.add_trace(
                        go.Scatter(
                            x=drawdown.index,
                            y=drawdown.values,
                            mode='lines',
                            name='Drawdown %',
                            line=dict(color='red', width=1),
                            fill='tozeroy',  # Fill to zero line
                            fillcolor='rgba(255, 0, 0, 0.2)',
                            showlegend=False,  # Don't clutter legend
                            connectgaps=True,
                            hovertemplate='Drawdown: %{y:.2f}%<br>Date: %{x}<extra></extra>',
                            yaxis='y2'
                        ),
                        row=row, col=1
                    )
                    
                    # Configure secondary y-axis for drawdown
                    fig.update_layout({f'yaxis{row}': dict(side='left', title='Portfolio Value')})
                    if row > 1:
                        fig.update_layout({f'yaxis{row + len(fig.data)}': dict(
                            side='right', 
                            title='Drawdown %', 
                            overlaying=f'y{row}',
                            range=[drawdown.min() * 1.1, 5]  # Small positive range for better visibility
                        )})
            
            # Update y-axis title
            fig.update_yaxes(title_text="Portfolio Value", row=row, col=1)
            
            return equity_aligned
            
        except Exception as e:
            logger.warning(f"Could not add equity curve: {e}")
            return pd.Series(dtype=float)
    
    def _finalize_chart_layout(self, fig: go.Figure, ohlcv: pd.DataFrame):
        """Finalize chart layout with timing-aware title and annotations."""
        # Update title with timing information
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
        
        # Add subtle timing annotation if enabled
        if self.signal_config.show_timing_indicator:
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xanchor='center',
                yanchor='bottom',
                text=f"Signal Timing: {self.signal_config.signal_timing_mode}",
                showarrow=False,
                font=dict(size=12)
            )
    
    def create_simple_candlestick(
        self,
        title: str = "Price Chart",
        show_volume: bool = True
    ) -> go.Figure:
        """Create a simple candlestick chart without signals or equity."""
        return self.create_main_chart(
            title=title,
            show_volume=show_volume,
            show_signals=False,
            show_equity=False
        )
    
    def save_chart(
        self,
        fig: go.Figure,
        filename: str,
        format: str = "html"
    ) -> str:
        """Save chart to file with enhanced configuration to prevent scaling issues."""
        filepath = Path(filename)
        
        if format.lower() == "html":
            # Enhanced config to prevent chart scaling issues
            config = {
                'displayModeBar': True,
                'displaylogo': False,
                'responsive': True,
                'scrollZoom': True,
                'doubleClick': 'reset+autosize',  # Reset zoom on double-click
                'modeBarButtonsToAdd': ['resetScale2d'],  # Add reset scale button
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],  # Remove problematic tools
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
                # Additional options to ensure proper rendering
                div_id="chart",
                full_html=True
            )
        elif format.lower() in ["png", "jpg", "jpeg", "pdf", "svg"]:
            fig.write_image(
                filepath.with_suffix(f'.{format.lower()}'),
                width=1920,
                height=1080,
                scale=2,
                # Ensure proper rendering
                engine="kaleido"
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Chart saved to {filepath}")
        return str(filepath)
    
    def get_chart_info(self) -> Dict[str, Any]:
        """Get information about the chart data for debugging."""
        # Get signals in dictionary format for compatibility
        signals_dict = self.signal_processor.get_signals_dict()
        
        return {
            'ohlcv_shape': self.data_processor.ohlcv_data.shape,
            'indicators_count': len(self.indicator_processor.indicators),
            'signals_extracted': {
                'long_entries': signals_dict['long_entries'].sum() if 'long_entries' in signals_dict else 0,
                'short_entries': signals_dict['short_entries'].sum() if 'short_entries' in signals_dict else 0,
                'exits': signals_dict.get('exits', signals_dict.get('long_exits', pd.Series(dtype=bool))).sum()
            },
            'portfolio_has_trades': hasattr(self.portfolio, 'trades') and len(getattr(self.portfolio.trades, 'records', [])) > 0,
            'signal_timing_mode': self.signal_config.signal_timing_mode,
            'execution_delay': self.signal_config.execution_delay
        }


# Convenience functions for backward compatibility and ease of use
def create_simple_candlestick(
    portfolio: vbt.Portfolio,
    data: vbt.Data,
    indicators: Optional[Dict[str, Any]] = None,
    title: str = "Price Chart"
) -> go.Figure:
    """Create a simple candlestick chart with indicators but no signals."""
    charts = TradingChartsEngine(portfolio, data, indicators)
    return charts.create_simple_candlestick(title=title)


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
    
    charts = TradingChartsEngine(portfolio, data, indicators, signals, 
                               chart_config=chart_config, signal_config=signal_config)
    return charts.create_main_chart()


def create_execution_timing_chart(
    portfolio: vbt.Portfolio,
    data: vbt.Data,
    indicators: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    title: str = "Trading Analysis - Execution Timing"
) -> go.Figure:
    """Create a chart with execution timing (realistic mode)."""
    return create_trading_analysis(portfolio, data, indicators, signals, title, "execution")


def create_signal_timing_chart(
    portfolio: vbt.Portfolio,
    data: vbt.Data,
    indicators: Optional[Dict[str, Any]] = None,
    signals: Optional[Dict[str, Any]] = None,
    title: str = "Trading Analysis - Signal Timing"
) -> go.Figure:
    """Create a chart with signal timing (analysis mode)."""
    return create_trading_analysis(portfolio, data, indicators, signals, title, "signal") 