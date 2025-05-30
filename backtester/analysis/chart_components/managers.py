"""
Chart State and Configuration Managers

This module contains manager components responsible for maintaining state,
configuration, and consistency across chart rendering operations. Managers
implement the IManager interface defined in base.py.

Module Structure:
================
1. LegendManager: Prevents legend duplication and manages legend appearance
2. LayoutManager: Manages chart layout and spacing (future)
3. ThemeManager: Manages consistent styling across components (future)

Purpose and Integration:
=======================
Managers solve cross-cutting concerns that affect multiple components:
- Legend deduplication across multiple renderers
- Consistent styling and theming
- Layout coordination between subplots
- State management during rendering

Data Flow:
==========
Renderers → LegendManager → Check if legend item should be shown
                         ↘ Track shown items to prevent duplicates

ChartBuilder → LayoutManager → Calculate optimal subplot heights
                            ↘ Coordinate spacing between elements

Related Modules:
===============
- base.py: Defines IManager interface
- renderers.py: Uses managers for legend and style decisions
- builders.py: Uses managers for layout calculations
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field

from backtester.analysis.chart_components.base import IManager, ComponentConfig
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class LegendConfig(ComponentConfig):
    """
    Configuration for legend management.
    
    Controls legend appearance and behavior across all chart components.
    """
    name: str = "LegendManager"
    orientation: str = "h"  # horizontal or "v" for vertical
    position: str = "top"   # top, bottom, left, right
    max_items: int = 20     # Maximum legend items before hiding
    priority_items: List[str] = field(default_factory=lambda: [
        "Long Entry", "Short Entry", "Exit",
        "Stop Loss", "Take Profit",
        "Portfolio Value"
    ])
    
    @property
    def plotly_config(self) -> Dict[str, Any]:
        """Convert to Plotly legend configuration."""
        # Map position to Plotly anchor points
        position_map = {
            "top": {"yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            "bottom": {"yanchor": "top", "y": -0.1, "xanchor": "right", "x": 1},
            "left": {"yanchor": "middle", "y": 0.5, "xanchor": "right", "x": -0.1},
            "right": {"yanchor": "middle", "y": 0.5, "xanchor": "left", "x": 1.02}
        }
        
        config = {
            "orientation": self.orientation,
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "rgba(0,0,0,0.1)",
            "borderwidth": 1
        }
        config.update(position_map.get(self.position, position_map["top"]))
        
        return config


class LegendManager(IManager):
    """
    Manages legend configuration and prevents item duplication.
    
    When multiple renderers add traces to a chart, they may try to add
    the same legend items multiple times (e.g., multiple "Long Entry" markers).
    This manager tracks which items have been shown and prevents duplicates.
    
    It also prioritizes important legend items when space is limited.
    
    Usage:
        legend_mgr = LegendManager(config)
        
        # In renderer:
        if legend_mgr.should_show_legend("Long Entry"):
            trace.showlegend = True
        else:
            trace.showlegend = False
    
    Integration Points:
    - Used by: All renderers (IndicatorRenderer, SignalRenderer)
    - Purpose: Prevent legend clutter and duplication
    - State: Tracks shown legend items per rendering session
    """
    
    def __init__(self, config: Optional[LegendConfig] = None):
        """
        Initialize the legend manager.
        
        Args:
            config: Legend configuration
        """
        self.config = config or LegendConfig()
        self.shown_items: Set[str] = set()
        self._item_counts: Dict[str, int] = {}
    
    def validate(self) -> bool:
        """Validate manager configuration."""
        return isinstance(self.config, LegendConfig)
    
    def should_show_legend(self, item_name: str) -> bool:
        """
        Determine if a legend item should be shown.
        
        Decision process:
        1. Check if item has already been shown
        2. Check if we've reached the maximum items
        3. Prioritize important items
        
        Args:
            item_name: Name of the legend item
            
        Returns:
            True if the item should be shown in legend
        """
        # Track item count
        self._item_counts[item_name] = self._item_counts.get(item_name, 0) + 1
        
        # Already shown? Don't show again
        if item_name in self.shown_items:
            return False
        
        # Check if we've hit the limit
        if len(self.shown_items) >= self.config.max_items:
            # Only show if it's a priority item
            if item_name not in self.config.priority_items:
                logger.debug(f"Legend item '{item_name}' hidden (max items reached)")
                return False
        
        # Mark as shown and return True
        self.shown_items.add(item_name)
        return True
    
    def reset(self) -> None:
        """
        Reset manager state for a new rendering session.
        
        Called before starting to render a new chart.
        """
        self.shown_items.clear()
        self._item_counts.clear()
        logger.debug("Legend manager reset")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current manager state for debugging.
        
        Returns:
            Dictionary with state information
        """
        return {
            "shown_items": list(self.shown_items),
            "item_counts": dict(self._item_counts),
            "total_items": len(self.shown_items),
            "config": {
                "max_items": self.config.max_items,
                "orientation": self.config.orientation,
                "position": self.config.position
            }
        }
    
    def get_plotly_config(self) -> Dict[str, Any]:
        """
        Get Plotly legend configuration.
        
        Returns:
            Dictionary of Plotly legend parameters
        """
        return self.config.plotly_config


@dataclass
class LayoutConfig(ComponentConfig):
    """
    Configuration for layout management.
    
    Controls subplot sizing, spacing, and arrangement.
    """
    name: str = "LayoutManager"
    
    # Height distribution (percentages)
    main_chart_height: float = 0.5      # Main price chart
    volume_chart_height: float = 0.1    # Volume subplot
    equity_chart_height: float = 0.2    # Equity curve subplot
    indicator_min_height: float = 0.12  # Minimum height per indicator
    
    # Spacing
    vertical_spacing: float = 0.03      # Space between subplots
    margin_top: int = 120               # Top margin (pixels)
    margin_bottom: int = 80             # Bottom margin
    margin_left: int = 80               # Left margin
    margin_right: int = 80              # Right margin
    
    # Responsive adjustments
    auto_adjust_heights: bool = True    # Automatically adjust heights
    max_subplots: int = 6               # Maximum number of subplots


class LayoutManager(IManager):
    """
    Manages chart layout and subplot arrangement.
    
    This manager calculates optimal subplot heights and spacing based on
    the content to be displayed. It ensures charts remain readable even
    with many indicators.
    
    Features:
    - Dynamic height calculation based on content
    - Minimum height guarantees for readability
    - Responsive spacing adjustments
    - Priority-based height allocation
    
    Usage:
        layout_mgr = LayoutManager(config)
        heights = layout_mgr.calculate_subplot_heights(
            has_volume=True,
            has_equity=True,
            n_indicators=3
        )
    
    Integration Points:
    - Used by: ChartBuilder for subplot creation
    - Purpose: Optimize chart layout for readability
    - Future: Could manage dynamic resizing
    """
    
    def __init__(self, config: Optional[LayoutConfig] = None):
        """
        Initialize the layout manager.
        
        Args:
            config: Layout configuration
        """
        self.config = config or LayoutConfig()
        self._layout_cache: Dict[str, Any] = {}
    
    def validate(self) -> bool:
        """Validate manager configuration."""
        # Check that heights are valid percentages
        total_min = (self.config.main_chart_height + 
                    self.config.volume_chart_height + 
                    self.config.equity_chart_height)
        
        if total_min > 1.0:
            logger.warning(f"Total minimum heights exceed 100%: {total_min}")
            return False
            
        return True
    
    def calculate_subplot_heights(
        self,
        has_volume: bool,
        has_equity: bool,
        n_indicators: int
    ) -> List[float]:
        """
        Calculate optimal heights for all subplots.
        
        Algorithm:
        1. Start with configured heights for main components
        2. Allocate remaining space to indicators
        3. Ensure minimum heights are respected
        4. Scale if total exceeds 100%
        
        Args:
            has_volume: Whether volume subplot is needed
            has_equity: Whether equity subplot is needed  
            n_indicators: Number of indicator subplots
            
        Returns:
            List of heights (0-1) for each subplot
        """
        heights = []
        
        # Main chart always present
        main_height = self.config.main_chart_height
        heights.append(main_height)
        
        # Optional volume chart
        volume_height = self.config.volume_chart_height if has_volume else 0
        if has_volume:
            heights.append(volume_height)
        
        # Optional equity chart
        equity_height = self.config.equity_chart_height if has_equity else 0
        if has_equity:
            heights.append(equity_height)
        
        # Calculate remaining space for indicators
        used_height = main_height + volume_height + equity_height
        remaining_height = 1.0 - used_height
        
        if n_indicators > 0:
            # Distribute remaining height among indicators
            indicator_height = max(
                self.config.indicator_min_height,
                remaining_height / n_indicators
            )
            
            # Add indicator heights
            heights.extend([indicator_height] * n_indicators)
            
            # Check if we exceeded 100%
            total_height = sum(heights)
            if total_height > 1.0 and self.config.auto_adjust_heights:
                # Scale all heights proportionally
                scale_factor = 1.0 / total_height
                heights = [h * scale_factor for h in heights]
                
                logger.debug(f"Scaled subplot heights by {scale_factor:.2f} to fit")
        
        # Cache the calculation
        cache_key = f"{has_volume}_{has_equity}_{n_indicators}"
        self._layout_cache[cache_key] = heights
        
        return heights
    
    def calculate_vertical_spacing(self, n_subplots: int) -> float:
        """
        Calculate optimal vertical spacing based on subplot count.
        
        More subplots = less spacing to maximize content area.
        
        Args:
            n_subplots: Total number of subplots
            
        Returns:
            Vertical spacing value (0-1)
        """
        if n_subplots <= 2:
            return 0.05
        elif n_subplots <= 4:
            return self.config.vertical_spacing
        else:
            # Reduce spacing for many subplots
            return 0.02
    
    def get_margin_dict(self) -> Dict[str, int]:
        """
        Get margin configuration for Plotly layout.
        
        Returns:
            Dictionary of margin values
        """
        return {
            "t": self.config.margin_top,
            "b": self.config.margin_bottom,
            "l": self.config.margin_left,
            "r": self.config.margin_right
        }
    
    def reset(self) -> None:
        """Reset manager state."""
        self._layout_cache.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current manager state."""
        return {
            "cached_layouts": list(self._layout_cache.keys()),
            "config": {
                "main_height": self.config.main_chart_height,
                "volume_height": self.config.volume_chart_height,
                "equity_height": self.config.equity_chart_height,
                "indicator_min_height": self.config.indicator_min_height
            }
        }


@dataclass  
class ThemeConfig(ComponentConfig):
    """
    Configuration for theme management.
    
    Defines colors, fonts, and visual styles.
    """
    name: str = "ThemeManager"
    
    # Plotly template
    template: str = "plotly_white"
    
    # Color schemes
    bullish_color: str = "#26a69a"     # Green for up moves
    bearish_color: str = "#ef5350"     # Red for down moves
    
    # Indicator colors (cycling)
    indicator_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ])
    
    # Signal colors
    signal_colors: Dict[str, str] = field(default_factory=lambda: {
        'long_entry': 'lime',
        'short_entry': 'orangered',
        'exit': 'purple',
        'stop_loss': 'red',
        'take_profit': 'green'
    })
    
    # Font settings
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16


class ThemeManager(IManager):
    """
    Manages consistent theming across chart components.
    
    Provides a centralized place for color schemes, fonts, and visual styles
    to ensure consistency across all chart elements.
    
    Features:
    - Predefined color schemes
    - Cycling colors for multiple indicators
    - Consistent font settings
    - Dark/light theme support (future)
    
    Usage:
        theme_mgr = ThemeManager(config)
        color = theme_mgr.get_indicator_color(index=0)
        signal_color = theme_mgr.get_signal_color("long_entry")
    
    Integration Points:
    - Used by: All renderers for consistent styling
    - Purpose: Centralize theme management
    - Future: Support multiple theme presets
    """
    
    def __init__(self, config: Optional[ThemeConfig] = None):
        """
        Initialize the theme manager.
        
        Args:
            config: Theme configuration
        """
        self.config = config or ThemeConfig()
        self._color_index = 0
    
    def validate(self) -> bool:
        """Validate theme configuration."""
        return len(self.config.indicator_colors) > 0
    
    def get_indicator_color(self, index: Optional[int] = None) -> str:
        """
        Get color for an indicator.
        
        Args:
            index: Specific color index, or None for next in cycle
            
        Returns:
            Hex color string
        """
        if index is not None:
            return self.config.indicator_colors[index % len(self.config.indicator_colors)]
        
        # Auto-increment and cycle
        color = self.config.indicator_colors[self._color_index]
        self._color_index = (self._color_index + 1) % len(self.config.indicator_colors)
        return color
    
    def get_signal_color(self, signal_type: str) -> str:
        """
        Get color for a signal type.
        
        Args:
            signal_type: Type of signal (long_entry, short_entry, etc.)
            
        Returns:
            Hex color string
        """
        return self.config.signal_colors.get(signal_type, '#000000')
    
    def get_volume_colors(self, is_bullish: List[bool]) -> List[str]:
        """
        Get colors for volume bars based on price movement.
        
        Args:
            is_bullish: List of boolean values (True for up bars)
            
        Returns:
            List of color strings with transparency
        """
        bullish_color = f"rgba{self._hex_to_rgba(self.config.bullish_color, 0.7)}"
        bearish_color = f"rgba{self._hex_to_rgba(self.config.bearish_color, 0.7)}"
        
        return [bullish_color if bull else bearish_color for bull in is_bullish]
    
    def get_font_dict(self) -> Dict[str, Any]:
        """
        Get font configuration for Plotly.
        
        Returns:
            Dictionary of font settings
        """
        return {
            "family": self.config.font_family,
            "size": self.config.font_size
        }
    
    def get_title_font_dict(self) -> Dict[str, Any]:
        """Get title font configuration."""
        return {
            "family": self.config.font_family,
            "size": self.config.title_font_size
        }
    
    def reset(self) -> None:
        """Reset color cycling."""
        self._color_index = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current theme state."""
        return {
            "template": self.config.template,
            "color_index": self._color_index,
            "theme_name": self.config.name
        }
    
    @staticmethod
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        """Convert hex color to RGBA tuple string."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"({r}, {g}, {b}, {alpha})"


# Module exports
__all__ = [
    'LegendManager',
    'LegendConfig',
    'LayoutManager',
    'LayoutConfig',
    'ThemeManager',
    'ThemeConfig'
]