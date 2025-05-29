"""
VectorBTPro Compatibility Layer

This module handles differences between vectorbtpro versions and provides
safe access to portfolio methods and properties.
"""

import pandas as pd
import numpy as np
import logging
from typing import Any, Optional, Union, Callable

logger = logging.getLogger(__name__)


class VBTCompatibilityLayer:
    """Handles differences between vectorbtpro versions and provides safe access methods."""
    
    @staticmethod
    def get_portfolio_returns(portfolio: Any) -> pd.Series:
        """
        Safely get portfolio returns handling both property and method access.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            
        Returns:
            Portfolio returns as pandas Series
        """
        try:
            if hasattr(portfolio, 'returns'):
                returns = portfolio.returns
                if callable(returns):
                    returns = returns()
                return returns
            elif hasattr(portfolio, 'value'):
                # Fallback: calculate returns from value
                value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                return value.pct_change()
            else:
                raise AttributeError("Portfolio object has no returns or value attribute")
        except Exception as e:
            logger.error(f"Error getting portfolio returns: {e}")
            raise
    
    @staticmethod
    def get_portfolio_value(portfolio: Any) -> pd.Series:
        """
        Safely get portfolio value handling both property and method access.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            
        Returns:
            Portfolio value as pandas Series
        """
        try:
            if hasattr(portfolio, 'value'):
                value = portfolio.value
                if callable(value):
                    value = value()
                return value
            elif hasattr(portfolio, 'total_value'):
                # Alternative attribute name
                value = portfolio.total_value
                if callable(value):
                    value = value()
                return value
            else:
                raise AttributeError("Portfolio object has no value attribute")
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            raise
    
    @staticmethod
    def get_portfolio_drawdowns(portfolio: Any) -> pd.Series:
        """
        Safely get portfolio drawdowns.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            
        Returns:
            Portfolio drawdowns as pandas Series
        """
        try:
            if hasattr(portfolio, 'drawdowns'):
                drawdowns = portfolio.drawdowns
                if callable(drawdowns):
                    drawdowns = drawdowns()
                
                # Check if it's a Drawdowns object with drawdown attribute
                if hasattr(drawdowns, 'drawdown'):
                    # It's a vectorbtpro Drawdowns object
                    dd_values = drawdowns.drawdown
                    if hasattr(dd_values, 'values') and hasattr(dd_values, 'index'):
                        # It's already a Series
                        return dd_values
                    elif hasattr(dd_values, 'values'):
                        # Get the index from portfolio value
                        value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                        # Make sure lengths match
                        if len(dd_values.values) == len(value):
                            return pd.Series(dd_values.values, index=value.index)
                        else:
                            # Try to get the actual drawdown series
                            if hasattr(drawdowns, 'max_drawdown'):
                                # Calculate drawdowns from value
                                value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                                cummax = value.expanding().max()
                                return (value - cummax) / cummax
                            else:
                                return dd_values
                    else:
                        return dd_values
                elif hasattr(drawdowns, 'values'):
                    # It's already a Series
                    return drawdowns
                else:
                    # Try to convert to Series
                    value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                    return pd.Series(drawdowns, index=value.index)
            else:
                # Fallback: calculate drawdowns from value
                value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                cummax = value.expanding().max()
                drawdowns = (value - cummax) / cummax
                return drawdowns
        except Exception as e:
            logger.error(f"Error getting portfolio drawdowns: {e}")
            # Last resort fallback
            try:
                value = VBTCompatibilityLayer.get_portfolio_value(portfolio)
                cummax = value.expanding().max()
                return (value - cummax) / cummax
            except:
                raise
    
    @staticmethod
    def get_portfolio_trades(portfolio: Any) -> Any:
        """
        Safely get portfolio trades/orders.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            
        Returns:
            Portfolio trades object
        """
        try:
            # Try different attribute names
            for attr in ['trades', 'orders', 'filled_orders']:
                if hasattr(portfolio, attr):
                    trades = getattr(portfolio, attr)
                    if callable(trades):
                        trades = trades()
                    return trades
            
            logger.warning("No trades attribute found on portfolio")
            return None
        except Exception as e:
            logger.error(f"Error getting portfolio trades: {e}")
            return None
    
    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 252, periods: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Returns series
            window: Rolling window size
            periods: Annualization factor (252 for daily data)
            
        Returns:
            Rolling Sharpe ratio
        """
        try:
            # Check if vectorbtpro has the method
            if hasattr(returns, 'vbt') and hasattr(returns.vbt, 'returns'):
                if hasattr(returns.vbt.returns, 'rolling_sharpe'):
                    return returns.vbt.returns.rolling_sharpe(window, periods=periods)
            
            # Fallback calculation
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            return rolling_mean / rolling_std * np.sqrt(periods)
        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}")
            raise
    
    @staticmethod
    def calculate_rolling_volatility(returns: pd.Series, window: int = 252, periods: int = 252) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            returns: Returns series
            window: Rolling window size
            periods: Annualization factor (252 for daily data)
            
        Returns:
            Rolling volatility
        """
        try:
            # Check if vectorbtpro has the method
            if hasattr(returns, 'vbt') and hasattr(returns.vbt, 'returns'):
                if hasattr(returns.vbt.returns, 'rolling_volatility'):
                    return returns.vbt.returns.rolling_volatility(window, periods=periods)
            
            # Fallback calculation
            return returns.rolling(window).std() * np.sqrt(periods)
        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {e}")
            raise
    
    @staticmethod
    def calculate_rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling maximum drawdown.
        
        Args:
            returns: Returns series
            window: Rolling window size
            
        Returns:
            Rolling maximum drawdown
        """
        try:
            # Check if vectorbtpro has the method
            if hasattr(returns, 'vbt') and hasattr(returns.vbt, 'returns'):
                if hasattr(returns.vbt.returns, 'rolling_max_drawdown'):
                    return returns.vbt.returns.rolling_max_drawdown(window)
            
            # Fallback calculation
            def calc_max_dd(returns_window):
                cum_returns = (1 + returns_window).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns - running_max) / running_max
                return drawdown.min()
            
            return returns.rolling(window).apply(calc_max_dd)
        except Exception as e:
            logger.error(f"Error calculating rolling max drawdown: {e}")
            raise
    
    @staticmethod
    def safe_plot_portfolio(portfolio: Any, subplots: list = None, **kwargs) -> Any:
        """
        Safely plot portfolio with fallback options.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            subplots: List of subplots to include
            **kwargs: Additional plotting arguments
            
        Returns:
            Plotly figure or None if plotting fails
        """
        try:
            # Try vectorbtpro's plot method
            if hasattr(portfolio, 'plot'):
                if subplots is None:
                    # Start with basic subplots we know should exist
                    subplots = ['value', 'drawdowns']
                
                try:
                    return portfolio.plot(subplots=subplots, **kwargs)
                except Exception as e:
                    logger.warning(f"Portfolio plot failed with subplots {subplots}: {e}")
                    # Try with minimal subplots
                    try:
                        return portfolio.plot(subplots=['value'], **kwargs)
                    except:
                        logger.warning("Portfolio plot failed even with minimal subplots")
                        return None
            else:
                logger.warning("Portfolio object has no plot method")
                return None
        except Exception as e:
            logger.error(f"Error plotting portfolio: {e}")
            return None
    
    @staticmethod
    def get_safe_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
        """
        Safely get an attribute from an object, handling both property and method access.
        
        Args:
            obj: Object to get attribute from
            attr_name: Name of the attribute
            default: Default value if attribute not found
            
        Returns:
            Attribute value or default
        """
        try:
            if hasattr(obj, attr_name):
                attr = getattr(obj, attr_name)
                if callable(attr):
                    return attr()
                return attr
            return default
        except Exception as e:
            logger.warning(f"Error getting attribute {attr_name}: {e}")
            return default 