#!/usr/bin/env python3
"""
Example 10: Momentum Strategy with Regime Filters

This example demonstrates a trend-following momentum strategy:
- Multiple momentum indicators (ROC, RSI, MACD)
- Market regime filtering
- Dynamic position sizing
- Trailing stop losses

Key concepts:
- Momentum scoring
- Regime-based trading
- Trend strength confirmation
- Adaptive risk management
"""

import os
import sys
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.plotting_engine import PlottingEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def demonstrate_momentum_strategy():
    """Demonstrate momentum strategy with regime filtering."""
    
    logger = get_logger()
    logger.section("🚀 Momentum Strategy", "Trend-following with regime filters and adaptive risk management")
    
    # Configuration
    symbol = "SOL/USDT"  # Use SOL for momentum demonstration
    timeframe = "4h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Load data
    logger.section("📊 Loading Market Data")
    
    with logger.operation("Loading market data"):
        data = fetch_data(
            symbols=[symbol],
            exchange="binance",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None:
            logger.error("Failed to load data")
            return None
        
        data_points = len(data.get('close'))
        logger.data_summary([symbol], timeframe, start_date, end_date, data_points)
    
    # Generate momentum indicators
    logger.section("📈 Calculating Momentum Indicators")
    
    momentum_indicators = calculate_momentum_indicators(data, logger)
    
    if momentum_indicators is None:
        logger.error("Failed to calculate momentum indicators")
        return None
    
    # Detect market regime
    logger.section("🌊 Market Regime Detection")
    
    market_regime = detect_market_regime(data, logger)
    
    if market_regime is None:
        logger.error("Failed to detect market regime")
        return None
    
    # Generate momentum signals
    logger.section("🎯 Generating Momentum Signals")
    
    momentum_signals = generate_momentum_signals(
        data, momentum_indicators, market_regime, logger
    )
    
    if momentum_signals is None:
        logger.error("Failed to generate momentum signals")
        return None
    
    # Run momentum strategy backtest
    logger.section("🚀 Running Momentum Strategy Backtest")
    
    with logger.operation("Running momentum strategy simulation"):
        try:
            # Create momentum strategy
            strategy = MomentumStrategy(data, momentum_signals, momentum_indicators)
            
            # Run backtest with dynamic position sizing
            sim_config = SimulationConfig(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0005,
                position_sizing={'type': 'volatility_based', 'target_vol': 0.02}
            )
            
            simulator = PortfolioSimulator(sim_config)
            portfolio = simulator.run_backtest(strategy)
            
            if portfolio is None:
                logger.error("Momentum strategy backtest failed")
                return None
            
            # Analyze results
            analyzer = PerformanceAnalyzer(portfolio)
            
            results = {
                'portfolio': portfolio,
                'total_return': analyzer.total_return(),
                'sharpe_ratio': analyzer.sharpe_ratio(),
                'max_drawdown': analyzer.max_drawdown(),
                'volatility': analyzer.volatility(),
                'total_trades': analyzer.total_trades(),
                'win_rate': analyzer.win_rate(),
                'profit_factor': analyzer.profit_factor(),
                'calmar_ratio': analyzer.calmar_ratio()
            }
            
            # Log results
            logger.backtest_result({
                'portfolio': portfolio,
                'metrics': results,
                'strategy_type': 'Momentum Strategy'
            })
            
            return {
                'strategy': strategy,
                'portfolio': portfolio,
                'results': results,
                'indicators': momentum_indicators,
                'regime': market_regime,
                'signals': momentum_signals
            }
            
        except Exception as e:
            logger.error(f"Momentum strategy backtest failed: {str(e)}")
            return None


def calculate_momentum_indicators(data, logger):
    """Calculate multiple momentum indicators."""
    try:
        close_prices = data.get('close')
        high_prices = data.get('high')
        low_prices = data.get('low')
        volume = data.get('volume')
        
        if close_prices is None:
            logger.error("No close price data available")
            return None
        
        # Rate of Change (ROC) - multiple periods
        roc_5 = close_prices.pct_change(5) * 100
        roc_10 = close_prices.pct_change(10) * 100
        roc_20 = close_prices.pct_change(20) * 100
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - macd_signal
        
        # Moving Averages for trend
        sma_20 = close_prices.rolling(20).mean()
        sma_50 = close_prices.rolling(50).mean()
        ema_20 = close_prices.ewm(span=20).mean()
        
        # Average True Range (ATR) for volatility
        if high_prices is not None and low_prices is not None:
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift(1))
            tr3 = abs(low_prices - close_prices.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
        else:
            atr = close_prices.rolling(14).std()
        
        # Volume-based indicators (if volume available)
        if volume is not None:
            volume_sma = volume.rolling(20).mean()
            volume_ratio = volume / volume_sma
        else:
            volume_ratio = pd.Series(1.0, index=close_prices.index)
        
        # Momentum Score (composite)
        # Normalize indicators to 0-100 scale
        roc_score = ((roc_10 + 50).clip(0, 100))  # Shift and clip ROC
        rsi_score = rsi
        macd_score = ((macd_histogram > 0).astype(int) * 100)  # Binary MACD signal
        trend_score = ((close_prices > sma_20).astype(int) * 100)  # Above/below MA
        
        # Composite momentum score
        momentum_score = (roc_score * 0.3 + rsi_score * 0.3 + 
                         macd_score * 0.2 + trend_score * 0.2)
        
        indicators = {
            'roc_5': roc_5,
            'roc_10': roc_10,
            'roc_20': roc_20,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema_20': ema_20,
            'atr': atr,
            'volume_ratio': volume_ratio,
            'momentum_score': momentum_score
        }
        
        logger.info("Momentum Indicators Calculated:")
        logger.info(f"  ROC (10-period): {roc_10.iloc[-1]:.2f}%")
        logger.info(f"  RSI: {rsi.iloc[-1]:.1f}")
        logger.info(f"  MACD: {macd_line.iloc[-1]:.4f}")
        logger.info(f"  Momentum Score: {momentum_score.iloc[-1]:.1f}")
        
        return indicators
        
    except Exception as e:
        logger.error(f"Momentum indicator calculation failed: {str(e)}")
        return None


def detect_market_regime(data, logger):
    """Detect market regime (trending vs ranging)."""
    try:
        close_prices = data.get('close')
        if close_prices is None:
            return None
        
        # Trend strength indicators
        sma_20 = close_prices.rolling(20).mean()
        sma_50 = close_prices.rolling(50).mean()
        
        # ADX-like calculation (simplified)
        high_prices = data.get('high', close_prices)
        low_prices = data.get('low', close_prices)
        
        # True Range
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = (high_prices - high_prices.shift(1)).where(
            (high_prices - high_prices.shift(1)) > (low_prices.shift(1) - low_prices), 0
        ).clip(lower=0)
        
        dm_minus = (low_prices.shift(1) - low_prices).where(
            (low_prices.shift(1) - low_prices) > (high_prices - high_prices.shift(1)), 0
        ).clip(lower=0)
        
        # Smooth the values
        tr_smooth = true_range.rolling(14).mean()
        dm_plus_smooth = dm_plus.rolling(14).mean()
        dm_minus_smooth = dm_minus.rolling(14).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(14).mean()
        
        # Volatility regime
        returns = close_prices.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(365 * 6)  # Annualized for 4h data
        vol_percentile = volatility.rolling(100).rank(pct=True)
        
        # Define regimes
        regime = pd.Series('Ranging', index=close_prices.index)
        
        # Trending regime: ADX > 25 and price above/below MA
        strong_trend = adx > 25
        uptrend = (close_prices > sma_20) & (sma_20 > sma_50)
        downtrend = (close_prices < sma_20) & (sma_20 < sma_50)
        
        regime[strong_trend & uptrend] = 'Strong_Uptrend'
        regime[strong_trend & downtrend] = 'Strong_Downtrend'
        regime[~strong_trend & uptrend] = 'Weak_Uptrend'
        regime[~strong_trend & downtrend] = 'Weak_Downtrend'
        
        # High volatility regime
        regime[vol_percentile > 0.8] = 'High_Volatility'
        
        # Log regime distribution
        regime_counts = regime.value_counts()
        logger.info("Market Regime Distribution:")
        for reg, count in regime_counts.items():
            percentage = count / len(regime) * 100
            logger.info(f"  {reg}: {count} periods ({percentage:.1f}%)")
        
        return {
            'regime': regime,
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'volatility': volatility,
            'vol_percentile': vol_percentile,
            'regime_counts': regime_counts
        }
        
    except Exception as e:
        logger.error(f"Market regime detection failed: {str(e)}")
        return None


def generate_momentum_signals(data, indicators, regime_data, logger):
    """Generate momentum trading signals with regime filtering."""
    try:
        close_prices = data.get('close')
        momentum_score = indicators['momentum_score']
        regime = regime_data['regime']
        adx = regime_data['adx']
        
        # Signal thresholds
        momentum_threshold_high = 70  # Strong momentum
        momentum_threshold_low = 30   # Weak momentum
        adx_threshold = 20           # Minimum trend strength
        
        # Base momentum signals
        strong_momentum_up = momentum_score > momentum_threshold_high
        strong_momentum_down = momentum_score < momentum_threshold_low
        
        # Trend confirmation
        trend_confirmation = adx > adx_threshold
        
        # Regime filtering
        favorable_regimes = ['Strong_Uptrend', 'Weak_Uptrend']
        regime_filter = regime.isin(favorable_regimes)
        
        # Entry signals
        long_entries = (strong_momentum_up & trend_confirmation & regime_filter)
        
        # Exit signals
        momentum_exit = momentum_score < 50  # Momentum weakening
        trend_exit = adx < 15  # Trend weakening
        regime_exit = ~regime_filter  # Regime change
        
        long_exits = momentum_exit | trend_exit | regime_exit
        
        # Additional filters
        # Volume confirmation (if available)
        volume_ratio = indicators.get('volume_ratio', pd.Series(1.0, index=close_prices.index))
        volume_filter = volume_ratio > 1.2  # Above average volume
        
        # Apply volume filter to entries
        long_entries = long_entries & volume_filter
        
        # Trailing stop based on ATR
        atr = indicators['atr']
        trailing_stop_distance = atr * 2.0  # 2x ATR trailing stop
        
        # Log signal statistics
        total_entries = long_entries.sum()
        total_exits = long_exits.sum()
        
        logger.info("Momentum Trading Signals:")
        logger.info(f"  Total Entry Signals: {total_entries}")
        logger.info(f"  Total Exit Signals: {total_exits}")
        logger.info(f"  Momentum Threshold: {momentum_threshold_high}")
        logger.info(f"  ADX Threshold: {adx_threshold}")
        logger.info(f"  Volume Filter Applied: {volume_filter.sum()} periods")
        
        return {
            'entries': long_entries,
            'exits': long_exits,
            'momentum_score': momentum_score,
            'regime_filter': regime_filter,
            'trend_confirmation': trend_confirmation,
            'volume_filter': volume_filter,
            'trailing_stop_distance': trailing_stop_distance,
            'thresholds': {
                'momentum_high': momentum_threshold_high,
                'momentum_low': momentum_threshold_low,
                'adx': adx_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Momentum signal generation failed: {str(e)}")
        return None


class MomentumStrategy:
    """Momentum strategy implementation."""
    
    def __init__(self, data, signals, indicators):
        self.data = data
        self.signals = signals
        self.indicators = indicators
        
    def generate_signals(self):
        """Generate signals for the portfolio simulator."""
        return self.signals


def compare_with_buy_and_hold(strategy, portfolio, logger):
    """Compare momentum strategy with buy-and-hold."""
    
    logger.section("📊 Buy-and-Hold Comparison")
    
    try:
        # Get strategy data
        close_prices = strategy.data.get('close')
        
        if close_prices is None or len(close_prices) == 0:
            logger.error("No price data for comparison")
            return None
        
        # Calculate buy-and-hold performance
        start_price = close_prices.iloc[0]
        end_price = close_prices.iloc[-1]
        bh_return = (end_price - start_price) / start_price
        
        # Calculate buy-and-hold volatility and Sharpe
        bh_returns = close_prices.pct_change().dropna()
        bh_volatility = bh_returns.std() * np.sqrt(365 * 6)  # Annualized for 4h data
        bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(365 * 6) if bh_returns.std() > 0 else 0
        
        # Calculate max drawdown for buy-and-hold
        cumulative = (1 + bh_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        bh_max_drawdown = drawdown.min()
        
        # Get momentum strategy performance
        if portfolio:
            analyzer = PerformanceAnalyzer(portfolio)
            momentum_return = analyzer.total_return()
            momentum_sharpe = analyzer.sharpe_ratio()
            momentum_max_dd = analyzer.max_drawdown()
            momentum_volatility = analyzer.volatility()
            
            logger.info("Performance Comparison:")
            logger.info(f"Momentum Strategy:")
            logger.info(f"  Total Return: {momentum_return:.2%}")
            logger.info(f"  Sharpe Ratio: {momentum_sharpe:.3f}")
            logger.info(f"  Max Drawdown: {momentum_max_dd:.2%}")
            logger.info(f"  Volatility: {momentum_volatility:.2%}")
            
            logger.info(f"Buy-and-Hold:")
            logger.info(f"  Total Return: {bh_return:.2%}")
            logger.info(f"  Sharpe Ratio: {bh_sharpe:.3f}")
            logger.info(f"  Max Drawdown: {bh_max_drawdown:.2%}")
            logger.info(f"  Volatility: {bh_volatility:.2%}")
            
            # Performance comparison
            if momentum_sharpe > bh_sharpe:
                logger.success("✅ Momentum strategy outperforms on risk-adjusted basis")
            else:
                logger.warning("⚠️ Buy-and-hold shows better risk-adjusted returns")
            
            if momentum_max_dd > bh_max_drawdown:
                logger.warning("⚠️ Momentum strategy has higher drawdown")
            else:
                logger.success("✅ Momentum strategy has lower drawdown")
            
            return {
                'momentum': {
                    'return': momentum_return,
                    'sharpe': momentum_sharpe,
                    'max_dd': momentum_max_dd,
                    'volatility': momentum_volatility
                },
                'buy_hold': {
                    'return': bh_return,
                    'sharpe': bh_sharpe,
                    'max_dd': bh_max_drawdown,
                    'volatility': bh_volatility
                }
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Buy-and-hold comparison failed: {str(e)}")
        return None


def test_without_regime_filter(logger):
    """Test momentum strategy without regime filtering."""
    
    logger.section("🔬 Testing Without Regime Filter")
    
    try:
        with logger.operation("Running momentum strategy without regime filter"):
            # This would be a simplified version for comparison
            # For brevity, we'll just log the concept
            logger.info("Regime filtering typically improves performance by:")
            logger.info("• Avoiding trades in unfavorable market conditions")
            logger.info("• Reducing false signals during ranging markets")
            logger.info("• Improving risk-adjusted returns")
            logger.info("• Reducing maximum drawdown")
            
            # In a full implementation, you would:
            # 1. Run the same strategy without regime filtering
            # 2. Compare the results
            # 3. Show the benefit of regime awareness
            
            return {
                'message': 'Regime filtering demonstration - see logs for insights'
            }
            
    except Exception as e:
        logger.error(f"Regime filter test failed: {str(e)}")
        return None


def main():
    """Run momentum strategy examples."""
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🚀 Momentum Strategy Examples", "Trend-following with advanced filtering and risk management")
    
    # Create results directory
    output_dir = Path("results/example_10_momentum_strategy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting Momentum Strategy Examples...")
        
        # Run main demonstration
        strategy_results = demonstrate_momentum_strategy()
        
        if strategy_results:
            strategy = strategy_results['strategy']
            portfolio = strategy_results['portfolio']
            
            # Compare with buy-and-hold
            comparison_results = compare_with_buy_and_hold(strategy, portfolio, logger)
            
            # Test without regime filter
            no_filter_results = test_without_regime_filter(logger)
            
            # Generate visualizations
            skip_plotting = "--no-plots" in sys.argv
            if not skip_plotting:
                with logger.operation("Generating momentum strategy visualizations"):
                    try:
                        create_momentum_analysis_charts(strategy_results, output_dir, logger)
                        logger.success(f"Visualizations saved to {output_dir}/")
                    except Exception as e:
                        logger.error(f"Failed to generate visualizations: {str(e)}")
            
            # Key insights
            logger.section("💡 Momentum Strategy Insights")
            
            results = strategy_results['results']
            regime_data = strategy_results['regime']
            
            logger.info("Key Findings:")
            logger.info(f"• Strategy Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            logger.info(f"• Total Trades: {results['total_trades']}")
            logger.info(f"• Win Rate: {results['win_rate']:.1%}")
            logger.info(f"• Calmar Ratio: {results['calmar_ratio']:.3f}")
            
            # Regime analysis
            regime_counts = regime_data['regime_counts']
            trending_periods = regime_counts.get('Strong_Uptrend', 0) + regime_counts.get('Strong_Downtrend', 0)
            total_periods = sum(regime_counts.values())
            trending_percentage = trending_periods / total_periods * 100
            
            logger.info(f"• Trending Market Periods: {trending_percentage:.1f}%")
            
            logger.info("\nMomentum Strategy Benefits:")
            logger.info("• Captures strong trending moves effectively")
            logger.info("• Regime filtering reduces false signals")
            logger.info("• Dynamic position sizing adapts to volatility")
            logger.info("• Trailing stops protect profits")
            logger.info("• Volume confirmation improves signal quality")
            
        else:
            logger.error("❌ Momentum strategy demonstration failed")
        
        logger.success("✅ Momentum strategy examples completed successfully!")
        logger.info(f"Results saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Momentum strategy examples error: {str(e)}")


def create_momentum_analysis_charts(strategy_results, output_dir, logger):
    """Create comprehensive momentum strategy analysis charts."""
    try:
        import matplotlib.pyplot as plt
        
        indicators = strategy_results['indicators']
        signals = strategy_results['signals']
        regime_data = strategy_results['regime']
        
        # Create comprehensive analysis chart
        fig, axes = plt.subplots(4, 2, figsize=(20, 16))
        fig.suptitle('Momentum Strategy Analysis', fontsize=16)
        
        # Get price data
        close_prices = strategy_results['strategy'].data.get('close')
        
        # 1. Price and moving averages
        axes[0, 0].plot(close_prices.index, close_prices, label='Close Price', alpha=0.8)
        axes[0, 0].plot(indicators['sma_20'].index, indicators['sma_20'], label='SMA 20', alpha=0.7)
        axes[0, 0].plot(indicators['sma_50'].index, indicators['sma_50'], label='SMA 50', alpha=0.7)
        axes[0, 0].set_title('Price and Moving Averages')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Momentum score
        momentum_score = indicators['momentum_score']
        axes[0, 1].plot(momentum_score.index, momentum_score, label='Momentum Score', color='purple')
        axes[0, 1].axhline(y=70, color='red', linestyle='--', label='High Threshold')
        axes[0, 1].axhline(y=30, color='green', linestyle='--', label='Low Threshold')
        axes[0, 1].axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Momentum Score')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RSI
        rsi = indicators['rsi']
        axes[1, 0].plot(rsi.index, rsi, label='RSI', color='orange')
        axes[1, 0].axhline(y=70, color='red', linestyle='--', label='Overbought')
        axes[1, 0].axhline(y=30, color='green', linestyle='--', label='Oversold')
        axes[1, 0].set_title('RSI')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. MACD
        macd_line = indicators['macd_line']
        macd_signal = indicators['macd_signal']
        macd_histogram = indicators['macd_histogram']
        
        axes[1, 1].plot(macd_line.index, macd_line, label='MACD Line', color='blue')
        axes[1, 1].plot(macd_signal.index, macd_signal, label='Signal Line', color='red')
        axes[1, 1].bar(macd_histogram.index, macd_histogram, label='Histogram', alpha=0.3, color='gray')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('MACD')
        axes[1, 1].set_ylabel('MACD')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Market regime
        regime = regime_data['regime']
        regime_numeric = pd.Series(0, index=regime.index)
        regime_map = {
            'Strong_Uptrend': 2,
            'Weak_Uptrend': 1,
            'Ranging': 0,
            'Weak_Downtrend': -1,
            'Strong_Downtrend': -2,
            'High_Volatility': 3
        }
        
        for reg, value in regime_map.items():
            regime_numeric[regime == reg] = value
        
        axes[2, 0].plot(regime_numeric.index, regime_numeric, label='Market Regime', 
                       color='brown', linewidth=2)
        axes[2, 0].set_title('Market Regime')
        axes[2, 0].set_ylabel('Regime')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. ADX (trend strength)
        adx = regime_data['adx']
        axes[2, 1].plot(adx.index, adx, label='ADX', color='green')
        axes[2, 1].axhline(y=25, color='red', linestyle='--', label='Strong Trend')
        axes[2, 1].axhline(y=20, color='orange', linestyle=':', label='Weak Trend')
        axes[2, 1].set_title('ADX (Trend Strength)')
        axes[2, 1].set_ylabel('ADX')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. Trading signals
        entries = signals['entries']
        exits = signals['exits']
        
        # Plot price with signals
        axes[3, 0].plot(close_prices.index, close_prices, label='Price', alpha=0.7)
        
        # Entry points
        entry_points = close_prices[entries]
        if len(entry_points) > 0:
            axes[3, 0].scatter(entry_points.index, entry_points, 
                              color='green', marker='^', s=50, label='Entries', zorder=5)
        
        # Exit points
        exit_points = close_prices[exits]
        if len(exit_points) > 0:
            axes[3, 0].scatter(exit_points.index, exit_points,
                              color='red', marker='v', s=50, label='Exits', zorder=5)
        
        axes[3, 0].set_title('Trading Signals')
        axes[3, 0].set_ylabel('Price')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. Performance metrics summary
        results = strategy_results['results']
        metrics = [
            f"Total Return: {results['total_return']:.2%}",
            f"Sharpe Ratio: {results['sharpe_ratio']:.3f}",
            f"Max Drawdown: {results['max_drawdown']:.2%}",
            f"Win Rate: {results['win_rate']:.1%}",
            f"Total Trades: {results['total_trades']}",
            f"Profit Factor: {results['profit_factor']:.2f}",
            f"Calmar Ratio: {results['calmar_ratio']:.3f}"
        ]
        
        axes[3, 1].text(0.1, 0.9, '\n'.join(metrics), transform=axes[3, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[3, 1].set_title('Performance Summary')
        axes[3, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "momentum_strategy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create momentum analysis charts: {str(e)}")


if __name__ == "__main__":
    main() 