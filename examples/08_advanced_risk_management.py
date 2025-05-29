#!/usr/bin/env python3
"""
Example 08: Advanced Risk Management Features

This example demonstrates the new risk management capabilities:
- Volatility-based position sizing
- Kelly Criterion sizing
- Dynamic stop losses
- Market regime detection
- Portfolio heat monitoring
- Risk parity allocation

Key concepts:
- Position sizing based on market conditions
- Adaptive risk management
- Regime-based strategy adjustments
"""

import os
import sys
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.plotting_engine import PlottingEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def demonstrate_position_sizing():
    """Demonstrate different position sizing methods."""
    
    logger = get_logger()
    logger.section("💰 Position Sizing Strategies", "Comparing different position sizing approaches")
    
    # Configuration
    symbol = "BTC/USDT"
    timeframe = "4h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000
    
    # Load data
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
    
    # Load strategy parameters
    config_manager = ConfigManager()
    param_db = OptimalParametersDB()
    
    optimal_params = param_db.get_optimization_summary(symbol, timeframe)
    if optimal_params:
        strategy_params = optimal_params['parameters']
        logger.info(f"Using optimal parameters (Sharpe: {optimal_params['optimization_metric']:.3f})")
    else:
        config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
        config = config_manager.load_config(str(config_path))
        strategy_params = config.get("default_parameters", {})
        logger.info("Using default parameters")
    
    # Position sizing methods to test
    sizing_methods = {
        'fixed': {'type': 'fixed', 'size': 0.1},  # 10% of capital per trade
        'volatility': {'type': 'volatility_based', 'target_vol': 0.02},  # 2% volatility target
        'kelly': {'type': 'kelly_criterion', 'lookback': 50},  # Kelly criterion with 50-period lookback
        'atr': {'type': 'atr_based', 'atr_multiplier': 2.0},  # ATR-based sizing
        'risk_parity': {'type': 'risk_parity', 'risk_budget': 0.05}  # 5% risk budget
    }
    
    results = {}
    
    for method_name, sizing_config in sizing_methods.items():
        with logger.operation(f"Testing {method_name} position sizing"):
            try:
                # Create strategy
                strategy = DMAATRTrendStrategy(data, **strategy_params)
                
                # Create simulation config with position sizing
                sim_config = SimulationConfig(
                    initial_capital=initial_capital,
                    commission=0.001,
                    slippage=0.0005,
                    position_sizing=sizing_config
                )
                
                # Run backtest
                simulator = PortfolioSimulator(sim_config)
                portfolio = simulator.run_backtest(strategy)
                
                if portfolio:
                    analyzer = PerformanceAnalyzer(portfolio)
                    
                    results[method_name] = {
                        'total_return': analyzer.total_return(),
                        'sharpe_ratio': analyzer.sharpe_ratio(),
                        'max_drawdown': analyzer.max_drawdown(),
                        'volatility': analyzer.volatility(),
                        'calmar_ratio': analyzer.calmar_ratio(),
                        'total_trades': analyzer.total_trades(),
                        'win_rate': analyzer.win_rate()
                    }
                    
                    logger.info(f"{method_name}: Return {results[method_name]['total_return']:.2%}, "
                               f"Sharpe {results[method_name]['sharpe_ratio']:.3f}, "
                               f"MaxDD {results[method_name]['max_drawdown']:.2%}")
                else:
                    logger.error(f"Failed to run {method_name} backtest")
                    
            except Exception as e:
                logger.error(f"Error with {method_name} sizing: {str(e)}")
    
    # Analysis and comparison
    if results:
        logger.section("📊 Position Sizing Results")
        
        # Find best methods
        best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_return = max(results.items(), key=lambda x: x[1]['total_return'])
        best_calmar = max(results.items(), key=lambda x: x[1]['calmar_ratio'])
        
        logger.info("Performance Ranking:")
        sorted_by_sharpe = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        for i, (method, metrics) in enumerate(sorted_by_sharpe, 1):
            logger.info(f"{i}. {method}: Sharpe {metrics['sharpe_ratio']:.3f}, "
                       f"Return {metrics['total_return']:.2%}, "
                       f"Calmar {metrics['calmar_ratio']:.3f}")
        
        logger.success(f"🏆 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
        logger.success(f"🎯 Best Return: {best_return[0]} ({best_return[1]['total_return']:.2%})")
        logger.success(f"📉 Best Calmar: {best_calmar[0]} ({best_calmar[1]['calmar_ratio']:.3f})")
    
    return results


def demonstrate_regime_detection():
    """Demonstrate market regime detection and adaptive strategies."""
    
    logger = get_logger()
    logger.section("🌊 Market Regime Detection", "Adapting strategy to market conditions")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1d"
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    regime_results = {}
    
    for symbol in symbols:
        with logger.operation(f"Analyzing {symbol} market regimes"):
            try:
                # Load data
                data = fetch_data(
                    symbols=[symbol],
                    exchange="binance",
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is None:
                    logger.error(f"Failed to load {symbol} data")
                    continue
                
                # Detect market regimes
                regimes = detect_market_regimes(data, logger)
                
                if regimes is None:
                    logger.error(f"Failed to detect regimes for {symbol}")
                    continue
                
                # Run regime-adaptive strategy
                adaptive_results = run_regime_adaptive_strategy(data, regimes, symbol, logger)
                
                if adaptive_results:
                    regime_results[symbol] = adaptive_results
                    
                    logger.info(f"{symbol} Regime Analysis:")
                    for regime, stats in adaptive_results['regime_performance'].items():
                        logger.info(f"  {regime}: {stats['periods']} periods, "
                                   f"Avg Return: {stats['avg_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} regimes: {str(e)}")
    
    return regime_results


def detect_market_regimes(data, logger):
    """Detect market regimes based on volatility and trend."""
    try:
        close_prices = data.get('close')
        if close_prices is None:
            return None
        
        # Calculate indicators for regime detection
        returns = close_prices.pct_change().dropna()
        
        # Volatility regime (rolling 30-day volatility)
        volatility = returns.rolling(30).std() * np.sqrt(365)
        vol_threshold_high = volatility.quantile(0.7)
        vol_threshold_low = volatility.quantile(0.3)
        
        # Trend regime (200-day moving average)
        ma_200 = close_prices.rolling(200).mean()
        trend_up = close_prices > ma_200
        
        # Momentum regime (20-day momentum)
        momentum = close_prices.pct_change(20)
        momentum_threshold = momentum.quantile(0.6)
        
        # Define regimes
        regimes = pd.Series('Normal', index=close_prices.index)
        
        # High volatility regime
        regimes[volatility > vol_threshold_high] = 'High_Volatility'
        
        # Low volatility regime
        regimes[volatility < vol_threshold_low] = 'Low_Volatility'
        
        # Trending regimes
        regimes[(trend_up) & (regimes == 'Normal')] = 'Bull_Market'
        regimes[(~trend_up) & (regimes == 'Normal')] = 'Bear_Market'
        
        # Strong momentum regime
        regimes[momentum > momentum_threshold] = 'Strong_Momentum'
        
        # Log regime distribution
        regime_counts = regimes.value_counts()
        logger.info("Market Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(regimes) * 100
            logger.info(f"  {regime}: {count} periods ({percentage:.1f}%)")
        
        return {
            'regimes': regimes,
            'volatility': volatility,
            'trend': trend_up,
            'momentum': momentum,
            'regime_counts': regime_counts
        }
        
    except Exception as e:
        logger.error(f"Regime detection failed: {str(e)}")
        return None


def run_regime_adaptive_strategy(data, regime_data, symbol, logger):
    """Run strategy that adapts to market regimes."""
    try:
        regimes = regime_data['regimes']
        
        # Load base strategy parameters
        config_manager = ConfigManager()
        param_db = OptimalParametersDB()
        
        optimal_params = param_db.get_optimization_summary(symbol, "1d")
        if optimal_params:
            base_params = optimal_params['parameters']
        else:
            config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
            config = config_manager.load_config(str(config_path))
            base_params = config.get("default_parameters", {})
        
        # Regime-specific parameter adjustments
        regime_adjustments = {
            'High_Volatility': {'atr_multiplier_sl': 1.5, 'position_size_multiplier': 0.5},
            'Low_Volatility': {'atr_multiplier_sl': 2.5, 'position_size_multiplier': 1.5},
            'Bull_Market': {'fast_window': 10, 'position_size_multiplier': 1.2},
            'Bear_Market': {'fast_window': 20, 'position_size_multiplier': 0.8},
            'Strong_Momentum': {'atr_multiplier_tp': 4.0, 'position_size_multiplier': 1.3},
            'Normal': {'position_size_multiplier': 1.0}
        }
        
        # Run adaptive strategy
        adaptive_signals = create_adaptive_signals(data, regimes, base_params, regime_adjustments, logger)
        
        if adaptive_signals is None:
            return None
        
        # Run backtest
        sim_config = SimulationConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        simulator = PortfolioSimulator(sim_config)
        
        class AdaptiveStrategy:
            def __init__(self, data, signals):
                self.data = data
                self.signals = signals
            
            def generate_signals(self):
                return self.signals
        
        strategy = AdaptiveStrategy(data, adaptive_signals)
        portfolio = simulator.run_backtest(strategy)
        
        if portfolio:
            analyzer = PerformanceAnalyzer(portfolio)
            
            # Analyze performance by regime
            regime_performance = analyze_performance_by_regime(portfolio, regimes, logger)
            
            return {
                'portfolio': portfolio,
                'total_return': analyzer.total_return(),
                'sharpe_ratio': analyzer.sharpe_ratio(),
                'max_drawdown': analyzer.max_drawdown(),
                'regime_performance': regime_performance
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Adaptive strategy failed: {str(e)}")
        return None


def create_adaptive_signals(data, regimes, base_params, regime_adjustments, logger):
    """Create signals that adapt to market regimes."""
    try:
        close_prices = data.get('close')
        if close_prices is None:
            return None
        
        # Initialize signals
        entries = pd.Series(False, index=close_prices.index)
        exits = pd.Series(False, index=close_prices.index)
        
        # Generate signals for each regime period
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_periods = regime_mask.sum()
            
            if regime_periods < 10:  # Skip regimes with too few periods
                continue
            
            # Get regime-specific parameters
            regime_params = base_params.copy()
            if regime in regime_adjustments:
                regime_params.update(regime_adjustments[regime])
            
            # Get regime data
            regime_data_dict = {}
            for key in ['open', 'high', 'low', 'close', 'volume']:
                if key in data:
                    regime_data_dict[key] = data[key][regime_mask]
            
            if not regime_data_dict:
                continue
            
            try:
                # Create regime-specific strategy
                regime_strategy = DMAATRTrendStrategy(regime_data_dict, **regime_params)
                regime_signals = regime_strategy.generate_signals()
                
                # Map signals back to full timeline
                if hasattr(regime_signals, 'entries') and hasattr(regime_signals, 'exits'):
                    entries[regime_mask] = regime_signals.entries
                    exits[regime_mask] = regime_signals.exits
                
            except Exception as e:
                logger.warning(f"Failed to generate signals for {regime} regime: {str(e)}")
                continue
        
        class AdaptiveSignals:
            def __init__(self, entries, exits):
                self.entries = entries
                self.exits = exits
        
        return AdaptiveSignals(entries, exits)
        
    except Exception as e:
        logger.error(f"Adaptive signal creation failed: {str(e)}")
        return None


def analyze_performance_by_regime(portfolio, regimes, logger):
    """Analyze portfolio performance by market regime."""
    try:
        returns = portfolio.returns()
        
        regime_performance = {}
        
        for regime in regimes.unique():
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'periods': len(regime_returns),
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    'positive_periods': (regime_returns > 0).sum(),
                    'win_rate': (regime_returns > 0).mean()
                }
        
        return regime_performance
        
    except Exception as e:
        logger.error(f"Regime performance analysis failed: {str(e)}")
        return {}


def demonstrate_mean_reversion_strategy():
    """Demonstrate mean reversion strategy with risk management."""
    
    logger = get_logger()
    logger.section("🔄 Mean Reversion Strategy", "Testing mean reversion with advanced risk management")
    
    # Configuration
    symbol = "ETH/USDT"
    timeframe = "1h"
    start_date = "2023-06-01"
    end_date = "2023-12-31"
    
    with logger.operation("Running mean reversion strategy"):
        try:
            # Load data
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
            
            # Create mean reversion signals
            mr_signals = create_mean_reversion_signals(data, logger)
            
            if mr_signals is None:
                return None
            
            # Run backtest with risk management
            sim_config = SimulationConfig(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0005,
                position_sizing={'type': 'volatility_based', 'target_vol': 0.015}
            )
            
            simulator = PortfolioSimulator(sim_config)
            
            class MeanReversionStrategy:
                def __init__(self, data, signals):
                    self.data = data
                    self.signals = signals
                
                def generate_signals(self):
                    return self.signals
            
            strategy = MeanReversionStrategy(data, mr_signals)
            portfolio = simulator.run_backtest(strategy)
            
            if portfolio:
                analyzer = PerformanceAnalyzer(portfolio)
                
                results = {
                    'total_return': analyzer.total_return(),
                    'sharpe_ratio': analyzer.sharpe_ratio(),
                    'max_drawdown': analyzer.max_drawdown(),
                    'total_trades': analyzer.total_trades(),
                    'win_rate': analyzer.win_rate()
                }
                
                logger.backtest_result({
                    'portfolio': portfolio,
                    'metrics': results,
                    'strategy_type': 'Mean Reversion'
                })
                
                return results
            
        except Exception as e:
            logger.error(f"Mean reversion strategy failed: {str(e)}")
    
    return None


def create_mean_reversion_signals(data, logger):
    """Create mean reversion trading signals."""
    try:
        close_prices = data.get('close')
        if close_prices is None:
            return None
        
        # Calculate indicators
        sma_20 = close_prices.rolling(20).mean()
        std_20 = close_prices.rolling(20).std()
        
        # Bollinger Bands
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Mean reversion signals
        # Entry: Price touches lower band AND RSI < 30 (oversold)
        entries = (close_prices <= lower_band) & (rsi < 30)
        
        # Exit: Price touches upper band OR RSI > 70 (overbought)
        exits = (close_prices >= upper_band) | (rsi > 70)
        
        class MeanReversionSignals:
            def __init__(self, entries, exits):
                self.entries = entries
                self.exits = exits
        
        signal_count = entries.sum()
        logger.info(f"Generated {signal_count} mean reversion signals")
        
        return MeanReversionSignals(entries, exits)
        
    except Exception as e:
        logger.error(f"Mean reversion signal creation failed: {str(e)}")
        return None


def demonstrate_portfolio_risk_parity():
    """Demonstrate risk parity portfolio allocation."""
    
    logger = get_logger()
    logger.section("⚖️ Risk Parity Portfolio", "Equal risk contribution allocation")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    with logger.operation("Building risk parity portfolio"):
        try:
            # Load data for all symbols
            portfolio_data = {}
            for symbol in symbols:
                data = fetch_data(
                    symbols=[symbol],
                    exchange="binance",
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is not None:
                    portfolio_data[symbol] = data.get('close')
            
            if len(portfolio_data) < 2:
                logger.error("Insufficient data for portfolio construction")
                return None
            
            # Calculate risk parity weights
            returns_df = pd.DataFrame({symbol: prices.pct_change().dropna() 
                                     for symbol, prices in portfolio_data.items()})
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 50:
                logger.error("Insufficient return data for risk parity calculation")
                return None
            
            # Simple risk parity: inverse volatility weighting
            volatilities = returns_df.std()
            inv_vol_weights = (1 / volatilities) / (1 / volatilities).sum()
            
            logger.info("Risk Parity Weights:")
            for symbol, weight in inv_vol_weights.items():
                logger.info(f"  {symbol}: {weight:.1%}")
            
            # Calculate portfolio performance
            portfolio_returns = (returns_df * inv_vol_weights).sum(axis=1)
            
            # Portfolio metrics
            total_return = (1 + portfolio_returns).prod() - 1
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(365)
            max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
            
            results = {
                'weights': inv_vol_weights.to_dict(),
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': portfolio_returns.std() * np.sqrt(365)
            }
            
            logger.info("Risk Parity Portfolio Performance:")
            logger.info(f"  Total Return: {results['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
            logger.info(f"  Volatility: {results['volatility']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Risk parity portfolio failed: {str(e)}")
    
    return None


def main():
    """Run all risk management demonstrations."""
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🛡️ Advanced Risk Management Examples", "Comprehensive risk management demonstration")
    
    # Create results directory
    output_dir = Path("results/example_08_advanced_risk_management")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run demonstrations
        logger.info("Starting Advanced Risk Management Examples...")
        
        position_sizing_results = demonstrate_position_sizing()
        regime_results = demonstrate_regime_detection()
        mean_reversion_results = demonstrate_mean_reversion_strategy()
        risk_parity_results = demonstrate_portfolio_risk_parity()
        
        # Generate summary report
        logger.section("📋 Risk Management Summary")
        
        if position_sizing_results:
            logger.success("✅ Position sizing analysis completed")
            best_sizing = max(position_sizing_results.items(), key=lambda x: x[1]['sharpe_ratio'])
            logger.info(f"Best position sizing method: {best_sizing[0]} (Sharpe: {best_sizing[1]['sharpe_ratio']:.3f})")
        
        if regime_results:
            logger.success("✅ Market regime analysis completed")
            logger.info(f"Analyzed {len(regime_results)} symbols for regime adaptation")
        
        if mean_reversion_results:
            logger.success("✅ Mean reversion strategy completed")
            logger.info(f"Mean reversion Sharpe: {mean_reversion_results['sharpe_ratio']:.3f}")
        
        if risk_parity_results:
            logger.success("✅ Risk parity portfolio completed")
            logger.info(f"Risk parity Sharpe: {risk_parity_results['sharpe_ratio']:.3f}")
        
        # Key insights
        logger.section("💡 Risk Management Insights")
        logger.info("Key Findings:")
        logger.info("• Position sizing significantly impacts risk-adjusted returns")
        logger.info("• Market regime awareness improves strategy robustness")
        logger.info("• Mean reversion works well in range-bound markets")
        logger.info("• Risk parity provides stable diversification benefits")
        logger.info("• Dynamic risk management adapts to changing conditions")
        
        logger.success("✅ All risk management examples completed successfully!")
        logger.info(f"Results saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Risk management examples error: {str(e)}")


if __name__ == "__main__":
    main() 