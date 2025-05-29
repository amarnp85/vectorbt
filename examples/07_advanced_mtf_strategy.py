#!/usr/bin/env python3
"""
Example 07: Advanced Multi-Timeframe Strategy using MTF Infrastructure

This example demonstrates how to use the backtester's MTF infrastructure
including the MTFDataHandler, MTFStrategy base class, and MTF plotting.

Key features:
- Uses the MTF_DMA_ATR_Strategy
- Demonstrates MTF data fetching and alignment
- Shows MTF-specific visualizations
- Compares performance with single-timeframe version
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
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.plotting_engine import PlottingEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def run_advanced_mtf_strategy():
    """
    Demonstrates the advanced MTF infrastructure with proper
    data handling, strategy implementation, and visualization.
    """
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🔬 Advanced Multi-Timeframe Strategy", "Using sophisticated MTF infrastructure and analysis")
    
    # Configuration
    symbol = "ETH/USDT"
    symbols = [symbol]  # MTF handlers expect a list
    exchange = "binance"
    timeframes = ["1h", "4h", "1d"]  # Multiple timeframes
    base_timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {timeframes} (Base: {base_timeframe})")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Initialize managers
    config_manager = ConfigManager()
    param_db = OptimalParametersDB()
    
    # Load MTF data
    logger.section("📊 Loading Multi-Timeframe Data")
    
    mtf_data = {}
    for tf in timeframes:
        with logger.operation(f"Fetching {tf} data"):
            data = fetch_data(
                symbols=symbols,
                timeframes=[tf],
                start_date=start_date,
                end_date=end_date,
                exchange=exchange
            )
            
            if data is None:
                logger.error(f"Failed to fetch {tf} data")
                return None
            
            mtf_data[tf] = data
            data_points = len(data.get('close')) if hasattr(data, 'get') else len(data)
            logger.info(f"{tf}: {data_points} candles loaded")
    
    # Load strategy parameters with MTF-specific enhancements
    logger.section("⚙️ Loading Advanced MTF Parameters")
    
    # Load base strategy parameters
    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
    config = config_manager.load_config(str(config_path))
    base_params = config.get("default_parameters", {})
    
    # Check for optimal parameters
    optimal_params = param_db.get_optimization_summary(symbol, base_timeframe)
    if optimal_params:
        base_params.update(optimal_params['parameters'])
        logger.info(f"Using optimal parameters (Sharpe: {optimal_params['optimization_metric']:.3f})")
    else:
        logger.info("Using default parameters")
    
    # Add MTF-specific parameters
    mtf_params = base_params.copy()
    mtf_params.update({
        'use_mtf_confirmation': True,
        'align_all_to_base': True,
        'mtf_weights': {'1h': 1.0, '4h': 1.5, '1d': 2.0},
        'trend_alignment_threshold': 0.6,
        'mtf_atr_multiplier': 1.2,
        'require_all_timeframes': False,  # Allow partial alignment
        'dominant_timeframe': '4h'  # Primary timeframe for trend
    })
    
    logger.info("MTF Configuration:")
    logger.info(f"  Weights: {mtf_params['mtf_weights']}")
    logger.info(f"  Alignment threshold: {mtf_params['trend_alignment_threshold']}")
    logger.info(f"  Dominant timeframe: {mtf_params['dominant_timeframe']}")
    
    # Initialize advanced MTF strategy
    logger.section("🧠 Implementing Advanced MTF Logic")
    
    mtf_strategy_results = {}
    
    with logger.operation("Running advanced MTF strategy"):
        try:
            # Create weighted MTF signals
            mtf_signals = create_weighted_mtf_signals(mtf_data, mtf_params, timeframes, logger)
            
            if mtf_signals is None:
                logger.error("Failed to create MTF signals")
                return None
            
            # Run MTF backtest
            base_data = mtf_data[base_timeframe]
            
            sim_config = SimulationConfig(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0005
            )
            
            simulator = PortfolioSimulator(sim_config)
            
            # Create strategy wrapper
            class AdvancedMTFStrategy:
                def __init__(self, data, signals, params):
                    self.data = data
                    self.signals = signals
                    self.params = params
                
                def generate_signals(self):
                    return self.signals
            
            mtf_strategy = AdvancedMTFStrategy(base_data, mtf_signals, mtf_params)
            mtf_portfolio = simulator.run_backtest(mtf_strategy)
            
            if mtf_portfolio:
                analyzer = PerformanceAnalyzer(mtf_portfolio)
                mtf_strategy_results = {
                    'portfolio': mtf_portfolio,
                    'total_return': analyzer.total_return(),
                    'sharpe_ratio': analyzer.sharpe_ratio(),
                    'max_drawdown': analyzer.max_drawdown(),
                    'volatility': analyzer.volatility(),
                    'win_rate': analyzer.win_rate(),
                    'profit_factor': analyzer.profit_factor(),
                    'total_trades': analyzer.total_trades()
                }
                
                logger.backtest_result({
                    'portfolio': mtf_portfolio,
                    'metrics': mtf_strategy_results,
                    'strategy_type': 'Advanced MTF'
                })
            else:
                logger.error("MTF portfolio simulation failed")
                return None
                
        except Exception as e:
            logger.error(f"Advanced MTF strategy failed: {str(e)}")
            return None
    
    # Compare with individual timeframe strategies
    logger.section("📈 Timeframe Performance Comparison")
    
    comparison_results = {'Advanced_MTF': mtf_strategy_results}
    
    for tf in timeframes:
        with logger.operation(f"Running {tf} individual strategy"):
            try:
                tf_data = mtf_data[tf]
                tf_strategy = DMAATRTrendStrategy(tf_data, **base_params)
                tf_portfolio = simulator.run_backtest(tf_strategy)
                
                if tf_portfolio:
                    tf_analyzer = PerformanceAnalyzer(tf_portfolio)
                    comparison_results[f'{tf}_Individual'] = {
                        'total_return': tf_analyzer.total_return(),
                        'sharpe_ratio': tf_analyzer.sharpe_ratio(),
                        'max_drawdown': tf_analyzer.max_drawdown(),
                        'total_trades': tf_analyzer.total_trades()
                    }
                    
                    logger.info(f"{tf}: Return {comparison_results[f'{tf}_Individual']['total_return']:.2%}, "
                               f"Sharpe {comparison_results[f'{tf}_Individual']['sharpe_ratio']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to run {tf} individual strategy: {str(e)}")
    
    # Advanced MTF analysis
    logger.section("🔍 Advanced MTF Analysis")
    
    # Signal quality analysis
    signal_quality = analyze_signal_quality(mtf_signals, mtf_data[base_timeframe], logger)
    
    # Timeframe contribution analysis
    tf_contributions = analyze_timeframe_contributions(mtf_data, mtf_params, logger)
    
    # Market regime analysis
    regime_analysis = analyze_market_regimes(mtf_data, mtf_strategy_results, logger)
    
    # Performance ranking
    logger.section("🏆 Performance Ranking")
    
    sorted_results = sorted(comparison_results.items(), 
                           key=lambda x: x[1]['sharpe_ratio'], 
                           reverse=True)
    
    logger.info("Ranking by Sharpe Ratio:")
    for i, (strategy, metrics) in enumerate(sorted_results, 1):
        logger.info(f"{i}. {strategy}: "
                   f"Sharpe {metrics['sharpe_ratio']:.3f}, "
                   f"Return {metrics['total_return']:.2%}, "
                   f"Trades {metrics['total_trades']}")
    
    best_strategy = sorted_results[0]
    logger.success(f"🏆 Best Strategy: {best_strategy[0]} "
                   f"(Sharpe: {best_strategy[1]['sharpe_ratio']:.3f})")
    
    # Generate advanced visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating advanced MTF visualizations"):
            try:
                output_dir = Path("results/example_07_advanced_mtf_strategy")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Advanced MTF performance chart
                create_advanced_mtf_charts(
                    mtf_portfolio, comparison_results, mtf_data, 
                    mtf_signals, output_dir, logger
                )
                
                # Signal quality heatmap
                create_signal_quality_heatmap(signal_quality, output_dir, logger)
                
                # Timeframe contribution chart
                create_contribution_chart(tf_contributions, output_dir, logger)
                
                # Market regime analysis chart
                create_regime_analysis_chart(regime_analysis, output_dir, logger)
                
                logger.success(f"Advanced visualizations saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate advanced visualizations: {str(e)}")
    
    # Advanced insights and recommendations
    logger.section("💡 Advanced MTF Strategy Insights")
    
    mtf_improvement = (mtf_strategy_results['sharpe_ratio'] / 
                      max([r['sharpe_ratio'] for k, r in comparison_results.items() if k != 'Advanced_MTF']))
    
    logger.info("Advanced MTF Analysis Results:")
    if mtf_improvement > 1.2:
        logger.success("✅ Advanced MTF provides significant improvement")
    elif mtf_improvement > 1.05:
        logger.info("✅ Advanced MTF provides modest improvement")
    else:
        logger.warning("⚠️ Advanced MTF may be over-complicating the strategy")
    
    logger.info("Key Insights:")
    logger.info("• Weighted timeframe signals can improve signal quality")
    logger.info("• Partial alignment allows more trading opportunities")
    logger.info("• Dominant timeframe helps maintain trend direction")
    logger.info("• Market regime awareness enhances strategy robustness")
    logger.info("• Signal quality metrics help optimize MTF parameters")
    
    return {
        'mtf_portfolio': mtf_portfolio,
        'mtf_results': mtf_strategy_results,
        'comparison_results': comparison_results,
        'signal_quality': signal_quality,
        'tf_contributions': tf_contributions,
        'regime_analysis': regime_analysis
    }


def create_weighted_mtf_signals(mtf_data, params, timeframes, logger):
    """Create weighted MTF signals using advanced logic."""
    try:
        base_timeframe = '1h'  # Assuming 1h is base
        base_data = mtf_data[base_timeframe]
        base_index = base_data.index if hasattr(base_data, 'index') else None
        
        if base_index is None:
            logger.error("Cannot access base timeframe index")
            return None
        
        # Generate individual timeframe signals
        tf_signals = {}
        for tf in timeframes:
            try:
                strategy = DMAATRTrendStrategy(mtf_data[tf], **params)
                signals = strategy.generate_signals()
                tf_signals[tf] = signals
            except Exception as e:
                logger.warning(f"Failed to generate {tf} signals: {str(e)}")
                continue
        
        if not tf_signals:
            logger.error("No timeframe signals generated")
            return None
        
        # Create weighted signals
        weights = params.get('mtf_weights', {tf: 1.0 for tf in timeframes})
        threshold = params.get('trend_alignment_threshold', 0.6)
        
        weighted_entries = pd.Series(0.0, index=base_index)
        weighted_exits = pd.Series(0.0, index=base_index)
        
        for i, timestamp in enumerate(base_index):
            entry_score = 0.0
            exit_score = 0.0
            total_weight = 0.0
            
            for tf, signals in tf_signals.items():
                weight = weights.get(tf, 1.0)
                
                try:
                    # Find closest signal
                    tf_data = mtf_data[tf]
                    tf_index = tf_data.index if hasattr(tf_data, 'index') else None
                    
                    if tf_index is not None:
                        closest_idx = tf_index.get_indexer([timestamp], method='ffill')[0]
                        
                        if 0 <= closest_idx < len(signals.entries):
                            if signals.entries.iloc[closest_idx]:
                                entry_score += weight
                            if signals.exits.iloc[closest_idx]:
                                exit_score += weight
                            total_weight += weight
                
                except Exception:
                    continue
            
            # Normalize scores
            if total_weight > 0:
                weighted_entries.iloc[i] = entry_score / total_weight
                weighted_exits.iloc[i] = exit_score / total_weight
        
        # Apply threshold
        final_entries = weighted_entries >= threshold
        final_exits = weighted_exits >= threshold
        
        class WeightedMTFSignals:
            def __init__(self, entries, exits, weights_entries, weights_exits):
                self.entries = entries
                self.exits = exits
                self.weight_entries = weights_entries
                self.weight_exits = weights_exits
        
        return WeightedMTFSignals(final_entries, final_exits, weighted_entries, weighted_exits)
        
    except Exception as e:
        logger.error(f"Weighted MTF signal creation failed: {str(e)}")
        return None


def analyze_signal_quality(signals, data, logger):
    """Analyze the quality of MTF signals."""
    try:
        if not hasattr(signals, 'entries') or not hasattr(signals, 'exits'):
            return {}
        
        # Basic signal statistics
        total_entries = signals.entries.sum()
        total_exits = signals.exits.sum()
        
        # Signal frequency
        data_length = len(data.index) if hasattr(data, 'index') else len(data)
        entry_frequency = total_entries / data_length if data_length > 0 else 0
        exit_frequency = total_exits / data_length if data_length > 0 else 0
        
        # Signal clustering (consecutive signals)
        entry_clusters = count_signal_clusters(signals.entries)
        exit_clusters = count_signal_clusters(signals.exits)
        
        quality_metrics = {
            'total_entries': total_entries,
            'total_exits': total_exits,
            'entry_frequency': entry_frequency,
            'exit_frequency': exit_frequency,
            'entry_clusters': entry_clusters,
            'exit_clusters': exit_clusters,
            'signal_balance': abs(total_entries - total_exits) / max(total_entries, total_exits, 1)
        }
        
        logger.info("Signal Quality Metrics:")
        logger.info(f"  Entry frequency: {entry_frequency:.3%}")
        logger.info(f"  Exit frequency: {exit_frequency:.3%}")
        logger.info(f"  Signal balance: {quality_metrics['signal_balance']:.3f}")
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Signal quality analysis failed: {str(e)}")
        return {}


def count_signal_clusters(signals):
    """Count consecutive signal clusters."""
    try:
        clusters = 0
        in_cluster = False
        
        for signal in signals:
            if signal and not in_cluster:
                clusters += 1
                in_cluster = True
            elif not signal:
                in_cluster = False
        
        return clusters
    except:
        return 0


def analyze_timeframe_contributions(mtf_data, params, logger):
    """Analyze how each timeframe contributes to the strategy."""
    try:
        contributions = {}
        weights = params.get('mtf_weights', {})
        
        for tf in mtf_data.keys():
            weight = weights.get(tf, 1.0)
            data_points = len(mtf_data[tf].get('close')) if hasattr(mtf_data[tf], 'get') else len(mtf_data[tf])
            
            contributions[tf] = {
                'weight': weight,
                'data_points': data_points,
                'relative_contribution': weight / sum(weights.values()) if weights else 0
            }
        
        logger.info("Timeframe Contributions:")
        for tf, contrib in contributions.items():
            logger.info(f"  {tf}: Weight {contrib['weight']:.1f}, "
                       f"Contribution {contrib['relative_contribution']:.1%}")
        
        return contributions
        
    except Exception as e:
        logger.error(f"Timeframe contribution analysis failed: {str(e)}")
        return {}


def analyze_market_regimes(mtf_data, strategy_results, logger):
    """Analyze strategy performance across different market regimes."""
    try:
        # Simple regime detection based on volatility and trend
        base_data = mtf_data.get('1h', list(mtf_data.values())[0])
        
        if not hasattr(base_data, 'get'):
            return {}
        
        close_prices = base_data.get('close')
        if close_prices is None:
            return {}
        
        # Calculate rolling volatility and trend
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(24).std()  # 24-hour rolling volatility
        trend = close_prices.rolling(48).mean()  # 48-hour trend
        
        # Define regimes
        high_vol_threshold = volatility.quantile(0.7)
        low_vol_threshold = volatility.quantile(0.3)
        
        regimes = pd.Series('Normal', index=close_prices.index)
        regimes[volatility > high_vol_threshold] = 'High_Volatility'
        regimes[volatility < low_vol_threshold] = 'Low_Volatility'
        
        # Add trend component
        price_above_trend = close_prices > trend
        regimes[price_above_trend & (regimes == 'Normal')] = 'Trending_Up'
        regimes[~price_above_trend & (regimes == 'Normal')] = 'Trending_Down'
        
        regime_counts = regimes.value_counts()
        
        logger.info("Market Regime Analysis:")
        for regime, count in regime_counts.items():
            percentage = count / len(regimes) * 100
            logger.info(f"  {regime}: {count} periods ({percentage:.1f}%)")
        
        return {
            'regimes': regimes,
            'regime_counts': regime_counts,
            'volatility': volatility,
            'trend': trend
        }
        
    except Exception as e:
        logger.error(f"Market regime analysis failed: {str(e)}")
        return {}


def create_advanced_mtf_charts(portfolio, comparison_results, mtf_data, signals, output_dir, logger):
    """Create advanced MTF visualization charts."""
    try:
        import matplotlib.pyplot as plt
        
        # Performance comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Advanced Multi-Timeframe Strategy Analysis', fontsize=16)
        
        # 1. Strategy comparison
        strategies = list(comparison_results.keys())
        sharpe_ratios = [comparison_results[s]['sharpe_ratio'] for s in strategies]
        returns = [comparison_results[s]['total_return'] * 100 for s in strategies]
        
        ax1.bar(strategies, sharpe_ratios, color=['red', 'blue', 'green', 'orange'])
        ax1.set_title('Sharpe Ratio Comparison')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Return comparison
        ax2.bar(strategies, returns, color=['red', 'blue', 'green', 'orange'])
        ax2.set_title('Total Return Comparison (%)')
        ax2.set_ylabel('Return (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Signal weights over time (if available)
        if hasattr(signals, 'weight_entries'):
            ax3.plot(signals.weight_entries.index, signals.weight_entries, 
                    label='Entry Weights', alpha=0.7)
            ax3.plot(signals.weight_exits.index, signals.weight_exits, 
                    label='Exit Weights', alpha=0.7)
            ax3.set_title('Signal Weights Over Time')
            ax3.set_ylabel('Weight')
            ax3.legend()
        
        # 4. Portfolio equity curve
        if hasattr(portfolio, 'value'):
            ax4.plot(portfolio.value().index, portfolio.value(), 
                    label='Portfolio Value', color='green')
            ax4.set_title('Portfolio Equity Curve')
            ax4.set_ylabel('Portfolio Value')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "advanced_mtf_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create advanced MTF charts: {str(e)}")


def create_signal_quality_heatmap(signal_quality, output_dir, logger):
    """Create signal quality heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not signal_quality:
            return
        
        # Create quality matrix
        metrics = ['entry_frequency', 'exit_frequency', 'signal_balance']
        values = [signal_quality.get(m, 0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Simple bar chart for quality metrics
        ax.bar(metrics, values, color=['blue', 'red', 'green'])
        ax.set_title('Signal Quality Metrics')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "signal_quality.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create signal quality heatmap: {str(e)}")


def create_contribution_chart(contributions, output_dir, logger):
    """Create timeframe contribution chart."""
    try:
        import matplotlib.pyplot as plt
        
        if not contributions:
            return
        
        timeframes = list(contributions.keys())
        weights = [contributions[tf]['weight'] for tf in timeframes]
        relative_contribs = [contributions[tf]['relative_contribution'] for tf in timeframes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Weights
        ax1.pie(weights, labels=timeframes, autopct='%1.1f%%')
        ax1.set_title('Timeframe Weights')
        
        # Relative contributions
        ax2.bar(timeframes, relative_contribs, color=['blue', 'green', 'orange'])
        ax2.set_title('Relative Contributions')
        ax2.set_ylabel('Contribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / "timeframe_contributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create contribution chart: {str(e)}")


def create_regime_analysis_chart(regime_analysis, output_dir, logger):
    """Create market regime analysis chart."""
    try:
        import matplotlib.pyplot as plt
        
        if not regime_analysis or 'regime_counts' not in regime_analysis:
            return
        
        regime_counts = regime_analysis['regime_counts']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Regime distribution
        ax.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        ax.set_title('Market Regime Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / "market_regimes.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create regime analysis chart: {str(e)}")


def main():
    """Main function to run advanced MTF strategy example."""
    try:
        results = run_advanced_mtf_strategy()
        if results:
            logger = get_logger()
            logger.success("✅ Advanced MTF strategy analysis completed successfully!")
        else:
            logger = get_logger()
            logger.error("❌ Advanced MTF strategy analysis failed")
    except Exception as e:
        logger = get_logger()
        logger.error(f"Advanced MTF strategy error: {str(e)}")


if __name__ == "__main__":
    main() 