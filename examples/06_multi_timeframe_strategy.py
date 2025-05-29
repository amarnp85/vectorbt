#!/usr/bin/env python3
"""
Example 06: Multi-Timeframe Strategy

This example demonstrates how to implement a multi-timeframe trading strategy:
- Using higher timeframes for trend direction
- Lower timeframes for entry/exit timing
- Proper data alignment between timeframes
- MTF signal confirmation

Key concepts:
- Timeframe hierarchy (1h, 4h, 1d)
- Trend alignment across timeframes
- Signal confirmation logic
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


def run_mtf_strategy():
    """
    Demonstrates multi-timeframe strategy implementation.
    
    Strategy Logic:
    1. Use 1d timeframe for overall trend direction
    2. Use 4h timeframe for intermediate trend confirmation
    3. Use 1h timeframe for precise entry/exit timing
    4. Only trade when all timeframes align
    """
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("📊 Multi-Timeframe Strategy", "Implementing trend alignment across multiple timeframes")
    
    # Configuration
    symbol = "ETH/USDT"
    timeframes = ["1h", "4h", "1d"]
    base_timeframe = "1h"  # Primary timeframe for trading
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframes: {', '.join(timeframes)} (Base: {base_timeframe})")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Initialize managers
    config_manager = ConfigManager()
    param_db = OptimalParametersDB()
    
    # Load data for all timeframes
    logger.section("📈 Loading Multi-Timeframe Data")
    
    mtf_data = {}
    for tf in timeframes:
        with logger.operation(f"Loading {tf} data"):
            data = fetch_data(
                symbols=[symbol],
                exchange="binance",
                timeframe=tf,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None:
                logger.error(f"Failed to load {tf} data")
                return None
            
            mtf_data[tf] = data
            data_points = len(data.get('close'))
            logger.info(f"{tf}: {data_points} candles")
    
    # Load strategy parameters for each timeframe
    logger.section("⚙️ Loading Strategy Parameters")
    
    mtf_params = {}
    for tf in timeframes:
        # Try to get optimal parameters for this timeframe
        optimal_params = param_db.get_optimization_summary(symbol, tf)
        
        if optimal_params:
            mtf_params[tf] = optimal_params['parameters']
            logger.info(f"{tf}: Using optimal parameters (Sharpe: {optimal_params['optimization_metric']:.3f})")
        else:
            # Load default parameters
            config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
            config = config_manager.load_config(str(config_path))
            mtf_params[tf] = config.get("default_parameters", {})
            logger.info(f"{tf}: Using default parameters")
    
    # Generate signals for each timeframe
    logger.section("🎯 Generating Multi-Timeframe Signals")
    
    mtf_signals = {}
    mtf_strategies = {}
    
    for tf in timeframes:
        with logger.operation(f"Generating {tf} signals"):
            try:
                strategy = DMAATRTrendStrategy(mtf_data[tf], **mtf_params[tf])
                signals = strategy.generate_signals()
                
                mtf_strategies[tf] = strategy
                mtf_signals[tf] = signals
                
                # Log signal summary
                if hasattr(signals, 'entries') and hasattr(signals, 'exits'):
                    entry_count = signals.entries.sum() if hasattr(signals.entries, 'sum') else 0
                    exit_count = signals.exits.sum() if hasattr(signals.exits, 'sum') else 0
                    logger.info(f"{tf}: {entry_count} entries, {exit_count} exits")
                
            except Exception as e:
                logger.error(f"Failed to generate {tf} signals: {str(e)}")
                return None
    
    # Implement MTF alignment logic
    logger.section("🔄 Implementing Multi-Timeframe Alignment")
    
    aligned_signals = create_aligned_signals(mtf_signals, mtf_data, base_timeframe, logger)
    
    if aligned_signals is None:
        logger.error("Failed to create aligned signals")
        return None
    
    # Run backtest with aligned signals
    logger.section("🚀 Running Multi-Timeframe Backtest")
    
    with logger.operation("Running aligned strategy backtest"):
        # Use base timeframe data for simulation
        base_data = mtf_data[base_timeframe]
        
        # Create simulation config
        sim_config = SimulationConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        # Run simulation with aligned signals
        simulator = PortfolioSimulator(sim_config)
        
        # Create a strategy wrapper for the aligned signals
        class AlignedStrategy:
            def __init__(self, data, signals):
                self.data = data
                self.signals = signals
            
            def generate_signals(self):
                return self.signals
        
        aligned_strategy = AlignedStrategy(base_data, aligned_signals)
        portfolio = simulator.run_backtest(aligned_strategy)
        
        if portfolio is None:
            logger.error("MTF backtest failed")
            return None
    
    # Analyze MTF performance
    with logger.operation("Analyzing MTF performance"):
        analyzer = PerformanceAnalyzer(portfolio)
        
        mtf_metrics = {
            'total_return': analyzer.total_return(),
            'sharpe_ratio': analyzer.sharpe_ratio(),
            'max_drawdown': analyzer.max_drawdown(),
            'volatility': analyzer.volatility(),
            'win_rate': analyzer.win_rate(),
            'profit_factor': analyzer.profit_factor(),
            'total_trades': analyzer.total_trades()
        }
        
        # Log MTF results
        logger.backtest_result({
            'portfolio': portfolio,
            'metrics': mtf_metrics,
            'strategy_type': 'Multi-Timeframe'
        })
    
    # Compare with single timeframe performance
    logger.section("📊 Single vs Multi-Timeframe Comparison")
    
    comparison_results = {}
    
    for tf in timeframes:
        with logger.operation(f"Running {tf} single-timeframe backtest"):
            try:
                # Run single timeframe strategy
                single_strategy = mtf_strategies[tf]
                single_portfolio = simulator.run_backtest(single_strategy)
                
                if single_portfolio:
                    single_analyzer = PerformanceAnalyzer(single_portfolio)
                    comparison_results[tf] = {
                        'total_return': single_analyzer.total_return(),
                        'sharpe_ratio': single_analyzer.sharpe_ratio(),
                        'max_drawdown': single_analyzer.max_drawdown(),
                        'total_trades': single_analyzer.total_trades()
                    }
                    
                    logger.info(f"{tf}: Return {comparison_results[tf]['total_return']:.2%}, "
                               f"Sharpe {comparison_results[tf]['sharpe_ratio']:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to run {tf} comparison: {str(e)}")
    
    # Add MTF results to comparison
    comparison_results['MTF_Aligned'] = mtf_metrics
    
    # Performance comparison summary
    logger.section("🏆 Performance Comparison Summary")
    
    best_sharpe = max(comparison_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    best_return = max(comparison_results.items(), key=lambda x: x[1]['total_return'])
    
    logger.info("Performance Ranking by Sharpe Ratio:")
    sorted_by_sharpe = sorted(comparison_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    for i, (method, metrics) in enumerate(sorted_by_sharpe, 1):
        logger.info(f"{i}. {method}: Sharpe {metrics['sharpe_ratio']:.3f}, "
                   f"Return {metrics['total_return']:.2%}")
    
    logger.success(f"🏆 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
    logger.success(f"🎯 Best Return: {best_return[0]} ({best_return[1]['total_return']:.2%})")
    
    # Generate visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating MTF visualizations"):
            try:
                output_dir = Path("results/example_06_multi_timeframe_strategy")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # MTF performance chart
                plotter = PlottingEngine(portfolio)
                plotter.plot_performance(
                    title=f"Multi-Timeframe Strategy Performance - {symbol}",
                    save_path=output_dir / "mtf_performance.png"
                )
                
                # Timeframe comparison chart
                create_mtf_comparison_chart(comparison_results, output_dir, logger)
                
                # Signal alignment visualization
                create_signal_alignment_chart(mtf_signals, mtf_data, output_dir, logger)
                
                logger.success(f"Visualizations saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {str(e)}")
    
    # Key insights and recommendations
    logger.section("💡 Multi-Timeframe Strategy Insights")
    
    mtf_vs_best_single = mtf_metrics['sharpe_ratio'] / max([r['sharpe_ratio'] for r in comparison_results.values() if r != mtf_metrics])
    
    logger.info("Key Findings:")
    if mtf_vs_best_single > 1.1:
        logger.success("✅ MTF alignment provides significant improvement")
    elif mtf_vs_best_single > 1.0:
        logger.info("✅ MTF alignment provides modest improvement")
    else:
        logger.warning("⚠️ MTF alignment may be over-filtering signals")
    
    logger.info("• Higher timeframes help identify major trend direction")
    logger.info("• Lower timeframes provide precise entry/exit timing")
    logger.info("• Signal alignment reduces false signals but may miss opportunities")
    logger.info("• Consider dynamic timeframe weights based on market conditions")
    
    return {
        'mtf_portfolio': portfolio,
        'mtf_metrics': mtf_metrics,
        'comparison_results': comparison_results,
        'mtf_signals': mtf_signals
    }


def create_aligned_signals(mtf_signals, mtf_data, base_timeframe, logger):
    """
    Create aligned signals that require confirmation across timeframes.
    
    Logic:
    - 1d timeframe: Overall trend direction
    - 4h timeframe: Intermediate confirmation
    - 1h timeframe: Entry timing
    """
    try:
        # Get base timeframe data for alignment
        base_data = mtf_data[base_timeframe]
        base_index = base_data.index if hasattr(base_data, 'index') else None
        
        if base_index is None:
            logger.error("Cannot access base timeframe index")
            return None
        
        # Initialize aligned signals
        aligned_entries = pd.Series(False, index=base_index)
        aligned_exits = pd.Series(False, index=base_index)
        
        # Simple alignment logic: require all timeframes to agree
        for i, timestamp in enumerate(base_index):
            try:
                # Check if all timeframes have bullish signals at this time
                all_bullish = True
                all_bearish = True
                
                for tf, signals in mtf_signals.items():
                    if hasattr(signals, 'entries') and hasattr(signals, 'exits'):
                        # Find closest timestamp in this timeframe
                        tf_data = mtf_data[tf]
                        tf_index = tf_data.index if hasattr(tf_data, 'index') else None
                        
                        if tf_index is not None:
                            # Find closest timestamp
                            closest_idx = tf_index.get_indexer([timestamp], method='ffill')[0]
                            
                            if closest_idx >= 0 and closest_idx < len(signals.entries):
                                # Check signal state
                                if not signals.entries.iloc[closest_idx]:
                                    all_bullish = False
                                if not signals.exits.iloc[closest_idx]:
                                    all_bearish = False
                
                # Set aligned signals
                if all_bullish:
                    aligned_entries.iloc[i] = True
                if all_bearish:
                    aligned_exits.iloc[i] = True
                    
            except Exception:
                continue
        
        # Create aligned signals object
        class AlignedSignals:
            def __init__(self, entries, exits):
                self.entries = entries
                self.exits = exits
        
        aligned_signals = AlignedSignals(aligned_entries, aligned_exits)
        
        # Log alignment summary
        entry_count = aligned_entries.sum()
        exit_count = aligned_exits.sum()
        logger.info(f"Aligned signals: {entry_count} entries, {exit_count} exits")
        
        return aligned_signals
        
    except Exception as e:
        logger.error(f"Signal alignment failed: {str(e)}")
        return None


def create_mtf_comparison_chart(comparison_results, output_dir, logger):
    """Create comparison chart for different timeframe approaches."""
    try:
        import matplotlib.pyplot as plt
        
        methods = list(comparison_results.keys())
        sharpe_ratios = [comparison_results[m]['sharpe_ratio'] for m in methods]
        returns = [comparison_results[m]['total_return'] * 100 for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sharpe ratio comparison
        bars1 = ax1.bar(methods, sharpe_ratios, color=['blue', 'green', 'orange', 'red'])
        ax1.set_title('Sharpe Ratio Comparison')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, sharpe_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Return comparison
        bars2 = ax2.bar(methods, returns, color=['blue', 'green', 'orange', 'red'])
        ax2.set_title('Total Return Comparison (%)')
        ax2.set_ylabel('Return (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, returns):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "timeframe_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create comparison chart: {str(e)}")


def create_signal_alignment_chart(mtf_signals, mtf_data, output_dir, logger):
    """Create visualization showing signal alignment across timeframes."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(mtf_signals), 1, figsize=(15, 4 * len(mtf_signals)))
        if len(mtf_signals) == 1:
            axes = [axes]
        
        for i, (tf, signals) in enumerate(mtf_signals.items()):
            data = mtf_data[tf]
            
            # Plot price
            if hasattr(data, 'get') and 'close' in data:
                close_prices = data.get('close')
                axes[i].plot(close_prices.index, close_prices, label='Close Price', alpha=0.7)
            
            # Plot signals
            if hasattr(signals, 'entries') and hasattr(signals, 'exits'):
                entry_points = signals.entries[signals.entries == True]
                exit_points = signals.exits[signals.exits == True]
                
                if len(entry_points) > 0:
                    axes[i].scatter(entry_points.index, 
                                   [close_prices.loc[idx] for idx in entry_points.index if idx in close_prices.index],
                                   color='green', marker='^', s=50, label='Entries')
                
                if len(exit_points) > 0:
                    axes[i].scatter(exit_points.index,
                                   [close_prices.loc[idx] for idx in exit_points.index if idx in close_prices.index],
                                   color='red', marker='v', s=50, label='Exits')
            
            axes[i].set_title(f'{tf} Timeframe Signals')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "signal_alignment.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create signal alignment chart: {str(e)}")


def main():
    """Main function to run multi-timeframe strategy example."""
    try:
        results = run_mtf_strategy()
        if results:
            logger = get_logger()
            logger.success("✅ Multi-timeframe strategy analysis completed successfully!")
        else:
            logger = get_logger()
            logger.error("❌ Multi-timeframe strategy analysis failed")
    except Exception as e:
        logger = get_logger()
        logger.error(f"Multi-timeframe strategy error: {str(e)}")


if __name__ == "__main__":
    main() 