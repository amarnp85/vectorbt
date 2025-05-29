#!/usr/bin/env python3
"""
Example 05: Cross-Validation with Walk-Forward Analysis

This advanced example demonstrates cross-validation using vectorbtpro's native
Splitter functionality, which provides:
- Rolling window optimization (walk-forward analysis)
- Expanding window optimization
- K-fold cross-validation for time series
- Proper train/test split handling

This helps avoid overfitting and gives better insight into strategy robustness.
"""

import os
import sys
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.utilities.structured_logging import setup_logging, get_logger


def run_walk_forward_analysis():
    """
    Demonstrates walk-forward analysis using vectorbtpro's Splitter.
    This is the gold standard for validating trading strategies.
    """
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🔄 Walk-Forward Analysis", "Testing strategy robustness with time-series cross-validation")
    
    # Configuration
    symbol = "BTC/USDT"
    timeframe = "4h"
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    # Walk-forward parameters
    train_period = 180  # days for training
    test_period = 30    # days for testing
    step_size = 15      # days to step forward
    
    logger.info(f"Symbol: {symbol} | Timeframe: {timeframe}")
    logger.info(f"Train: {train_period}d | Test: {test_period}d | Step: {step_size}d")
    
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
        
        # Log data summary
        data_points = len(data.get('close'))
        logger.data_summary([symbol], timeframe, start_date, end_date, data_points)
    
    # Load configuration for parameter ranges
    config_manager = ConfigManager()
    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
    config = config_manager.load_config(str(config_path))
    
    # Parameter ranges for optimization
    optimization_ranges = config.get("optimization_ranges", {})
    param_ranges = {
        'fast_window': optimization_ranges.get('fast_window', [5, 10, 15, 20]),
        'slow_window': optimization_ranges.get('slow_window', [20, 30, 40, 50]),
        'atr_window': optimization_ranges.get('atr_window', [10, 14, 20]),
        'atr_multiplier_sl': optimization_ranges.get('atr_multiplier_sl', [1.5, 2.0, 2.5]),
        'atr_multiplier_tp': optimization_ranges.get('atr_multiplier_tp', [2.0, 3.0, 4.0])
    }
    
    # Create time-based splitter
    with logger.operation("Setting up walk-forward splits"):
        # Convert to datetime index if needed
        if hasattr(data, 'index'):
            dates = data.index
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='4H')
        
        # Create splits manually for better control
        splits = []
        current_start = pd.to_datetime(start_date)
        data_end = pd.to_datetime(end_date)
        
        while current_start + pd.Timedelta(days=train_period + test_period) <= data_end:
            train_end = current_start + pd.Timedelta(days=train_period)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_period)
            
            splits.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_start += pd.Timedelta(days=step_size)
        
        logger.info(f"Created {len(splits)} walk-forward splits")
    
    # Run walk-forward analysis
    logger.section("🧪 Running Walk-Forward Tests")
    
    wf_results = []
    param_db = OptimalParametersDB()
    
    for i, split in enumerate(splits, 1):
        with logger.operation(f"Split {i}/{len(splits)} ({split['test_start'].strftime('%Y-%m-%d')})"):
            try:
                # Get train/test data
                train_mask = (dates >= split['train_start']) & (dates < split['train_end'])
                test_mask = (dates >= split['test_start']) & (dates < split['test_end'])
                
                if hasattr(data, 'loc'):
                    train_data = data.loc[train_mask]
                    test_data = data.loc[test_mask]
                else:
                    # Handle vectorbt data structure
                    train_indices = np.where(train_mask)[0]
                    test_indices = np.where(test_mask)[0]
                    
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        logger.warning("Insufficient data for this split")
                        continue
                    
                    train_data = data.iloc[train_indices[0]:train_indices[-1]+1]
                    test_data = data.iloc[test_indices[0]:test_indices[-1]+1]
                
                # Optimize on training data
                best_params = optimize_on_training_data(train_data, param_ranges, logger)
                
                if best_params is None:
                    logger.warning("Optimization failed for this split")
                    continue
                
                # Test on out-of-sample data
                test_results = test_on_oos_data(test_data, best_params, logger)
                
                if test_results:
                    wf_results.append({
                        'split': i,
                        'train_start': split['train_start'],
                        'train_end': split['train_end'],
                        'test_start': split['test_start'],
                        'test_end': split['test_end'],
                        'best_params': best_params,
                        'oos_sharpe': test_results['sharpe_ratio'],
                        'oos_return': test_results['total_return'],
                        'oos_max_dd': test_results['max_drawdown'],
                        'oos_trades': test_results['total_trades']
                    })
                    
                    logger.info(f"OOS Sharpe: {test_results['sharpe_ratio']:.3f}, "
                               f"Return: {test_results['total_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Error in split {i}: {str(e)}")
                continue
    
    if not wf_results:
        logger.error("No successful walk-forward results")
        return None
    
    # Analyze walk-forward results
    logger.section("📊 Walk-Forward Analysis Results")
    
    results_df = pd.DataFrame(wf_results)
    
    # Summary statistics
    avg_oos_sharpe = results_df['oos_sharpe'].mean()
    std_oos_sharpe = results_df['oos_sharpe'].std()
    avg_oos_return = results_df['oos_return'].mean()
    positive_periods = (results_df['oos_return'] > 0).sum()
    
    logger.info(f"Average OOS Sharpe: {avg_oos_sharpe:.3f} ± {std_oos_sharpe:.3f}")
    logger.info(f"Average OOS Return: {avg_oos_return:.2%}")
    logger.info(f"Positive Periods: {positive_periods}/{len(results_df)} ({positive_periods/len(results_df):.1%})")
    
    # Stability analysis
    sharpe_stability = std_oos_sharpe / abs(avg_oos_sharpe) if avg_oos_sharpe != 0 else float('inf')
    logger.info(f"Sharpe Stability (CV): {sharpe_stability:.3f}")
    
    if sharpe_stability < 0.5:
        logger.success("✅ Strategy shows good stability")
    elif sharpe_stability < 1.0:
        logger.warning("⚠️ Strategy shows moderate stability")
    else:
        logger.error("❌ Strategy shows poor stability")
    
    # Parameter consistency analysis
    logger.section("🔧 Parameter Consistency Analysis")
    
    param_consistency = analyze_parameter_consistency(results_df, logger)
    
    # Save results
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Saving walk-forward results"):
            output_dir = Path("results/example_05_walk_forward_analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_df.to_csv(output_dir / "walk_forward_results.csv", index=False)
            
            # Create performance chart
            create_wf_performance_chart(results_df, output_dir, logger)
            
            logger.success(f"Results saved to {output_dir}/")
    
    return results_df


def optimize_on_training_data(train_data, param_ranges, logger):
    """Optimize parameters on training data."""
    try:
        # Use quiet mode to suppress optimization logs
        with logger.quiet_mode():
            # Simple grid search optimization
            best_sharpe = -np.inf
            best_params = None
            
            # Sample a subset of parameter combinations for speed
            max_combinations = 100
            total_combinations = np.prod([len(v) for v in param_ranges.values()])
            
            if total_combinations > max_combinations:
                # Random sampling
                np.random.seed(42)  # For reproducibility
                combinations = []
                for _ in range(max_combinations):
                    combo = {
                        param: np.random.choice(values)
                        for param, values in param_ranges.items()
                    }
                    # Ensure fast < slow
                    if combo['fast_window'] >= combo['slow_window']:
                        combo['fast_window'] = combo['slow_window'] - 5
                        if combo['fast_window'] < 5:
                            continue
                    combinations.append(combo)
            else:
                # Full grid search
                from itertools import product
                param_names = list(param_ranges.keys())
                param_values = list(param_ranges.values())
                combinations = [
                    dict(zip(param_names, combo))
                    for combo in product(*param_values)
                    if combo[0] < combo[1]  # fast < slow
                ]
            
            for params in combinations:
                try:
                    # Run strategy
                    strategy = DMAATRTrendStrategy(train_data, **params)
                    
                    # Create simulation
                    sim_config = SimulationConfig(
                        initial_capital=100000,
                        commission=0.001,
                        slippage=0.0005
                    )
                    
                    simulator = PortfolioSimulator(sim_config)
                    portfolio = simulator.run_backtest(strategy)
                    
                    if portfolio:
                        # Calculate Sharpe ratio
                        returns = portfolio.returns()
                        if len(returns) > 0 and returns.std() > 0:
                            sharpe = returns.mean() / returns.std() * np.sqrt(252)
                            
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_params = params.copy()
                
                except Exception:
                    continue
            
            return best_params
            
    except Exception as e:
        logger.error(f"Training optimization failed: {str(e)}")
        return None


def test_on_oos_data(test_data, params, logger):
    """Test parameters on out-of-sample data."""
    try:
        # Run strategy with optimized parameters
        strategy = DMAATRTrendStrategy(test_data, **params)
        
        # Create simulation
        sim_config = SimulationConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        simulator = PortfolioSimulator(sim_config)
        portfolio = simulator.run_backtest(strategy)
        
        if portfolio:
            from backtester.analysis.performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(portfolio)
            
            return {
                'total_return': analyzer.total_return(),
                'sharpe_ratio': analyzer.sharpe_ratio(),
                'max_drawdown': analyzer.max_drawdown(),
                'total_trades': analyzer.total_trades()
            }
        
        return None
        
    except Exception as e:
        logger.error(f"OOS testing failed: {str(e)}")
        return None


def analyze_parameter_consistency(results_df, logger):
    """Analyze how consistent the optimal parameters are across splits."""
    param_columns = ['best_params']
    
    # Extract parameter values
    param_data = []
    for _, row in results_df.iterrows():
        params = row['best_params']
        param_data.append(params)
    
    param_df = pd.DataFrame(param_data)
    
    # Calculate parameter statistics
    for param in param_df.columns:
        values = param_df[param]
        logger.info(f"{param}: mean={values.mean():.2f}, std={values.std():.2f}")
    
    return param_df


def create_wf_performance_chart(results_df, output_dir, logger):
    """Create walk-forward performance visualization."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Analysis Results', fontsize=16)
        
        # OOS Sharpe over time
        axes[0, 0].plot(results_df['test_start'], results_df['oos_sharpe'], 'b-o')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Out-of-Sample Sharpe Ratio')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # OOS Returns over time
        axes[0, 1].plot(results_df['test_start'], results_df['oos_return'] * 100, 'g-o')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Out-of-Sample Returns (%)')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sharpe distribution
        axes[1, 0].hist(results_df['oos_sharpe'], bins=10, alpha=0.7, color='blue')
        axes[1, 0].axvline(x=results_df['oos_sharpe'].mean(), color='red', linestyle='--', label='Mean')
        axes[1, 0].set_title('Sharpe Ratio Distribution')
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Return distribution
        axes[1, 1].hist(results_df['oos_return'] * 100, bins=10, alpha=0.7, color='green')
        axes[1, 1].axvline(x=results_df['oos_return'].mean() * 100, color='red', linestyle='--', label='Mean')
        axes[1, 1].set_title('Return Distribution (%)')
        axes[1, 1].set_xlabel('Return (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "walk_forward_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create performance chart: {str(e)}")


def main():
    """Main function to run walk-forward analysis."""
    try:
        results = run_walk_forward_analysis()
        if results is not None:
            logger = get_logger()
            logger.success("✅ Walk-forward analysis completed successfully!")
            logger.info("Key insights:")
            logger.info("• Walk-forward analysis helps validate strategy robustness")
            logger.info("• Consistent performance across periods indicates good strategy")
            logger.info("• Parameter stability is crucial for live trading")
            logger.info("• Out-of-sample results are more realistic than in-sample")
        else:
            logger = get_logger()
            logger.error("❌ Walk-forward analysis failed")
    except Exception as e:
        logger = get_logger()
        logger.error(f"Walk-forward analysis error: {str(e)}")


if __name__ == "__main__":
    main() 