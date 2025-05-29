#!/usr/bin/env python3
"""
Example 02: Parameter Optimization

This example demonstrates how to find optimal parameters for your strategy using
vectorbtpro's advanced optimization capabilities:
- Using @vbt.parameterized decorator for efficient parameter broadcasting
- Testing different parameter combinations with grid search
- Finding the best performing parameters
- Storing optimal parameters in database for future use

This builds on Example 01 by showing how to improve strategy performance.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules directly
from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.utilities.structured_logging import setup_logging


@vbt.parameterized(merge_func="concat")
def optimize_dma_strategy(data, fast_window, slow_window, atr_window, atr_multiplier_sl, atr_multiplier_tp):
    """
    Parameterized strategy function for optimization using vectorbtpro.
    
    This function will be called with different parameter combinations
    and return the performance metric for each combination.
    """
    try:
        # Create strategy parameters
        params = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'atr_window': atr_window,
            'atr_multiplier_sl': atr_multiplier_sl,
            'atr_multiplier_tp': atr_multiplier_tp,
            'use_volume_filter': False  # Keep simple for optimization
        }
        
        # Initialize strategy
        strategy = DMAATRTrendStrategy(data, params)
        
        # Calculate indicators and generate signals
        strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Use PortfolioSimulator for proper long/short signal handling
        from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
        
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=0.001,
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        # Return the optimization metric (Sharpe ratio)
        return portfolio.sharpe_ratio
        
    except Exception as e:
        # Return NaN for failed combinations
        return np.nan


def safe_extract_metric(metric):
    """Safely extract metric value, handling Series and scalar cases."""
    if hasattr(metric, 'iloc'):
        return float(metric.iloc[0])
    else:
        return float(metric)


def safe_extract_int_metric(metric):
    """Safely extract integer metric value, handling Series and scalar cases."""
    if hasattr(metric, 'iloc'):
        return int(metric.iloc[0])
    elif callable(metric):
        result = metric()
        if hasattr(result, 'iloc'):
            return int(result.iloc[0])
        else:
            return int(result)
    else:
        return int(metric)


def convert_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def main():
    """Run parameter optimization example."""
    # Setup clean structured logging
    logger = setup_logging("INFO")
    
    # Configuration
    symbol = "BTC/USDT"
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Start with a clean section header
    logger.section("🔧 Parameter Optimization", f"Optimizing {symbol} on {timeframe} timeframe")
    
    # Initialize managers
    param_db = OptimalParametersDB()
    
    # Check if already optimized
    existing_params = param_db.get_optimization_summary(symbol, timeframe)
    if existing_params:
        logger.info(f"Found previous optimization for {symbol}")
        logger.info(f"Previous Sharpe: {existing_params['optimization_metric']:.3f}")
        logger.info(f"Last optimized: {existing_params['created_at'][:10]}")
        
        # Ask if user wants to re-optimize
        logger.info("Re-running optimization to demonstrate the process...")
    
    # Load data with operation tracking
    with logger.operation("Loading market data"):
        data = fetch_data(
            symbols=[symbol],
            exchange="binance",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None:
            logger.error("Failed to load market data")
            return
        
        # Log data summary
        data_points = len(data.get('close'))
        logger.data_summary([symbol], timeframe, start_date, end_date, data_points)
    
    # Load configuration for parameter ranges
    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
    config_manager = ConfigManager()
    config = config_manager.load_config(str(config_path))
    
    # Map config parameters to strategy parameters
    optimization_ranges = config.get("optimization_ranges", {})
    
    # Define parameter ranges for optimization (using vectorbtpro best practices)
    param_ranges = {
        'fast_window': optimization_ranges.get('short_ma_window', [10, 15, 20, 25, 30]),
        'slow_window': optimization_ranges.get('long_ma_window', [40, 50, 60, 70, 80]),
        'atr_window': optimization_ranges.get('atr_period', [10, 14, 20]),
        'atr_multiplier_sl': optimization_ranges.get('sl_atr_multiplier', [1.5, 2.0, 2.5, 3.0]),
        'atr_multiplier_tp': optimization_ranges.get('tp_atr_multiplier', [3.0, 4.0, 5.0, 6.0])
    }
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_ranges.values():
        total_combinations *= len(values)
    
    # Log optimization configuration
    logger.section("⚙️ Optimization Configuration")
    
    # Create parameter ranges table
    param_data = []
    for param, values in param_ranges.items():
        param_data.append({
            'Parameter': param.replace('_', ' ').title(),
            'Values': str(values),
            'Count': len(values)
        })
    
    logger.table("Parameter Ranges", param_data)
    logger.info(f"Total combinations to test: {total_combinations:,}")
    
    # Output directory
    symbol_safe = symbol.replace('/', '_')
    output_dir = f"results/symbols/{symbol_safe}/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization using vectorbtpro's parameterized approach
    try:
        start_time = time.time()
        
        with logger.operation("Running parameter optimization", total_combinations):
            # Use quiet mode during optimization to avoid log spam
            with logger.quiet_mode():
                # Also suppress standard Python logging during optimization
                import logging
                strategy_logger = logging.getLogger('backtester.strategies.dma_atr_trend_strategy')
                original_level = strategy_logger.level
                strategy_logger.setLevel(logging.WARNING)  # Suppress INFO logs
                
                try:
                    sharpe_results = optimize_dma_strategy(
                        data,
                        fast_window=vbt.Param(param_ranges['fast_window'], condition="fast_window < slow_window"),
                        slow_window=vbt.Param(param_ranges['slow_window']),
                        atr_window=vbt.Param(param_ranges['atr_window']),
                        atr_multiplier_sl=vbt.Param(param_ranges['atr_multiplier_sl']),
                        atr_multiplier_tp=vbt.Param(param_ranges['atr_multiplier_tp'])
                    )
                finally:
                    # Restore original logging level
                    strategy_logger.setLevel(original_level)
        
        optimization_time = time.time() - start_time
        
        # Process results
        if sharpe_results is not None and not sharpe_results.empty:
            # Find best parameters
            valid_results = sharpe_results.dropna()
            if len(valid_results) == 0:
                logger.error("No valid optimization results found")
                return
            
            best_idx = valid_results.idxmax()
            best_sharpe = valid_results.loc[best_idx]
            
            # Extract best parameters from the index
            if isinstance(best_idx, tuple):
                # Multi-index case
                param_names = sharpe_results.index.names
                best_params = dict(zip(param_names, best_idx))
            else:
                # Single parameter case (shouldn't happen with multiple params)
                best_params = {'fast_window': best_idx}
            
            # Convert to native Python types
            best_params = convert_for_json(best_params)
            
            # Log optimization results
            optimization_results = {
                'best_params': best_params,
                'best_metric': float(best_sharpe),
                'total_trials': len(sharpe_results),
                'valid_trials': len(valid_results),
                'optimization_time': optimization_time
            }
            logger.optimization_result(optimization_results)
            
            # Run full backtest with best parameters
            with logger.operation("Running validation backtest"):
                # Create strategy with best parameters
                strategy = DMAATRTrendStrategy(data, best_params)
                strategy.init_indicators()
                signals = strategy.generate_signals()
                
                # Create detailed portfolio
                portfolio = vbt.Portfolio.from_signals(
                    data.get('close'),
                    signals['entries'],
                    signals['exits'],
                    direction="both",
                    freq=data.wrapper.freq
                )
                
                # Calculate detailed performance metrics with proper type handling
                performance_metrics = {
                    'sharpe_ratio': safe_extract_metric(portfolio.sharpe_ratio),
                    'total_return': safe_extract_metric(portfolio.total_return),
                    'max_drawdown': safe_extract_metric(portfolio.max_drawdown),
                    'win_rate': safe_extract_metric(portfolio.trades.win_rate),
                    'total_trades': safe_extract_int_metric(portfolio.trades.count)
                }
                
                # Add profit factor if available
                if hasattr(portfolio.trades, 'profit_factor'):
                    performance_metrics['profit_factor'] = safe_extract_metric(portfolio.trades.profit_factor)
                else:
                    performance_metrics['profit_factor'] = 0.0
                
                # Convert to native Python types
                performance_metrics = convert_for_json(performance_metrics)
                
                # Log backtest results
                backtest_results = {'portfolio': portfolio}
                logger.backtest_result(backtest_results)
            
            # Store the results in database
            with logger.operation("Saving optimization results"):
                optimization_stats = {
                    'total_combinations': total_combinations,
                    'valid_combinations': len(valid_results),
                    'optimization_time': optimization_time
                }
                
                param_db.store_optimization_result(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name="DMAATRTrendStrategy",
                    best_params=best_params,
                    performance_metrics=performance_metrics,
                    parameter_ranges=convert_for_json(param_ranges),
                    optimization_stats=optimization_stats
                )
                
                # Save results to CSV
                results_df = pd.DataFrame({
                    'sharpe_ratio': sharpe_results
                }).reset_index()
                results_df.to_csv(os.path.join(output_dir, "optimization_results.csv"), index=False)
                
                # Save best parameters
                import json
                with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
                    json.dump(best_params, f, indent=2)
                
                # Save performance metrics
                with open(os.path.join(output_dir, "performance_metrics.json"), 'w') as f:
                    json.dump(performance_metrics, f, indent=2)
            
            # Compare with previous (if exists)
            if existing_params:
                old_sharpe = existing_params['optimization_metric']
                new_sharpe = performance_metrics['sharpe_ratio']
                improvement = ((new_sharpe - old_sharpe) / abs(old_sharpe) * 100) if old_sharpe != 0 else 0
                
                logger.section("📊 Performance Comparison")
                comparison_data = [
                    {'Metric': 'Previous Sharpe', 'Value': f"{old_sharpe:.3f}"},
                    {'Metric': 'New Sharpe', 'Value': f"{new_sharpe:.3f}"},
                    {'Metric': 'Improvement', 'Value': f"{improvement:+.1f}%"}
                ]
                logger.table("Optimization Comparison", comparison_data)
            
            # Final summary
            logger.section("✅ Optimization Complete")
            logger.success(f"Results saved to: {output_dir}")
            logger.info(f"Database location: {param_db.db_path}")
            
            # Show generated files
            files_data = [
                {'File': 'optimization_results.csv', 'Description': 'All tested parameter combinations'},
                {'File': 'best_params.json', 'Description': 'Best parameters found'},
                {'File': 'performance_metrics.json', 'Description': 'Detailed performance metrics'}
            ]
            logger.table("Generated Files", files_data)
            
        else:
            logger.error("Optimization failed - no results returned")
            
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        return
    
    # Next steps
    logger.section("🚀 Next Steps")
    next_steps = [
        "Review optimization_results.csv to see all tested combinations",
        "The optimal parameters are now stored in the database",
        "Run example 01 again to see it use the optimal parameters",
        "Use strategy_tester.py for testing multiple symbols/timeframes",
        "Try optimizing other symbols or timeframes"
    ]
    
    for i, step in enumerate(next_steps, 1):
        logger.info(f"{i}. {step}")


if __name__ == "__main__":
    main() 