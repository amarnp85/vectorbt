#!/usr/bin/env python3
"""
Strategy Tester - Consolidated Testing Workflow

This script provides a complete workflow for testing strategies with consistent date ranges:
1. Load data for specified symbols and timeframes
2. Check for existing optimal parameters in database
3. Run optimization if needed or requested
4. Run backtests with optimal/default parameters using fixed date ranges
5. Generate performance analysis and visualizations
6. Store results and date ranges in database for future reference

Key Features:
- Fixed default date ranges for consistent comparisons
- Configurable date ranges (ratio-based or explicit dates)
- Date range storage in database for walk-forward analysis reference
- Consolidated workflow without unnecessary complexity

Usage Examples:
    # Test with default fixed date range
    python examples/strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize
    
    # Test with custom date range
    python strategy_tester.py --symbols BTC/USDT --timeframes 4h --start-date 2023-01-01 --end-date 2023-12-31
    
    # Test multiple symbols with optimization
    python strategy_tester.py --symbols BTC/USDT,ETH/USDT --timeframes 4h,1d --optimize
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import time
import json
from typing import Dict, Any, Tuple, List, Optional
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules directly
from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.trading_charts_refactored import TradingChartsEngine
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.utilities.structured_logging import setup_logging

# Default date ranges for consistent backtesting
DEFAULT_DATE_RANGES = {
    'recent_data': {
        'start_date': '2020-01-01',
        'end_date': '2023-12-30',
        'description': 'Recent 18-month period for current market analysis'
    },
    'full_data': {
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'description': 'Full historical data for comprehensive analysis'
    },
    'optimization_period': {
        'start_date': '2022-01-01',
        'end_date': '2023-12-31',
        'description': 'Optimization period (24 months)'
    },
    'validation_period': {
        'start_date': '2024-01-01',
        'end_date': '2024-06-30',
        'description': 'Validation period (6 months)'
    }
}

# Default period to use if none specified
DEFAULT_PERIOD = 'recent_data'


@vbt.parameterized(merge_func="concat")
def optimize_dma_strategy(data, fast_window, slow_window, atr_window, atr_multiplier_sl, atr_multiplier_tp):
    """
    Parameterized strategy function for optimization using vectorbtpro.
    """
    try:
        params = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            'atr_window': atr_window,
            'atr_multiplier_sl': atr_multiplier_sl,
            'atr_multiplier_tp': atr_multiplier_tp,
            'use_volume_filter': False
        }
        
        strategy = DMAATRTrendStrategy(data, params)
        strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Use PortfolioSimulator for proper long/short signal handling
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=0.001,
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        return portfolio.sharpe_ratio
        
    except Exception as e:
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


def get_default_params():
    """Get default strategy parameters."""
    return {
        'fast_window': 10,
        'slow_window': 30,
        'atr_window': 14,
        'atr_multiplier_sl': 2.0,
        'atr_multiplier_tp': 3.0,
        'use_volume_filter': False
    }


def fix_parameter_types(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fix parameter types to ensure proper data types."""
    if not params:
        return params
    
    fixed_params = {}
    for key, value in params.items():
        if key in ['fast_window', 'slow_window', 'atr_window']:
            # These should be integers
            try:
                fixed_params[key] = int(value)
            except (ValueError, TypeError):
                fixed_params[key] = value
        elif key in ['atr_multiplier_sl', 'atr_multiplier_tp']:
            # These should be floats
            try:
                fixed_params[key] = float(value)
            except (ValueError, TypeError):
                fixed_params[key] = value
        else:
            fixed_params[key] = value
    
    return fixed_params


def get_date_range(args):
    """
    Get the date range to use for backtesting.
    
    Priority:
    1. Explicit start_date/end_date from args
    2. Named period from args
    3. Default period
    
    Returns:
        Tuple of (start_date, end_date, period_description)
    """
    # If explicit dates provided, use them
    if args.start_date and args.end_date:
        return args.start_date, args.end_date, f"Custom range: {args.start_date} to {args.end_date}"
    
    # If named period provided, use it
    if args.period and args.period in DEFAULT_DATE_RANGES:
        period_config = DEFAULT_DATE_RANGES[args.period]
        return period_config['start_date'], period_config['end_date'], period_config['description']
    
    # Use default period
    period_config = DEFAULT_DATE_RANGES[DEFAULT_PERIOD]
    return period_config['start_date'], period_config['end_date'], period_config['description']


def optimize_parameters(symbol, timeframe, data, param_ranges, logger, force_reoptimize=False, date_info=None):
    """
    Run parameter optimization and store results with date range information.
    
    Returns:
        Tuple of (success, best_params, performance_metrics, optimization_results)
    """
    param_db = OptimalParametersDB()
    
    # Check if already optimized (unless forcing)
    if not force_reoptimize:
        existing_params = param_db.get_optimization_summary(symbol, timeframe)
        if existing_params and existing_params.get('optimization_metric', 0) > 0:
            # Only use existing if it has a valid optimization metric
            fixed_params = fix_parameter_types(existing_params['parameters'])
            logger.info(f"Using existing optimization for {symbol} {timeframe} (Sharpe: {existing_params['optimization_metric']:.3f})")
            return True, fixed_params, existing_params['performance'], None
        elif existing_params:
            logger.warning(f"Found invalid optimization for {symbol} {timeframe} (Sharpe: {existing_params.get('optimization_metric', 0):.3f}) - re-optimizing")
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_ranges.values():
        total_combinations *= len(values)
    
    logger.info(f"Running optimization for {symbol} {timeframe}: {total_combinations:,} combinations")
    
    # Run optimization
    start_time = time.time()
    
    with logger.quiet_mode():
        strategy_logger = logging.getLogger('backtester.strategies.dma_atr_trend_strategy')
        original_level = strategy_logger.level
        strategy_logger.setLevel(logging.WARNING)
        
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
            strategy_logger.setLevel(original_level)
    
    optimization_time = time.time() - start_time
    
    # Process results
    if sharpe_results is not None and not sharpe_results.empty:
        valid_results = sharpe_results.dropna()
        if len(valid_results) == 0:
            logger.error(f"No valid optimization results for {symbol} {timeframe}")
            return False, None, None, None
        
        best_idx = valid_results.idxmax()
        best_sharpe = valid_results.loc[best_idx]
        
        # Extract best parameters
        if isinstance(best_idx, tuple):
            param_names = sharpe_results.index.names
            best_params = dict(zip(param_names, best_idx))
        else:
            best_params = {'fast_window': best_idx}
        
        best_params = convert_for_json(best_params)
        
        # Run validation backtest with best parameters
        strategy = DMAATRTrendStrategy(data, best_params)
        strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Use PortfolioSimulator for proper long/short signal handling
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=0.001,
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(portfolio, signals=signals)
        
        # Get all metrics
        metrics = {}
        metrics.update(analyzer.get_returns_metrics())
        metrics.update(analyzer.get_trade_metrics())
        metrics.update(analyzer.get_drawdown_metrics())
        metrics.update(analyzer.get_risk_metrics())
        
        # Add metadata and parameter information
        metrics['data_points'] = len(data.get('close'))
        metrics['param_source'] = 'optimized'
        metrics['parameters'] = best_params.copy()
        
        # Add parameter summary for display
        param_summary = f"F{best_params.get('fast_window', '?')}/S{best_params.get('slow_window', '?')}/A{best_params.get('atr_window', '?')}"
        metrics['param_summary'] = param_summary
        
        logger.info(f"✅ {symbol} {timeframe}: Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe_ratio', 0):.2f} ({param_summary})")
        
        # Create comprehensive optimization analysis
        optimization_analysis = create_optimization_analysis(
            sharpe_results, best_params, metrics, param_ranges, 
            symbol, timeframe, optimization_time, date_info
        )
        
        # Store results in database with date range information
        optimization_stats = {
            'total_combinations': total_combinations,
            'valid_combinations': len(valid_results),
            'optimization_time': optimization_time,
            'data_points': len(data.get('close'))
        }
        
        # Add date range information if provided
        if date_info:
            optimization_stats.update({
                'start_date': date_info['start_date'],
                'end_date': date_info['end_date'],
                'period_description': date_info['description'],
                'date_range_type': date_info.get('type', 'custom')
            })
        
        optimization_stats = convert_for_json(optimization_stats)
        
        param_db.store_optimization_result(
            symbol=symbol,
            timeframe=timeframe,
            strategy_name="DMAATRTrendStrategy",
            best_params=best_params,
            performance_metrics=metrics,
            parameter_ranges=convert_for_json(param_ranges),
            optimization_stats=optimization_stats
        )
        
        return True, best_params, metrics, optimization_analysis
    
    else:
        logger.error(f"Optimization failed for {symbol} {timeframe}")
        return False, None, None, None


def create_optimization_analysis(sharpe_results, best_params, performance_metrics, param_ranges, 
                                symbol, timeframe, optimization_time, date_info):
    """
    Create comprehensive optimization analysis.
    
    Returns:
        Dictionary with analysis results and file paths
    """
    import pandas as pd
    import numpy as np
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame({'sharpe_ratio': sharpe_results}).reset_index()
    valid_results = results_df.dropna()
    
    # Basic statistics
    stats = {
        'total_combinations': len(results_df),
        'valid_combinations': len(valid_results),
        'success_rate': len(valid_results) / len(results_df) * 100,
        'best_sharpe': valid_results['sharpe_ratio'].max(),
        'worst_sharpe': valid_results['sharpe_ratio'].min(),
        'mean_sharpe': valid_results['sharpe_ratio'].mean(),
        'std_sharpe': valid_results['sharpe_ratio'].std(),
        'optimization_time': optimization_time
    }
    
    # Parameter sensitivity analysis
    param_sensitivity = {}
    for param in param_ranges.keys():
        if param in valid_results.columns:
            param_groups = valid_results.groupby(param)['sharpe_ratio'].agg(['mean', 'std', 'count'])
            param_sensitivity[param] = {
                'best_value': valid_results.loc[valid_results['sharpe_ratio'].idxmax(), param],
                'mean_by_value': param_groups['mean'].to_dict(),
                'std_by_value': param_groups['std'].to_dict(),
                'count_by_value': param_groups['count'].to_dict()
            }
    
    # Top performers analysis
    top_n = min(20, len(valid_results))
    top_performers = valid_results.nlargest(top_n, 'sharpe_ratio')
    
    # Parameter frequency in top performers
    param_frequency = {}
    for param in param_ranges.keys():
        if param in top_performers.columns:
            freq = top_performers[param].value_counts()
            param_frequency[param] = freq.to_dict()
    
    return {
        'statistics': stats,
        'parameter_sensitivity': param_sensitivity,
        'top_performers': top_performers.to_dict('records'),
        'parameter_frequency': param_frequency,
        'best_parameters': best_params,
        'performance_metrics': performance_metrics,
        'date_info': date_info,
        'all_results': valid_results.to_dict('records')
    }


def create_streamlined_optimization_visualization(optimization_analysis, symbol, timeframe, opt_dir):
    """
    Create streamlined optimization analysis visualizations using Plotly.
    Only creates essential charts, removes redundant ones.
    
    Args:
        optimization_analysis: Dictionary with optimization analysis results
        symbol: Trading symbol
        timeframe: Timeframe
        opt_dir: Output directory for saving charts
        
    Returns:
        List of created visualization file paths
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import logging
    
    logger = logging.getLogger(__name__)
    created_files = []
    
    try:
        # Create visualizations subdirectory
        viz_dir = os.path.join(opt_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Parameter Sensitivity Heatmap (keep this - most useful)
        try:
            param_sensitivity = optimization_analysis.get('parameter_sensitivity', {})
            if param_sensitivity:
                # Create heatmap data
                heatmap_data = []
                param_names = []
                
                for param, analysis in param_sensitivity.items():
                    param_names.append(param.replace('_', ' ').title())
                    values = []
                    labels = []
                    
                    for value, mean_sharpe in analysis['mean_by_value'].items():
                        values.append(mean_sharpe)
                        labels.append(str(value))
                    
                    heatmap_data.append(values)
                
                if heatmap_data:
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data,
                        x=labels if len(set(len(row) for row in heatmap_data)) == 1 else None,
                        y=param_names,
                        colorscale='RdYlGn',
                        colorbar=dict(title="Mean Sharpe Ratio"),
                        hovertemplate='Parameter: %{y}<br>Value: %{x}<br>Mean Sharpe: %{z:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Parameter Sensitivity - {symbol} {timeframe}",
                        xaxis_title="Parameter Values",
                        yaxis_title="Parameters",
                        height=400 + len(param_names) * 30,
                        template="plotly_white"
                    )
                    
                    heatmap_file = os.path.join(viz_dir, "parameter_heatmap.html")
                    fig.write_html(heatmap_file)
                    created_files.append(heatmap_file)
        
        except Exception as e:
            logger.warning(f"Could not create parameter heatmap: {e}")
        
        # 2. Performance Distribution (keep this - shows optimization quality)
        try:
            all_results = optimization_analysis.get('all_results', [])
            if all_results:
                sharpe_values = [result['sharpe_ratio'] for result in all_results if 'sharpe_ratio' in result]
                
                if sharpe_values:
                    # Create performance distribution chart
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=["Sharpe Ratio Distribution", "Top vs Bottom Performers"],
                        specs=[[{"type": "histogram"}, {"type": "box"}]]
                    )
                    
                    # Histogram of Sharpe ratios
                    fig.add_trace(
                        go.Histogram(
                            x=sharpe_values,
                            nbinsx=30,
                            name="Sharpe Distribution",
                            marker_color='lightblue',
                            opacity=0.7
                        ),
                        row=1, col=1
                    )
                    
                    # Box plot comparing top vs bottom performers
                    sorted_results = sorted(all_results, key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
                    top_10_percent = int(len(sorted_results) * 0.1)
                    bottom_10_percent = int(len(sorted_results) * 0.1)
                    
                    top_sharpe = [r['sharpe_ratio'] for r in sorted_results[:top_10_percent]]
                    bottom_sharpe = [r['sharpe_ratio'] for r in sorted_results[-bottom_10_percent:]]
                    
                    fig.add_trace(
                        go.Box(y=top_sharpe, name="Top 10%", marker_color='green'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Box(y=bottom_sharpe, name="Bottom 10%", marker_color='red'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        title=f"Performance Distribution - {symbol} {timeframe}",
                        height=400,
                        showlegend=False,
                        template="plotly_white"
                    )
                    
                    # Update axis labels
                    fig.update_xaxes(title_text="Sharpe Ratio", row=1, col=1)
                    fig.update_yaxes(title_text="Frequency", row=1, col=1)
                    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
                    
                    distribution_file = os.path.join(viz_dir, "performance_distribution.html")
                    fig.write_html(distribution_file)
                    created_files.append(distribution_file)
        
        except Exception as e:
            logger.warning(f"Could not create performance distribution chart: {e}")
        
        # 3. Top Performers Analysis (keep this - shows best parameter combinations)
        try:
            top_performers = optimization_analysis.get('top_performers', [])
            if top_performers and len(top_performers) >= 5:
                # Create parallel coordinates plot for top performers
                import pandas as pd
                df_top = pd.DataFrame(top_performers[:10])  # Top 10
                
                # Prepare data for parallel coordinates - only numeric columns
                dimensions = []
                for col in df_top.columns:
                    if col != 'sharpe_ratio' and pd.api.types.is_numeric_dtype(df_top[col]):
                        dimensions.append(dict(
                            range=[df_top[col].min(), df_top[col].max()],
                            label=col.replace('_', ' ').title(),
                            values=df_top[col]
                        ))
                
                # Add Sharpe ratio as the color dimension
                if pd.api.types.is_numeric_dtype(df_top['sharpe_ratio']):
                    dimensions.append(dict(
                        range=[df_top['sharpe_ratio'].min(), df_top['sharpe_ratio'].max()],
                        label='Sharpe Ratio',
                        values=df_top['sharpe_ratio']
                    ))
                
                # Only create plot if we have enough numeric dimensions
                if len(dimensions) >= 2:
                    fig = go.Figure(data=go.Parcoords(
                        line=dict(
                            color=df_top['sharpe_ratio'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        dimensions=dimensions
                    ))
                    
                    fig.update_layout(
                        title=f"Top 10 Performers - {symbol} {timeframe}",
                        height=500,
                        template="plotly_white"
                    )
                    
                    parallel_file = os.path.join(viz_dir, "top_performers_parallel.html")
                    fig.write_html(parallel_file)
                    created_files.append(parallel_file)
        
        except Exception as e:
            logger.warning(f"Could not create top performers analysis: {e}")
        
        logger.info(f"Created {len(created_files)} streamlined optimization visualization files")
        return created_files
        
    except Exception as e:
        logger.error(f"Failed to create streamlined optimization visualizations: {e}")
        return created_files


def save_optimization_analysis(optimization_analysis, symbol, timeframe, output_dir):
    """
    Save optimization analysis to files and create visualizations.
    
    Returns:
        List of created file paths (including visualizations)
    """
    import json
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    
    # Create symbol-specific optimization directory organized by timeframe
    symbol_safe = symbol.replace('/', '_')
    opt_dir = f"results/symbols/{symbol_safe}/optimization/{timeframe}"
    os.makedirs(opt_dir, exist_ok=True)
    
    created_files = []
    
    # 1. Save optimization summary (keep this one)
    summary = {
        'symbol': symbol,
        'timeframe': timeframe,
        'optimization_date': datetime.now().isoformat(),
        'statistics': optimization_analysis['statistics'],
        'best_parameters': optimization_analysis['best_parameters'],
        'performance_metrics': optimization_analysis['performance_metrics'],
        'date_range': optimization_analysis.get('date_info', {})
    }
    
    summary_file = os.path.join(opt_dir, "optimization_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    created_files.append(summary_file)
    
    # 2. Save top performers as CSV (keep this one)
    top_performers_df = pd.DataFrame(optimization_analysis['top_performers'])
    top_file = os.path.join(opt_dir, "top_performers.csv")
    top_performers_df.to_csv(top_file, index=False)
    created_files.append(top_file)
    
    # 3. Create parameter analysis CSV for easy viewing (keep this one)
    param_analysis_data = []
    for param, analysis in optimization_analysis['parameter_sensitivity'].items():
        for value, mean_sharpe in analysis['mean_by_value'].items():
            param_analysis_data.append({
                'parameter': param,
                'value': value,
                'mean_sharpe': mean_sharpe,
                'std_sharpe': analysis['std_by_value'].get(value, 0),
                'count': analysis['count_by_value'].get(value, 0),
                'is_best': value == analysis['best_value']
            })
    
    if param_analysis_data:
        param_df = pd.DataFrame(param_analysis_data)
        param_file = os.path.join(opt_dir, "parameter_analysis.csv")
        param_df.to_csv(param_file, index=False)
        created_files.append(param_file)
    
    # 4. Create streamlined visualizations (remove redundant ones)
    try:
        visualization_files = create_streamlined_optimization_visualization(optimization_analysis, symbol, timeframe, opt_dir)
        created_files.extend(visualization_files)
        logger.info(f"Created {len(visualization_files)} interactive visualization files")
        
        # Categorize files for better reporting
        data_files = [f for f in created_files if f.endswith(('.csv', '.json'))]
        viz_files = [f for f in created_files if f.endswith('.html')]
        
        logger.info(f"Created {len(data_files)} analysis files and {len(viz_files)} interactive visualizations")
        
        # Log the visualization files specifically
        if viz_files:
            logger.info("📊 Interactive visualizations created:")
            for viz_file in viz_files:
                viz_name = os.path.basename(viz_file).replace('.html', '').replace('_', ' ').title()
                logger.info(f"  📈 {viz_name}")
        
        # Log the data files
        if data_files:
            logger.info("📄 Analysis data files created:")
            for data_file in data_files:
                file_name = os.path.basename(data_file)
                logger.info(f"  📊 {file_name}")
    except Exception as e:
        logger.warning(f"Could not create optimization visualizations: {e}")
    
    return created_files


def run_backtest(symbol, timeframe, data, strategy_params, portfolio_params, logger, param_source="unknown"):
    """
    Run backtest with enhanced parameter information.
    
    Returns:
        Tuple of (success, results_dict)
    """
    try:
        # Fix parameter types
        strategy_params = fix_parameter_types(strategy_params)
        
        # Initialize strategy
        strategy = DMAATRTrendStrategy(data, strategy_params)
        indicators = strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Create portfolio
        sim_config = SimulationConfig(
            init_cash=portfolio_params.get("init_cash", 100000),
            fees=portfolio_params.get("fees", 0.001),
            slippage=portfolio_params.get("slippage", 0.0005),
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(portfolio, signals=signals)
        
        # Get all metrics
        metrics = {}
        metrics.update(analyzer.get_returns_metrics())
        metrics.update(analyzer.get_trade_metrics())
        metrics.update(analyzer.get_drawdown_metrics())
        metrics.update(analyzer.get_risk_metrics())
        
        # Add metadata and parameter information
        metrics['data_points'] = len(data.get('close'))
        metrics['param_source'] = param_source
        metrics['parameters'] = strategy_params.copy()
        
        # Add parameter summary for display
        param_summary = f"F{strategy_params.get('fast_window', '?')}/S{strategy_params.get('slow_window', '?')}/A{strategy_params.get('atr_window', '?')}"
        metrics['param_summary'] = param_summary
        
        logger.info(f"✅ {symbol} {timeframe}: Return={metrics.get('total_return', 0):.2%}, Sharpe={metrics.get('sharpe_ratio', 0):.2f} ({param_summary})")
        
        return True, {
            'metrics': metrics,
            'portfolio': portfolio,
            'data': data,
            'indicators': indicators,
            'signals': signals,
            'strategy_params': strategy_params,
            'analyzer': analyzer
        }
        
    except Exception as e:
        logger.error(f"Error running backtest for {symbol} {timeframe}: {str(e)}")
        return False, None


def reset_parameters(symbol, timeframe, logger):
    """Reset stored parameters for a symbol/timeframe combination."""
    param_db = OptimalParametersDB()
    
    try:
        param_db.clear_symbol_data(symbol, timeframe)
        logger.info(f"✅ Reset parameters for {symbol} {timeframe}")
        return True
    except Exception as e:
        logger.error(f"Failed to reset parameters for {symbol} {timeframe}: {str(e)}")
        return False


def main():
    """Main strategy testing function."""
    parser = argparse.ArgumentParser(description='Strategy Tester - Consolidated Testing Workflow')
    
    # Input parameters
    parser.add_argument('--symbols', type=str, default='BTC/USDT',
                       help='Comma-separated list of symbols (e.g., BTC/USDT,ETH/USDT)')
    parser.add_argument('--timeframes', type=str, default='4h',
                       help='Comma-separated list of timeframes (e.g., 1h,4h,1d)')
    
    # Date range options (multiple ways to specify)
    parser.add_argument('--period', type=str, choices=list(DEFAULT_DATE_RANGES.keys()),
                       help=f'Use predefined period: {", ".join(DEFAULT_DATE_RANGES.keys())}')
    parser.add_argument('--start-date', type=str,
                       help='Start date for data (YYYY-MM-DD) - overrides period')
    parser.add_argument('--end-date', type=str,
                       help='End date for data (YYYY-MM-DD) - overrides period')
    
    # Processing options
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--force', action='store_true',
                       help='Force re-optimization even if already exists')
    parser.add_argument('--reset-params', action='store_true',
                       help='Reset stored parameters before testing')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results/general/testing',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse input parameters
    symbols = [s.strip() for s in args.symbols.split(',')]
    timeframes = [t.strip() for t in args.timeframes.split(',')]
    
    # Get date range
    start_date, end_date, period_description = get_date_range(args)
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    # Configuration summary
    total_combinations = len(symbols) * len(timeframes)
    logger.section("🧪 Strategy Testing Workflow", 
                   f"Testing {len(symbols)} symbols × {len(timeframes)} timeframes = {total_combinations} combinations")
    
    # Show configuration table
    config_data = [
        {'Setting': 'Symbols', 'Value': ', '.join(symbols)},
        {'Setting': 'Timeframes', 'Value': ', '.join(timeframes)},
        {'Setting': 'Date Range', 'Value': f"{start_date} to {end_date}"},
        {'Setting': 'Period Type', 'Value': period_description},
        {'Setting': 'Optimize Parameters', 'Value': str(args.optimize)},
        {'Setting': 'Force Re-optimization', 'Value': str(args.force)},
        {'Setting': 'Reset Parameters', 'Value': str(args.reset_params)},
        {'Setting': 'Generate Plots', 'Value': str(not args.no_plots)},
        {'Setting': 'Output Directory', 'Value': args.output_dir}
    ]
    logger.table("Configuration", config_data)
    
    # Show available periods for reference
    if not args.start_date or not args.end_date:
        logger.section("📅 Available Date Periods")
        period_data = []
        for period_name, period_config in DEFAULT_DATE_RANGES.items():
            is_current = (period_name == args.period) or (period_name == DEFAULT_PERIOD and not args.period)
            status = "✅ CURRENT" if is_current else ""
            period_data.append({
                'Period': period_name,
                'Start': period_config['start_date'],
                'End': period_config['end_date'],
                'Description': period_config['description'],
                'Status': status
            })
        logger.table("Date Periods", period_data)
    
    # Initialize managers
    config_manager = ConfigManager()
    param_db = OptimalParametersDB()
    
    # Load configuration for parameter ranges
    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
    config = config_manager.load_config(str(config_path))
    
    # Get parameter ranges for optimization
    optimization_ranges = config.get("optimization_ranges", {})
    param_ranges = {
        'fast_window': optimization_ranges.get('short_ma_window', [10, 15, 20, 25, 30]),
        'slow_window': optimization_ranges.get('long_ma_window', [40, 50, 60, 70, 80]),
        'atr_window': optimization_ranges.get('atr_period', [10, 14, 20]),
        'atr_multiplier_sl': optimization_ranges.get('sl_atr_multiplier', [1.5, 2.0, 2.5, 3.0]),
        'atr_multiplier_tp': optimization_ranges.get('tp_atr_multiplier', [3.0, 4.0, 5.0, 6.0])
    }
    
    # Portfolio parameters
    portfolio_params = config_manager.get_portfolio_params()
    
    # Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Date range information for storage
    date_info = {
        'start_date': start_date,
        'end_date': end_date,
        'description': period_description,
        'type': 'predefined' if args.period else 'custom'
    }
    
    # Reset parameters if requested
    if args.reset_params:
        logger.section("🔄 Resetting Parameters")
        for symbol, timeframe in product(symbols, timeframes):
            reset_parameters(symbol, timeframe, logger)
    
    # Process each symbol/timeframe combination
    logger.section("📊 Processing Combinations")
    
    all_results = []
    successful_tests = 0
    failed_tests = 0
    
    for symbol, timeframe in product(symbols, timeframes):
        logger.info(f"\n--- Processing {symbol} {timeframe} ---")
        
        # Load data
        with logger.operation(f"Loading data for {symbol} {timeframe}"):
            data = fetch_data(
                symbols=[symbol],
                exchange="binance",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None:
                logger.error(f"Failed to load data for {symbol} {timeframe}")
                failed_tests += 1
                continue
            
            data_points = len(data.get('close'))
            logger.success(f"Loaded {data_points} data points")
        
        # Get or optimize parameters
        strategy_params = None
        param_source = "default"
        
        if args.optimize:
            # Run optimization
            with logger.operation(f"Optimizing parameters for {symbol} {timeframe}"):
                success, best_params, performance, optimization_analysis = optimize_parameters(
                    symbol, timeframe, data, param_ranges, logger, args.force, date_info
                )
                
                if not success:
                    logger.error(f"Optimization failed for {symbol} {timeframe}")
                    continue
                
                # Save optimization analysis if new optimization was performed
                if optimization_analysis:
                    with logger.operation(f"Saving optimization analysis for {symbol} {timeframe}"):
                        created_files = save_optimization_analysis(optimization_analysis, symbol, timeframe, output_dir)
                        
                        # Categorize files for better reporting
                        data_files = [f for f in created_files if f.endswith(('.csv', '.json'))]
                        viz_files = [f for f in created_files if f.endswith('.html')]
                        
                        logger.info(f"Created {len(data_files)} analysis files and {len(viz_files)} interactive visualizations")
                        
                        # Log the visualization files specifically
                        if viz_files:
                            logger.info("📊 Interactive visualizations created:")
                            for viz_file in viz_files:
                                viz_name = os.path.basename(viz_file).replace('.html', '').replace('_', ' ').title()
                                logger.info(f"  📈 {viz_name}")
                        
                        # Log the data files
                        if data_files:
                            logger.info("📄 Analysis data files created:")
                            for data_file in data_files:
                                file_name = os.path.basename(data_file)
                                logger.info(f"  📊 {file_name}")
        
        else:
            # Use existing parameters
            best_params = param_db.get_optimal_params(symbol, timeframe)
            if not best_params:
                logger.warning(f"No optimal parameters found for {symbol} {timeframe}, using defaults")
                best_params = get_default_params()
            else:
                logger.info(f"Using existing parameters for {symbol} {timeframe}")
        
        # Store parameter source for results
        param_source = "optimized" if args.optimize else "database" if best_params != get_default_params() else "default"
        
        # Run backtest
        with logger.operation(f"Running backtest for {symbol} {timeframe}"):
            success, results = run_backtest(
                symbol, timeframe, data, best_params, portfolio_params, logger, param_source
            )
            
            if not success:
                failed_tests += 1
                continue
            
            # Add metadata
            results['symbol'] = symbol
            results['timeframe'] = timeframe
            results['param_source'] = param_source
            results['date_range'] = date_info
            
            all_results.append(results)
            successful_tests += 1
        
        # Create visualizations
        if not args.no_plots:
            with logger.operation(f"Creating visualizations for {symbol} {timeframe}"):
                try:
                    trading_charts = TradingChartsEngine(
                        results['portfolio'], 
                        results['data'], 
                        indicators=results['indicators']
                    )
                    
                    plot_configs = [
                        ("main_chart.html", "Main Trading Chart", 
                         lambda: trading_charts.create_main_chart(
                             title=f"{symbol} {timeframe} - Trading Analysis ({start_date} to {end_date})",
                             show_volume=True,
                             show_signals=True,
                             show_equity=True
                         )),
                        ("strategy_analysis.html", "Strategy Performance Analysis", 
                         lambda: trading_charts.create_main_chart(
                             title=f"{symbol} {timeframe} - Strategy Analysis ({start_date} to {end_date})",
                             show_volume=False,
                             show_signals=True,
                             show_equity=True
                         )),
                    ]
                    
                    plots_created = []
                    symbol_safe = symbol.replace('/', '_')
                    symbol_plots_dir = f"results/symbols/{symbol_safe}/plots"
                    os.makedirs(symbol_plots_dir, exist_ok=True)
                    
                    for filename, plot_name, plot_func in plot_configs:
                        try:
                            fig = plot_func()
                            # Save to symbol-specific plots directory
                            symbol_output_path = os.path.join(symbol_plots_dir, f"{symbol_safe}_{timeframe}_{filename}")
                            fig.write_html(
                                symbol_output_path,
                                config={'displayModeBar': False},
                                include_plotlyjs='cdn'
                            )
                            
                            # Also save to general output directory for comparison
                            general_output_path = os.path.join(output_dir, f"{symbol_safe}_{timeframe}_{filename}")
                            fig.write_html(
                                general_output_path,
                                config={'displayModeBar': False},
                                include_plotlyjs='cdn'
                            )
                            plots_created.append(general_output_path)
                            
                        except Exception as e:
                            logger.warning(f"Failed to create {plot_name}: {str(e)}")
                    
                    results['plots'] = plots_created
                    
                except Exception as e:
                    logger.warning(f"Failed to create visualizations for {symbol} {timeframe}: {str(e)}")
    
    # Generate summary report
    logger.section("📊 Testing Results")
    
    # Summary statistics
    summary_data = [
        {'Metric': 'Total Combinations', 'Value': f"{total_combinations}"},
        {'Metric': 'Successful', 'Value': f"{successful_tests}"},
        {'Metric': 'Failed', 'Value': f"{failed_tests}"},
        {'Metric': 'Success Rate', 'Value': f"{successful_tests/total_combinations*100:.1f}%"},
        {'Metric': 'Date Range', 'Value': f"{start_date} to {end_date}"},
        {'Metric': 'Period Type', 'Value': period_description}
    ]
    logger.table("Summary", summary_data)
    
    if all_results:
        # Create results DataFrame with parameter information
        results_data = []
        for result in all_results:
            metrics = result['metrics']
            params = metrics.get('parameters', {})
            
            results_data.append({
                'symbol': result['symbol'],
                'timeframe': result['timeframe'],
                'param_source': result['param_source'],
                'param_summary': metrics.get('param_summary', 'N/A'),
                'fast_window': params.get('fast_window', 'N/A'),
                'slow_window': params.get('slow_window', 'N/A'),
                'atr_window': params.get('atr_window', 'N/A'),
                'atr_multiplier_sl': params.get('atr_multiplier_sl', 'N/A'),
                'atr_multiplier_tp': params.get('atr_multiplier_tp', 'N/A'),
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_trades': metrics.get('total_trades', 0),
                'avg_trade_duration': metrics.get('avg_trade_duration', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'data_points': metrics.get('data_points', 0),
                'start_date': start_date,
                'end_date': end_date,
                'period_description': period_description
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        results_df.to_csv(os.path.join(output_dir, "testing_results.csv"), index=False)
        
        # Top performers with parameter information
        logger.section("🏆 Top Performers")
        top_performers = results_df.nlargest(min(10, len(results_df)), 'sharpe_ratio')
        
        top_data = []
        for _, row in top_performers.iterrows():
            top_data.append({
                'Symbol': row['symbol'],
                'Timeframe': row['timeframe'],
                'Params': row['param_source'],
                'Config': row['param_summary'],
                'Sharpe': f"{row['sharpe_ratio']:.3f}",
                'Return': f"{row['total_return']:.2%}",
                'Max DD': f"{row['max_drawdown']:.2%}",
                'Win Rate': f"{row['win_rate']:.1%}",
                'Trades': str(int(row['total_trades']))
            })
        
        logger.table("Best Performing Combinations", top_data)
        
        # Parameter source analysis
        param_source_summary = results_df.groupby('param_source').agg({
            'sharpe_ratio': ['count', 'mean'],
            'total_return': 'mean'
        }).round(3)
        
        logger.section("📈 Parameter Source Analysis")
        source_data = []
        for source in param_source_summary.index:
            count = param_source_summary.loc[source, ('sharpe_ratio', 'count')]
            avg_sharpe = param_source_summary.loc[source, ('sharpe_ratio', 'mean')]
            avg_return = param_source_summary.loc[source, ('total_return', 'mean')]
            
            source_data.append({
                'Parameter Source': source.title(),
                'Count': str(int(count)),
                'Avg Sharpe': f"{avg_sharpe:.3f}",
                'Avg Return': f"{avg_return:.2%}"
            })
        
        logger.table("Parameter Source Performance", source_data)
        
        # Final summary
        logger.section("✅ Testing Complete")
        logger.success(f"Results saved to: {output_dir}")
        logger.info(f"Database location: {param_db.db_path}")
        
        # Show generated files
        files_data = [
            {'File': 'testing_results.csv', 'Description': 'Complete testing results with parameters and date ranges'}
        ]
        
        if not args.no_plots:
            plot_count = sum(len(result.get('plots', [])) for result in all_results)
            if plot_count > 0:
                files_data.append({'File': f'{plot_count} HTML plots', 'Description': 'Trading visualizations'})
        
        logger.table("Generated Files", files_data)
        
    else:
        logger.error("No successful tests - no results to analyze")
    
    # Next steps
    logger.section("🚀 Next Steps")
    next_steps = [
        "Review testing_results.csv for detailed performance and parameter analysis",
        "Date ranges are stored in database for walk-forward analysis reference",
        "Use different periods (--period) to test strategy across different market conditions",
        "Compare results across different date ranges for robustness analysis",
        "Parameters and date ranges are ready for walk-forward analysis implementation"
    ]
    
    for i, step in enumerate(next_steps, 1):
        logger.info(f"{i}. {step}")


if __name__ == "__main__":
    main() 