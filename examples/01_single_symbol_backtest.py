#!/usr/bin/env python3
"""
Example 01: Single Symbol Backtest

This example demonstrates the basic usage of the backtesting framework:
- Loading data for a single symbol
- Running a backtest with optimal or default parameters
- Viewing performance metrics
- Generating basic visualizations

This is the starting point for understanding how the framework works.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules directly
from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.trading_charts_refactored import TradingChartsEngine
from backtester.analysis.trading_signals import SignalConfig
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.utilities.structured_logging import setup_logging


def main():
    """Run a single symbol backtest example."""
    # Setup logging
    logger = setup_logging("INFO")
    
    # Configuration
    symbol = "SOL/USDT"
    timeframe = "4h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"  # Full year - SL/TP labels will be visible throughout
    
    logger.section("📊 Single Symbol Backtest Example", 
                   f"Symbol: {symbol} | Timeframe: {timeframe} | Period: {start_date} to {end_date}")
    
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
            return
        
        data_points = len(data.get('close'))
        logger.success(f"Data loaded successfully: {data_points} candles")
    
    # Load configuration and check for optimal parameters
    with logger.operation("Loading configuration"):
        config_manager = ConfigManager()
        config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
        config = config_manager.load_config(str(config_path))
        
        # Check for optimal parameters in database
        param_db = OptimalParametersDB()
        optimal_params = param_db.get_optimal_params(symbol, timeframe)
        
        if optimal_params:
            logger.info(f"Using stored optimal parameters for {symbol}")
            strategy_params = optimal_params
            param_source = "optimal"
        else:
            logger.info(f"Using default parameters from config")
            strategy_params = config_manager.get_strategy_params()
            param_source = "default"
        
        # Portfolio parameters
        portfolio_params = config_manager.get_portfolio_params()
        
        # Log strategy configuration
        logger.strategy_config("DMA ATR Trend Strategy", strategy_params)
    
    # Output directory
    symbol_safe = symbol.replace('/', '_')
    output_dir = f"results/symbols/{symbol_safe}"
    plots_dir = f"{output_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Run backtest
    with logger.operation("Running backtest"):
        # Initialize strategy
        strategy = DMAATRTrendStrategy(data, strategy_params)
        indicators = strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Create portfolio
        # Use configuration-based position sizing if available
        if config and 'portfolio_parameters' in config:
            # Create SimulationConfig from configuration
            sim_config = SimulationConfig.from_config_dict(config)
        else:
            # Fallback to manual configuration (backwards compatibility)
            sim_config = SimulationConfig(
                init_cash=portfolio_params.get("init_cash", 100000),
                fees=portfolio_params.get("fees", 0.001),
                slippage=portfolio_params.get("slippage", 0.0005),
                freq=data.wrapper.freq if hasattr(data, "wrapper") else "D",
                position_size_mode=PositionSizeMode.PERCENT_EQUITY,
                position_size_value=portfolio_params.get("position_size_value", 0.95)
            )
        
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        logger.success("Backtest completed successfully")
    
    # Analyze performance
    with logger.operation("Analyzing performance"):
        analyzer = PerformanceAnalyzer(portfolio, signals=signals)
        
        # Get all metrics
        metrics = {}
        metrics.update(analyzer.get_returns_metrics())
        metrics.update(analyzer.get_trade_metrics())
        metrics.update(analyzer.get_drawdown_metrics())
        metrics.update(analyzer.get_risk_metrics())
        
        # Save metrics
        metrics_file = os.path.join(output_dir, "performance_metrics.csv")
        analyzer.export_results(metrics_file)
        
        # Display results using structured logging
        logger.backtest_result({'portfolio': portfolio})
    
    # Create visualizations
    logger.section("📈 Generating Trading Visualizations")
    
    # Debug: Log SL/TP signal information
    if signals:
        logger.info("SL/TP Signal Analysis:")
        sl_price_count = (~signals.get('sl_price_levels', pd.Series()).isna()).sum() if 'sl_price_levels' in signals else 0
        tp_price_count = (~signals.get('tp_price_levels', pd.Series()).isna()).sum() if 'tp_price_levels' in signals else 0
        sl_pct_count = (~signals.get('sl_levels', pd.Series()).isna()).sum() if 'sl_levels' in signals else 0
        tp_pct_count = (~signals.get('tp_levels', pd.Series()).isna()).sum() if 'tp_levels' in signals else 0
        total_entries = signals.get('long_entries', pd.Series()).sum() + signals.get('short_entries', pd.Series()).sum()
        
        logger.info(f"  - Total entry signals: {total_entries}")
        logger.info(f"  - SL price levels: {sl_price_count}")
        logger.info(f"  - TP price levels: {tp_price_count}")
        logger.info(f"  - SL percentage levels: {sl_pct_count}")
        logger.info(f"  - TP percentage levels: {tp_pct_count}")
        
        if sl_price_count > 0 or tp_price_count > 0:
            logger.success("✅ SL/TP data available - symbols should appear on charts!")
        else:
            logger.warning("⚠️  No SL/TP price data found - check strategy configuration")
    
    # Create enhanced trading charts with stop levels
    # IMPORTANT: Configure signal rendering to show SL/TP symbols
    signal_config = SignalConfig(
        signal_timing_mode="execution",
        show_signals=True,
        show_stop_levels=True     # Enable SL/TP horizontal dash symbols
    )
    
    trading_charts = TradingChartsEngine(
        portfolio=portfolio, 
        data=data, 
        indicators=indicators, 
        signals=signals,
        signal_config=signal_config  # Pass the enhanced signal configuration
    )
    
    plot_configs = [
        ("main_chart.html", "Main Trading Chart", 
         lambda: trading_charts.create_main_chart(
             title=f"{symbol} {timeframe} - Trading Analysis (with SL/TP Symbols)",
             show_volume=True,
             show_signals=True,
             show_equity=True
         )),
        ("simple_chart.html", "Simple Candlestick Chart", 
         lambda: trading_charts.create_simple_candlestick(
             title=f"{symbol} {timeframe} - Price Chart"
         )),
    ]
    
    plots_created = []
    with logger.operation("Creating plots", total_items=len(plot_configs)):
        for i, (filename, plot_name, plot_func) in enumerate(plot_configs):
            try:
                logger.update_progress(i + 1, len(plot_configs), f"Creating {plot_name}")
                fig = plot_func()
                
                # Save plot
                output_path = os.path.join(plots_dir, f"{symbol_safe}_{timeframe}_{filename}")
                fig.write_html(
                    output_path,
                    config={'displayModeBar': False},
                    include_plotlyjs='cdn'
                )
                plots_created.append(filename)
                logger.debug(f"Created {plot_name} successfully")
                
            except Exception as e:
                logger.warning(f"Failed to create {plot_name}: {str(e)}")
    
    # Final summary
    logger.section("✅ Backtest Complete")
    
    if metrics:
        # Create a summary table
        summary_data = [
            {"Metric": "Parameter Source", "Value": param_source.title()},
            {"Metric": "Total Return", "Value": f"{metrics.get('total_return', 0):.2%}"},
            {"Metric": "Sharpe Ratio", "Value": f"{metrics.get('sharpe_ratio', 0):.2f}"},
            {"Metric": "Max Drawdown", "Value": f"{metrics.get('max_drawdown', 0):.2%}"},
            {"Metric": "Win Rate", "Value": f"{metrics.get('win_rate', 0):.2%}"},
            {"Metric": "Total Trades", "Value": f"{metrics.get('total_trades', 0)}"},
            {"Metric": "Avg Trade Duration", "Value": f"{metrics.get('avg_trade_duration', 0):.1f} bars"},
        ]
        logger.table("Performance Summary", summary_data)
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info("\nGenerated files:")
        logger.info("  - performance_metrics.csv: Detailed performance statistics")
        for plot_file in plots_created:
            logger.info(f"  - {plot_file}")
    else:
        logger.error("Backtest failed - no metrics generated")
    
    logger.info("\n" + "=" * 50)
    logger.info("Next steps:")
    logger.info("1. Review the generated HTML files for visual analysis")
    logger.info("2. Check performance_metrics.csv for detailed statistics")
    logger.info("3. Run example 02 to optimize parameters")
    logger.info("4. Use strategy_tester.py for testing multiple symbols/timeframes")


if __name__ == "__main__":
    main() 