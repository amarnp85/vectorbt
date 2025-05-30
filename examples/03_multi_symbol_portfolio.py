#!/usr/bin/env python3
"""
Example 03: Multi-Symbol Portfolio Backtest

This example demonstrates portfolio backtesting with multiple assets:
- Loading data for multiple symbols simultaneously
- Running strategy on a portfolio of assets
- Analyzing portfolio-level performance
- Understanding diversification benefits

This builds on Examples 01-02 by expanding to multiple assets.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import vectorbtpro as vbt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules directly
from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.portfolio import PortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis import PerformanceAnalyzer, TradingChartsEngine
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.utilities.structured_logging import setup_logging, get_logger


def main():
    """Run multi-symbol portfolio backtest example."""
    # Setup structured logging
    logger = setup_logging("INFO")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.section("💼 Multi-Symbol Portfolio Backtest", 
                   f"Symbols: {', '.join(symbols)} | Timeframe: {timeframe} | Period: {start_date} to {end_date}")
    
    # Initialize database manager
    param_db = OptimalParametersDB()
    
    # Check which symbols have optimal parameters
    with logger.operation("Checking optimal parameters"):
        optimal_params_available = {}
        for symbol in symbols:
            params = param_db.get_optimization_summary(symbol, timeframe)
            if params:
                optimal_params_available[symbol] = params
                logger.info(f"✅ {symbol}: Optimized (Sharpe: {params['optimization_metric']:.3f})")
            else:
                optimal_params_available[symbol] = None
                logger.info(f"❌ {symbol}: Using default parameters")
    
    # Load data
    with logger.operation("Loading market data"):
        data = fetch_data(
            symbols=symbols,
            exchange="binance",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None:
            logger.error("Failed to load data")
            return
        
        # Log data summary
        data_points = len(data.get('close'))
        logger.data_summary(symbols, timeframe, start_date, end_date, data_points)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
    config_manager = ConfigManager()
    config = config_manager.load_config(str(config_path))
    
    # Strategy selection approach
    # For multi-symbol portfolios, we can:
    # 1. Use the same parameters for all symbols (consensus approach)
    # 2. Use optimal parameters for each symbol (if available)
    # This example uses approach #1 for simplicity
    
    primary_symbol = symbols[0]
    optimal_params = optimal_params_available.get(primary_symbol)
    
    if optimal_params:
        logger.info(f"Using optimal parameters from {primary_symbol}")
        strategy_params = optimal_params['parameters']
    else:
        logger.info("Using default parameters from config")
        strategy_params = config.get("default_parameters", {})
    
    # Run backtest with portfolio approach
    with logger.operation("Running portfolio backtest"):
        # Create simulation config for portfolio
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            position_size_mode=PositionSizeMode.PERCENT_EQUITY,
            position_size_value=0.25  # 25% of equity per position (4 symbols = 100%)
        )
        
        # For multi-symbol portfolios, we need to generate signals per symbol
        all_signals = {}
        all_indicators = {}
        
        for symbol in symbols:
            # Get single symbol data
            if hasattr(data, 'select'):
                symbol_data = data.select(symbol)
            else:
                # If data doesn't have select method, extract manually
                symbol_data = vbt.Data.from_data({
                    'close': data.get('close')[symbol] if isinstance(data.get('close'), pd.DataFrame) else data.get('close'),
                    'high': data.get('high')[symbol] if isinstance(data.get('high'), pd.DataFrame) else data.get('high'),
                    'low': data.get('low')[symbol] if isinstance(data.get('low'), pd.DataFrame) else data.get('low'),
                    'open': data.get('open')[symbol] if isinstance(data.get('open'), pd.DataFrame) else data.get('open'),
                    'volume': data.get('volume')[symbol] if isinstance(data.get('volume'), pd.DataFrame) else data.get('volume')
                })
            
            # Use optimal parameters if available for this symbol
            symbol_optimal = optimal_params_available.get(symbol)
            if symbol_optimal:
                symbol_params = symbol_optimal['parameters']
            else:
                symbol_params = strategy_params
            
            # Initialize strategy for this symbol
            strategy = DMAATRTrendStrategy(symbol_data, symbol_params)
            
            # Calculate indicators and generate signals
            indicators = strategy.init_indicators()
            signals = strategy.generate_signals()
            
            all_indicators[symbol] = indicators
            all_signals[symbol] = signals
        
        # Combine signals from all symbols for portfolio simulation
        # For now, we'll use a simple approach of concatenating signals
        # Note: This is a simplified approach - in practice, you'd want more sophisticated signal combination
        
        # Use the first symbol's signals as a template and combine others
        combined_signals = all_signals[symbols[0]].copy()
        combined_indicators = all_indicators[symbols[0]].copy()
        
        # Run simulation with the first symbol's signals for demonstration
        # In a real multi-symbol strategy, you'd want to properly combine signals
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(combined_signals)
        
        if portfolio is None:
            logger.error("Portfolio simulation failed")
            return
    
    # Analyze results
    with logger.operation("Analyzing portfolio performance"):
        analyzer = PerformanceAnalyzer(portfolio)
        
        # Portfolio-level metrics
        returns_metrics = analyzer.get_returns_metrics()
        trade_metrics = analyzer.get_trade_metrics()
        
        portfolio_metrics = {
            'total_return': returns_metrics.get('Total Return', 0),
            'sharpe_ratio': returns_metrics.get('Sharpe Ratio', 0),
            'max_drawdown': returns_metrics.get('Max Drawdown', 0),
            'volatility': returns_metrics.get('Volatility', 0),
            'win_rate': trade_metrics.get('Win Rate', 0),
            'profit_factor': trade_metrics.get('Profit Factor', 0),
            'total_trades': trade_metrics.get('Total Trades', 0)
        }
        
        # Log portfolio results
        logger.backtest_result({
            'portfolio': portfolio,
            'metrics': portfolio_metrics,
            'symbols': symbols,
            'timeframe': timeframe
        })
    
    # Individual symbol analysis
    logger.section("📊 Individual Symbol Performance")
    
    individual_results = {}
    for symbol in symbols:
        try:
            # Get symbol-specific data
            symbol_data = data.select(symbol) if hasattr(data, 'select') else data
            
            # Use optimal parameters if available for this symbol
            symbol_optimal = optimal_params_available.get(symbol)
            if symbol_optimal:
                symbol_params = symbol_optimal['parameters']
            else:
                symbol_params = strategy_params
            
            # Run individual backtest
            symbol_strategy = DMAATRTrendStrategy(symbol_data, symbol_params)
            symbol_indicators = symbol_strategy.init_indicators()
            symbol_signals = symbol_strategy.generate_signals()
            
            # Create individual simulator
            symbol_simulator = PortfolioSimulator(symbol_data, sim_config)
            symbol_portfolio = symbol_simulator.simulate_from_signals(symbol_signals)
            
            if symbol_portfolio:
                symbol_analyzer = PerformanceAnalyzer(symbol_portfolio)
                symbol_returns = symbol_analyzer.get_returns_metrics()
                symbol_trades = symbol_analyzer.get_trade_metrics()
                
                individual_results[symbol] = {
                    'total_return': symbol_returns.get('Total Return', 0),
                    'sharpe_ratio': symbol_returns.get('Sharpe Ratio', 0),
                    'max_drawdown': symbol_returns.get('Max Drawdown', 0),
                    'total_trades': symbol_trades.get('Total Trades', 0)
                }
                
                logger.info(f"{symbol}: Return {individual_results[symbol]['total_return']:.2%}, "
                           f"Sharpe {individual_results[symbol]['sharpe_ratio']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {str(e)}")
            individual_results[symbol] = None
    
    # Generate visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating portfolio visualizations"):
            try:
                # Create output directory
                output_dir = Path("results/example_03_multi_symbol_portfolio")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create trading charts engine
                charts_engine = TradingChartsEngine(portfolio, data, combined_indicators, combined_signals)
                
                # Portfolio performance chart
                main_chart = charts_engine.create_main_chart(
                    title=f"Multi-Symbol Portfolio Performance ({', '.join(symbols)})"
                )
                charts_engine.save_chart(main_chart, output_dir / "portfolio_performance.html")
                
                # Strategy analysis chart
                analysis_chart = charts_engine.create_strategy_analysis_chart(
                    title="Portfolio Strategy Analysis"
                )
                charts_engine.save_chart(analysis_chart, output_dir / "strategy_analysis.html")
                
                logger.success(f"Visualizations saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate plots: {str(e)}")
    
    # Summary
    logger.section("📈 Portfolio Summary")
    logger.info(f"Portfolio Return: {portfolio_metrics['total_return']:.2%}")
    logger.info(f"Portfolio Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
    logger.info(f"Total Trades: {portfolio_metrics['total_trades']}")
    
    # Diversification benefit analysis
    avg_individual_return = np.mean([r['total_return'] for r in individual_results.values() if r])
    avg_individual_sharpe = np.mean([r['sharpe_ratio'] for r in individual_results.values() if r])
    
    logger.info(f"\nDiversification Analysis:")
    logger.info(f"Average Individual Return: {avg_individual_return:.2%}")
    logger.info(f"Portfolio Return: {portfolio_metrics['total_return']:.2%}")
    logger.info(f"Average Individual Sharpe: {avg_individual_sharpe:.3f}")
    logger.info(f"Portfolio Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}")
    
    if portfolio_metrics['sharpe_ratio'] > avg_individual_sharpe:
        logger.success("✅ Portfolio shows diversification benefit!")
    else:
        logger.warning("⚠️ Portfolio may be over-diversified or poorly balanced")


if __name__ == "__main__":
    main() 