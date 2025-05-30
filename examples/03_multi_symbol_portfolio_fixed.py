#!/usr/bin/env python3
"""
Example 03: Multi-Symbol Portfolio Backtest (CORRECTED VERSION)

This example demonstrates PROPER portfolio backtesting with multiple assets:
- Loading data for multiple symbols simultaneously
- Using VectorBT Pro's native multi-symbol capabilities
- Running true multi-symbol strategy with cross-symbol analysis
- Analyzing portfolio-level performance with proper diversification
- Understanding how VectorBT Pro handles multi-symbol portfolios

Key Improvements:
1. Uses MultiSymbolDMAATRStrategy instead of individual single-symbol strategies
2. Uses MultiAssetPortfolioSimulator with cash_sharing=True
3. Generates signals for all symbols simultaneously in a single DataFrame
4. Applies cross-symbol filters (correlation, market regime)
5. Proper portfolio-level position sizing and risk management
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
from backtester.strategies.multi_symbol_dma_atr_strategy import MultiSymbolDMAATRStrategy
from backtester.portfolio import MultiAssetPortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis import PerformanceAnalyzer, TradingChartsEngine
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.utilities.structured_logging import setup_logging, get_logger


def main():
    """Run proper multi-symbol portfolio backtest example."""
    # Setup structured logging
    logger = setup_logging("INFO")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.section("💼 Multi-Symbol Portfolio Backtest (CORRECTED)", 
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
    
    # Load data as a single multi-symbol vbt.Data object
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
        
        # Verify this is proper multi-symbol data
        if not hasattr(data, 'symbols') or len(data.symbols) != len(symbols):
            logger.error("Data is not properly formatted as multi-symbol vbt.Data object")
            return
        
        # Log data summary
        data_points = len(data.get('close'))
        logger.data_summary(symbols, timeframe, start_date, end_date, data_points)
        logger.info(f"Multi-symbol data confirmed: {len(data.symbols)} symbols")
    
    # Strategy parameter selection
    # For multi-symbol portfolios, we can use consensus parameters or symbol-specific parameters
    # This example uses consensus parameters from the best-performing symbol
    
    best_symbol = None
    best_sharpe = -999
    
    for symbol, params in optimal_params_available.items():
        if params and params['optimization_metric'] > best_sharpe:
            best_sharpe = params['optimization_metric']
            best_symbol = symbol
    
    if best_symbol:
        logger.info(f"Using optimal parameters from best symbol: {best_symbol} (Sharpe: {best_sharpe:.3f})")
        strategy_params = optimal_params_available[best_symbol]['parameters']
    else:
        logger.info("Using default parameters from config")
        # Load default parameters
        config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
        config_manager = ConfigManager()
        config = config_manager.load_config(str(config_path))
        strategy_params = config.get("default_parameters", {})
    
    # Add multi-symbol specific parameters
    multi_symbol_params = {
        **strategy_params,
        'correlation_lookback': 60,
        'relative_strength_lookback': 20,
        'use_symbol_ranking': True,
        'max_active_symbols': 3,  # Trade top 3 symbols at any time
        'max_correlation': 0.7,   # Reduce positions if correlation > 70%
        'min_market_breadth': 0.25  # Need 25% of symbols trending up for long signals
    }
    
    # Run multi-symbol portfolio backtest
    with logger.operation("Running multi-symbol portfolio backtest"):
        # Initialize PROPER multi-symbol strategy
        strategy = MultiSymbolDMAATRStrategy(data, multi_symbol_params)
        
        # Validate the strategy
        if not strategy.validate_parameters():
            logger.error("Strategy parameter validation failed")
            return
        
        # Calculate indicators and generate signals for ALL symbols simultaneously
        logger.info("Calculating indicators for all symbols...")
        indicators = strategy.init_indicators()
        
        logger.info("Generating signals with cross-symbol filters...")
        signals = strategy.generate_signals()
        
        # Log signal summary for all symbols
        for signal_type in ['long_entries', 'long_exits', 'short_entries', 'short_exits']:
            if signal_type in signals:
                total_signals = signals[signal_type].sum().sum()
                logger.info(f"{signal_type}: {total_signals} total across all symbols")
        
        # Create simulation config for TRUE multi-symbol portfolio
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=0.001,
            slippage=0.0005,
            position_size_mode=PositionSizeMode.PERCENT_EQUITY,
            position_size_value=0.2,  # 20% per position, allowing up to 5 positions
            cash_sharing=True,  # CRITICAL: Share cash across all symbols
            max_leverage=1.0
        )
        
        # Use MultiAssetPortfolioSimulator for proper multi-symbol simulation
        simulator = MultiAssetPortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        if portfolio is None:
            logger.error("Portfolio simulation failed")
            return
        
        logger.info(f"Portfolio simulation completed with {len(data.symbols)} symbols")
    
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
    
    # Individual symbol contribution analysis
    logger.section("📊 Symbol Contribution Analysis")
    
    # Analyze each symbol's contribution to the portfolio
    symbol_contributions = {}
    
    try:
        # Get portfolio value and positions
        portfolio_value = portfolio.value()
        positions = portfolio.position()
        
        for symbol in symbols:
            if symbol in positions.columns:
                symbol_positions = positions[symbol]
                symbol_trades = symbol_positions.count()
                symbol_pnl = (positions[symbol] * data.close[symbol].pct_change()).sum()
                
                symbol_contributions[symbol] = {
                    'trades': symbol_trades,
                    'pnl_contribution': symbol_pnl,
                    'avg_position': symbol_positions.mean()
                }
                
                logger.info(f"{symbol}: {symbol_trades} trades, "
                           f"PnL contribution: {symbol_pnl:.2f}")
            else:
                symbol_contributions[symbol] = {
                    'trades': 0,
                    'pnl_contribution': 0,
                    'avg_position': 0
                }
                logger.info(f"{symbol}: No trades")
                
    except Exception as e:
        logger.warning(f"Could not analyze symbol contributions: {e}")
    
    # Compare with equal-weight baseline
    logger.section("📈 Diversification Analysis")
    
    try:
        # Calculate equal-weight portfolio returns for comparison
        equal_weight_returns = data.close.pct_change().mean(axis=1)
        equal_weight_total_return = (1 + equal_weight_returns).prod() - 1
        equal_weight_volatility = equal_weight_returns.std() * np.sqrt(252)
        equal_weight_sharpe = equal_weight_returns.mean() * 252 / equal_weight_volatility
        
        logger.info(f"Strategy Portfolio:")
        logger.info(f"  Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Volatility: {portfolio_metrics['volatility']:.2%}")
        
        logger.info(f"Equal-Weight Baseline:")
        logger.info(f"  Return: {equal_weight_total_return:.2%}")
        logger.info(f"  Sharpe: {equal_weight_sharpe:.3f}")
        logger.info(f"  Volatility: {equal_weight_volatility:.2%}")
        
        if portfolio_metrics['sharpe_ratio'] > equal_weight_sharpe:
            improvement = portfolio_metrics['sharpe_ratio'] - equal_weight_sharpe
            logger.success(f"✅ Strategy outperforms equal-weight by {improvement:.3f} Sharpe points!")
        else:
            underperformance = equal_weight_sharpe - portfolio_metrics['sharpe_ratio']
            logger.warning(f"⚠️ Strategy underperforms equal-weight by {underperformance:.3f} Sharpe points")
            
    except Exception as e:
        logger.warning(f"Could not perform diversification analysis: {e}")
    
    # Generate visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating portfolio visualizations"):
            try:
                # Create output directory
                output_dir = Path("results/example_03_multi_symbol_portfolio_fixed")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create trading charts engine with multi-symbol data
                charts_engine = TradingChartsEngine(portfolio, data, indicators, signals)
                
                # Portfolio performance chart
                main_chart = charts_engine.create_main_chart(
                    title=f"Multi-Symbol Portfolio Performance - CORRECTED ({', '.join(symbols)})"
                )
                charts_engine.save_chart(main_chart, output_dir / "portfolio_performance.html")
                
                # Strategy analysis chart
                analysis_chart = charts_engine.create_strategy_analysis_chart(
                    title="Multi-Symbol Portfolio Strategy Analysis"
                )
                charts_engine.save_chart(analysis_chart, output_dir / "strategy_analysis.html")
                
                logger.success(f"Visualizations saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate plots: {str(e)}")
    
    # Strategy insights and recommendations
    logger.section("💡 Strategy Insights")
    
    logger.info("Key Features of this Multi-Symbol Implementation:")
    logger.info("• Uses VectorBT Pro's native multi-symbol capabilities")
    logger.info("• Applies cross-symbol correlation filtering")
    logger.info("• Includes market regime detection")
    logger.info("• Symbol ranking and selection (top 3 most promising)")
    logger.info("• Portfolio-level cash sharing and position sizing")
    logger.info("• Proper diversification benefits")
    
    if portfolio_metrics['total_trades'] > 0:
        logger.success("✅ Multi-symbol portfolio strategy executed successfully!")
        logger.info(f"Final Portfolio Metrics:")
        logger.info(f"  Total Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
        logger.info(f"  Total Trades: {portfolio_metrics['total_trades']}")
    else:
        logger.warning("⚠️ No trades were generated - consider adjusting strategy parameters")
    
    return portfolio_metrics


if __name__ == "__main__":
    main()