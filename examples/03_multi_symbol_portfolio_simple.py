#!/usr/bin/env python3
"""
Example 03: Multi-Symbol Portfolio Backtest (SIMPLE VERSION)

This is a SIMPLIFIED version that demonstrates the correct VectorBT Pro approach
to multi-symbol portfolios without the complex cross-symbol analysis.

Key Concepts Demonstrated:
1. Multi-symbol vbt.Data object
2. Signals as DataFrames with multiple columns (one per symbol)
3. MultiAssetPortfolioSimulator with cash_sharing=True
4. Proper portfolio-level diversification
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbtpro as vbt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.portfolio import MultiAssetPortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis import PerformanceAnalyzer, TradingChartsEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def create_simple_ma_crossover_signals(data: vbt.Data, fast_period=20, slow_period=50):
    """
    Create simple moving average crossover signals for all symbols.
    
    This demonstrates the CORRECT way to generate multi-symbol signals:
    - Use VBT's broadcasting to calculate indicators for all symbols at once
    - Return signals as DataFrames with columns matching data.symbols
    """
    close = data.close  # This is a DataFrame with symbols as columns
    
    # Calculate moving averages for ALL symbols at once (VBT broadcasting)
    fast_ma_raw = vbt.MA.run(close, window=fast_period).ma
    slow_ma_raw = vbt.MA.run(close, window=slow_period).ma
    
    # Extract values and reset column names to match original symbols
    # VBT creates MultiIndex columns like (window, symbol), we want just symbol
    fast_ma = pd.DataFrame(fast_ma_raw.values, index=fast_ma_raw.index, columns=close.columns)
    slow_ma = pd.DataFrame(slow_ma_raw.values, index=slow_ma_raw.index, columns=close.columns)
    
    # Generate crossover signals for ALL symbols at once
    long_entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    long_exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    # For simplicity, no short signals in this example
    short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
    short_exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    
    # Calculate ATR-based stop levels
    atr_raw = vbt.ATR.run(data.high, data.low, data.close, window=14).atr
    # Fix ATR column alignment too
    atr = pd.DataFrame(atr_raw.values, index=atr_raw.index, columns=close.columns)
    
    # Stop levels as percentages
    sl_levels = (atr / close) * 2.0  # 2x ATR stop loss
    tp_levels = (atr / close) * 3.0  # 3x ATR take profit
    
    signals = {
        'long_entries': long_entries,
        'long_exits': long_exits,
        'short_entries': short_entries,
        'short_exits': short_exits,
        'sl_levels': sl_levels,
        'tp_levels': tp_levels
    }
    
    return signals


def main():
    """Run simplified multi-symbol portfolio backtest."""
    logger = setup_logging("INFO")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.section("💼 Multi-Symbol Portfolio Backtest (SIMPLE)", 
                   f"Symbols: {', '.join(symbols)} | Timeframe: {timeframe}")
    
    # Load multi-symbol data
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
        
        # Verify multi-symbol structure
        logger.info(f"Data shape: {data.close.shape}")
        logger.info(f"Symbols: {list(data.close.columns)}")
        logger.info(f"Date range: {data.close.index[0]} to {data.close.index[-1]}")
    
    # Generate signals for all symbols
    with logger.operation("Generating signals"):
        signals = create_simple_ma_crossover_signals(data, fast_period=20, slow_period=50)
        
        # Log signal summary
        for signal_type, signal_df in signals.items():
            if isinstance(signal_df, pd.DataFrame):
                total_signals = signal_df.sum().sum()
                logger.info(f"{signal_type}: {total_signals} total signals across {len(signal_df.columns)} symbols")
    
    # Configure portfolio simulation
    sim_config = SimulationConfig(
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        position_size_mode=PositionSizeMode.PERCENT_EQUITY,
        position_size_value=0.25,  # 25% per position
        cash_sharing=True,  # CRITICAL: Share cash across symbols
        max_leverage=1.0
    )
    
    # Run multi-symbol portfolio simulation
    with logger.operation("Running portfolio simulation"):
        simulator = MultiAssetPortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        if portfolio is None:
            logger.error("Portfolio simulation failed")
            return
        
        logger.info("Portfolio simulation completed successfully")
    
    # Analyze results
    with logger.operation("Analyzing performance"):
        analyzer = PerformanceAnalyzer(portfolio)
        
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
        
        logger.backtest_result({
            'portfolio': portfolio,
            'metrics': portfolio_metrics,
            'symbols': symbols,
            'timeframe': timeframe
        })
    
    # Show individual symbol contributions
    logger.section("📊 Symbol Analysis")
    
    try:
        # Get trade information
        trades = portfolio.trades.records_readable
        
        if len(trades) > 0:
            symbol_stats = trades.groupby('Column').agg({
                'PnL': ['count', 'sum', 'mean'],
                'Return': 'mean'
            }).round(4)
            
            logger.info("Symbol-wise trade statistics:")
            for symbol in symbol_stats.index:
                count = symbol_stats.loc[symbol, ('PnL', 'count')]
                total_pnl = symbol_stats.loc[symbol, ('PnL', 'sum')]
                avg_return = symbol_stats.loc[symbol, ('Return', 'mean')]
                logger.info(f"{symbol}: {count} trades, PnL: {total_pnl:.2f}, Avg Return: {avg_return:.2%}")
        else:
            logger.warning("No trades were executed")
            
    except Exception as e:
        logger.warning(f"Could not analyze symbol contributions: {e}")
    
    # Compare with buy-and-hold
    logger.section("📈 Benchmark Comparison")
    
    try:
        # Calculate equal-weight buy and hold
        initial_price = data.close.iloc[0]
        final_price = data.close.iloc[-1]
        equal_weight_return = ((final_price / initial_price).mean() - 1)
        
        logger.info(f"Strategy Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"Equal-Weight Buy & Hold: {equal_weight_return:.2%}")
        
        if portfolio_metrics['total_return'] > equal_weight_return:
            outperformance = portfolio_metrics['total_return'] - equal_weight_return
            logger.success(f"✅ Strategy outperforms by {outperformance:.2%}")
        else:
            underperformance = equal_weight_return - portfolio_metrics['total_return']
            logger.warning(f"⚠️ Strategy underperforms by {underperformance:.2%}")
            
    except Exception as e:
        logger.warning(f"Could not calculate benchmark comparison: {e}")
    
    # Generate visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating visualizations"):
            try:
                output_dir = Path("results/example_03_multi_symbol_simple")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create charts with the signals we generated  
                fast_ma_raw = vbt.MA.run(data.close, window=20).ma
                slow_ma_raw = vbt.MA.run(data.close, window=50).ma
                atr_raw = vbt.ATR.run(data.high, data.low, data.close, window=14).atr
                
                indicators = {
                    'fast_ma': pd.DataFrame(fast_ma_raw.values, index=fast_ma_raw.index, columns=data.close.columns),
                    'slow_ma': pd.DataFrame(slow_ma_raw.values, index=slow_ma_raw.index, columns=data.close.columns),
                    'atr': pd.DataFrame(atr_raw.values, index=atr_raw.index, columns=data.close.columns)
                }
                
                charts_engine = TradingChartsEngine(portfolio, data, indicators, signals)
                
                main_chart = charts_engine.create_main_chart(
                    title=f"Simple Multi-Symbol Portfolio ({', '.join(symbols)})"
                )
                charts_engine.save_chart(main_chart, output_dir / "portfolio_performance.html")
                
                logger.success(f"Charts saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate plots: {str(e)}")
    
    # Summary
    logger.section("💡 Key Insights")
    
    logger.info("This example demonstrates:")
    logger.info("• Proper multi-symbol data handling with vbt.Data")
    logger.info("• Signal generation using VBT broadcasting across symbols")
    logger.info("• Portfolio simulation with cash sharing")
    logger.info("• Diversification benefits from multi-symbol approach")
    
    if portfolio_metrics['total_trades'] > 0:
        logger.success("✅ Multi-symbol portfolio executed successfully!")
        logger.info(f"Portfolio Performance:")
        logger.info(f"  Return: {portfolio_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe: {portfolio_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Trades: {portfolio_metrics['total_trades']}")
    else:
        logger.warning("⚠️ No trades executed - signals may be too restrictive")
    
    return portfolio_metrics


if __name__ == "__main__":
    main()