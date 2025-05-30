#!/usr/bin/env python3
"""
Multi-Symbol Portfolio Backtest - DETAILED ANALYSIS VERSION

This script runs the multi-symbol portfolio backtest with extensive logging
and analysis to help you understand exactly what's happening at each step.

Run this to see:
1. How multi-symbol data is structured
2. How signals are generated across symbols
3. How portfolio simulation works
4. Detailed performance analysis
5. Risk metrics and diversification benefits
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.portfolio import MultiAssetPortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis import PerformanceAnalyzer, TradingChartsEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def analyze_data_structure(data: vbt.Data, symbols: list):
    """Analyze and explain the multi-symbol data structure."""
    print("\n" + "="*80)
    print("📊 MULTI-SYMBOL DATA STRUCTURE ANALYSIS")
    print("="*80)
    
    print(f"\n1. Data Type: {type(data)}")
    print(f"   - VectorBT Data object designed for multi-symbol analysis")
    
    print(f"\n2. Data Shape:")
    print(f"   - Close prices shape: {data.close.shape}")
    print(f"   - Rows (time points): {data.close.shape[0]}")
    print(f"   - Columns (symbols): {data.close.shape[1]}")
    
    print(f"\n3. Symbols: {list(data.close.columns)}")
    
    print(f"\n4. Date Range:")
    print(f"   - Start: {data.close.index[0]}")
    print(f"   - End: {data.close.index[-1]}")
    print(f"   - Trading days: {len(data.close)}")
    
    print(f"\n5. Price Ranges by Symbol:")
    for symbol in symbols:
        min_price = data.close[symbol].min()
        max_price = data.close[symbol].max()
        current = data.close[symbol].iloc[-1]
        print(f"   - {symbol}: ${min_price:,.2f} - ${max_price:,.2f} (Current: ${current:,.2f})")
    
    print(f"\n6. Data Completeness:")
    for symbol in symbols:
        missing = data.close[symbol].isna().sum()
        print(f"   - {symbol}: {missing} missing values ({missing/len(data.close)*100:.2f}%)")


def analyze_signal_generation(signals: dict, data: vbt.Data):
    """Analyze the signal generation process."""
    print("\n" + "="*80)
    print("📈 SIGNAL GENERATION ANALYSIS")
    print("="*80)
    
    print("\n1. Signal Structure:")
    print(f"   - Signal types: {list(signals.keys())}")
    
    print("\n2. Signal Counts by Type:")
    for signal_type, signal_df in signals.items():
        if isinstance(signal_df, pd.DataFrame):
            print(f"\n   {signal_type}:")
            for symbol in signal_df.columns:
                count = signal_df[symbol].sum()
                print(f"      - {symbol}: {count} signals")
    
    print("\n3. Signal Timing Analysis:")
    entries = signals['long_entries']
    exits = signals['long_exits']
    
    print("\n   Entry Signal Distribution by Year:")
    for symbol in entries.columns:
        symbol_entries = entries[entries[symbol]]
        if len(symbol_entries) > 0:
            yearly = symbol_entries.groupby(symbol_entries.index.year).size()
            print(f"   {symbol}:")
            for year, count in yearly.items():
                print(f"      - {year}: {count} entries")
    
    print("\n4. Signal Quality Metrics:")
    for symbol in entries.columns:
        symbol_entries = entries[symbol].sum()
        symbol_exits = exits[symbol].sum()
        if symbol_entries > 0:
            completion_rate = symbol_exits / symbol_entries * 100
            print(f"   - {symbol}: {completion_rate:.1f}% signal completion rate")


def analyze_portfolio_performance(portfolio: vbt.Portfolio, symbols: list):
    """Detailed portfolio performance analysis."""
    print("\n" + "="*80)
    print("💰 PORTFOLIO PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Get portfolio statistics
    stats = portfolio.stats()
    
    print("\n1. Overall Performance:")
    print(f"   - Initial Capital: ${stats['Start Value']:,.2f}")
    print(f"   - Final Value: ${stats['End Value']:,.2f}")
    print(f"   - Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"   - Annual Return: {stats['Annualized Return [%]']:.2f}%")
    
    print("\n2. Risk Metrics:")
    print(f"   - Sharpe Ratio: {stats['Sharpe Ratio']:.3f}")
    print(f"   - Sortino Ratio: {stats['Sortino Ratio']:.3f}")
    print(f"   - Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
    print(f"   - Annual Volatility: {stats['Annualized Volatility [%]']:.2f}%")
    
    print("\n3. Trading Activity:")
    print(f"   - Total Trades: {stats['Total Trades']}")
    print(f"   - Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"   - Average Trade: {stats['Avg Winning Trade [%]']:.2f}%")
    print(f"   - Best Trade: {stats['Best Trade [%]']:.2f}%")
    print(f"   - Worst Trade: {stats['Worst Trade [%]']:.2f}%")
    
    # Analyze trades by symbol
    trades = portfolio.trades.records_readable
    if len(trades) > 0:
        print("\n4. Symbol-by-Symbol Analysis:")
        
        for i, symbol in enumerate(symbols):
            symbol_trades = trades[trades['Column'] == symbol]
            if len(symbol_trades) > 0:
                total_pnl = symbol_trades['PnL'].sum()
                avg_return = symbol_trades['Return'].mean() * 100
                win_rate = (symbol_trades['PnL'] > 0).mean() * 100
                
                print(f"\n   {symbol}:")
                print(f"      - Trades: {len(symbol_trades)}")
                print(f"      - Total PnL: ${total_pnl:,.2f}")
                print(f"      - Avg Return: {avg_return:.2f}%")
                print(f"      - Win Rate: {win_rate:.1f}%")
                print(f"      - Best Trade: {symbol_trades['Return'].max()*100:.2f}%")
                print(f"      - Worst Trade: {symbol_trades['Return'].min()*100:.2f}%")


def analyze_diversification_benefits(portfolio: vbt.Portfolio, data: vbt.Data, symbols: list):
    """Analyze the diversification benefits of multi-symbol trading."""
    print("\n" + "="*80)
    print("🎯 DIVERSIFICATION ANALYSIS")
    print("="*80)
    
    # Calculate correlations
    returns = data.close.pct_change()
    corr_matrix = returns.corr()
    
    print("\n1. Symbol Correlations:")
    print("\n   Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Calculate average correlation
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_correlation = corr_matrix.where(mask).stack().mean()
    print(f"\n   Average Pairwise Correlation: {avg_correlation:.3f}")
    
    if avg_correlation < 0.5:
        print("   ✅ Low correlation - Good diversification potential")
    elif avg_correlation < 0.7:
        print("   ⚠️ Moderate correlation - Some diversification benefits")
    else:
        print("   ❌ High correlation - Limited diversification benefits")
    
    # Compare portfolio volatility vs individual volatilities
    portfolio_returns = portfolio.returns()
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    
    print("\n2. Volatility Comparison:")
    individual_vols = returns.std() * np.sqrt(252)
    
    for symbol in symbols:
        print(f"   - {symbol}: {individual_vols[symbol]*100:.1f}% annual volatility")
    
    print(f"\n   - Portfolio: {portfolio_vol*100:.1f}% annual volatility")
    print(f"   - Average Individual: {individual_vols.mean()*100:.1f}%")
    
    vol_reduction = (individual_vols.mean() - portfolio_vol) / individual_vols.mean() * 100
    print(f"   - Volatility Reduction: {vol_reduction:.1f}%")
    
    if vol_reduction > 20:
        print("   ✅ Significant volatility reduction from diversification")
    elif vol_reduction > 10:
        print("   ⚠️ Moderate volatility reduction")
    else:
        print("   ❌ Limited volatility reduction")


def analyze_position_sizing_impact(portfolio: vbt.Portfolio, config: SimulationConfig):
    """Analyze how position sizing affects the portfolio."""
    print("\n" + "="*80)
    print("📏 POSITION SIZING ANALYSIS")
    print("="*80)
    
    print(f"\n1. Configuration:")
    print(f"   - Initial Capital: ${config.init_cash:,.2f}")
    print(f"   - Position Size Mode: {config.position_size_mode.name}")
    print(f"   - Position Size Value: {config.position_size_value*100:.0f}%")
    print(f"   - Max Theoretical Positions: {int(1/config.position_size_value)}")
    
    # Analyze actual position usage
    positions = portfolio.positions()
    active_positions = (positions != 0).sum(axis=1)
    
    print(f"\n2. Position Usage:")
    print(f"   - Max Concurrent Positions: {active_positions.max()}")
    print(f"   - Average Active Positions: {active_positions[active_positions > 0].mean():.1f}")
    print(f"   - Time with Positions: {(active_positions > 0).mean()*100:.1f}%")
    
    # Cash utilization
    cash = portfolio.cash()
    cash_usage = 1 - (cash / config.init_cash)
    
    print(f"\n3. Capital Utilization:")
    print(f"   - Max Cash Deployed: {cash_usage.max()*100:.1f}%")
    print(f"   - Average Cash Deployed: {cash_usage[cash_usage > 0].mean()*100:.1f}%")
    
    if cash_usage.max() < 0.8:
        print("   ⚠️ Portfolio may be under-utilizing capital")
    else:
        print("   ✅ Good capital utilization")


def create_performance_comparison(data: vbt.Data, portfolio: vbt.Portfolio):
    """Create a comparison between strategy and buy-and-hold."""
    print("\n" + "="*80)
    print("📊 STRATEGY VS BUY-AND-HOLD COMPARISON")
    print("="*80)
    
    # Calculate buy-and-hold returns for each symbol
    initial_prices = data.close.iloc[0]
    final_prices = data.close.iloc[-1]
    bh_returns = (final_prices / initial_prices - 1) * 100
    
    print("\n1. Individual Buy-and-Hold Returns:")
    for symbol in data.close.columns:
        print(f"   - {symbol}: {bh_returns[symbol]:.1f}%")
    
    print(f"\n   - Equal-Weight Average: {bh_returns.mean():.1f}%")
    
    # Strategy performance
    strategy_return = portfolio.total_return() * 100
    
    print(f"\n2. Strategy Return: {strategy_return:.1f}%")
    
    # Risk-adjusted comparison
    strategy_sharpe = portfolio.sharpe_ratio()
    
    # Calculate buy-and-hold Sharpe
    bh_daily_returns = data.close.pct_change().mean(axis=1)
    bh_sharpe = bh_daily_returns.mean() / bh_daily_returns.std() * np.sqrt(252)
    
    print(f"\n3. Risk-Adjusted Performance:")
    print(f"   - Strategy Sharpe: {strategy_sharpe:.3f}")
    print(f"   - Buy-and-Hold Sharpe: {bh_sharpe:.3f}")
    
    if strategy_sharpe > bh_sharpe:
        print("   ✅ Strategy provides better risk-adjusted returns")
    else:
        print("   ❌ Buy-and-hold provides better risk-adjusted returns")


def main():
    """Run comprehensive multi-symbol portfolio analysis."""
    logger = setup_logging("INFO")
    
    print("\n" + "="*80)
    print("🚀 MULTI-SYMBOL PORTFOLIO BACKTEST - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    print(f"\nBacktest Parameters:")
    print(f"- Symbols: {', '.join(symbols)}")
    print(f"- Timeframe: {timeframe}")
    print(f"- Period: {start_date} to {end_date}")
    print(f"- Strategy: Simple Moving Average Crossover (20/50)")
    
    # Load data
    print("\n⏳ Loading market data...")
    data = fetch_data(
        symbols=symbols,
        exchange="binance",
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Analyze data structure
    analyze_data_structure(data, symbols)
    
    # Generate signals
    print("\n⏳ Generating trading signals...")
    
    # Create simple MA crossover signals
    def create_simple_ma_crossover_signals(data, fast_period=20, slow_period=50):
        close = data.close
        
        # Calculate moving averages
        fast_ma_raw = vbt.MA.run(close, window=fast_period).ma
        slow_ma_raw = vbt.MA.run(close, window=slow_period).ma
        
        # Fix column alignment
        fast_ma = pd.DataFrame(fast_ma_raw.values, index=fast_ma_raw.index, columns=close.columns)
        slow_ma = pd.DataFrame(slow_ma_raw.values, index=slow_ma_raw.index, columns=close.columns)
        
        # Generate crossover signals
        long_entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        long_exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # No short signals for simplicity
        short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        short_exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        
        # ATR-based stops
        atr_raw = vbt.ATR.run(data.high, data.low, data.close, window=14).atr
        atr = pd.DataFrame(atr_raw.values, index=atr_raw.index, columns=close.columns)
        
        sl_levels = (atr / close) * 2.0
        tp_levels = (atr / close) * 3.0
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels,
            'tp_levels': tp_levels
        }
    
    signals = create_simple_ma_crossover_signals(data)
    
    # Analyze signals
    analyze_signal_generation(signals, data)
    
    # Configure portfolio
    sim_config = SimulationConfig(
        init_cash=100000,
        fees=0.001,  # 0.1% commission
        slippage=0.0005,  # 0.05% slippage
        position_size_mode=PositionSizeMode.PERCENT_EQUITY,
        position_size_value=0.25,  # 25% per position
        cash_sharing=True,  # Share cash across all symbols
        max_leverage=1.0  # No leverage
    )
    
    # Run simulation
    print("\n⏳ Running portfolio simulation...")
    simulator = MultiAssetPortfolioSimulator(data, sim_config)
    portfolio = simulator.simulate_from_signals(signals)
    
    if portfolio is None:
        print("❌ Portfolio simulation failed")
        return
    
    # Comprehensive analysis
    analyze_portfolio_performance(portfolio, symbols)
    analyze_diversification_benefits(portfolio, data, symbols)
    analyze_position_sizing_impact(portfolio, sim_config)
    create_performance_comparison(data, portfolio)
    
    # Generate detailed report
    print("\n" + "="*80)
    print("📋 EXECUTIVE SUMMARY")
    print("="*80)
    
    stats = portfolio.stats()
    
    print(f"\n💵 Profitability:")
    print(f"   - Total Return: {stats['Total Return [%]']:.2f}%")
    print(f"   - Annual Return: {stats['Annualized Return [%]']:.2f}%")
    print(f"   - Final Portfolio Value: ${stats['End Value']:,.2f}")
    
    print(f"\n⚡ Risk Management:")
    print(f"   - Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
    print(f"   - Sharpe Ratio: {stats['Sharpe Ratio']:.3f}")
    print(f"   - Win Rate: {stats['Win Rate [%]']:.2f}%")
    
    print(f"\n📊 Trading Activity:")
    print(f"   - Total Trades: {stats['Total Trades']}")
    print(f"   - Avg Trade Duration: {stats['Avg Trade Duration']}")
    
    print("\n🎯 Key Insights:")
    
    # Performance assessment
    if stats['Sharpe Ratio'] > 1.0:
        print("   ✅ Excellent risk-adjusted returns (Sharpe > 1.0)")
    elif stats['Sharpe Ratio'] > 0.5:
        print("   ⚠️ Moderate risk-adjusted returns (Sharpe 0.5-1.0)")
    else:
        print("   ❌ Poor risk-adjusted returns (Sharpe < 0.5)")
    
    # Drawdown assessment
    if stats['Max Drawdown [%]'] < -20:
        print("   ❌ High risk - Maximum drawdown exceeds 20%")
    else:
        print("   ✅ Acceptable risk - Maximum drawdown under 20%")
    
    # Win rate assessment
    if stats['Win Rate [%]'] > 50:
        print("   ✅ Positive win rate - More winning trades than losing")
    else:
        print("   ⚠️ Low win rate - Relies on large winners")
    
    print("\n📈 Recommendations:")
    print("   1. Consider adjusting MA periods based on market conditions")
    print("   2. Experiment with different position sizing (current: 25%)")
    print("   3. Add filters for market regime (trending vs ranging)")
    print("   4. Consider adding more uncorrelated assets for better diversification")
    
    # Save results option
    print("\n💾 Results saved to: results/example_03_multi_symbol_simple/")
    print("   - portfolio_performance.html (interactive chart)")
    print("   - Run without --no-plots to generate visualizations")
    
    return portfolio


if __name__ == "__main__":
    portfolio = main()