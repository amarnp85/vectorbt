#!/usr/bin/env python3
"""
Analyze Multi-Symbol Portfolio Results

This script runs the multi-symbol portfolio backtest and produces
a comprehensive analysis report with actionable insights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

# Import our backtesting framework
from backtester.data import fetch_data
from backtester.portfolio import MultiAssetPortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.analysis import PerformanceAnalyzer
# We'll define the function here instead of importing
import vectorbtpro as vbt


def generate_analysis_report(portfolio, data, symbols):
    """Generate a comprehensive analysis report."""
    
    print("\n" + "="*80)
    print("📊 MULTI-SYMBOL PORTFOLIO ANALYSIS REPORT")
    print("="*80)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get portfolio statistics
    stats = portfolio.stats()
    trades = portfolio.trades.records_readable
    
    print("\n" + "-"*50)
    print("1. EXECUTIVE SUMMARY")
    print("-"*50)
    
    print(f"\n💰 Financial Performance:")
    print(f"   Initial Capital:     ${100000:>12,.2f}")
    print(f"   Final Value:         ${stats['End Value']:>12,.2f}")
    print(f"   Total Return:        {stats['Total Return [%]']:>12.2f}%")
    print(f"   Annual Return:       {stats.get('Annualized Return [%]', stats.get('Annual Return [%]', 0)):>12.2f}%")
    
    print(f"\n📉 Risk Metrics:")
    print(f"   Max Drawdown:        {stats.get('Max Drawdown [%]', 0):>12.2f}%")
    print(f"   Annual Volatility:   {stats.get('Annualized Volatility [%]', stats.get('Annual Volatility [%]', 0)):>12.2f}%")
    print(f"   Sharpe Ratio:        {stats.get('Sharpe Ratio', 0):>12.3f}")
    print(f"   Sortino Ratio:       {stats.get('Sortino Ratio', 0):>12.3f}")
    
    print(f"\n📈 Trading Activity:")
    print(f"   Total Trades:        {stats.get('Total Trades', 0):>12.0f}")
    print(f"   Win Rate:            {stats.get('Win Rate [%]', 0):>12.2f}%")
    print(f"   Profit Factor:       {stats.get('Profit Factor', 0):>12.2f}")
    print(f"   Avg Trade Duration:  {stats.get('Avg Trade Duration', 'N/A'):>12}")
    
    # Symbol-level analysis
    print("\n" + "-"*50)
    print("2. SYMBOL PERFORMANCE BREAKDOWN")
    print("-"*50)
    
    symbol_stats = []
    for i, symbol in enumerate(symbols):
        symbol_trades = trades[trades['Column'] == symbol]
        if len(symbol_trades) > 0:
            stats_dict = {
                'Symbol': symbol,
                'Trades': len(symbol_trades),
                'Total PnL': symbol_trades['PnL'].sum(),
                'Avg Return': symbol_trades['Return'].mean() * 100,
                'Win Rate': (symbol_trades['PnL'] > 0).mean() * 100,
                'Best Trade': symbol_trades['Return'].max() * 100,
                'Worst Trade': symbol_trades['Return'].min() * 100,
                'Contribution': symbol_trades['PnL'].sum() / trades['PnL'].sum() * 100
            }
            symbol_stats.append(stats_dict)
    
    df_symbols = pd.DataFrame(symbol_stats)
    if len(df_symbols) > 0:
        df_symbols = df_symbols.sort_values('Total PnL', ascending=False)
        
        for _, row in df_symbols.iterrows():
            print(f"\n{row['Symbol']}:")
            print(f"   Trades:        {row['Trades']:>6.0f}")
            print(f"   Total PnL:     ${row['Total PnL']:>10,.2f}")
            print(f"   Avg Return:    {row['Avg Return']:>6.2f}%")
            print(f"   Win Rate:      {row['Win Rate']:>6.1f}%")
            print(f"   Best/Worst:    {row['Best Trade']:>6.2f}% / {row['Worst Trade']:>6.2f}%")
            print(f"   Contribution:  {row['Contribution']:>6.1f}% of total PnL")
    
    # Risk analysis
    print("\n" + "-"*50)
    print("3. RISK ANALYSIS")
    print("-"*50)
    
    returns = portfolio.returns
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    
    print(f"\nVolatility Analysis:")
    print(f"   Current 30-day Vol:  {rolling_vol.iloc[-1]:>6.2f}%")
    print(f"   Average Vol:         {rolling_vol.mean():>6.2f}%")
    print(f"   Max Vol:             {rolling_vol.max():>6.2f}%")
    print(f"   Min Vol:             {rolling_vol.min():>6.2f}%")
    
    # Drawdown analysis
    drawdowns = portfolio.drawdowns
    drawdown_periods = (drawdowns < 0).astype(int).groupby((drawdowns >= 0).cumsum()).sum()
    significant_drawdowns = drawdown_periods[drawdown_periods > 20]  # Drawdowns lasting > 20 days
    
    print(f"\nDrawdown Analysis:")
    print(f"   Number of Drawdowns:     {len(drawdown_periods)}")
    print(f"   Significant Drawdowns:   {len(significant_drawdowns)} (>20 days)")
    print(f"   Longest Drawdown:        {drawdown_periods.max()} days")
    print(f"   Average Recovery Time:   {drawdown_periods[drawdown_periods > 0].mean():.1f} days")
    
    # Correlation analysis
    print("\n" + "-"*50)
    print("4. DIVERSIFICATION ANALYSIS")
    print("-"*50)
    
    # Calculate returns for each symbol
    symbol_returns = data.close.pct_change()
    correlation_matrix = symbol_returns.corr()
    
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3).to_string())
    
    # Average correlation
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    avg_corr = correlation_matrix.where(mask).stack().mean()
    
    print(f"\nAverage Pairwise Correlation: {avg_corr:.3f}")
    if avg_corr < 0.5:
        print("✅ Low correlation - Excellent diversification")
    elif avg_corr < 0.7:
        print("⚠️  Moderate correlation - Decent diversification")
    else:
        print("❌ High correlation - Poor diversification")
    
    # Market regime analysis
    print("\n" + "-"*50)
    print("5. MARKET REGIME ANALYSIS")
    print("-"*50)
    
    # Calculate market breadth
    positive_returns = (symbol_returns > 0).sum(axis=1) / len(symbols) * 100
    
    print(f"\nMarket Breadth Statistics:")
    print(f"   Average Breadth:     {positive_returns.mean():>6.1f}% symbols positive")
    print(f"   Current Breadth:     {positive_returns.iloc[-1]:>6.1f}% symbols positive")
    
    # Bull/Bear market classification
    market_return = symbol_returns.mean(axis=1).cumsum()
    market_ma = market_return.rolling(50).mean()
    bull_market = (market_return > market_ma).sum() / len(market_return) * 100
    
    print(f"\nMarket Regime:")
    print(f"   Time in Bull Market: {bull_market:>6.1f}%")
    print(f"   Time in Bear Market: {100-bull_market:>6.1f}%")
    
    # Recommendations
    print("\n" + "-"*50)
    print("6. STRATEGIC RECOMMENDATIONS")
    print("-"*50)
    
    print("\n📋 Based on the analysis:")
    
    # Performance-based recommendations
    if stats['Sharpe Ratio'] < 0.5:
        print("\n⚠️  Low Sharpe Ratio:")
        print("   - Consider adjusting MA periods (try 10/30 or 30/60)")
        print("   - Add trend filters to avoid choppy markets")
        print("   - Implement volatility-based position sizing")
    
    if stats['Win Rate [%]'] < 40:
        print("\n⚠️  Low Win Rate:")
        print("   - Current system relies on few large winners")
        print("   - Consider tighter stop losses")
        print("   - Add confirmation indicators (RSI, Volume)")
    
    if stats['Max Drawdown [%]'] < -20:
        print("\n⚠️  High Drawdown Risk:")
        print("   - Reduce position size (try 15-20% per position)")
        print("   - Implement portfolio-level stop loss")
        print("   - Consider max exposure limits")
    
    # Symbol-specific recommendations
    if len(df_symbols) > 0:
        underperformers = df_symbols[df_symbols['Total PnL'] < 0]
        if len(underperformers) > 0:
            print(f"\n⚠️  Underperforming Symbols ({', '.join(underperformers['Symbol'].tolist())}):")
            print("   - Consider removing or reducing allocation")
            print("   - May need different parameters per symbol")
            print("   - Check if symbol is suitable for trend following")
    
    # Diversification recommendations
    if avg_corr > 0.7:
        print("\n⚠️  High Correlation Between Assets:")
        print("   - Add non-crypto assets (commodities, forex)")
        print("   - Include defensive assets")
        print("   - Consider market-neutral strategies")
    
    print("\n✅ Positive Aspects:")
    if stats['Sharpe Ratio'] > 0.7:
        print("   - Good risk-adjusted returns")
    if stats['Win Rate [%]'] > 50:
        print("   - Solid win rate indicates consistent edge")
    if avg_corr < 0.5:
        print("   - Excellent diversification across symbols")
    if stats['Max Drawdown [%]'] > -15:
        print("   - Drawdown within acceptable limits")
    
    # Action items
    print("\n" + "-"*50)
    print("7. ACTION ITEMS")
    print("-"*50)
    
    print("\n1️⃣  Immediate Actions:")
    print("   □ Review and optimize MA periods per symbol")
    print("   □ Implement proper position sizing based on volatility")
    print("   □ Add market regime filter (trend vs ranging)")
    
    print("\n2️⃣  Short-term Improvements:")
    print("   □ Test with different stop loss/take profit ratios")
    print("   □ Add volume confirmation for signals")
    print("   □ Implement correlation-based position reduction")
    
    print("\n3️⃣  Long-term Enhancements:")
    print("   □ Develop symbol-specific parameter optimization")
    print("   □ Add machine learning for market regime detection")
    print("   □ Implement dynamic portfolio rebalancing")
    
    return stats


def create_simple_ma_crossover_signals(data, fast_period=20, slow_period=50):
    """Create simple MA crossover signals for all symbols."""
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


def main():
    """Run analysis and generate report."""
    
    print("🚀 Running Multi-Symbol Portfolio Analysis...")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Fetch data
    print("📊 Loading market data...")
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
    
    # Generate signals
    print("📈 Generating trading signals...")
    signals = create_simple_ma_crossover_signals(data)
    
    # Configure and run simulation
    print("💼 Running portfolio simulation...")
    sim_config = SimulationConfig(
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        position_size_mode=PositionSizeMode.PERCENT_EQUITY,
        position_size_value=0.25,
        cash_sharing=True
    )
    
    simulator = MultiAssetPortfolioSimulator(data, sim_config)
    portfolio = simulator.simulate_from_signals(signals)
    
    if portfolio is None:
        print("❌ Simulation failed")
        return
    
    # Generate comprehensive report
    stats = generate_analysis_report(portfolio, data, symbols)
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print("="*80)
    
    return portfolio, stats


if __name__ == "__main__":
    portfolio, stats = main()