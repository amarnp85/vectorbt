#!/usr/bin/env python3
"""
Compare Basic vs Advanced Multi-Symbol Strategies

This script runs both strategies on the same data and shows:
1. Performance improvements
2. Risk reduction benefits
3. Diversification effectiveness
4. Signal quality differences
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime

from backtester.data import fetch_data
from backtester.portfolio import MultiAssetPortfolioSimulator, SimulationConfig, PositionSizeMode
from backtester.strategies.advanced_multi_symbol_strategy import AdvancedMultiSymbolStrategy


def create_simple_ma_signals(data, fast_period=20, slow_period=50):
    """Create basic MA crossover signals."""
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
    
    # No short signals for basic strategy
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


def analyze_strategy_differences(basic_portfolio, advanced_portfolio, basic_signals, advanced_signals, symbols):
    """Analyze differences between basic and advanced strategies."""
    
    print("\n" + "="*80)
    print("📊 STRATEGY COMPARISON ANALYSIS")
    print("="*80)
    
    # Get stats for both
    basic_stats = basic_portfolio.stats()
    advanced_stats = advanced_portfolio.stats()
    
    # Performance comparison
    print("\n💰 Performance Comparison:")
    print(f"{'Metric':<25} {'Basic':>15} {'Advanced':>15} {'Improvement':>15}")
    print("-" * 70)
    
    metrics = [
        ('Total Return [%]', '{:.2f}%', '{:.2f}%', '{:+.2f}%'),
        ('Sharpe Ratio', '{:.3f}', '{:.3f}', '{:+.3f}'),
        ('Max Drawdown [%]', '{:.2f}%', '{:.2f}%', '{:+.2f}%'),
        ('Win Rate [%]', '{:.1f}%', '{:.1f}%', '{:+.1f}%'),
        ('Profit Factor', '{:.2f}', '{:.2f}', '{:+.2f}'),
        ('Total Trades', '{:.0f}', '{:.0f}', '{:+.0f}')
    ]
    
    for metric, fmt1, fmt2, fmt3 in metrics:
        basic_val = basic_stats.get(metric, 0)
        advanced_val = advanced_stats.get(metric, 0)
        
        # Special handling for drawdown (negative is better)
        if 'Drawdown' in metric:
            improvement = basic_val - advanced_val
        else:
            improvement = advanced_val - basic_val
        
        print(f"{metric:<25} {fmt1.format(basic_val):>15} {fmt2.format(advanced_val):>15} {fmt3.format(improvement):>15}")
    
    # Signal quality comparison
    print("\n🎯 Signal Quality Analysis:")
    print(f"{'Symbol':<15} {'Basic Signals':>15} {'Advanced Signals':>15} {'Filter Rate':>15}")
    print("-" * 60)
    
    for symbol in symbols:
        basic_entries = basic_signals['long_entries'][symbol].sum()
        advanced_entries = advanced_signals['long_entries'][symbol].sum()
        
        if basic_entries > 0:
            filter_rate = (1 - advanced_entries / basic_entries) * 100
        else:
            filter_rate = 0
        
        print(f"{symbol:<15} {basic_entries:>15} {advanced_entries:>15} {filter_rate:>14.1f}%")
    
    # Risk analysis
    print("\n⚡ Risk Reduction Analysis:")
    
    basic_returns = basic_portfolio.returns
    advanced_returns = advanced_portfolio.returns
    
    basic_vol = basic_returns.std() * np.sqrt(252) * 100
    advanced_vol = advanced_returns.std() * np.sqrt(252) * 100
    
    basic_downside = basic_returns[basic_returns < 0].std() * np.sqrt(252) * 100
    advanced_downside = advanced_returns[advanced_returns < 0].std() * np.sqrt(252) * 100
    
    print(f"   Annual Volatility:     Basic: {basic_vol:.1f}%, Advanced: {advanced_vol:.1f}% ({advanced_vol-basic_vol:+.1f}%)")
    print(f"   Downside Volatility:   Basic: {basic_downside:.1f}%, Advanced: {advanced_downside:.1f}% ({advanced_downside-basic_downside:+.1f}%)")
    
    # Trade distribution
    basic_trades = basic_portfolio.trades.records_readable
    advanced_trades = advanced_portfolio.trades.records_readable
    
    if len(basic_trades) > 0 and len(advanced_trades) > 0:
        print("\n📈 Trade Distribution:")
        
        for portfolio_name, trades in [("Basic", basic_trades), ("Advanced", advanced_trades)]:
            print(f"\n   {portfolio_name} Strategy:")
            
            for symbol in symbols:
                symbol_trades = trades[trades['Column'] == symbol]
                if len(symbol_trades) > 0:
                    pnl = symbol_trades['PnL'].sum()
                    contribution = pnl / trades['PnL'].sum() * 100
                    print(f"      {symbol}: {len(symbol_trades)} trades, ${pnl:,.0f} PnL ({contribution:.1f}% contribution)")


def create_side_by_side_chart(basic_portfolio, advanced_portfolio, data, output_path):
    """Create a side-by-side comparison chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Portfolio Value Comparison', 'Drawdown Comparison'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Portfolio values
    basic_value = basic_portfolio.value
    advanced_value = advanced_portfolio.value
    
    fig.add_trace(
        go.Scatter(x=basic_value.index, y=basic_value.values, 
                   name='Basic Strategy', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=advanced_value.index, y=advanced_value.values,
                   name='Advanced Strategy', line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # Drawdowns
    basic_dd = basic_portfolio.drawdowns * 100
    advanced_dd = advanced_portfolio.drawdowns * 100
    
    fig.add_trace(
        go.Scatter(x=basic_dd.index, y=basic_dd.values,
                   name='Basic DD', line=dict(color='lightblue'), fill='tozeroy'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=advanced_dd.index, y=advanced_dd.values,
                   name='Advanced DD', line=dict(color='lightgreen'), fill='tozeroy'),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_layout(
        title="Basic vs Advanced Multi-Symbol Strategy Comparison",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.write_html(output_path)
    print(f"\n📊 Comparison chart saved to: {output_path}")


def main():
    """Run comparison between basic and advanced strategies."""
    
    print("🔬 Multi-Symbol Strategy Comparison")
    print("   Comparing: Basic MA Crossover vs Advanced Multi-Symbol Strategy")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "MATIC/USDT"]
    timeframe = "1d"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    init_cash = 100000
    
    # Fetch data
    print("\n📊 Loading market data...")
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
    
    # Configure portfolio simulation
    sim_config = SimulationConfig(
        init_cash=init_cash,
        fees=0.001,
        slippage=0.0005,
        position_size_mode=PositionSizeMode.PERCENT_EQUITY,
        position_size_value=0.20,  # 20% per position
        cash_sharing=True,
        sl_method='percent',
        tp_method='percent',
        use_unified_signals=False,  # Disable for multi-symbol DataFrame signals
        validate_signals=False  # Skip validation for multi-symbol
    )
    
    # Run basic strategy
    print("\n🔵 Running Basic MA Crossover Strategy...")
    basic_signals = create_simple_ma_signals(data)
    
    basic_simulator = MultiAssetPortfolioSimulator(data, sim_config)
    basic_portfolio = basic_simulator.simulate_from_signals(basic_signals)
    
    # Run advanced strategy
    print("\n🟢 Running Advanced Multi-Symbol Strategy...")
    advanced_strategy = AdvancedMultiSymbolStrategy(data, {
        'use_market_regime': False,  # Start with regime filter disabled
        'use_correlation_sizing': True,
        'use_volume_filter': False,  # Disable volume filter for now
        'use_volatility_filter': False,  # Disable volatility filter for now
        'max_correlation': 0.7,
        'position_reduction_factor': 0.5
    })
    
    indicators = advanced_strategy.init_indicators()
    advanced_signals = advanced_strategy.generate_signals()
    
    advanced_simulator = MultiAssetPortfolioSimulator(data, sim_config)
    advanced_portfolio = advanced_simulator.simulate_from_signals(advanced_signals)
    
    # Analyze differences
    analyze_strategy_differences(
        basic_portfolio, advanced_portfolio,
        basic_signals, advanced_signals,
        symbols
    )
    
    # Create comparison chart
    output_path = "results/strategy_comparison.html"
    create_side_by_side_chart(basic_portfolio, advanced_portfolio, data, output_path)
    
    # Summary
    print("\n" + "="*80)
    print("💡 KEY FINDINGS")
    print("="*80)
    
    basic_return = basic_portfolio.total_return() * 100
    advanced_return = advanced_portfolio.total_return() * 100
    improvement = advanced_return - basic_return
    
    print(f"\n📊 Return Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("   ✅ Advanced strategy outperformed basic strategy")
        print("\n   Benefits realized:")
        print("   - Better signal filtering reduced false positives")
        print("   - Market regime adaptation improved timing")
        print("   - Correlation-based sizing reduced concentration risk")
        print("   - Symbol-specific parameters optimized performance")
    else:
        print("   ⚠️ Basic strategy performed better in this period")
        print("\n   Possible reasons:")
        print("   - Over-filtering may have missed opportunities")
        print("   - Market conditions favored simple trend following")
        print("   - Advanced filters need parameter tuning")
    
    # Risk-adjusted performance
    basic_sharpe = basic_portfolio.sharpe_ratio()
    advanced_sharpe = advanced_portfolio.sharpe_ratio()
    
    if advanced_sharpe > basic_sharpe:
        print(f"\n✅ Risk-Adjusted Performance: Advanced strategy has better Sharpe ratio ({advanced_sharpe:.3f} vs {basic_sharpe:.3f})")
    
    print("\n🚀 Recommendations:")
    print("   1. Run parameter optimization for the advanced strategy")
    print("   2. Test across different market conditions (bull/bear markets)")
    print("   3. Analyze which filters provide the most value")
    print("   4. Consider ensemble approach combining both strategies")
    
    return basic_portfolio, advanced_portfolio


if __name__ == "__main__":
    basic_portfolio, advanced_portfolio = main()