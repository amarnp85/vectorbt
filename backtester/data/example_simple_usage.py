#!/usr/bin/env python3
"""
Simple Data Module Usage Example

This example demonstrates how easy it is to use the data module
with its simplified interface.
"""

from backtester.data import (
    fetch_data,
    fetch_top_symbols,
    quick_fetch,
    update_data,
    load_cached,
    get_cache_info
)


def main():
    print("Data Module Simple Usage Example")
    print("=" * 50)
    
    # 1. Quick single symbol fetch
    print("\n1. Quick fetch for prototyping:")
    btc_data = quick_fetch('BTC/USDT', days=30)
    if btc_data:
        print(f"   - Fetched {len(btc_data)} candles")
        print(f"   - Latest close: ${btc_data.close.iloc[-1]:,.2f}")
    
    # 2. Fetch multiple symbols
    print("\n2. Fetch multiple symbols:")
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    data = fetch_data(symbols, timeframe='1d', start_date='2024-01-01')
    if data:
        print(f"   - Symbols: {data.symbols}")
        print(f"   - Date range: {data.index[0]} to {data.index[-1]}")
    
    # 3. Get top symbols by volume
    print("\n3. Get top symbols by volume:")
    top_data = fetch_top_symbols(limit=5)
    if top_data:
        print(f"   - Top 5 symbols: {top_data.symbols}")
    
    # 4. Load cached data
    print("\n4. Load cached data (no fetching):")
    cached_data = load_cached(symbols=['BTC/USDT'])
    if cached_data:
        print(f"   - Loaded from cache: {cached_data.symbols}")
    
    # 5. Check cache info
    print("\n5. Cache information:")
    info = get_cache_info()
    print(f"   - Total cached files: {info.get('pickle_files', 0)}")
    print(f"   - Total size: {info.get('total_size_mb', 0):.2f} MB")
    
    # 6. Working with VectorBT features
    print("\n6. VectorBT features:")
    if btc_data:
        # Calculate indicators
        rsi = btc_data.run('talib:RSI', 14)
        sma = btc_data.run('talib:SMA', 20)
        
        print(f"   - Latest RSI(14): {rsi.iloc[-1]:.2f}")
        print(f"   - Latest SMA(20): ${sma.iloc[-1]:,.2f}")
        
        # Access returns
        returns = btc_data.returns
        print(f"   - Average daily return: {returns.mean():.4%}")
    
    print("\n✅ That's it! Simple and clean interface.")
    print("   No need to understand the internal architecture.")
    print("   Everything happens automatically behind the scenes.")


if __name__ == "__main__":
    main() 