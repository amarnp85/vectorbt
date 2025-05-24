#!/usr/bin/env python3
"""Test VBT Data Integration Patterns.

This script demonstrates and tests various VBT data access patterns
that are commonly used in strategy development.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import vectorbtpro as vbt
from backtester.data.fetching.data_fetcher_new import (
    fetch_data, quick_fetch, test_vbt_integration
)
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_data_access():
    """Test basic VBT data access patterns."""
    print("\n" + "="*50)
    print("Testing Basic VBT Data Access Patterns")
    print("="*50)
    
    # Fetch some sample data
    data = quick_fetch('BTC/USDT', days=30, timeframe='1h')
    if data is None:
        print("❌ Failed to fetch data")
        return False
    
    print(f"✅ Fetched data for {data.symbols}")
    
    # Test 1: Magnet features (data.open, data.close, etc.)
    print("\n1. Testing magnet features:")
    try:
        print(f"   - data.open shape: {data.open.shape}")
        print(f"   - data.close shape: {data.close.shape}")
        print(f"   - data.high shape: {data.high.shape}")
        print(f"   - data.low shape: {data.low.shape}")
        print(f"   - data.volume shape: {data.volume.shape}")
        print("   ✅ Magnet features working")
    except Exception as e:
        print(f"   ❌ Magnet features failed: {e}")
        return False
    
    # Test 2: get() method
    print("\n2. Testing get() method:")
    try:
        open_price = data.get('Open')
        close_price = data.get('Close')
        print(f"   - data.get('Open') shape: {open_price.shape}")
        print(f"   - data.get('Close') shape: {close_price.shape}")
        print("   ✅ get() method working")
    except Exception as e:
        print(f"   ❌ get() method failed: {e}")
        return False
    
    # Test 3: Returns calculation
    print("\n3. Testing returns calculation:")
    try:
        returns = data.returns
        print(f"   - data.returns shape: {returns.shape}")
        print(f"   - Mean return: {returns.mean():.4%}")
        print("   ✅ Returns calculation working")
    except Exception as e:
        print(f"   ❌ Returns calculation failed: {e}")
    
    # Test 4: HLC/OHLC calculations
    print("\n4. Testing HLC/OHLC calculations:")
    try:
        hlc3 = data.hlc3
        ohlc4 = data.ohlc4
        print(f"   - data.hlc3 shape: {hlc3.shape}")
        print(f"   - data.ohlc4 shape: {ohlc4.shape}")
        print("   ✅ HLC/OHLC calculations working")
    except Exception as e:
        print(f"   ❌ HLC/OHLC calculations failed: {e}")
    
    return True


def test_indicator_integration():
    """Test indicator integration with data.run()."""
    print("\n" + "="*50)
    print("Testing Indicator Integration")
    print("="*50)
    
    # Fetch data
    data = quick_fetch('ETH/USDT', days=60, timeframe='1h')
    if data is None:
        print("❌ Failed to fetch data")
        return False
    
    # Test various indicators
    indicators_to_test = [
        ('talib:SMA', 20),
        ('talib:EMA', 20),
        ('talib:RSI', 14),
        ('talib:BBANDS', 20),
    ]
    
    for indicator, period in indicators_to_test:
        try:
            result = data.run(indicator, period)
            print(f"✅ {indicator}({period}) executed successfully")
            # Print some info about the result
            if hasattr(result, 'shape'):
                print(f"   Shape: {result.shape}")
            elif hasattr(result, 'real'):
                print(f"   Output shape: {result.real.shape}")
        except Exception as e:
            print(f"❌ {indicator}({period}) failed: {e}")
    
    # Test RSI strategy pattern from tutorial
    print("\n5. Testing RSI strategy pattern:")
    try:
        open_price = data.get('Open')
        close_price = data.get('Close')
        
        # Run RSI on open price
        rsi = vbt.RSI.run(open_price, window=14)
        
        # Generate signals
        entries = rsi.rsi.vbt.crossed_below(30)
        exits = rsi.rsi.vbt.crossed_above(70)
        
        print(f"   - RSI calculated: {rsi.rsi.shape}")
        print(f"   - Entry signals: {entries.sum().sum()}")
        print(f"   - Exit signals: {exits.sum().sum()}")
        print("   ✅ RSI strategy pattern working")
        
    except Exception as e:
        print(f"   ❌ RSI strategy pattern failed: {e}")
    
    return True


def test_multi_timeframe():
    """Test multi-timeframe analysis patterns."""
    print("\n" + "="*50)
    print("Testing Multi-Timeframe Analysis")
    print("="*50)
    
    # Fetch hourly data
    h1_data = quick_fetch('BTC/USDT', days=30, timeframe='1h')
    if h1_data is None:
        print("❌ Failed to fetch hourly data")
        return False
    
    print(f"✅ Fetched H1 data: {h1_data.wrapper.shape}")
    
    # Test resampling to different timeframes
    timeframes_to_test = ['4h', '1d']
    
    for tf in timeframes_to_test:
        try:
            resampled = h1_data.resample(tf)
            print(f"✅ Resampled to {tf}: {resampled.wrapper.shape}")
            
            # Verify OHLCV is correct after resampling
            print(f"   - Open preserved: {resampled.open.iloc[0] == h1_data.open.iloc[0]}")
            print(f"   - High is max: {resampled.high.iloc[0] >= h1_data.high.iloc[0]}")
            print(f"   - Low is min: {resampled.low.iloc[0] <= h1_data.low.iloc[0]}")
            
        except Exception as e:
            print(f"❌ Resampling to {tf} failed: {e}")
    
    return True


def test_multi_symbol():
    """Test multi-symbol data handling."""
    print("\n" + "="*50)
    print("Testing Multi-Symbol Data Handling")
    print("="*50)
    
    # Fetch multiple symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    data = fetch_data(symbols, timeframe='1d', start_date='2024-01-01')
    
    if data is None:
        print("❌ Failed to fetch multi-symbol data")
        return False
    
    print(f"✅ Fetched data for {len(data.symbols)} symbols")
    print(f"   Symbols: {data.symbols}")
    print(f"   Shape: {data.wrapper.shape}")
    
    # Test symbol selection
    for symbol in data.symbols[:2]:  # Test first 2
        try:
            selected = data.select(symbol)
            print(f"✅ Selected {symbol}: shape {selected.wrapper.shape}")
        except Exception as e:
            print(f"❌ Failed to select {symbol}: {e}")
    
    # Test indicator on multi-symbol data
    try:
        rsi = vbt.RSI.run(data.close, window=14)
        print(f"✅ RSI on multi-symbol data: {rsi.rsi.shape}")
        print(f"   Columns: {list(rsi.rsi.columns)}")
    except Exception as e:
        print(f"❌ RSI on multi-symbol data failed: {e}")
    
    return True


def run_integration_tests():
    """Run comprehensive integration tests."""
    print("\n" + "="*50)
    print("Running Comprehensive Integration Tests")
    print("="*50)
    
    # Quick fetch some data for testing
    data = quick_fetch('BTC/USDT', days=7, timeframe='1h')
    if data is None:
        print("❌ Failed to fetch test data")
        return
    
    # Run automated tests
    results = test_vbt_integration(data)
    
    print("\nIntegration Test Results:")
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}: {'PASSED' if passed else 'FAILED'}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


def main():
    """Run all tests."""
    print("VectorBT PRO Data Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Data Access", test_basic_data_access),
        ("Indicator Integration", test_indicator_integration),
        ("Multi-Timeframe Analysis", test_multi_timeframe),
        ("Multi-Symbol Handling", test_multi_symbol),
        ("Comprehensive Integration", run_integration_tests),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main() 