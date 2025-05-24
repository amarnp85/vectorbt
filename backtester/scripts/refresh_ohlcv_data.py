#!/usr/bin/env python3
"""Refresh OHLCV Data

Refresh cached data that has incomplete OHLCV structure to enable storage resampling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt
from backtester.data.storage.data_storage import data_storage
from backtester.data.fetching.data_fetcher_new import fetch_data

def refresh_incomplete_ohlcv_data():
    """Refresh cached data that only has close prices instead of full OHLCV."""
    
    print("🔄 OHLCV Data Refresh Utility")
    print("=" * 50)
    
    # Check existing 4h data
    print("\n📊 Checking existing 4h cache...")
    existing_4h = data_storage.load_data('binance', '4h', market_type='spot')
    
    if existing_4h is None:
        print("   ❌ No 4h data found")
        return False
    
    # Check if it has proper OHLCV structure
    has_full_ohlcv = (
        hasattr(existing_4h, 'has_ohlcv') and existing_4h.has_ohlcv and
        hasattr(existing_4h, 'wrapper') and len(existing_4h.wrapper.columns) >= 5
    )
    
    print(f"   Type: {type(existing_4h)}")
    print(f"   Symbols: {len(existing_4h.symbols)} symbols")
    print(f"   Shape: {existing_4h.wrapper.shape}")
    print(f"   Columns: {list(existing_4h.wrapper.columns)}")
    print(f"   Has OHLCV: {existing_4h.has_ohlcv if hasattr(existing_4h, 'has_ohlcv') else 'Unknown'}")
    
    if has_full_ohlcv:
        print("   ✅ 4h data has complete OHLCV structure - no refresh needed")
        return True
    
    print("   ⚠️ 4h data has incomplete OHLCV structure - refreshing...")
    
    # Get symbols from existing data
    symbols = list(existing_4h.symbols)
    print(f"   📋 Refreshing {len(symbols)} symbols: {symbols}")
    
    # Backup existing file
    backup_success = _backup_existing_data('binance', '4h', 'spot')
    if backup_success:
        print("   💾 Created backup of existing 4h data")
    
    # Fetch fresh OHLCV data
    print("\n🔄 Fetching fresh OHLCV data...")
    fresh_data = fetch_data(
        symbols=symbols,
        exchange_id='binance',
        timeframe='4h',
        start_date=None,  # Get from inception
        end_date=None,    # Get latest
        use_cache=False,  # Don't use cache to avoid recursion
        market_type='spot',
        prefer_resampling=False  # Direct fetch
    )
    
    if fresh_data is None:
        print("   ❌ Failed to fetch fresh data")
        return False
    
    # Verify fresh data has OHLCV
    fresh_has_ohlcv = (
        hasattr(fresh_data, 'has_ohlcv') and fresh_data.has_ohlcv and
        hasattr(fresh_data, 'wrapper') and len(fresh_data.wrapper.columns) >= 5
    )
    
    print(f"   ✅ Fresh data fetched successfully")
    print(f"   Shape: {fresh_data.wrapper.shape}")
    print(f"   Columns: {list(fresh_data.wrapper.columns)}")
    print(f"   Has OHLCV: {fresh_data.has_ohlcv if hasattr(fresh_data, 'has_ohlcv') else 'Unknown'}")
    
    if not fresh_has_ohlcv:
        print("   ❌ Fresh data also lacks OHLCV structure - this indicates a deeper issue")
        return False
    
    # Save the fresh data (this will overwrite the incomplete cache)
    print("\n💾 Saving fresh OHLCV data...")
    save_success = data_storage.save_data(fresh_data, 'binance', '4h', 'spot')
    
    if save_success:
        print("   ✅ Fresh OHLCV data saved successfully")
        print("   🎯 Storage resampling should now work properly")
        return True
    else:
        print("   ❌ Failed to save fresh data")
        return False

def _backup_existing_data(exchange_id: str, timeframe: str, market_type: str) -> bool:
    """Create a backup of existing data before refresh."""
    
    try:
        import shutil
        from datetime import datetime
        
        # Get existing file path
        pickle_path = data_storage._get_pickle_path(exchange_id, timeframe, market_type)
        
        if not os.path.exists(pickle_path):
            return False
        
        # Create backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{pickle_path}.backup_{timestamp}"
        
        shutil.copy2(pickle_path, backup_path)
        return True
        
    except Exception as e:
        print(f"   ⚠️ Failed to create backup: {e}")
        return False

def test_resampling_after_refresh():
    """Test storage resampling after refresh to verify it works."""
    
    print("\n🧪 Testing Storage Resampling After Refresh")
    print("-" * 50)
    
    try:
        # Try to resample 4h to 1d data using new core architecture
        from backtester.data.fetching.core import CacheHandler
        
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Use the new core cache handler for resampling test
        cache_handler = CacheHandler('binance', '1d', 'spot')
        result = cache_handler.try_load_from_lower_timeframe(
            symbols=test_symbols,
            end_date=None,
            require_fresh=False
        )
        
        if result is not None:
            print(f"   ✅ Storage resampling test successful!")
            print(f"   Shape: {result.wrapper.shape}")
            print(f"   Columns: {list(result.wrapper.columns)}")
            print(f"   Has OHLCV: {result.has_ohlcv if hasattr(result, 'has_ohlcv') else 'Unknown'}")
            return True
        else:
            print(f"   ❌ Storage resampling test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Storage resampling test error: {e}")
        return False

if __name__ == "__main__":
    success = refresh_incomplete_ohlcv_data()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 OHLCV data refresh completed successfully!")
        print("💡 You can now use storage resampling functionality.")
        
        # Test resampling
        test_success = test_resampling_after_refresh()
        if test_success:
            print("✅ Storage resampling is now working properly!")
        else:
            print("⚠️ Storage resampling still has issues - manual investigation needed.")
    else:
        print("\n" + "=" * 50)
        print("❌ OHLCV data refresh failed")
        print("💡 Manual intervention may be required.") 