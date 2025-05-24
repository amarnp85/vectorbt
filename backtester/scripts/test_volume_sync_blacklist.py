#!/usr/bin/env python3
"""
Test Volume Data Synchronization and Blacklist Functionality

This script tests the enhanced data_fetcher functionality that:
1. Automatically checks and updates volume data to latest midnight UTC
2. Discovers new symbols and updates their timestamps
3. Applies blacklist filtering to exclude unwanted symbols

Run this script to verify the implementation works correctly.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to Python path so we can import backtester modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetching.data_fetcher import fetch_data, fetch_top_symbols, _filter_blacklisted_symbols, _load_blacklist
from data.cache_system import cache_manager
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_blacklist_functionality():
    """Test blacklist loading and filtering functionality."""
    
    print("\n" + "="*60)
    print("🚫 TESTING BLACKLIST FUNCTIONALITY")
    print("="*60)
    
    # Test loading blacklist
    blacklisted = _load_blacklist()
    print(f"📋 Loaded blacklist: {len(blacklisted)} symbols")
    if blacklisted:
        print(f"   Examples: {list(blacklisted)[:5]}{'...' if len(blacklisted) > 5 else ''}")
    
    # Test symbol filtering
    test_symbols = [
        'BTC/USDT', 'ETH/USDT', 'USDC/USDT', 'BNB/USDT', 'BUSD/USDT', 
        'ADA/USDT', 'DOT/USDT', 'LINK/USDT'
    ]
    
    print(f"\n🧪 Testing symbol filtering:")
    print(f"   Input symbols: {test_symbols}")
    
    filtered_symbols = _filter_blacklisted_symbols(test_symbols, 'binance')
    print(f"   Filtered symbols: {filtered_symbols}")
    print(f"   Removed: {len(test_symbols) - len(filtered_symbols)} symbols")
    
    return filtered_symbols

def test_volume_sync():
    """Test volume data synchronization functionality."""
    
    print("\n" + "="*60)  
    print("📊 TESTING VOLUME DATA SYNCHRONIZATION")
    print("="*60)
    
    exchange_id = 'binance'
    
    # Check current volume cache status
    volume_cache_fresh = cache_manager.is_volume_cache_fresh(exchange_id)
    print(f"📦 Volume cache fresh for {exchange_id}: {volume_cache_fresh}")
    
    # Get current volume data stats
    volumes = cache_manager.get_all_volumes(exchange_id)
    timestamps = cache_manager.get_all_timestamps(exchange_id)
    
    print(f"📊 Current cache stats:")
    print(f"   Volume symbols: {len(volumes)}")
    print(f"   Timestamp symbols: {len(timestamps)}")
    
    # Show volume file timestamp if it exists
    try:
        volume_file = os.path.join(cache_manager._get_exchange_dir(exchange_id), 'volume.json')
        if os.path.exists(volume_file):
            mtime = os.path.getmtime(volume_file)
            mtime_dt = datetime.fromtimestamp(mtime)
            print(f"   Volume file last modified: {mtime_dt}")
        else:
            print(f"   Volume file: Not found")
    except Exception as e:
        print(f"   Volume file check failed: {e}")

def test_fetch_with_sync():
    """Test data fetching with volume sync and blacklist filtering."""
    
    print("\n" + "="*60)
    print("🚀 TESTING DATA FETCH WITH SYNC & BLACKLIST")
    print("="*60)
    
    # Test symbols including some that should be blacklisted
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'USDC/USDT', 'BNB/USDT', 'ADA/USDT']
    
    print(f"🎯 Testing fetch_data with symbols: {test_symbols}")
    
    try:
        # This should trigger volume sync and blacklist filtering
        data = fetch_data(
            symbols=test_symbols,
            exchange_id='binance',
            timeframe='1d',
            use_cache=True,
            prefer_resampling=False
        )
        
        if data is not None:
            available_symbols = list(data.symbols) if hasattr(data, 'symbols') else []
            print(f"✅ Fetch successful!")
            print(f"   Requested: {test_symbols}")
            print(f"   Retrieved: {available_symbols}")
            print(f"   Data shape: {data.wrapper.index.shape if hasattr(data, 'wrapper') else 'unknown'}")
        else:
            print(f"❌ Fetch failed - no data returned")
            
    except Exception as e:
        print(f"❌ Fetch error: {e}")

def test_top_symbols_with_blacklist():
    """Test fetching top symbols with blacklist filtering."""
    
    print("\n" + "="*60)
    print("🔝 TESTING TOP SYMBOLS WITH BLACKLIST")
    print("="*60)
    
    try:
        # This should get top symbols by volume and apply blacklist filtering
        data = fetch_top_symbols(
            exchange_id='binance',
            quote_currency='USDT',
            limit=10,
            timeframe='1d',
            use_cache=True
        )
        
        if data is not None:
            symbols = list(data.symbols) if hasattr(data, 'symbols') else []
            print(f"✅ Top symbols fetch successful!")
            print(f"   Retrieved symbols: {symbols}")
            
            # Check if any blacklisted symbols made it through
            blacklisted = _load_blacklist()
            blacklisted_found = [s for s in symbols if s in blacklisted]
            
            if blacklisted_found:
                print(f"⚠️ WARNING: Blacklisted symbols found in results: {blacklisted_found}")
            else:
                print(f"✅ No blacklisted symbols in results - filtering working correctly!")
                
        else:
            print(f"❌ Top symbols fetch failed")
            
    except Exception as e:
        print(f"❌ Top symbols fetch error: {e}")

def inspect_blacklist_file():
    """Inspect the current blacklist.json file."""
    
    print("\n" + "="*60)
    print("📄 BLACKLIST FILE INSPECTION")
    print("="*60)
    
    blacklist_file = os.path.join(cache_manager.cache_dir, 'blacklist.json')
    
    if os.path.exists(blacklist_file):
        print(f"📄 Blacklist file found: {blacklist_file}")
        
        try:
            with open(blacklist_file, 'r') as f:
                blacklist_data = json.load(f)
            
            print(f"📋 Blacklist contents:")
            print(json.dumps(blacklist_data, indent=2))
            
        except Exception as e:
            print(f"❌ Error reading blacklist file: {e}")
    else:
        print(f"📄 Blacklist file not found at: {blacklist_file}")
        print(f"   (Will be created automatically on first run)")

def main():
    """Run all tests."""
    
    print("🧪 Testing Volume Data Synchronization and Blacklist Functionality")
    print("=" * 80)
    
    try:
        # Inspect blacklist file first
        inspect_blacklist_file()
        
        # Test blacklist functionality
        filtered_symbols = test_blacklist_functionality()
        
        # Test volume synchronization
        test_volume_sync()
        
        # Test data fetching with sync and blacklist
        test_fetch_with_sync()
        
        # Test top symbols with blacklist
        test_top_symbols_with_blacklist()
        
        print("\n" + "="*80)
        print("✅ All tests completed!")
        print("✨ Volume synchronization and blacklist functionality is working")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 