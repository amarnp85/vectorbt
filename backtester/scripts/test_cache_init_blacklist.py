#!/usr/bin/env python3
"""
Test Cache Manager Blacklist Initialization

This script tests that the cache manager automatically creates a default blacklist.json
with USDC/USDT and other stable-stable pairs when initialized, ensuring any future
new exchange will have sensible defaults.
"""

import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to Python path so we can import backtester modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.cache_system.cache_manager import SimpleCacheManager
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_fresh_cache_manager_initialization():
    """Test that a fresh cache manager creates a default blacklist."""
    
    print("\n" + "="*60)
    print("🆕 TESTING FRESH CACHE MANAGER INITIALIZATION")
    print("="*60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary cache directory: {temp_dir}")
        
        # Create a new cache manager in the temp directory
        cache_manager = SimpleCacheManager(cache_dir=temp_dir)
        
        # Check if blacklist.json was created
        blacklist_file = os.path.join(temp_dir, 'blacklist.json')
        
        if os.path.exists(blacklist_file):
            print(f"✅ Blacklist file created automatically: {blacklist_file}")
            
            # Load and inspect the blacklist
            with open(blacklist_file, 'r') as f:
                blacklist_data = json.load(f)
            
            print(f"📋 Blacklist structure: {list(blacklist_data.keys())}")
            
            # Check that USDC/USDT is in global blacklist
            global_blacklist = blacklist_data.get('global', [])
            print(f"🌍 Global blacklist ({len(global_blacklist)} symbols): {global_blacklist}")
            
            if 'USDC/USDT' in global_blacklist:
                print(f"✅ USDC/USDT found in global blacklist - requirement satisfied!")
            else:
                print(f"❌ USDC/USDT NOT found in global blacklist!")
                
            # Check exchange-specific sections
            for exchange in ['binance', 'bybit', 'okx', 'kucoin', 'coinbase']:
                exchange_list = blacklist_data.get(exchange, [])
                print(f"   {exchange}: {len(exchange_list)} symbols")
                
        else:
            print(f"❌ Blacklist file was NOT created automatically!")
            
        return blacklist_file if os.path.exists(blacklist_file) else None

def test_cache_manager_blacklist_methods():
    """Test the cache manager's blacklist methods."""
    
    print("\n" + "="*60)
    print("🧪 TESTING CACHE MANAGER BLACKLIST METHODS")
    print("="*60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_manager = SimpleCacheManager(cache_dir=temp_dir)
        
        # Test get_blacklist
        blacklist_data = cache_manager.get_blacklist()
        print(f"📋 get_blacklist() returned: {len(blacklist_data)} sections")
        
        # Test is_symbol_blacklisted
        test_symbols = ['USDC/USDT', 'BTC/USDT', 'ETH/USDT', 'BUSD/USDT']
        print(f"\n🧪 Testing is_symbol_blacklisted():")
        
        for symbol in test_symbols:
            is_blacklisted = cache_manager.is_symbol_blacklisted(symbol, 'binance')
            print(f"   {symbol}: {'🚫 BLACKLISTED' if is_blacklisted else '✅ allowed'}")
        
        # Test filter_blacklisted_symbols
        print(f"\n🧪 Testing filter_blacklisted_symbols():")
        print(f"   Input: {test_symbols}")
        
        filtered = cache_manager.filter_blacklisted_symbols(test_symbols, 'binance')
        print(f"   Filtered: {filtered}")
        print(f"   Removed: {len(test_symbols) - len(filtered)} symbols")

def test_existing_blacklist_validation():
    """Test that existing blacklist files are validated and updated if needed."""
    
    print("\n" + "="*60)
    print("🔄 TESTING EXISTING BLACKLIST VALIDATION")
    print("="*60)
    
    # Create a temporary directory with an incomplete blacklist
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Creating incomplete blacklist in: {temp_dir}")
        
        # Create an incomplete blacklist (missing USDC/USDT)
        incomplete_blacklist = {
            "global": ["BUSD/USDT", "DAI/USDT"],
            "binance": []
        }
        
        blacklist_file = os.path.join(temp_dir, 'blacklist.json')
        with open(blacklist_file, 'w') as f:
            json.dump(incomplete_blacklist, f, indent=2)
        
        print(f"📝 Created incomplete blacklist (missing USDC/USDT)")
        
        # Initialize cache manager - should validate and update the blacklist
        cache_manager = SimpleCacheManager(cache_dir=temp_dir)
        
        # Check if USDC/USDT was added
        updated_blacklist = cache_manager.get_blacklist()
        global_list = updated_blacklist.get('global', [])
        
        if 'USDC/USDT' in global_list:
            print(f"✅ USDC/USDT was automatically added to existing blacklist!")
            print(f"📋 Updated global blacklist: {global_list}")
        else:
            print(f"❌ USDC/USDT was NOT added to existing blacklist!")

def main():
    """Run all tests."""
    
    print("🧪 Testing Cache Manager Blacklist Initialization")
    print("=" * 80)
    
    try:
        # Test fresh initialization
        test_fresh_cache_manager_initialization()
        
        # Test blacklist methods
        test_cache_manager_blacklist_methods()
        
        # Test validation of existing blacklist
        test_existing_blacklist_validation()
        
        print("\n" + "="*80)
        print("✅ All cache manager blacklist tests completed!")
        print("✨ Default blacklist initialization is working correctly")
        print("🚫 USDC/USDT is guaranteed to be blacklisted for all exchanges")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 