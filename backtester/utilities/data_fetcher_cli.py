#!/usr/bin/env python3
"""
Command Line Interface for the Data Fetcher

This CLI allows testing and using the refactored data fetcher from the command line.
Supports various operations like fetch, update, storage info, and top symbols.
"""

import argparse
import sys
import json
from typing import List, Optional

def fetch_command(args):
    """Handle fetch command"""
    from backtester.data.fetching.data_fetcher import fetch_data
    
    symbols = args.symbols if args.symbols else ['BTCUSDT']
    
    print(f"🚀 Fetching {len(symbols)} symbols from {args.exchange}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Date range: {args.start_date or 'inception'} → {args.end_date or 'latest'}")
    
    data = fetch_data(
        symbols=symbols,
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache,
        market_type=args.market_type,
        prefer_resampling=not args.no_resampling
    )
    
    if data is not None:
        print(f"\n✅ Fetch completed successfully!")
        print(f"   Symbols: {list(data.symbols)}")
        print(f"   Shape: {data.wrapper.shape}")
        print(f"   Date range: {data.wrapper.index[0]} to {data.wrapper.index[-1]}")
        
        # Show sample data if requested
        if args.show_data:
            print(f"\n📊 Sample data (first 5 rows):")
            if hasattr(data, 'close'):
                print(data.close.head())
            else:
                print(data.get().head())
        
        return True
    else:
        print("❌ Fetch failed!")
        return False

def top_symbols_command(args):
    """Handle top symbols command"""
    from backtester.data.fetching.data_fetcher import fetch_top_symbols
    
    print(f"📊 Fetching top {args.limit} symbols by volume")
    print(f"   Exchange: {args.exchange}")
    print(f"   Quote currency: {args.quote_currency}")
    
    data = fetch_top_symbols(
        exchange_id=args.exchange,
        quote_currency=args.quote_currency,
        market_type=args.market_type,
        limit=args.limit,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=not args.no_cache
    )
    
    if data is not None:
        print(f"\n✅ Top symbols fetch completed!")
        print(f"   Symbols: {list(data.symbols)}")
        print(f"   Shape: {data.wrapper.shape}")
        
        if args.show_data:
            print(f"\n📊 Sample data (first 5 rows):")
            if hasattr(data, 'close'):
                print(data.close.head())
            else:
                print(data.get().head())
        
        return True
    else:
        print("❌ Top symbols fetch failed!")
        return False

def update_command(args):
    """Handle update command"""
    from backtester.data.fetching.data_fetcher import update_data
    
    print(f"🔄 Updating cached data")
    print(f"   Exchange: {args.exchange}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Market type: {args.market_type}")
    
    success = update_data(
        exchange_id=args.exchange,
        timeframe=args.timeframe,
        symbols=args.symbols,
        market_type=args.market_type
    )
    
    if success:
        print("✅ Update completed successfully!")
        return True
    else:
        print("❌ Update failed!")
        return False

def storage_info_command(args):
    """Handle storage info command"""
    from backtester.data.fetching.data_fetcher import get_storage_info, get_resampling_info
    
    print("📊 Storage Information")
    print("=" * 50)
    
    storage_info = get_storage_info()
    
    print(f"Storage Directory: {storage_info['storage_dir']}")
    print(f"Total Files: {storage_info['pickle_files']}")
    print(f"Total Size: {storage_info['total_size_mb']:.2f} MB")
    
    if args.detailed and 'files' in storage_info:
        print(f"\n📁 Detailed File Information:")
        for filename, info in storage_info['files'].items():
            print(f"\n  {filename}:")
            print(f"    Exchange: {info['exchange']}")
            print(f"    Market: {info['market_type']}")
            print(f"    Timeframe: {info['timeframe']}")
            print(f"    Size: {info['size_mb']:.2f} MB")
            print(f"    Symbols: {info['symbol_count']} ({', '.join(info['symbols'][:5])}{'...' if len(info['symbols']) > 5 else ''})")
            print(f"    Date Range: {info['date_range'][0]} → {info['date_range'][1]}")
    
    if args.resampling_info:
        print(f"\n🔄 Resampling Information:")
        resampling_info = get_resampling_info()
        print(f"Supported Timeframes: {', '.join(resampling_info['supported_timeframes'])}")
        print(f"Resampling Type: {resampling_info['resampling_type']}")
    
    return True

def test_command(args):
    """Handle test command"""
    print("🧪 Running Data Fetcher Tests")
    print("=" * 50)
    
    # Import our test script
    try:
        from test_data_fetcher import main as run_tests
        return run_tests()
    except ImportError:
        print("❌ Test script not found. Please ensure test_data_fetcher.py is in the current directory.")
        return False

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Data Fetcher CLI - Test and use the refactored data fetcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fetch BTCUSDT --timeframe 1d --start-date 2024-01-01 --end-date 2024-01-10
  %(prog)s fetch BTCUSDT ETHUSDT --exchange binance --timeframe 4h
  %(prog)s top-symbols --limit 5 --quote-currency USDT
  %(prog)s update --exchange binance --timeframe 1d
  %(prog)s storage-info --detailed
  %(prog)s test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--exchange', default='binance', 
                              help='Exchange ID (default: binance)')
    common_parser.add_argument('--timeframe', default='1d', 
                              help='Timeframe (default: 1d)')
    common_parser.add_argument('--market-type', default='spot', 
                              help='Market type (default: spot)')
    common_parser.add_argument('--no-cache', action='store_true', 
                              help='Disable caching')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', parents=[common_parser], 
                                        help='Fetch data for symbols')
    fetch_parser.add_argument('symbols', nargs='*', 
                             help='Symbols to fetch (default: BTCUSDT)')
    fetch_parser.add_argument('--start-date', 
                             help='Start date (YYYY-MM-DD or None for inception)')
    fetch_parser.add_argument('--end-date', 
                             help='End date (YYYY-MM-DD or None for latest)')
    fetch_parser.add_argument('--no-resampling', action='store_true', 
                             help='Disable resampling fallback')
    fetch_parser.add_argument('--show-data', action='store_true', 
                             help='Show sample data')
    
    # Top symbols command  
    top_parser = subparsers.add_parser('top-symbols', parents=[common_parser], 
                                      help='Fetch top symbols by volume')
    top_parser.add_argument('--limit', type=int, default=10, 
                           help='Number of top symbols (default: 10)')
    top_parser.add_argument('--quote-currency', default='USDT', 
                           help='Quote currency filter (default: USDT)')
    top_parser.add_argument('--start-date', 
                           help='Start date (YYYY-MM-DD)')
    top_parser.add_argument('--end-date', 
                           help='End date (YYYY-MM-DD)')
    top_parser.add_argument('--show-data', action='store_true', 
                           help='Show sample data')
    
    # Update command
    update_parser = subparsers.add_parser('update', parents=[common_parser], 
                                         help='Update cached data')
    update_parser.add_argument('symbols', nargs='*', 
                              help='Specific symbols to update (default: all)')
    
    # Storage info command
    storage_parser = subparsers.add_parser('storage-info', 
                                          help='Show storage information')
    storage_parser.add_argument('--detailed', action='store_true', 
                               help='Show detailed file information')
    storage_parser.add_argument('--resampling-info', action='store_true', 
                               help='Show resampling capabilities')
    
    # Test command
    subparsers.add_parser('test', help='Run test suite')
    
    return parser

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'fetch':
            success = fetch_command(args)
        elif args.command == 'top-symbols':
            success = top_symbols_command(args)
        elif args.command == 'update':
            success = update_command(args)
        elif args.command == 'storage-info':
            success = storage_info_command(args)
        elif args.command == 'test':
            success = test_command(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 