#!/usr/bin/env python3
"""
Simple Cache CLI

A streamlined CLI for essential cache operations.
Replaces the complex metadata_cli.py with focused, practical commands.

Commands:
- info: Show cache statistics
- clear: Clear cache data
- fetch: Fetch market data and cache it
- timestamps: Get inception timestamps for symbols

Usage:
    python -m backtester.data.cache_cli info
    python -m backtester.data.cache_cli clear --exchange binance
    python -m backtester.data.cache_cli fetch --exchange binance --limit 100
    python -m backtester.data.cache_cli timestamps --exchange binance BTC/USDT ETH/USDT
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any

from .cache_manager import cache_manager
from .metadata_fetcher import data_fetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cmd_info(args: argparse.Namespace) -> int:
    """Show cache information and statistics."""
    try:
        stats = cache_manager.get_cache_stats()
        
        print("\n=== Cache Statistics ===")
        print(f"Total exchanges: {len(stats['exchanges'])}")
        print(f"Volume symbols: {stats['volume_symbols']}")
        print(f"Timestamp symbols: {stats['timestamp_symbols']}")
        print(f"Failed symbols: {stats['failed_symbols']}")
        
        if stats['by_exchange']:
            print("\n=== By Exchange ===")
            for exchange, data in sorted(stats['by_exchange'].items()):
                print(f"\n{exchange.upper()}:")
                print(f"  Volume symbols: {data['volume_symbols']}")
                print(f"  Timestamp symbols: {data['timestamp_symbols']}")
                print(f"  Failed symbols: {data['failed_symbols']}")
                print(f"  Volume cache fresh: {'Yes' if data['volume_cache_fresh'] else 'No'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return 1

def cmd_clear(args: argparse.Namespace) -> int:
    """Clear cache data."""
    try:
        if args.exchange and args.type:
            desc = f"{args.type} cache for {args.exchange}"
        elif args.exchange:
            desc = f"all caches for {args.exchange}"
        elif args.type:
            desc = f"{args.type} cache for all exchanges"
        else:
            desc = "all caches"
        
        print(f"Clearing {desc}...")
        
        success = cache_manager.clear_cache(
            exchange_id=args.exchange,
            cache_type=args.type
        )
        
        if success:
            print(f"Successfully cleared {desc}")
            return 0
        else:
            print(f"Failed to clear {desc}")
            return 1
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return 1

def cmd_fetch(args: argparse.Namespace) -> int:
    """Fetch market data and cache it."""
    try:
        print(f"Fetching market data for {args.exchange}...")
        
        # Fetch market data
        market_data = data_fetcher.get_market_data(
            exchange_id=args.exchange,
            quote_currency=args.quote,
            market_types=args.market_types.split(',') if args.market_types else None,
            limit=args.limit,
            top_by_volume=True,
            use_cache=not args.force_refresh,
            force_refresh=args.force_refresh
        )
        
        if not market_data:
            print("No market data retrieved")
            return 1
        
        # Show summary
        symbols_with_volume = sum(1 for data in market_data.values() if 'volume' in data)
        print(f"\nRetrieved {len(market_data)} symbols")
        print(f"Symbols with volume: {symbols_with_volume}")
        
        # Show top symbols if requested
        if args.show_top:
            print(f"\nTop {min(args.show_top, len(market_data))} symbols by volume:")
            sorted_symbols = sorted(
                market_data.items(),
                key=lambda x: x[1].get('volume', 0),
                reverse=True
            )
            
            for i, (symbol, data) in enumerate(sorted_symbols[:args.show_top], 1):
                volume = data.get('volume', 'N/A')
                market_type = data.get('type', 'unknown')
                print(f"  {i}. {symbol} ({market_type}) - Volume: {volume}")
        
        # Save to file if requested
        if args.output:
            output_data = {
                'exchange': args.exchange,
                'timestamp': datetime.now().isoformat(),
                'filters': {
                    'quote_currency': args.quote,
                    'market_types': args.market_types.split(',') if args.market_types else None,
                    'limit': args.limit
                },
                'summary': {
                    'total_symbols': len(market_data),
                    'symbols_with_volume': symbols_with_volume
                },
                'data': market_data
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"Saved data to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return 1

def cmd_timestamps(args: argparse.Namespace) -> int:
    """Get inception timestamps for symbols."""
    try:
        # Determine which symbols to process
        if args.all:
            # Get all symbols that have volume data
            all_volumes = cache_manager.get_all_volumes(args.exchange)
            if not all_volumes:
                print(f"No volume data found for {args.exchange}")
                return 1
            symbols_to_process = list(all_volumes.keys())
            print(f"Fetching timestamps for ALL {len(symbols_to_process)} symbols with volume data on {args.exchange}...")
        elif args.symbols:
            symbols_to_process = args.symbols
            print(f"Getting timestamps for {len(args.symbols)} specified symbols on {args.exchange}...")
        else:
            print("No symbols specified. Use specific symbols or --all flag")
            return 1
        
        # Process in batches if we have many symbols
        if len(symbols_to_process) > args.batch_size:
            print(f"Processing {len(symbols_to_process)} symbols in batches of {args.batch_size}...")
            
            all_timestamps = {}
            total_processed = 0
            
            for i in range(0, len(symbols_to_process), args.batch_size):
                batch = symbols_to_process[i:i+args.batch_size]
                batch_num = (i // args.batch_size) + 1
                total_batches = (len(symbols_to_process) + args.batch_size - 1) // args.batch_size
                
                print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} symbols) ---")
                
                batch_timestamps = data_fetcher.get_inception_timestamps(
                    exchange_id=args.exchange,
                    symbols=batch,
                    concurrent_requests=args.concurrent,
                    use_cache=not args.force_refresh
                )
                
                all_timestamps.update(batch_timestamps)
                total_processed += len(batch_timestamps)
                
                print(f"Batch {batch_num} complete: {len(batch_timestamps)}/{len(batch)} successful")
                print(f"Total progress: {total_processed}/{len(symbols_to_process)} timestamps retrieved")
            
            timestamps = all_timestamps
        else:
            # Process all at once for smaller numbers
            timestamps = data_fetcher.get_inception_timestamps(
                exchange_id=args.exchange,
                symbols=symbols_to_process,
                concurrent_requests=args.concurrent,
                use_cache=not args.force_refresh
            )
        
        if not timestamps:
            print("No timestamps retrieved")
            return 1
        
        # Show summary
        print(f"\n=== FINAL RESULTS ===")
        print(f"Successfully retrieved {len(timestamps)} timestamps out of {len(symbols_to_process)} symbols")
        print(f"Success rate: {len(timestamps)/len(symbols_to_process)*100:.1f}%")
        
        # Show sample results (not all to avoid spam)
        sample_size = min(10, len(timestamps))
        if len(timestamps) > sample_size:
            print(f"\nShowing first {sample_size} results:")
        else:
            print(f"\nAll {len(timestamps)} results:")
            
        for i, (symbol, timestamp) in enumerate(list(timestamps.items())[:sample_size]):
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
            print(f"  {symbol}: {date} (timestamp: {timestamp})")
        
        if len(timestamps) > sample_size:
            print(f"  ... and {len(timestamps) - sample_size} more")
        
        # Show failed symbols
        failed_symbols = [s for s in symbols_to_process if s not in timestamps]
        if failed_symbols:
            print(f"\nFailed to retrieve {len(failed_symbols)} symbols:")
            for symbol in failed_symbols[:5]:  # Show first 5 failures
                print(f"  {symbol}")
            if len(failed_symbols) > 5:
                print(f"  ... and {len(failed_symbols) - 5} more")
        
        # Save to file if requested
        if args.output:
            output_data = {
                'exchange': args.exchange,
                'timestamp': datetime.now().isoformat(),
                'total_requested': len(symbols_to_process),
                'total_retrieved': len(timestamps),
                'success_rate': len(timestamps)/len(symbols_to_process)*100,
                'symbols_requested': symbols_to_process if not args.all else f"All {len(symbols_to_process)} symbols with volume data",
                'timestamps': {}
            }
            
            for symbol, timestamp in timestamps.items():
                output_data['timestamps'][symbol] = {
                    'timestamp': timestamp,
                    'date': datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')
                }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved {len(timestamps)} timestamps to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting timestamps: {e}")
        return 1

def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Simple Cache CLI for cryptocurrency exchange data'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show cache statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache data')
    clear_parser.add_argument(
        '--exchange', '-e',
        help='Exchange to clear (if not specified, clears all exchanges)'
    )
    clear_parser.add_argument(
        '--type', '-t',
        choices=['volume', 'timestamps', 'failed_symbols'],
        help='Cache type to clear (if not specified, clears all types)'
    )
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch and cache market data')
    fetch_parser.add_argument(
        '--exchange', '-e',
        required=True,
        help='Exchange to fetch data from'
    )
    fetch_parser.add_argument(
        '--quote', '-q',
        help='Filter by quote currency (e.g., USDT)'
    )
    fetch_parser.add_argument(
        '--market-types', '-m',
        help='Comma-separated market types (e.g., spot,swap)'
    )
    fetch_parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Maximum number of symbols to fetch'
    )
    fetch_parser.add_argument(
        '--force-refresh', '-f',
        action='store_true',
        help='Force refresh even if cache is fresh'
    )
    fetch_parser.add_argument(
        '--show-top', '-s',
        type=int,
        default=10,
        help='Show top N symbols by volume (default: 10)'
    )
    fetch_parser.add_argument(
        '--output', '-o',
        help='Save results to JSON file'
    )
    
    # Timestamps command
    timestamps_parser = subparsers.add_parser('timestamps', help='Get inception timestamps')
    timestamps_parser.add_argument(
        '--exchange', '-e',
        required=True,
        help='Exchange to get timestamps from'
    )
    timestamps_parser.add_argument(
        'symbols',
        nargs='*',
        help='Symbols to get timestamps for (optional if using --all)'
    )
    timestamps_parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Fetch timestamps for all symbols with volume data'
    )
    timestamps_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=50,
        help='Number of symbols to process at once (default: 50)'
    )
    timestamps_parser.add_argument(
        '--concurrent', '-c',
        type=int,
        default=5,
        help='Number of concurrent requests (default: 5)'
    )
    timestamps_parser.add_argument(
        '--force-refresh', '-f',
        action='store_true',
        help='Force refresh timestamps from API'
    )
    timestamps_parser.add_argument(
        '--output', '-o',
        help='Save results to JSON file'
    )
    
    return parser

def main() -> int:
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        if args.command == 'info':
            return cmd_info(args)
        elif args.command == 'clear':
            return cmd_clear(args)
        elif args.command == 'fetch':
            return cmd_fetch(args)
        elif args.command == 'timestamps':
            return cmd_timestamps(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 