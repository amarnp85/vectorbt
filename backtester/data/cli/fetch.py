#!/usr/bin/env python3
"""CLI Tool for Data Fetching

This script provides command-line interface for fetching cryptocurrency data
using the VBT-native approach with support for spot/swap markets and
historical data from inception.

Examples:
    # Fetch top 10 spot symbols for 1h timeframe from inception
    python -m backtester.data.cli.fetch --exchange binance --market spot --timeframe 1h --top 10 --inception

    # Fetch specific symbols for swap market
    python -m backtester.data.cli.fetch --exchange binance --market swap --timeframe 4h --symbols BTC/USDT,ETH/USDT

    # Fetch top 5 symbols with date range
    python -m backtester.data.cli.fetch --exchange binance --top 5 --start "7 days ago" --end "1 day ago"
"""

import argparse
import sys
import os
from typing import List

from backtester.data import fetch_data, fetch_top_symbols, data_storage


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string."""
    return [s.strip() for s in symbols_str.split(",") if s.strip()]


def format_data_summary(data, exchange_id: str, market_type: str, timeframe: str):
    """Format and display data summary."""
    if data is None:
        print("❌ No data fetched")
        return

    print(f"\n✅ Data operation completed successfully!")
    print(f"📊 Exchange: {exchange_id.upper()}")
    print(f"📊 Market: {market_type.upper()}")
    print(f"📊 Timeframe: {timeframe}")
    print(f"📊 Symbols: {list(data.symbols)}")
    print(f"📊 Shape: {data.wrapper.shape}")

    if hasattr(data.wrapper, "index") and len(data.wrapper.index) > 0:
        print(f"📊 Date range: {data.wrapper.index[0]} to {data.wrapper.index[-1]}")
        print(f"📊 Data points per symbol: {len(data.wrapper.index)}")

        # Check if this looks like fresh data (recent end date)
        from datetime import datetime, timezone

        end_date = data.wrapper.index[-1]
        if hasattr(end_date, "tz_convert"):
            end_date = end_date.tz_convert("UTC")

        time_since_last = datetime.now(timezone.utc) - end_date.to_pydatetime()
        hours_since = time_since_last.total_seconds() / 3600

        if hours_since < 2:  # Very recent data
            print(
                f"📊 Data freshness: ✅ Very fresh (updated {hours_since:.1f} hours ago)"
            )
        elif hours_since < 24:  # Recent data
            print(f"📊 Data freshness: ✅ Fresh (updated {hours_since:.1f} hours ago)")
        else:  # Older data
            days_since = hours_since / 24
            print(
                f"📊 Data freshness: ⚠️ Older data (updated {days_since:.1f} days ago)"
            )

    # Volume rankings already shown during fetch process - no need to duplicate


def show_storage_summary():
    """Display current storage summary."""
    print(f"\n{'='*60}")
    print(f"💾 Storage Summary")
    print(f"{'='*60}")

    summary = data_storage.get_storage_summary()

    print(f"📂 Storage directory: {summary['storage_dir']}")
    print(f"📁 Total files: {summary['pickle_files']}")
    print(f"💾 Total size: {summary['total_size_mb']:.2f} MB")

    if summary["files"]:
        print(f"\n📋 Stored Data Files:")
        for filename, details in summary["files"].items():
            print(f"\n   📄 {filename}")
            print(f"      Exchange: {details['exchange'].upper()}")
            print(f"      Market: {details['market_type'].upper()}")
            print(f"      Timeframe: {details['timeframe']}")
            print(f"      Size: {details['size_mb']} MB")
            print(f"      Symbols: {details['symbol_count']} total")
            if details["symbols"]:
                print(f"      Sample: {', '.join(details['symbols'][:5])}")
            if details["date_range"]:
                print(
                    f"      Date range: {details['date_range'][0]} to {details['date_range'][1]}"
                )
    else:
        print(f"\n📭 No data files found")


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for fetching cryptocurrency data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --exchange binance --market spot --timeframe 1h --top 10 --inception
  %(prog)s --exchange binance --market swap --timeframe 4h --symbols BTC/USDT,ETH/USDT
  %(prog)s --exchange binance --top 5 --start "7 days ago" --end "1 day ago"
  %(prog)s --storage-summary
        """,
    )

    # Main operation mode
    parser.add_argument(
        "--storage-summary", action="store_true", help="Show storage summary and exit"
    )

    # Data selection
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., BTC/USDT,ETH/USDT)",
    )
    data_group.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top symbols by volume to fetch (default: 10)",
    )

    # Exchange and market settings
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange identifier (default: binance)",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["spot", "swap"],
        default="spot",
        help="Market type: spot or swap (default: spot)",
    )
    parser.add_argument(
        "--quote",
        type=str,
        default="USDT",
        help="Quote currency for top symbols filtering (default: USDT)",
    )

    # Time settings
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Timeframe (e.g., 1m, 5m, 1h, 4h, 1d) (default: 1d)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help='Start date (e.g., "7 days ago", "2024-01-01"). If not specified, uses cached inception dates.',
    )
    parser.add_argument(
        "--end", type=str, help='End date (e.g., "1 day ago", "2024-12-31")'
    )
    parser.add_argument(
        "--inception",
        action="store_true",
        help="Force fetch from inception (maximum available history) even if start/end dates are specified",
    )

    # Cache settings
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching (fetch fresh data)"
    )

    # Output settings
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (minimal output)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose mode (detailed output)"
    )

    args = parser.parse_args()

    # Handle storage summary mode
    if args.storage_summary:
        show_storage_summary()
        return

    # Validate arguments for fetch mode
    if not args.storage_summary and not args.symbols and not args.top:
        parser.error("Must specify either --symbols or --top")

    if args.inception and (args.start or args.end):
        parser.error("Cannot use --inception with --start or --end")

    # Set up date parameters
    start_date = None
    end_date = None

    if args.inception:
        # For inception, we'll use a very early date to get maximum history
        start_date = "2009-01-01"  # Bitcoin genesis
        # Don't override end_date here - let it use the default logic below
    elif args.start or args.end:
        # Only use provided dates if specifically given
        start_date = args.start
    else:
        # Default behavior: use cached inception dates (start_date = None triggers this)
        start_date = None

    # Default to "now" for end_date unless explicitly provided
    if args.end is not None:
        end_date = args.end
    else:
        end_date = "now"  # Always default to current time to ensure latest data

    if not args.quiet:
        print(f"🚀 Fetching Data")
        print(f"={'='*50}")
        print(f"Exchange: {args.exchange.upper()}")
        print(f"Market: {args.market.upper()}")
        print(f"Timeframe: {args.timeframe}")
        if args.symbols:
            symbols = parse_symbols(args.symbols)
            print(f"Symbols: {symbols}")
        else:
            print(f"Top symbols: {args.top} (quote: {args.quote})")
        if args.inception:
            print(f"Date range: FROM INCEPTION")
        elif start_date or end_date:
            print(f"Date range: {start_date or 'earliest'} to {end_date or 'latest'}")
        else:
            print(f"Date range: FROM CACHED INCEPTION to {end_date or 'latest'}")
        print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
        print(f"{'='*50}")

    try:
        # Fetch data
        if args.symbols:
            # Fetch specific symbols
            symbols = parse_symbols(args.symbols)
            if args.verbose:
                print(f"📡 Fetching {len(symbols)} specific symbols...")

            data = fetch_data(
                symbols=symbols,
                exchange_id=args.exchange,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=not args.no_cache,
                market_type=args.market,
            )
        else:
            # Fetch top symbols by volume
            if args.verbose:
                print(f"📡 Fetching top {args.top} symbols by volume...")

            data = fetch_top_symbols(
                exchange_id=args.exchange,
                quote_currency=args.quote,
                market_type=args.market,
                limit=args.top,
                timeframe=args.timeframe,
                start_date=start_date,
                end_date=end_date,
                use_cache=not args.no_cache,
            )

        if not args.quiet:
            format_data_summary(data, args.exchange, args.market, args.timeframe)

        if data is not None:
            print(f"\n💾 Data saved to VBT storage")
            if args.verbose:
                show_storage_summary()
            return 0
        else:
            print(f"\n❌ Failed to fetch data")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 