#!/usr/bin/env python3
# backtester/utilities/exchange_info.py
"""
Command-line utility for working with exchange information.

This script provides a convenient interface to the exchange_config module.
"""

import argparse
import sys
import logging
from backtester.data.exchange_config import (
    list_available_exchanges,
    get_exchange_info,
    get_exchange_timeframes,
    is_futures_exchange,
    data_fetching_example,
)
import vectorbtpro as vbt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="VectorBT Pro Exchange Utilities")
    parser.add_argument(
        "--list", action="store_true", help="List all available exchanges"
    )
    parser.add_argument(
        "--info", metavar="EXCHANGE", help="Get information about an exchange"
    )
    parser.add_argument(
        "--timeframes", metavar="EXCHANGE", help="List timeframes for an exchange"
    )
    parser.add_argument(
        "--example",
        metavar="EXCHANGE",
        help="Show data fetching example for an exchange",
    )
    parser.add_argument(
        "--test",
        metavar="EXCHANGE",
        help="Test if data can be fetched from an exchange",
    )
    args = parser.parse_args()

    # Print banner
    print(
        """\n🔍 VectorBT Pro Exchange Info\n
    IMPORTANT NOTES:
    - API keys are NOT required for data fetching
    - VectorBT Pro is the primary library for all functionality\n"""
    )

    # Handle list command
    if args.list:
        exchanges = list_available_exchanges()
        print(f"Available exchanges for VectorBT Pro ({len(exchanges)}):\n")
        for i, exchange in enumerate(exchanges):
            print(f"  {exchange}", end=", " if (i + 1) % 5 != 0 else "\n")
        print("\n")

    # Handle info command
    elif args.info:
        try:
            info = get_exchange_info(args.info)
            print(f"Exchange: {info['name']} ({info['id']})")
            print(
                f"Supports OHLCV data fetching: {'Yes' if info['has_fetchOHLCV'] else 'No'}"
            )
            print(
                f"Requires API keys for data fetching: {info['requires_api_key_for_data']}"
            )
            print(f"Available timeframes: {len(info['timeframes'])}")
            print(f"Futures exchange: {is_futures_exchange(args.info)}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Handle timeframes command
    elif args.timeframes:
        try:
            timeframes = get_exchange_timeframes(args.timeframes)
            print(f"Timeframes for {args.timeframes}:")
            for tf in sorted(timeframes.keys()):
                print(f"  {tf}")
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Handle example command
    elif args.example:
        try:
            example = data_fetching_example(exchange=args.example)
            print(f"Example for fetching data from {args.example}:\n")
            print(example)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Handle test command
    elif args.test:
        try:
            print(f"Testing connection to {args.test}...")
            print(
                "Attempting to fetch a small amount of data using VectorBT Pro CCXTData..."
            )

            # Simple test - try to fetch minimal data
            symbol = "BTC/USDT" if not is_futures_exchange(args.test) else "BTC/USD"
            print(f"Using symbol: {symbol}")

            # This just verifies we can connect - we don't actually process the data
            try:
                ohlcv = vbt.CCXTData.fetch(
                    symbols=[symbol],
                    exchange=args.test,
                    timeframe="1d",  # Daily timeframe for minimal data
                    start="1 day ago",  # Minimal timeframe
                )
                print(f"✅ Success! Successfully fetched data from {args.test}")
                print(
                    f"Data shape: {ohlcv.shape if hasattr(ohlcv, 'shape') else 'N/A'}"
                )
            except Exception as e:
                print(f"❌ Error fetching data: {str(e)}")
                sys.exit(1)
        except Exception as e:
            print(f"Error: {str(e)}")
            sys.exit(1)

    # Show help if no arguments provided
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
