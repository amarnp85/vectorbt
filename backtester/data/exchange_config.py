#!/usr/bin/env python3
# backtester/data/exchange_config.py
"""
Simple utilities for working with cryptocurrency exchanges via VectorBT Pro.

IMPORTANT NOTES:
- This module provides informational helpers only
- API keys are NOT required for data fetching with VectorBT Pro
- VectorBT Pro is the primary library for all functionality
- Direct CCXT usage is limited to informational purposes only
"""

import logging
from typing import Dict, List, Any
import vectorbtpro as vbt

# Import ccxt for informational purposes only
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_available_exchanges() -> List[str]:
    """
    List all exchanges available for VectorBT Pro backtesting.

    This function uses CCXT for INFORMATIONAL PURPOSES ONLY to provide
    a list of exchanges that can be used with VectorBT Pro's CCXTData.

    Returns
    -------
    list
        List of available exchange IDs
    """
    # Use CCXT to get the list of available exchanges (informational only)
    return sorted(ccxt.exchanges)


def get_exchange_info(exchange_id: str) -> Dict[str, Any]:
    """
    Get informational details about an exchange.

    This function is for INFORMATIONAL PURPOSES ONLY and doesn't affect
    actual data fetching with VectorBT Pro.

    Parameters
    ----------
    exchange_id : str
        The ID of the exchange (e.g., 'binance', 'binanceusdm')

    Returns
    -------
    dict
        Information about the exchange's features and capabilities
    """
    # Validate exchange ID
    if exchange_id not in ccxt.exchanges:
        available = ", ".join(sorted(ccxt.exchanges[:10])) + "..."
        raise ValueError(
            f"Exchange '{exchange_id}' not found. Examples of available exchanges: {available}"
        )

    # Create exchange instance without any credentials (for information only)
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()

    # Extract helpful information
    info = {
        "id": exchange_id,
        "name": exchange.name if hasattr(exchange, "name") else exchange_id,
        "timeframes": getattr(exchange, "timeframes", {}),
        "has_fetchOHLCV": exchange.has.get("fetchOHLCV", False),
        "requires_api_key_for_data": False,  # Most exchanges allow data fetching without keys
    }

    return info


def get_exchange_timeframes(exchange_id: str) -> Dict[str, str]:
    """
    Get available timeframes for an exchange.

    Parameters
    ----------
    exchange_id : str
        The ID of the exchange (e.g., 'binance')

    Returns
    -------
    dict
        Dictionary of available timeframes
    """
    info = get_exchange_info(exchange_id)
    return info.get("timeframes", {})


def data_fetching_example(
    symbol: str = "BTC/USDT", exchange: str = "binance", timeframe: str = "1h"
) -> str:
    """
    Return example code for fetching data with VectorBT Pro.

    Parameters
    ----------
    symbol : str, default "BTC/USDT"
        Symbol to use in the example
    exchange : str, default "binance"
        Exchange to use in the example
    timeframe : str, default "1h"
        Timeframe to use in the example

    Returns
    -------
    str
        Example code snippet
    """
    return f"""
# Example for fetching OHLCV data with VectorBT Pro:
ohlcv = vbt.CCXTData.fetch(
    symbols=['{symbol}'],
    exchange='{exchange}',
    timeframe='{timeframe}',
    start='1 month ago'
)

# Process data with VectorBT Pro:
# vbt.OHLCV.from_data(ohlcv)

# Note: No API keys needed for data fetching!
"""


def is_futures_exchange(exchange_id: str) -> bool:
    """
    Check if an exchange ID is for a futures market.

    Parameters
    ----------
    exchange_id : str
        The exchange ID to check

    Returns
    -------
    bool
        True if it's a futures exchange, False otherwise
    """
    futures_exchanges = {
        "binanceusdm": True,  # USDT-margined futures on Binance
        "binancecoinm": True,  # COIN-margined futures on Binance
        "bitmex": True,
        "bybit": False,  # Bybit has both spot and futures
        "deribit": True,
        "ftx": False,  # FTX has both spot and futures
        "krakenfutures": True,
        "kucoinfutures": True,
    }

    return futures_exchanges.get(exchange_id, False)


# Command-line interface for direct usage
if __name__ == "__main__":
    import argparse
    import sys

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
