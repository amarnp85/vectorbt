#!/usr/bin/env python3
"""
Data refresh CLI tool for updating cached data.

This module provides command-line functionality to refresh cached data
for specified symbols, timeframes, and exchanges.
"""

import argparse
import sys
from typing import List, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Use the simplified interface
from backtester.data import fetch_data
from backtester.data.storage.data_storage import data_storage
from backtester.utilities.structured_logging import setup_logging

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Refresh cached cryptocurrency data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh all cached data for Binance daily timeframe
  python -m backtester.data.cli.refresh --exchange binance --timeframe 1d
  
  # Refresh specific symbols
  python -m backtester.data.cli.refresh --symbols BTC/USDT ETH/USDT --timeframe 1h
  
  # Refresh with custom date range
  python -m backtester.data.cli.refresh --symbols BTC/USDT --start 2024-01-01 --end 2024-03-01
        """
    )
    
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        help='Symbols to refresh (e.g., BTC/USDT ETH/USDT). If not specified, refreshes all cached symbols.'
    )
    
    parser.add_argument(
        '--exchange', '-e',
        default='binance',
        choices=['binance', 'bybit', 'hyperliquid'],
        help='Exchange to refresh data from (default: binance)'
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        default='1d',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Timeframe to refresh (default: 1d)'
    )
    
    parser.add_argument(
        '--market-type', '-m',
        default='spot',
        choices=['spot', 'swap'],
        help='Market type (default: spot)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date for refresh (YYYY-MM-DD). If not specified, keeps existing start date.'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date for refresh (YYYY-MM-DD). If not specified, uses current date.'
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        help='Number of days back from today to refresh. Overrides --start if specified.'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if data appears up-to-date'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be refreshed without actually doing it'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def get_cached_symbols(exchange: str, timeframe: str, market_type: str) -> List[str]:
    """Get list of symbols that are currently cached."""
    try:
        # Load cached data to get symbols
        cached_data = data_storage.load_data(exchange, timeframe, None, market_type)
        if cached_data is not None:
            return list(cached_data.symbols)
    except Exception as e:
        logger.warning(f"Could not load cached symbols: {e}")
    
    return []


def refresh_data(
    symbols: List[str],
    exchange: str,
    timeframe: str,
    market_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """
    Refresh data for specified symbols.
    
    Args:
        symbols: List of symbols to refresh
        exchange: Exchange ID
        timeframe: Timeframe string
        market_type: Market type (spot/swap)
        start_date: Start date for data
        end_date: End date for data
        force: Force refresh even if up-to-date
        dry_run: Show what would be done without doing it
        
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"DRY RUN: Would refresh {len(symbols)} symbols")
        for symbol in symbols:
            logger.info(f"  - {symbol}")
        return True
    
    logger.info(f"Refreshing {len(symbols)} symbols from {exchange} {timeframe}")
    
    try:
        # Fetch data using the simplified interface
        data = fetch_data(
            symbols=symbols,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            market_type=market_type
        )
        
        if data is not None:
            logger.info(f"Successfully refreshed data for {len(data.symbols)} symbols")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            return True
        else:
            logger.error("Failed to refresh data")
            return False
            
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        return False


def main():
    """Main entry point for the refresh CLI."""
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging("data_refresh", log_level=log_level)
    
    # Determine symbols to refresh
    if args.symbols:
        symbols = args.symbols
        logger.info(f"Refreshing specified symbols: {symbols}")
    else:
        # Get all cached symbols
        symbols = get_cached_symbols(args.exchange, args.timeframe, args.market_type)
        if not symbols:
            logger.warning("No cached symbols found to refresh")
            return 1
        logger.info(f"Found {len(symbols)} cached symbols to refresh")
    
    # Determine date range
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')
    
    if args.days_back:
        start_date = (datetime.now() - timedelta(days=args.days_back)).strftime('%Y-%m-%d')
        logger.info(f"Refreshing last {args.days_back} days of data")
    else:
        start_date = args.start
    
    # Refresh data
    success = refresh_data(
        symbols=symbols,
        exchange=args.exchange,
        timeframe=args.timeframe,
        market_type=args.market_type,
        start_date=start_date,
        end_date=end_date,
        force=args.force,
        dry_run=args.dry_run
    )
    
    if success:
        logger.info("✅ Data refresh completed successfully")
        return 0
    else:
        logger.error("❌ Data refresh failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 