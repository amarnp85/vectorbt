#!/usr/bin/env python3
"""Standalone Data Interpolation Script

Interpolate missing data points in VBT cached cryptocurrency data.

This script provides manual control over data interpolation with various
strategies appropriate for financial time series data.

Features:
- Multiple interpolation strategies (financial, linear, time-aware)
- Backup creation before modification
- Validation of interpolated data
- Support for specific exchanges/timeframes/markets

Usage:
    python -m backtester.data.cli.interpolate [options]

Examples:
    # Interpolate all Binance 5m data with financial strategy
    python -m backtester.data.cli.interpolate --exchange binance --timeframe 5m

    # Use linear interpolation for specific market
    python -m backtester.data.cli.interpolate --exchange binance --market spot --strategy linear

    # Dry run to see what would be interpolated
    python -m backtester.data.cli.interpolate --dry-run --exchange binance
"""

import argparse
import sys
import os

from backtester.data.storage.data_storage import data_storage
from backtester.data.health_check.data_healthcheck import StreamlinedHealthChecker


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate missing data points in VBT cached cryptocurrency data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interpolation Strategies:
  financial_forward_fill  OHLC uses last known close, Volume=0 (recommended for crypto)
  linear                  Linear interpolation between known points
  time_aware             Time-weighted interpolation considering gap duration

Examples:
  %(prog)s --exchange binance --timeframe 5m
  %(prog)s --exchange binance --strategy linear --backup
  %(prog)s --dry-run --exchange binance --market spot
        """,
    )

    parser.add_argument(
        "--exchange", type=str, required=True, help="Exchange to interpolate (required)"
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["spot", "swap"],
        help="Market type (default: all available)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Timeframe to interpolate (default: all available)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["financial_forward_fill", "linear", "time_aware"],
        default="financial_forward_fill",
        help="Interpolation strategy (default: financial_forward_fill)",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=10000,
        help="Maximum gap size to interpolate (default: 10000)",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before interpolation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be interpolated without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Interpolate even if no significant gaps found",
    )

    args = parser.parse_args()

    print("📊 VBT Data Interpolation Tool")
    print("=" * 50)
    print(f"Exchange: {args.exchange}")
    print(f"Strategy: {args.strategy}")
    print(f"Max gap size: {args.max_gap}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    print()

    # Get available data files
    available_data = data_storage.list_available_data()

    if not available_data:
        print("❌ No cached data found")
        return 1

    # Filter files based on arguments
    filtered_files = []
    for file_info in available_data:
        if file_info["exchange"] != args.exchange:
            continue
        if args.market and file_info["market"] != args.market:
            continue
        if args.timeframe and file_info["timeframe"] != args.timeframe:
            continue
        filtered_files.append(file_info)

    if not filtered_files:
        print(f"❌ No data files found for {args.exchange}")
        if args.market:
            print(f"   Market: {args.market}")
        if args.timeframe:
            print(f"   Timeframe: {args.timeframe}")
        return 1

    print(f"📁 Found {len(filtered_files)} data files to process")
    print()

    # Initialize health checker for interpolation functionality
    checker = StreamlinedHealthChecker(
        enable_interpolation=True, interpolation_strategy=args.strategy
    )

    processed_count = 0
    interpolated_count = 0

    for file_info in filtered_files:
        exchange = file_info["exchange"]
        market = file_info["market"]
        timeframe = file_info["timeframe"]

        print(f"🔍 Processing {exchange}/{market}/{timeframe}...")

        try:
            # Load data
            data = data_storage.load_data(exchange, timeframe, market_type=market)

            if data is None:
                print(f"   ❌ Cannot load data file")
                continue

            # Check for gaps
            gap_info = checker.check_data_gaps(data, exchange, market, timeframe)
            total_missing = gap_info.get("total_missing", 0)

            if total_missing == 0:
                print(f"   ✅ No gaps found")
                processed_count += 1
                continue

            if total_missing > args.max_gap:
                print(
                    f"   ⚠️  Gap too large: {total_missing} > {args.max_gap} (skipping)"
                )
                processed_count += 1
                continue

            if not args.force and total_missing < 100:
                print(
                    f"   ℹ️  Minor gaps: {total_missing} periods (use --force to interpolate)"
                )
                processed_count += 1
                continue

            print(f"   📊 Found {total_missing} missing periods")

            if args.dry_run:
                print(f"   🔄 Would interpolate using {args.strategy} strategy")
            else:
                # Create backup if requested
                if args.backup:
                    backup_path = data_storage.backup_data_file(
                        exchange, timeframe, market
                    )
                    if backup_path:
                        print(f"   💾 Backup created: {backup_path}")

                # Perform interpolation
                interpolated_data = checker.interpolate_missing_data(
                    data, exchange, market, timeframe, args.strategy
                )

                if interpolated_data is not None:
                    # Save interpolated data
                    success = data_storage.save_data(
                        interpolated_data, exchange, timeframe, market
                    )

                    if success:
                        print(f"   ✅ Interpolation complete and saved")
                        interpolated_count += 1
                    else:
                        print(f"   ❌ Failed to save interpolated data")
                else:
                    print(f"   ❌ Interpolation failed")

            processed_count += 1

        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            continue

    print()
    print("=" * 50)
    print("📊 INTERPOLATION SUMMARY")
    print(f"Files processed: {processed_count}")

    if args.dry_run:
        print("Mode: Dry run (no changes made)")
    else:
        print(f"Files interpolated: {interpolated_count}")

        if interpolated_count > 0:
            print()
            print("✅ Interpolation complete!")
            print("   Run health check to verify improvements:")
            print(
                f"   python -m backtester.data.health_check.data_healthcheck --exchange {args.exchange}"
            )
        elif processed_count > 0:
            print("ℹ️  No files required interpolation")
        else:
            print("❌ No files were processed successfully")

    return 0


if __name__ == "__main__":
    exit(main()) 