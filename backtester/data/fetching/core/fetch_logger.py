#!/usr/bin/env python3
"""Fetch Logger - Structured logging for fetch operations.

This module provides clean, structured logging for data fetch operations,
tracking the flow and results of each step with detailed tables and validation.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import vectorbtpro as vbt

logger = logging.getLogger(__name__)
console = Console()


class FetchLogger:
    """Track and display fetch operation state with detailed progress and validation."""

    def __init__(self, symbols: List[str], exchange_id: str, timeframe: str):
        """Initialize fetch logger."""
        self.requested_symbols = symbols
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.filtered_symbols = []
        self.operation_path = []
        self.cache_state = {}
        self.final_result = None
        self.start_time = time.time()
        self.symbol_progress = {}  # Track per-symbol status
        self.current_operation = None
        self.fetch_stages = []  # Track each stage of the fetch

    def _get_utc_time(self) -> str:
        """Get current UTC time as formatted string."""
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def log_start(self, start_date: Optional[str], end_date: Optional[str]):
        """Log the start of a fetch operation."""
        self.start_time = time.time()
        date_range = f"{start_date or 'inception'} → {end_date or 'now'}"
        utc_time = self._get_utc_time()

        console.print(
            Panel(
                f"[bold cyan]{self.exchange_id.upper()}[/bold cyan] | "
                f"[cyan]{self.timeframe}[/cyan] | "
                f"[cyan]{len(self.requested_symbols)} symbols[/cyan] | "
                f"[dim]{date_range}[/dim]\n"
                f"[dim]Started: {utc_time}[/dim]",
                title="📊 Data Fetch",
                expand=False,
                style="cyan",
            )
        )

        # Log requested symbols for debugging
        if len(self.requested_symbols) <= 10:
            symbols_str = ", ".join(self.requested_symbols)
            logger.debug(f"[{utc_time}] Requested symbols: {symbols_str}")
        else:
            logger.debug(
                f"[{utc_time}] Requested {len(self.requested_symbols)} symbols: {self.requested_symbols[:5]}..."
            )

    def log_filter_result(self, filtered_symbols: List[str]):
        """Log blacklist filtering results."""
        self.filtered_symbols = filtered_symbols

        # Initialize symbol progress tracking
        for symbol in filtered_symbols:
            self.symbol_progress[symbol] = {
                "status": "pending",
                "source": None,
                "time": None,
            }

    def log_cache_result(self, status: str, details: Dict[str, Any]):
        """Log cache check results with detailed symbol breakdown."""
        utc_time = self._get_utc_time()
        self.cache_state = details
        self.operation_path.append(("cache", status))

        # Create cache results table
        cache_table = Table(title=f"💾 Cache Analysis ({utc_time})", box=box.ROUNDED)
        cache_table.add_column("Category", style="cyan")
        cache_table.add_column("Count", justify="right")
        cache_table.add_column("Symbols", style="dim")
        cache_table.add_column("Action", style="yellow")

        if status == "hit":
            cache_table.add_row(
                "✅ Found",
                str(len(self.filtered_symbols)),
                ", ".join(self.filtered_symbols[:3])
                + ("..." if len(self.filtered_symbols) > 3 else ""),
                "Use cached data",
            )
            # Mark all symbols as cached
            for symbol in self.filtered_symbols:
                self.symbol_progress[symbol] = {
                    "status": "cached",
                    "source": "cache",
                    "time": 0,
                }

        elif status == "partial":
            details.get("available", 0)
            details.get("missing", 0)
            details.get("stale", 0)

            cached_symbols = details.get("cached_symbols", set())
            missing_symbols = details.get("missing_symbols", set())
            stale_symbols = details.get("stale_symbols", set())

            if cached_symbols:
                cache_table.add_row(
                    "✅ Fresh",
                    str(len(cached_symbols)),
                    ", ".join(list(cached_symbols)[:3])
                    + ("..." if len(cached_symbols) > 3 else ""),
                    "Use from cache",
                )
                for symbol in cached_symbols:
                    if symbol in self.symbol_progress:
                        self.symbol_progress[symbol] = {
                            "status": "cached",
                            "source": "cache",
                            "time": 0,
                        }

            if stale_symbols:
                cache_table.add_row(
                    "🔄 Stale",
                    str(len(stale_symbols)),
                    ", ".join(list(stale_symbols)[:3])
                    + ("..." if len(stale_symbols) > 3 else ""),
                    "Update recent data",
                )

            if missing_symbols:
                cache_table.add_row(
                    "❌ Missing",
                    str(len(missing_symbols)),
                    ", ".join(list(missing_symbols)[:3])
                    + ("..." if len(missing_symbols) > 3 else ""),
                    "Fetch from inception",
                )

        else:
            cache_table.add_row(
                "❌ Miss", "0", "No cached data found", "Fetch all from exchange"
            )

        console.print(cache_table)

        # Record fetch stage
        self.fetch_stages.append(
            {
                "stage": "cache",
                "status": status,
                "utc_time": utc_time,
                "details": details,
            }
        )

    def log_resample_result(self, success: bool, source_tf: Optional[str] = None):
        """Log resampling attempt results."""
        utc_time = self._get_utc_time()
        self.operation_path.append(("resample", "success" if success else "fail"))

        # Create resample results table
        resample_table = Table(
            title=f"🔄 Resample Analysis ({utc_time})", box=box.ROUNDED
        )
        resample_table.add_column("Result", style="cyan")
        resample_table.add_column("Details")
        resample_table.add_column("Action", style="yellow")

        if success:
            resample_table.add_row(
                "✅ Success",
                (
                    f"Resampled from {source_tf}"
                    if source_tf
                    else "Resampled data available"
                ),
                "Use resampled data",
            )
            logger.info(
                f"[{utc_time}] Successfully resampled from {source_tf} timeframe"
            )

            # Mark symbols as resampled
            for symbol in self.filtered_symbols:
                if self.symbol_progress[symbol]["status"] == "pending":
                    self.symbol_progress[symbol] = {
                        "status": "resampled",
                        "source": f"resample_{source_tf}",
                        "time": 0,
                    }
        else:
            resample_table.add_row(
                "⏭️ Skip", "No suitable lower timeframe data", "Continue to exchange"
            )
            logger.debug(
                f"[{utc_time}] No suitable lower timeframe data for resampling"
            )

        console.print(resample_table)

        # Record fetch stage
        self.fetch_stages.append(
            {
                "stage": "resample",
                "status": "success" if success else "skip",
                "utc_time": utc_time,
                "source_tf": source_tf,
            }
        )

    def log_exchange_start(self, symbols: List[str]):
        """Log the start of exchange data fetching."""
        utc_time = self._get_utc_time()
        self.current_operation = "exchange"

        # Create exchange start table
        exchange_table = Table(
            title=f"🔄 Exchange Fetch Starting ({utc_time})", box=box.ROUNDED
        )
        exchange_table.add_column("Metric", style="cyan")
        exchange_table.add_column("Value")
        exchange_table.add_column("Details", style="dim")

        exchange_table.add_row(
            "Symbols",
            str(len(symbols)),
            ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
        )
        exchange_table.add_row("Exchange", self.exchange_id.upper(), "Direct API fetch")
        exchange_table.add_row("Timeframe", self.timeframe, "Target resolution")
        exchange_table.add_row("Started", utc_time, "UTC timestamp")

        console.print(exchange_table)
        logger.info(
            f"[{utc_time}] Starting exchange fetch for {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}"
        )

        # Track operation start time
        for symbol in symbols:
            if symbol in self.symbol_progress:
                self.symbol_progress[symbol]["start_time"] = time.time()

    def log_exchange_result(
        self,
        success: bool,
        symbol_count: Optional[int] = None,
        fetch_time: Optional[float] = None,
    ):
        """Log exchange fetch results with detailed symbol breakdown."""
        utc_time = self._get_utc_time()
        self.operation_path.append(("exchange", "success" if success else "fail"))

        # Create exchange results table
        exchange_table = Table(
            title=f"📊 Exchange Results ({utc_time})", box=box.ROUNDED
        )
        exchange_table.add_column("Metric", style="cyan")
        exchange_table.add_column("Value", justify="right")
        exchange_table.add_column("Details")

        if success:
            exchange_table.add_row(
                "✅ Status", "Success", "Data retrieved successfully"
            )
            if symbol_count:
                exchange_table.add_row(
                    "📈 Symbols", str(symbol_count), "Successfully fetched"
                )
            if fetch_time:
                exchange_table.add_row(
                    "⏱️ Duration", f"{fetch_time:.1f}s", "Total fetch time"
                )

            # Show symbol breakdown if we have detailed progress
            self._add_symbol_summary_to_table(exchange_table)

        else:
            exchange_table.add_row("❌ Status", "Failed", "Exchange fetch unsuccessful")
            if fetch_time:
                exchange_table.add_row(
                    "⏱️ Duration", f"{fetch_time:.1f}s", "Time before failure"
                )
            self._add_symbol_summary_to_table(exchange_table, show_failures=True)

        console.print(exchange_table)

        # Record fetch stage
        self.fetch_stages.append(
            {
                "stage": "exchange",
                "status": "success" if success else "fail",
                "utc_time": utc_time,
                "symbol_count": symbol_count,
                "fetch_time": fetch_time,
            }
        )

    def _add_symbol_summary_to_table(self, table: Table, show_failures: bool = False):
        """Add symbol breakdown to results table."""
        if not self.symbol_progress:
            return

        successful = sum(
            1 for s in self.symbol_progress.values() if s["status"] == "success"
        )
        cached = sum(
            1 for s in self.symbol_progress.values() if s["status"] == "cached"
        )
        resampled = sum(
            1 for s in self.symbol_progress.values() if s["status"] == "resampled"
        )
        failed = sum(
            1 for s in self.symbol_progress.values() if s["status"] == "failed"
        )

        if cached > 0:
            table.add_row("💾 Cached", str(cached), "From existing cache")
        if resampled > 0:
            table.add_row("🔄 Resampled", str(resampled), "From lower timeframes")
        if successful > 0:
            table.add_row("🌐 Fetched", str(successful), "From exchange API")
        if failed > 0:
            table.add_row("❌ Failed", str(failed), "Could not retrieve", style="red")

            # Log failed symbols if requested
            if show_failures:
                failed_symbols = [
                    sym
                    for sym, data in self.symbol_progress.items()
                    if data["status"] == "failed"
                ]
                logger.error(f"Failed symbols: {failed_symbols}")

    def log_symbol_progress(
        self, completed: int, total: int, current_symbol: Optional[str] = None
    ):
        """Log progress during symbol fetching."""
        utc_time = self._get_utc_time()

        if current_symbol:
            console.print(
                f"      📈 Progress ({utc_time}): {completed}/{total} symbols | Current: {current_symbol}"
            )
        else:
            console.print(
                f"      📈 Progress ({utc_time}): {completed}/{total} symbols"
            )

        # Update symbol status
        if current_symbol and current_symbol in self.symbol_progress:
            self.symbol_progress[current_symbol]["status"] = "fetching"

    def log_symbol_success(self, symbol: str, data_points: Optional[int] = None):
        """Log successful fetch for a specific symbol."""
        if symbol in self.symbol_progress:
            start_time = self.symbol_progress[symbol].get("start_time", time.time())
            fetch_time = time.time() - start_time
            self.symbol_progress[symbol].update(
                {
                    "status": "success",
                    "source": "exchange",
                    "time": fetch_time,
                    "data_points": data_points,
                }
            )

        msg = f"      ✅ {symbol}: "
        if data_points:
            msg += f"{data_points} data points"
        else:
            msg += "fetched"
        logger.debug(msg)

    def log_symbol_failure(self, symbol: str, error: str):
        """Log failed fetch for a specific symbol."""
        if symbol in self.symbol_progress:
            start_time = self.symbol_progress[symbol].get("start_time", time.time())
            fetch_time = time.time() - start_time
            self.symbol_progress[symbol].update(
                {
                    "status": "failed",
                    "source": "exchange",
                    "time": fetch_time,
                    "error": error,
                }
            )

        console.print(f"      ❌ {symbol}: [red]{error}[/red]")
        logger.warning(f"Failed to fetch {symbol}: {error}")

    def log_final_result(self, data: Optional[vbt.Data], saved: bool = False):
        """Log the final result with comprehensive validation."""
        utc_time = self._get_utc_time()
        self.final_result = data
        total_time = time.time() - self.start_time

        if data is None:
            console.print(
                Panel(
                    f"[red]Fetch failed[/red] - No data retrieved\n"
                    f"[dim]Completed: {utc_time}[/dim]",
                    style="red",
                    expand=False,
                )
            )
            logger.error(f"[{utc_time}] Data fetch failed after {total_time:.1f}s")
            return

        # Determine if this was a cache hit (no actual fetching needed)
        was_cache_hit = (
            len(self.operation_path) == 1
            and self.operation_path[0][0] == "cache"
            and self.operation_path[0][1] == "hit"
        )

        # Create comprehensive results table
        results_table = Table(title=f"🎯 Final Results ({utc_time})", box=box.ROUNDED)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", justify="right")
        results_table.add_column("Details", style="dim")

        # Basic metrics
        symbol_count = len(data.symbols) if hasattr(data, "symbols") else 0

        if was_cache_hit:
            results_table.add_row(
                "✅ Status", "Cache Hit", "No fetch needed - data already fresh"
            )
        else:
            results_table.add_row("✅ Status", "Success", "Data fetch completed")

        results_table.add_row(
            "📊 Symbols",
            str(symbol_count),
            ", ".join(data.symbols[:3]) + ("..." if symbol_count > 3 else ""),
        )
        results_table.add_row(
            "⏱️ Total Time", f"{total_time:.1f}s", "End-to-end duration"
        )
        results_table.add_row(
            "🔄 Path", self._get_path_description(), "Data source strategy"
        )

        if saved:
            results_table.add_row("💾 Cached", "Yes", "Saved for future use")

        # Data validation
        try:
            if hasattr(data, "index") and len(data.index) > 0:
                start_date = data.index[0].strftime("%Y-%m-%d %H:%M")
                end_date = data.index[-1].strftime("%Y-%m-%d %H:%M")
                total_candles = len(data.index)

                results_table.add_row("📅 Start Date", start_date, "First data point")
                results_table.add_row("📅 End Date", end_date, "Last data point")
                results_table.add_row(
                    "📊 Candles", str(total_candles), "Total data points"
                )

                # Validate data completeness
                validation_result = self._validate_data_completeness(data)
                if validation_result["complete"]:
                    results_table.add_row(
                        "✅ Completeness",
                        "Good",
                        f"Expected: {validation_result['expected']}, Found: {validation_result['actual']}",
                    )
                else:
                    results_table.add_row(
                        "⚠️ Completeness",
                        "Issues",
                        f"Missing {validation_result['missing']} candles",
                    )
        except Exception as e:
            results_table.add_row(
                "❌ Validation", "Error", f"Could not validate: {str(e)}"
            )

        console.print(results_table)

        # Show fetch stages summary
        self._log_fetch_stages_summary(utc_time)

        if was_cache_hit:
            logger.info(
                f"[{utc_time}] Cache hit completed in {total_time:.1f}s: {symbol_count} symbols (no fetch needed)"
            )
        else:
            logger.info(
                f"[{utc_time}] Data fetch completed in {total_time:.1f}s: {symbol_count} symbols via {self._get_path_description()}"
            )

        # Log performance metrics if we have detailed tracking
        if self.symbol_progress:
            # Fix: Properly handle None values in time calculations
            valid_times = []
            for s in self.symbol_progress.values():
                time_val = s.get("time")
                if (
                    time_val is not None
                    and isinstance(time_val, (int, float))
                    and time_val > 0
                ):
                    valid_times.append(time_val)

            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                logger.debug(
                    f"[{utc_time}] Average fetch time per symbol: {avg_time:.2f}s"
                )
                logger.debug(
                    f"[{utc_time}] Performance calculated from {len(valid_times)}/{len(self.symbol_progress)} symbols"
                )

    def _validate_data_completeness(self, data: vbt.Data) -> Dict[str, Any]:
        """Validate data completeness and continuity."""
        try:
            from .freshness_checker import FreshnessChecker

            if not hasattr(data, "index") or len(data.index) == 0:
                return {"complete": False, "error": "No data index"}

            # Calculate expected number of candles
            start_time = data.index[0]
            end_time = data.index[-1]
            tf_minutes = FreshnessChecker.parse_timeframe_minutes(self.timeframe)

            expected_periods = (
                int((end_time - start_time).total_seconds() / (tf_minutes * 60)) + 1
            )
            actual_periods = len(data.index)

            # Allow for some tolerance (market closures, etc.)
            tolerance = max(1, int(expected_periods * 0.05))  # 5% tolerance
            is_complete = abs(expected_periods - actual_periods) <= tolerance

            return {
                "complete": is_complete,
                "expected": expected_periods,
                "actual": actual_periods,
                "missing": max(0, expected_periods - actual_periods),
                "tolerance": tolerance,
            }

        except Exception as e:
            return {"complete": False, "error": str(e)}

    def _log_fetch_stages_summary(self, utc_time: str):
        """Log a summary of all fetch stages."""
        if not self.fetch_stages:
            return

        stages_table = Table(title="📋 Fetch Stages Summary", box=box.ROUNDED)
        stages_table.add_column("Stage", style="cyan")
        stages_table.add_column("Result")
        stages_table.add_column("Time", style="dim")
        stages_table.add_column("Details")

        for stage in self.fetch_stages:
            stage_name = stage["stage"].title()
            status = stage["status"]
            stage_time = stage["utc_time"]

            if status == "success" or status == "hit":
                result = "✅ Success"
            elif status == "skip":
                result = "⏭️ Skipped"
            elif status == "partial":
                result = "🔍 Partial"
            else:
                result = "❌ Failed"

            details = ""
            if stage["stage"] == "cache" and "details" in stage:
                d = stage["details"]
                if "available" in d:
                    available = d.get("available", 0) or 0
                    missing = d.get("missing", 0) or 0
                    stale = d.get("stale", 0) or 0
                    details = f"{available} cached, {missing} missing, {stale} stale"
                else:
                    symbols_count = d.get("symbols", 0) or 0
                    details = f"{symbols_count} symbols"
            elif stage["stage"] == "resample" and "source_tf" in stage:
                source_tf = stage.get("source_tf")
                if source_tf is not None:
                    details = f"From {source_tf}"
                else:
                    details = "Source timeframe unknown"
            elif stage["stage"] == "exchange" and "symbol_count" in stage:
                symbol_count = stage.get("symbol_count")
                if symbol_count is not None:
                    details = f"{symbol_count} symbols"
                    fetch_time = stage.get("fetch_time")
                    if fetch_time is not None:
                        details += f" in {fetch_time:.1f}s"
                else:
                    details = "Symbol count unknown"

            stages_table.add_row(stage_name, result, stage_time, details)

        console.print(stages_table)

    def _get_path_description(self) -> str:
        """Get a description of the fetch path taken."""
        if not self.operation_path:
            return "unknown path"

        # Find the successful operation
        for source, status in reversed(self.operation_path):
            if status == "success" or status == "hit":
                if source == "cache" and status == "hit":
                    return "from cache"
                elif source == "cache" and "partial" in str(self.cache_state):
                    return "mixed sources"
                elif source == "resample":
                    return "resampled"
                elif source == "exchange":
                    return "from exchange"

        return "fetch failed"
