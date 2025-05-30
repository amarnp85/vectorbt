#!/usr/bin/env python3
"""
Structured Logging System for Backtester

A complete overhaul of the logging system designed to provide clean, structured,
and informative output without the massive repetition and verbosity of the current system.

Key Features:
- Clean, structured output using Rich tables and panels
- Context-aware logging that groups related operations
- Progress tracking for long-running operations
- Intelligent log level management with smart deduplication
- Performance metrics tracking
- Minimal repetition with smart deduplication
- Beautiful console output with colors and formatting

This system replaces the current enhanced_logging.py with a fundamentally
different approach focused on user experience and information density.
"""

import sys
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.align import Align


# Global console instance
console = Console()

# Configure loguru to be less verbose
logger.remove()
logger.add(
    sys.stderr,
    level="WARNING",  # Only show warnings and errors in console
    format="<red>{level}</red>: {message}",
    colorize=True
)


@dataclass
class OperationContext:
    """Context for tracking operations."""
    name: str
    start_time: float = field(default_factory=time.time)
    total_items: Optional[int] = None
    completed_items: int = 0
    status: str = "running"
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time
    
    @property
    def duration_str(self) -> str:
        duration = self.duration
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"


class StructuredLogger:
    """
    Main structured logger that provides clean, organized output.
    
    This logger focuses on:
    1. Grouping related operations
    2. Showing progress for long-running tasks
    3. Providing summary information
    4. Minimizing repetitive output
    """
    
    def __init__(self, name: str = "backtester", level: str = "INFO"):
        """Initialize the structured logger."""
        self.name = name
        self.level = level
        self.console = console
        self._operation_stack: List[OperationContext] = []
        self._lock = threading.Lock()
        self._last_progress_update = 0
        self._progress_throttle = 0.1  # Update progress at most every 100ms
        
        # Configure loguru for file logging (minimal)
        log_file = Path("logs") / f"{name}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz"
        )
    
    def section(self, title: str, subtitle: str = None):
        """Start a new section with a clear header."""
        # Don't show sections in quiet mode
        if self.level == "ERROR":
            return
            
        if subtitle:
            content = f"[dim]{subtitle}[/dim]"
            panel = Panel(content, title=f"[bold cyan]{title}[/bold cyan]", 
                         border_style="cyan", padding=(0, 1))
        else:
            panel = Panel(f"[bold cyan]{title}[/bold cyan]", border_style="cyan")
        
        self.console.print()
        self.console.print(panel)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        import os
        if os.environ.get('QUIET_MODE') == '1':
            return
        if self.level != "ERROR":
            self.console.print(f"ℹ️  {message}", style="blue")
        logger.info(message)
    
    def success(self, message: str, **kwargs):
        """Log a success message."""
        import os
        if os.environ.get('QUIET_MODE') == '1':
            return
        if self.level != "ERROR":
            self.console.print(f"✅ {message}", style="green")
        logger.success(message)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        import os
        if os.environ.get('QUIET_MODE') == '1':
            return
        if self.level != "ERROR":
            self.console.print(f"⚠️  {message}", style="yellow")
        logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log an error message."""
        self.console.print(f"❌ {message}", style="red")
        logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message (only in debug mode)."""
        import os
        if os.environ.get('QUIET_MODE') == '1':
            return
        if self.level == "DEBUG":
            self.console.print(f"🔍 {message}", style="dim")
        logger.debug(message)
    
    @contextmanager
    def operation(self, name: str, total_items: Optional[int] = None):
        """Context manager for tracking operations."""
        op = OperationContext(name=name, total_items=total_items)
        
        with self._lock:
            self._operation_stack.append(op)
        
        # Only show operation messages if not in quiet mode (ERROR level)
        if self.level != "ERROR":
            # Show operation start
            if total_items:
                self.console.print(f"🚀 {name} (0/{total_items})")
            else:
                self.console.print(f"🚀 {name}")
        
        try:
            yield op
        finally:
            with self._lock:
                if self._operation_stack and self._operation_stack[-1] == op:
                    self._operation_stack.pop()
            
            # Only show completion message if not in quiet mode
            if self.level != "ERROR":
                # Show operation completion
                op.status = "completed"
                if op.total_items:
                    self.console.print(f"✅ {name} completed ({op.completed_items}/{op.total_items}) in {op.duration_str}")
                else:
                    self.console.print(f"✅ {name} completed in {op.duration_str}")
    
    def update_progress(self, completed: int, total: Optional[int] = None, details: str = None):
        """Update progress for the current operation."""
        current_time = time.time()
        if current_time - self._last_progress_update < self._progress_throttle:
            return  # Throttle updates
        
        self._last_progress_update = current_time
        
        with self._lock:
            if self._operation_stack:
                op = self._operation_stack[-1]
                op.completed_items = completed
                if total:
                    op.total_items = total
                
                # Only show progress updates for significant operations
                if op.total_items and op.total_items > 10:
                    progress_pct = (completed / op.total_items) * 100
                    if details:
                        self.console.print(f"⏳ {op.name}: {completed}/{op.total_items} ({progress_pct:.1f}%) - {details}", style="dim")
                    else:
                        self.console.print(f"⏳ {op.name}: {completed}/{op.total_items} ({progress_pct:.1f}%)", style="dim")
    
    def table(self, title: str, data: List[Dict[str, Any]], max_rows: int = 20):
        """Display data in a formatted table."""
        if not data:
            self.console.print(f"📊 {title}: No data")
            return
        
        # Limit rows if too many
        display_data = data[:max_rows]
        
        table = Table(title=title, box=box.ROUNDED)
        
        # Add columns based on first row
        for key in display_data[0].keys():
            table.add_column(str(key).replace('_', ' ').title(), style="cyan")
        
        # Add rows
        for row in display_data:
            table.add_row(*[str(v) for v in row.values()])
        
        if len(data) > max_rows:
            table.caption = f"Showing {max_rows} of {len(data)} rows"
        
        self.console.print(table)
    
    @contextmanager
    def quiet_mode(self):
        """Context manager for suppressing verbose output during optimization loops."""
        original_level = self.level
        self.level = "ERROR"
        
        # Also disable console output temporarily
        original_console = self.console
        from rich.console import Console
        from io import StringIO
        self.console = Console(file=StringIO(), quiet=True)
        
        try:
            yield
        finally:
            self.level = original_level
            self.console = original_console


class BacktestLogger(StructuredLogger):
    """Specialized logger for backtesting operations."""
    
    def __init__(self, name: str = "backtester"):
        super().__init__(name)
        self._optimization_stats = {}
        self._backtest_stats = {}
        # Add logger attribute for backward compatibility
        self.logger = self
    
    def data_summary(self, symbols: List[str], timeframe: str, start_date: str, end_date: str, data_points: int):
        """Log data loading summary."""
        self.section("📊 Data Summary")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Symbols", ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""))
        table.add_row("Timeframe", timeframe)
        table.add_row("Period", f"{start_date} → {end_date}")
        table.add_row("Data Points", f"{data_points:,}")
        table.add_row("Symbol Count", str(len(symbols)))
        
        self.console.print(table)
    
    def strategy_config(self, strategy_name: str, params: Dict[str, Any]):
        """Log strategy configuration."""
        self.section("⚙️ Strategy Configuration")
        
        # Filter out less important parameters
        important_params = {k: v for k, v in params.items() 
                          if not k.startswith('_') and k not in ['clean_signals', 'use_cache']}
        
        table = Table(box=box.ROUNDED)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Strategy", strategy_name)
        for key, value in important_params.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            table.add_row(formatted_key, formatted_value)
        
        self.console.print(table)
    
    def signal_summary(self, stats: Dict[str, Any]):
        """Log signal generation summary."""
        # Don't show signal summary in quiet mode or if no signals
        if self.level == "ERROR":
            return
            
        # Check if we have any actual signals
        has_signals = any(
            isinstance(data, dict) and data.get('total_signals', 0) > 0 
            for data in stats.values()
        )
        
        if not has_signals:
            return
            
        self.section("📡 Signal Summary")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Signal Type", style="cyan")
        table.add_column("Count", style="white")
        table.add_column("Rate", style="white")
        
        for signal_type, data in stats.items():
            if isinstance(data, dict) and 'total_signals' in data:
                count = data['total_signals']
                rate = data.get('signal_rate', 0)
                if count > 0:  # Only show rows with actual signals
                    table.add_row(signal_type.replace('_', ' ').title(), str(count), f"{rate:.2%}")
        
        # Only print table if it has rows
        if table.row_count > 0:
            self.console.print(table)
    
    def backtest_result(self, results: Dict[str, Any]):
        """Log backtest results summary."""
        portfolio = results.get('portfolio')
        if not portfolio:
            return
        
        self.section("📈 Backtest Results")
        
        # Extract key metrics using safe extraction
        try:
            def safe_extract_metric(metric):
                """Safely extract metric value, handling Series and scalar cases."""
                if hasattr(metric, 'iloc'):
                    return float(metric.iloc[0])
                else:
                    return float(metric)
            
            def safe_extract_int_metric(metric):
                """Safely extract integer metric value, handling Series and scalar cases."""
                if hasattr(metric, 'iloc'):
                    return int(metric.iloc[0])
                elif callable(metric):
                    result = metric()
                    if hasattr(result, 'iloc'):
                        return int(result.iloc[0])
                    else:
                        return int(result)
                else:
                    return int(metric)
            
            total_return = safe_extract_metric(portfolio.total_return) if hasattr(portfolio, 'total_return') else 0.0
            sharpe_ratio = safe_extract_metric(portfolio.sharpe_ratio) if hasattr(portfolio, 'sharpe_ratio') else 0.0
            max_drawdown = safe_extract_metric(portfolio.max_drawdown) if hasattr(portfolio, 'max_drawdown') else 0.0
            win_rate = safe_extract_metric(portfolio.trades.win_rate) if hasattr(portfolio, 'trades') and hasattr(portfolio.trades, 'win_rate') else 0.0
            total_trades = safe_extract_int_metric(portfolio.trades.count) if hasattr(portfolio, 'trades') and hasattr(portfolio.trades, 'count') else 0
        except (AttributeError, TypeError, ValueError):
            self.warning("Could not extract all performance metrics")
            return
        
        table = Table(box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="white")
        
        # Add metrics with status indicators
        table.add_row("Total Return", f"{total_return:.2%}", "🟢" if total_return > 0 else "🔴")
        table.add_row("Sharpe Ratio", f"{sharpe_ratio:.2f}", "🟢" if sharpe_ratio > 1 else "🟡" if sharpe_ratio > 0 else "🔴")
        table.add_row("Max Drawdown", f"{max_drawdown:.2%}", "🟢" if abs(max_drawdown) < 0.1 else "🟡" if abs(max_drawdown) < 0.2 else "🔴")
        table.add_row("Win Rate", f"{win_rate:.2%}", "🟢" if win_rate > 0.5 else "🟡" if win_rate > 0.3 else "🔴")
        table.add_row("Total Trades", str(total_trades), "🟢" if total_trades > 10 else "🟡" if total_trades > 0 else "🔴")
        
        self.console.print(table)
    
    def optimization_progress(self, completed: int, total: int, current_params: Dict[str, Any] = None, best_metric: float = None):
        """Log optimization progress (throttled)."""
        # Only update every 10 iterations or significant milestones
        if completed % max(1, total // 20) == 0 or completed == total:
            progress_pct = (completed / total) * 100
            
            status_msg = f"Optimization: {completed}/{total} ({progress_pct:.1f}%)"
            if best_metric is not None:
                status_msg += f" | Best: {best_metric:.4f}"
            
            self.console.print(f"⏳ {status_msg}", style="dim")
    
    def optimization_result(self, results: Dict[str, Any]):
        """Log optimization results."""
        best_params = results.get('best_params', {})
        best_metric = results.get('best_metric')
        total_trials = results.get('total_trials', 0)
        
        self.section("🏆 Optimization Results")
        
        if best_metric is not None:
            self.success(f"Best metric: {best_metric:.4f} (from {total_trials:,} trials)")
            
            # Show best parameters
            if best_params:
                table = Table(title="Best Parameters", box=box.ROUNDED)
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="white")
                
                for key, value in best_params.items():
                    if not key.startswith('_'):
                        formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                        table.add_row(key.replace('_', ' ').title(), formatted_value)
                
                self.console.print(table)
        else:
            self.error("Optimization failed to find valid results")


# Global logger instance
_global_logger = None


def get_logger(name: str = "backtester") -> BacktestLogger:
    """Get or create a global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = BacktestLogger(name)
    return _global_logger


def setup_logging(level: str = "INFO", quiet: bool = False):
    """Setup global logging configuration."""
    global _global_logger
    
    if quiet:
        # Minimal logging mode
        logger.remove()
        logger.add(sys.stderr, level="ERROR", format="{message}")
        _global_logger = BacktestLogger("backtester")
        _global_logger.level = "ERROR"
    else:
        _global_logger = BacktestLogger("backtester")
        _global_logger.level = level
    
    return _global_logger


def log_operation(name: str, total_items: Optional[int] = None):
    """Decorator for logging operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with get_logger().operation(name, total_items):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_info(message: str):
    """Log an info message."""
    get_logger().info(message)


def log_success(message: str):
    """Log a success message."""
    get_logger().success(message)


def log_warning(message: str):
    """Log a warning message."""
    get_logger().warning(message)


def log_error(message: str):
    """Log an error message."""
    get_logger().error(message)


@contextmanager
def quiet_logging():
    """Context manager for temporarily suppressing verbose logging."""
    global _global_logger
    
    # Get all active loggers
    main_logger = get_logger()
    
    # Store original states
    original_level = main_logger.level
    original_console = main_logger.console
    
    # Set to quiet mode
    main_logger.level = "ERROR"
    from rich.console import Console
    from io import StringIO
    main_logger.console = Console(file=StringIO(), quiet=True)
    
    # Also disable loguru
    logger.disable("backtester")
    
    try:
        yield
    finally:
        # Restore original states
        main_logger.level = original_level
        main_logger.console = original_console
        logger.enable("backtester") 