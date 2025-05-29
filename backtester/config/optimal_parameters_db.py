"""
Database-based Optimal Parameters Manager

This module provides a more scalable solution for storing and retrieving optimal parameters
using SQLite database instead of JSON files. This allows for better organization,
querying, and handling of multiple symbols, timeframes, and strategies.
"""

import sqlite3
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class OptimalParametersDB:
    """
    Database-based manager for optimal parameters storage and retrieval.
    
    Uses SQLite for efficient storage and querying of optimization results
    across multiple symbols, timeframes, and strategies.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Optional path to SQLite database file
        """
        if db_path is None:
            storage_dir = Path(__file__).parent / "optimal_parameters"
            storage_dir.mkdir(exist_ok=True)
            db_path = storage_dir / "optimal_parameters.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main optimization results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON string
                    performance_metrics TEXT NOT NULL,  -- JSON string
                    optimization_metric REAL,  -- Primary metric (e.g., Sharpe ratio)
                    total_return REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON string for additional info
                    is_active BOOLEAN DEFAULT 1,  -- Whether this is the current best
                    UNIQUE(symbol, timeframe, strategy_name, is_active)
                )
            """)
            
            # Optimization history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    optimization_metric REAL,
                    total_combinations INTEGER,
                    valid_combinations INTEGER,
                    optimization_time_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Parameter ranges table (for tracking what was tested)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_ranges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id INTEGER,
                    parameter_name TEXT NOT NULL,
                    parameter_values TEXT NOT NULL,  -- JSON array
                    FOREIGN KEY (optimization_id) REFERENCES optimization_history (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_strategy 
                ON optimization_results (symbol, timeframe, strategy_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_metric 
                ON optimization_results (optimization_metric DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON optimization_results (created_at DESC)
            """)
            
            conn.commit()
    
    def store_optimization_result(self,
                                 symbol: str,
                                 timeframe: str,
                                 strategy_name: str,
                                 best_params: Dict[str, Any],
                                 performance_metrics: Dict[str, Any],
                                 parameter_ranges: Optional[Dict[str, List]] = None,
                                 optimization_stats: Optional[Dict[str, Any]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store optimization results in the database.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            strategy_name: Name of the strategy
            best_params: Dictionary of optimal parameters
            performance_metrics: Dictionary of performance metrics
            parameter_ranges: Dictionary of parameter ranges tested
            optimization_stats: Statistics about the optimization run
            metadata: Additional metadata
            
        Returns:
            ID of the stored optimization result
        """
        # Extract key metrics first (outside try block)
        optimization_metric = performance_metrics.get('sharpe_ratio', 0)
        total_return = performance_metrics.get('total_return', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        total_trades = performance_metrics.get('total_trades', 0)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                # First, delete any existing active records to avoid constraint issues
                cursor.execute("""
                    DELETE FROM optimization_results 
                    WHERE symbol = ? AND timeframe = ? AND strategy_name = ? AND is_active = 1
                """, (symbol, timeframe, strategy_name))
                
                # Insert new result
                cursor.execute("""
                    INSERT INTO optimization_results (
                        symbol, timeframe, strategy_name, parameters, performance_metrics,
                        optimization_metric, total_return, max_drawdown, win_rate, total_trades,
                        metadata, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    symbol, timeframe, strategy_name,
                    json.dumps(best_params), json.dumps(performance_metrics),
                    optimization_metric, total_return, max_drawdown, win_rate, total_trades,
                    json.dumps(metadata or {})
                ))
                
                result_id = cursor.lastrowid
                
                # Store optimization history
                if optimization_stats:
                    cursor.execute("""
                        INSERT INTO optimization_history (
                            symbol, timeframe, strategy_name, parameters, performance_metrics,
                            optimization_metric, total_combinations, valid_combinations,
                            optimization_time_seconds, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timeframe, strategy_name,
                        json.dumps(best_params), json.dumps(performance_metrics),
                        optimization_metric,
                        optimization_stats.get('total_combinations', 0),
                        optimization_stats.get('valid_combinations', 0),
                        optimization_stats.get('optimization_time', 0),
                        json.dumps(optimization_stats)
                    ))
                    
                    history_id = cursor.lastrowid
                    
                    # Store parameter ranges
                    if parameter_ranges:
                        for param_name, param_values in parameter_ranges.items():
                            cursor.execute("""
                                INSERT INTO parameter_ranges (
                                    optimization_id, parameter_name, parameter_values
                                ) VALUES (?, ?, ?)
                            """, (history_id, param_name, json.dumps(param_values)))
                
                conn.commit()
                
            except sqlite3.IntegrityError as e:
                # If we still get a constraint error, log it and continue
                logger.warning(f"Database constraint error for {symbol} {timeframe}: {e}")
                
                # Try a more aggressive cleanup and retry
                cursor.execute("""
                    DELETE FROM optimization_results 
                    WHERE symbol = ? AND timeframe = ? AND strategy_name = ?
                """, (symbol, timeframe, strategy_name))
                
                # Retry the insert
                cursor.execute("""
                    INSERT INTO optimization_results (
                        symbol, timeframe, strategy_name, parameters, performance_metrics,
                        optimization_metric, total_return, max_drawdown, win_rate, total_trades,
                        metadata, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    symbol, timeframe, strategy_name,
                    json.dumps(best_params), json.dumps(performance_metrics),
                    optimization_metric, total_return, max_drawdown, win_rate, total_trades,
                    json.dumps(metadata or {})
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                
        logger.info(f"Stored optimization result for {symbol} ({timeframe}) - {strategy_name}")
        return result_id
    
    def get_optimal_params(self, 
                          symbol: str, 
                          timeframe: str, 
                          strategy_name: str = "DMAATRTrendStrategy") -> Optional[Dict[str, Any]]:
        """
        Get optimal parameters for a symbol/timeframe/strategy combination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Strategy name
            
        Returns:
            Dictionary of optimal parameters or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT parameters FROM optimization_results
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ? AND is_active = 1
                ORDER BY optimization_metric DESC
                LIMIT 1
            """, (symbol, timeframe, strategy_name))
            
            result = cursor.fetchone()
            
            if result:
                params = json.loads(result[0])
                # Apply type conversion to ensure numeric values are properly typed
                return self._convert_parameter_types(params)
            
            return None
    
    def _convert_parameter_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parameter types to ensure compatibility with strategy classes.
        
        Args:
            params: Raw parameters dictionary
            
        Returns:
            Parameters with proper types
        """
        converted = {}
        
        # Define expected numeric parameters
        numeric_params = {
            'fast_window', 'slow_window', 'atr_window', 'atr_period',
            'atr_multiplier', 'atr_multiplier_sl', 'atr_multiplier_tp',
            'short_ma_window', 'long_ma_window', 'trend_ma_window',
            'volume_window', 'volume_factor', 'adx_period', 'adx_threshold',
            'min_signal_gap', 'signal_confirmation_bars', 'max_position_size',
            'risk_per_trade'
        }
        
        # Define expected boolean parameters
        boolean_params = {
            'use_volume_filter', 'enable_short_trades', 'use_adx_filter'
        }
        
        for key, value in params.items():
            if key in numeric_params:
                # Convert to float first, then to int if it's a whole number
                try:
                    float_val = float(value)
                    if float_val.is_integer():
                        converted[key] = int(float_val)
                    else:
                        converted[key] = float_val
                except (ValueError, TypeError):
                    converted[key] = value  # Keep original if conversion fails
            elif key in boolean_params:
                # Convert to boolean
                if isinstance(value, str):
                    converted[key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted[key] = bool(value)
            else:
                # Keep as-is for string parameters
                converted[key] = value
        
        return converted
    
    def get_performance_metrics(self, 
                               symbol: str, 
                               timeframe: str, 
                               strategy_name: str = "DMAATRTrendStrategy") -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for optimal parameters.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Strategy name
            
        Returns:
            Dictionary of performance metrics or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT performance_metrics FROM optimization_results
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ? AND is_active = 1
                ORDER BY optimization_metric DESC
                LIMIT 1
            """, (symbol, timeframe, strategy_name))
            
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            
        return None
    
    def get_optimization_summary(self, 
                                symbol: str, 
                                timeframe: str, 
                                strategy_name: str = "DMAATRTrendStrategy") -> Optional[Dict[str, Any]]:
        """
        Get complete optimization summary.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Strategy name
            
        Returns:
            Complete summary dictionary or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT parameters, performance_metrics, optimization_metric, 
                       total_return, max_drawdown, win_rate, total_trades,
                       created_at, updated_at, metadata
                FROM optimization_results
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ? AND is_active = 1
                ORDER BY optimization_metric DESC
                LIMIT 1
            """, (symbol, timeframe, strategy_name))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'strategy_name': strategy_name,
                    'parameters': json.loads(result[0]),
                    'performance': json.loads(result[1]),
                    'optimization_metric': result[2],
                    'total_return': result[3],
                    'max_drawdown': result[4],
                    'win_rate': result[5],
                    'total_trades': result[6],
                    'created_at': result[7],
                    'updated_at': result[8],
                    'metadata': json.loads(result[9]) if result[9] else {}
                }
            
        return None
    
    def list_optimized_combinations(self, 
                                   strategy_name: Optional[str] = None,
                                   min_sharpe: Optional[float] = None) -> pd.DataFrame:
        """
        List all optimized symbol/timeframe combinations.
        
        Args:
            strategy_name: Optional filter by strategy name
            min_sharpe: Optional minimum Sharpe ratio filter
            
        Returns:
            DataFrame with optimization results
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT symbol, timeframe, strategy_name, optimization_metric as sharpe_ratio,
                       total_return, max_drawdown, win_rate, total_trades,
                       created_at, updated_at
                FROM optimization_results
                WHERE is_active = 1
            """
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if min_sharpe:
                query += " AND optimization_metric >= ?"
                params.append(min_sharpe)
            
            query += " ORDER BY optimization_metric DESC"
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_optimization_history(self, 
                                symbol: str, 
                                timeframe: str, 
                                strategy_name: str = "DMAATRTrendStrategy") -> pd.DataFrame:
        """
        Get optimization history for a symbol/timeframe/strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Strategy name
            
        Returns:
            DataFrame with historical optimization results
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM optimization_history
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ?
                ORDER BY created_at DESC
            """
            
            return pd.read_sql_query(query, conn, params=(symbol, timeframe, strategy_name))
    
    def bulk_optimize_symbols(self, 
                             symbols: List[str], 
                             timeframes: List[str],
                             strategy_name: str = "DMAATRTrendStrategy") -> Dict[Tuple[str, str], bool]:
        """
        Check which symbol/timeframe combinations have been optimized.
        
        Args:
            symbols: List of symbols to check
            timeframes: List of timeframes to check
            strategy_name: Strategy name
            
        Returns:
            Dictionary mapping (symbol, timeframe) tuples to optimization status
        """
        results = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for symbol in symbols:
                for timeframe in timeframes:
                    cursor.execute("""
                        SELECT COUNT(*) FROM optimization_results
                        WHERE symbol = ? AND timeframe = ? AND strategy_name = ? AND is_active = 1
                    """, (symbol, timeframe, strategy_name))
                    
                    count = cursor.fetchone()[0]
                    results[(symbol, timeframe)] = count > 0
        
        return results
    
    def export_to_csv(self, output_path: str, strategy_name: Optional[str] = None):
        """
        Export optimization results to CSV.
        
        Args:
            output_path: Path to output CSV file
            strategy_name: Optional filter by strategy name
        """
        df = self.list_optimized_combinations(strategy_name=strategy_name)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported optimization results to {output_path}")
    
    def clear_symbol_data(self, 
                         symbol: str, 
                         timeframe: str, 
                         strategy_name: str = "DMAATRTrendStrategy"):
        """
        Clear stored data for a specific symbol/timeframe/strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy_name: Strategy name
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete from all tables
            cursor.execute("""
                DELETE FROM optimization_results
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ?
            """, (symbol, timeframe, strategy_name))
            
            cursor.execute("""
                DELETE FROM optimization_history
                WHERE symbol = ? AND timeframe = ? AND strategy_name = ?
            """, (symbol, timeframe, strategy_name))
            
            conn.commit()
            
        logger.info(f"Cleared data for {symbol} ({timeframe}) - {strategy_name}")
    
    def get_best_performers(self, 
                           limit: int = 10, 
                           strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get top performing optimizations.
        
        Args:
            limit: Number of results to return
            strategy_name: Optional filter by strategy name
            
        Returns:
            DataFrame with top performers
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT symbol, timeframe, strategy_name, optimization_metric as sharpe_ratio,
                       total_return, max_drawdown, win_rate, total_trades, created_at
                FROM optimization_results
                WHERE is_active = 1
            """
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY optimization_metric DESC LIMIT ?"
            params.append(limit)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def __repr__(self) -> str:
        """String representation."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM optimization_results WHERE is_active = 1")
            count = cursor.fetchone()[0]
        
        return f"OptimalParametersDB({count} active optimizations)"

    def get_all_optimal_params(self) -> List[Dict[str, Any]]:
        """
        Get all optimal parameters from the database.
        
        Returns:
            List of all parameter dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, timeframe, strategy_name, parameters, 
                       performance_metrics, created_at, updated_at
                FROM optimization_results
                WHERE is_active = 1
                ORDER BY symbol, timeframe, strategy_name
            """)
            
            results = cursor.fetchall()
            
            all_params = []
            for result in results:
                params_dict = {
                    'symbol': result[0],
                    'timeframe': result[1],
                    'strategy_name': result[2],
                    'parameters': json.loads(result[3]) if result[3] else {},
                    'performance_metrics': json.loads(result[4]) if result[4] else {},
                    'last_updated': result[6],  # updated_at
                    'optimization_count': 1  # Default since we don't track this in current schema
                }
                all_params.append(params_dict)
            
            return all_params

    def list_optimized_combinations(self) -> List[Dict[str, str]]:
        """
        List all symbol/timeframe/strategy combinations that have optimal parameters.
        
        Returns:
            List of dictionaries with symbol, timeframe, and strategy_name
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT symbol, timeframe, strategy_name
                FROM optimization_results
                WHERE is_active = 1
                ORDER BY symbol, timeframe, strategy_name
            """)
            
            results = cursor.fetchall()
            
            combinations = []
            for result in results:
                combinations.append({
                    'symbol': result[0],
                    'timeframe': result[1],
                    'strategy_name': result[2]
                })
            
            return combinations

    def backup_to_json(self, output_path: Optional[str] = None) -> str:
        """
        Create a JSON backup of all optimization results.
        
        Args:
            output_path: Optional path for backup file
            
        Returns:
            Path to backup file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(self.db_path).parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            output_path = backup_dir / f"database_backup_{timestamp}.json"
        
        # Get all parameters from database
        all_params = self.get_all_optimal_params()
        
        # Convert to JSON format
        backup_data = {}
        for params in all_params:
            key = f"{params['strategy_name']}_{params['symbol'].replace('/', '_')}_{params['timeframe']}"
            backup_data[key] = {
                'symbol': params['symbol'],
                'timeframe': params['timeframe'],
                'strategy': params['strategy_name'],
                'parameters': params['parameters'],
                'performance': params.get('performance_metrics', {}),
                'last_updated': params.get('last_updated', ''),
                'optimization_count': params.get('optimization_count', 0)
            }
        
        # Save backup
        with open(output_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Created database backup: {output_path}")
        return str(output_path) 