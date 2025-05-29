"""
Performance Analyzer Module

Provides comprehensive performance analysis functionality for backtesting results.
Calculates various performance metrics, risk measures, and trade statistics using
vectorbtpro's Portfolio functionality.

Key Features:
- Portfolio-level performance metrics (returns, Sharpe, Sortino, Calmar ratios)
- Trade-level analysis (win rate, profit factor, average trade metrics)
- Risk-adjusted returns and drawdown analysis
- Benchmark comparison and relative performance
- Multi-timeframe analysis support
- Export capabilities for results

Usage:
    analyzer = PerformanceAnalyzer(portfolio)
    metrics = analyzer.get_summary_stats()
    trade_metrics = analyzer.get_trade_metrics()
    report = analyzer.export_results("performance_report.csv")
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis class for portfolio backtesting results.

    This class provides detailed analysis of portfolio performance using vectorbtpro's
    Portfolio functionality, including returns-based metrics, trade analysis, and
    risk measures.
    """

    def __init__(self, portfolio: vbt.Portfolio, benchmark: Optional[pd.Series] = None, signals: Optional[Dict[str, Any]] = None):
        """
        Initialize performance analyzer with portfolio results.

        Args:
            portfolio (vbt.Portfolio): Portfolio simulation results from vectorbtpro
            benchmark (Optional[pd.Series]): Benchmark price series for comparison
            signals (Optional[Dict[str, Any]]): Strategy signals containing SL/TP levels
        """
        if not isinstance(portfolio, vbt.Portfolio):
            raise TypeError("portfolio must be a vectorbtpro Portfolio object")

        self.portfolio = portfolio
        self.benchmark = benchmark
        self.signals = signals  # Store strategy signals for trade enhancement
        self._cache = {}  # Cache for expensive calculations

        logger.info("Performance analyzer initialized with portfolio")
        if benchmark is not None:
            logger.info("Benchmark data provided for relative performance analysis")
        if signals is not None:
            logger.info("Strategy signals provided for enhanced trade analysis")

    def get_summary_stats(self) -> pd.Series:
        """
        Get comprehensive summary performance statistics.

        Returns:
            pd.Series: Summary statistics from vectorbtpro's stats() method
        """
        if "summary_stats" not in self._cache:
            try:
                self._cache["summary_stats"] = self.portfolio.stats()
                logger.info("Summary statistics calculated successfully")
            except Exception as e:
                logger.error(f"Failed to calculate summary statistics: {e}")
                raise

        return self._cache["summary_stats"]

    def get_returns_metrics(self) -> Dict[str, float]:
        """
        Get detailed returns-based performance metrics.

        Returns:
            Dict[str, float]: Dictionary of returns metrics
        """
        if "returns_metrics" not in self._cache:
            try:
                # Helper function to safely call portfolio methods
                def safe_call(method_or_prop):
                    try:
                        return (
                            method_or_prop()
                            if callable(method_or_prop)
                            else method_or_prop
                        )
                    except (AttributeError, TypeError, ValueError):
                        return method_or_prop

                metrics = {
                    "total_return": safe_call(self.portfolio.total_return),
                    "annualized_return": safe_call(self.portfolio.annualized_return),
                    "sharpe_ratio": safe_call(self.portfolio.sharpe_ratio),
                    "sortino_ratio": safe_call(self.portfolio.sortino_ratio),
                    "calmar_ratio": safe_call(self.portfolio.calmar_ratio),
                    "max_drawdown": safe_call(self.portfolio.max_drawdown),
                    "volatility": safe_call(self.portfolio.annualized_volatility),
                    "skewness": self.portfolio.returns.skew(),
                    "kurtosis": self.portfolio.returns.kurtosis(),
                }

                # Add additional metrics if available
                try:
                    metrics["omega_ratio"] = safe_call(self.portfolio.omega_ratio)
                except (AttributeError, TypeError):
                    logger.debug("Omega ratio not available")

                try:
                    metrics["tail_ratio"] = safe_call(self.portfolio.tail_ratio)
                except (AttributeError, TypeError):
                    logger.debug("Tail ratio not available")

                self._cache["returns_metrics"] = metrics
                logger.info("Returns metrics calculated successfully")

            except Exception as e:
                logger.error(f"Failed to calculate returns metrics: {e}")
                raise

        return self._cache["returns_metrics"]

    def get_trade_metrics(self) -> Dict[str, float]:
        """
        Get trade-based performance metrics.

        Returns:
            Dict[str, float]: Dictionary of trade metrics
        """
        if "trade_metrics" not in self._cache:
            try:
                # Helper function to safely call portfolio methods
                def safe_call(method_or_prop):
                    try:
                        return (
                            method_or_prop()
                            if callable(method_or_prop)
                            else method_or_prop
                        )
                    except (AttributeError, TypeError, ValueError):
                        return method_or_prop

                # Helper function to safely extract values from vectorbtpro objects
                def safe_extract(obj, default=0):
                    try:
                        if hasattr(obj, "values"):
                            val = obj.values
                            return (
                                val
                                if np.isscalar(val)
                                else (val[0] if len(val) > 0 else default)
                            )
                        elif hasattr(obj, "iloc") and len(obj) > 0:
                            return obj.iloc[0] if np.isscalar(obj.iloc[0]) else default
                        elif np.isscalar(obj):
                            return obj
                        else:
                            return default
                    except (AttributeError, TypeError, ValueError, IndexError):
                        return default

                # Helper function to safely get array values
                def safe_array(obj, default_array=None):
                    try:
                        if hasattr(obj, "values"):
                            return obj.values
                        elif hasattr(obj, "to_numpy"):
                            return obj.to_numpy()
                        elif hasattr(obj, "__array__"):
                            return np.array(obj)
                        else:
                            return (
                                default_array
                                if default_array is not None
                                else np.array([])
                            )
                    except (AttributeError, TypeError, ValueError):
                        return (
                            default_array if default_array is not None else np.array([])
                        )

                trades = self.portfolio.trades

                # Get basic trade counts
                total_trades = safe_extract(safe_call(trades.count))
                winning_trades_count = safe_extract(safe_call(trades.winning.count))
                losing_trades_count = safe_extract(safe_call(trades.losing.count))

                # Ensure total_trades is a scalar
                if isinstance(total_trades, pd.Series):
                    total_trades = total_trades.iloc[0] if len(total_trades) > 0 else 0
                elif hasattr(total_trades, "item"):
                    total_trades = total_trades.item()

                metrics = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades_count,
                    "losing_trades": losing_trades_count,
                    "win_rate": (
                        winning_trades_count / total_trades if total_trades > 0 else 0
                    ),
                    "profit_factor": safe_extract(safe_call(trades.profit_factor)),
                    "expectancy": 0,  # Will calculate below
                    "avg_trade_pnl": 0,
                    "avg_trade_return": 0,
                    "avg_winning_trade": 0,
                    "avg_losing_trade": 0,
                    "largest_winning_trade": 0,
                    "largest_losing_trade": 0,
                    "max_consecutive_wins": 0,
                    "max_consecutive_losses": 0,
                    "avg_trade_duration": 0,
                    "max_trade_duration": 0,
                    "min_trade_duration": 0,
                }

                # Only calculate detailed metrics if we have trades
                if total_trades > 0:
                    # Get trade PnL and returns safely
                    try:
                        trade_pnl_array = safe_array(trades.pnl)
                        if len(trade_pnl_array) > 0:
                            metrics["avg_trade_pnl"] = np.mean(trade_pnl_array)
                    except Exception as e:
                        logger.debug(f"Could not calculate avg trade PnL: {e}")

                    try:
                        trade_returns_array = safe_array(trades.returns)
                        if len(trade_returns_array) > 0:
                            metrics["avg_trade_return"] = np.mean(trade_returns_array)
                    except Exception as e:
                        logger.debug(f"Could not calculate avg trade return: {e}")

                    # Get winning trade metrics
                    try:
                        winning_trades = trades.winning
                        winning_pnl_array = safe_array(winning_trades.pnl)
                        if len(winning_pnl_array) > 0:
                            metrics["avg_winning_trade"] = np.mean(winning_pnl_array)
                            metrics["largest_winning_trade"] = np.max(winning_pnl_array)
                    except Exception as e:
                        logger.debug(f"Could not calculate winning trade metrics: {e}")

                    # Get losing trade metrics
                    try:
                        losing_trades = trades.losing
                        losing_pnl_array = safe_array(losing_trades.pnl)
                        if len(losing_pnl_array) > 0:
                            metrics["avg_losing_trade"] = np.mean(losing_pnl_array)
                            metrics["largest_losing_trade"] = np.min(losing_pnl_array)
                    except Exception as e:
                        logger.debug(f"Could not calculate losing trade metrics: {e}")

                    # Get duration metrics
                    try:
                        duration_array = safe_array(trades.duration)
                        if len(duration_array) > 0:
                            # Convert timedelta to days if needed
                            if hasattr(duration_array[0], "days"):
                                duration_days = np.array(
                                    [d.days for d in duration_array]
                                )
                            else:
                                duration_days = duration_array

                            metrics["avg_trade_duration"] = np.mean(duration_days)
                            metrics["max_trade_duration"] = np.max(duration_days)
                            metrics["min_trade_duration"] = np.min(duration_days)
                    except Exception as e:
                        logger.debug(f"Could not calculate duration metrics: {e}")

                    # Calculate expectancy
                    if metrics["win_rate"] > 0 and (
                        metrics["avg_winning_trade"] != 0
                        or metrics["avg_losing_trade"] != 0
                    ):
                        metrics["expectancy"] = (
                            metrics["win_rate"] * metrics["avg_winning_trade"]
                            + (1 - metrics["win_rate"]) * metrics["avg_losing_trade"]
                        )
                    else:
                        metrics["expectancy"] = 0

                    # Calculate consecutive wins/losses if possible
                    try:
                        trade_pnl_array = safe_array(trades.pnl)
                        if len(trade_pnl_array) > 0:
                            trade_outcomes = trade_pnl_array > 0
                            consecutive_wins = self._calculate_max_consecutive(
                                pd.Series(trade_outcomes), True
                            )
                            consecutive_losses = self._calculate_max_consecutive(
                                pd.Series(trade_outcomes), False
                            )
                            metrics["max_consecutive_wins"] = consecutive_wins
                            metrics["max_consecutive_losses"] = consecutive_losses
                    except Exception as e:
                        logger.debug(
                            f"Could not calculate consecutive wins/losses: {e}"
                        )

                self._cache["trade_metrics"] = metrics
                logger.info("Trade metrics calculated successfully")

            except Exception as e:
                logger.error(f"Failed to calculate trade metrics: {e}")
                raise

        return self._cache["trade_metrics"]

    def get_drawdown_metrics(self) -> Dict[str, float]:
        """
        Get drawdown-related metrics.

        Returns:
            Dict[str, float]: Dictionary of drawdown metrics
        """
        if "drawdown_metrics" not in self._cache:
            try:
                # Helper function to safely call portfolio methods
                def safe_call(method_or_prop):
                    try:
                        return (
                            method_or_prop()
                            if callable(method_or_prop)
                            else method_or_prop
                        )
                    except (AttributeError, TypeError, ValueError):
                        return method_or_prop

                # Helper function to safely extract numeric values
                def safe_extract_numeric(obj, default=0.0):
                    try:
                        if hasattr(obj, "values"):
                            val = obj.values
                            if hasattr(val, "__len__") and len(val) > 0:
                                return (
                                    float(val[0])
                                    if np.isscalar(val[0])
                                    else float(val.flat[0])
                                )
                            else:
                                return default
                        elif isinstance(obj, np.ndarray):
                            if obj.size > 0:
                                return float(obj.flat[0])
                            else:
                                return default
                        elif hasattr(obj, "iloc") and len(obj) > 0:
                            return float(obj.iloc[0])
                        elif hasattr(obj, "days"):  # timedelta
                            return float(obj.days)
                        elif np.isscalar(obj):
                            return float(obj) if not pd.isna(obj) else default
                        else:
                            return default
                    except (AttributeError, TypeError, ValueError, IndexError):
                        return default

                drawdowns = self.portfolio.drawdowns

                # Get max drawdown value safely
                max_dd = safe_call(self.portfolio.max_drawdown)
                if isinstance(max_dd, pd.Series):
                    max_dd = max_dd.iloc[0] if len(max_dd) > 0 else 0
                elif hasattr(max_dd, "item"):
                    max_dd = max_dd.item()
                max_dd = float(max_dd) if max_dd is not None else 0.0

                # Get total return safely
                total_ret = safe_call(self.portfolio.total_return)
                if isinstance(total_ret, pd.Series):
                    total_ret = total_ret.iloc[0] if len(total_ret) > 0 else 0
                elif hasattr(total_ret, "item"):
                    total_ret = total_ret.item()
                total_ret = float(total_ret) if total_ret is not None else 0.0

                metrics = {
                    "max_drawdown": max_dd,
                    "avg_drawdown": safe_extract_numeric(drawdowns.drawdown.mean()),
                    "max_drawdown_duration": safe_extract_numeric(
                        drawdowns.duration.max()
                    ),
                    "avg_drawdown_duration": safe_extract_numeric(
                        drawdowns.duration.mean()
                    ),
                    "recovery_factor": (
                        abs(total_ret / max_dd) if max_dd != 0 else np.inf
                    ),
                    "drawdown_count": safe_extract_numeric(safe_call(drawdowns.count)),
                }

                # Calculate time to recovery for max drawdown
                try:
                    dd_count = safe_call(drawdowns.count)
                    if isinstance(dd_count, pd.Series):
                        dd_count = dd_count.iloc[0] if len(dd_count) > 0 else 0
                    elif hasattr(dd_count, "item"):
                        dd_count = dd_count.item()
                    dd_count = float(dd_count) if dd_count is not None else 0

                    if dd_count > 0:
                        max_dd_idx = drawdowns.drawdown.idxmin()
                        max_dd_duration = drawdowns.duration.loc[max_dd_idx]
                        metrics["max_drawdown_recovery_time"] = safe_extract_numeric(
                            max_dd_duration
                        )
                    else:
                        metrics["max_drawdown_recovery_time"] = 0.0
                except Exception as e:
                    logger.debug(f"Could not calculate max drawdown recovery time: {e}")
                    metrics["max_drawdown_recovery_time"] = 0.0

                self._cache["drawdown_metrics"] = metrics
                logger.info("Drawdown metrics calculated successfully")

            except Exception as e:
                logger.error(f"Failed to calculate drawdown metrics: {e}")
                raise

        return self._cache["drawdown_metrics"]

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get risk-related performance metrics.

        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        if "risk_metrics" not in self._cache:
            try:
                # Helper function to safely call portfolio methods
                def safe_call(method_or_prop):
                    try:
                        return (
                            method_or_prop()
                            if callable(method_or_prop)
                            else method_or_prop
                        )
                    except (AttributeError, TypeError, ValueError):
                        return method_or_prop

                returns = self.portfolio.returns

                metrics = {
                    "volatility": safe_call(self.portfolio.annualized_volatility),
                    "downside_volatility": returns[returns < 0].std() * np.sqrt(252),
                    "var_95": returns.quantile(0.05),
                    "var_99": returns.quantile(0.01),
                    "cvar_95": returns[returns <= returns.quantile(0.05)].mean(),
                    "cvar_99": returns[returns <= returns.quantile(0.01)].mean(),
                    "max_daily_loss": returns.min(),
                    "max_daily_gain": returns.max(),
                }

                # Calculate additional risk metrics
                if len(returns) > 0:
                    # Ulcer Index
                    try:
                        drawdown_series = safe_call(self.portfolio.drawdown)
                        ulcer_index = np.sqrt((drawdown_series**2).mean())
                        metrics["ulcer_index"] = ulcer_index

                        # Pain Index (average drawdown)
                        metrics["pain_index"] = drawdown_series.mean()
                    except (AttributeError, TypeError, ValueError):
                        logger.debug("Could not calculate ulcer/pain index")

                self._cache["risk_metrics"] = metrics
                logger.info("Risk metrics calculated successfully")

            except Exception as e:
                logger.error(f"Failed to calculate risk metrics: {e}")
                raise

        return self._cache["risk_metrics"]

    def get_benchmark_comparison(self) -> Optional[Dict[str, float]]:
        """
        Get benchmark comparison metrics if benchmark is provided.

        Returns:
            Optional[Dict[str, float]]: Benchmark comparison metrics or None
        """
        if self.benchmark is None:
            logger.info("No benchmark provided for comparison")
            return None

        if "benchmark_metrics" not in self._cache:
            try:
                # Helper function to safely call portfolio methods
                def safe_call(method_or_prop):
                    try:
                        return (
                            method_or_prop()
                            if callable(method_or_prop)
                            else method_or_prop
                        )
                    except (AttributeError, TypeError, ValueError):
                        return method_or_prop

                logger.debug("Starting benchmark comparison calculation")

                # Align benchmark with portfolio dates
                portfolio_returns = self.portfolio.returns
                logger.debug(f"Portfolio returns shape: {portfolio_returns.shape}")

                benchmark_returns = (
                    self.benchmark.pct_change(fill_method=None)
                    .reindex(portfolio_returns.index)
                    .fillna(0)
                )
                logger.debug(f"Benchmark returns shape: {benchmark_returns.shape}")

                # Calculate benchmark metrics
                benchmark_total_return = (1 + benchmark_returns).prod() - 1
                logger.debug(
                    f"Benchmark total return calculated: {benchmark_total_return}"
                )

                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                logger.debug(f"Benchmark volatility calculated: {benchmark_volatility}")

                # Safe calculation of benchmark Sharpe ratio
                benchmark_std = benchmark_returns.std()
                logger.debug(f"Benchmark std: {benchmark_std}")

                if float(benchmark_std) > 0:
                    benchmark_sharpe = (
                        benchmark_returns.mean() / benchmark_std * np.sqrt(252)
                    )
                else:
                    benchmark_sharpe = 0
                logger.debug(f"Benchmark Sharpe calculated: {benchmark_sharpe}")

                # Calculate relative metrics
                excess_returns = portfolio_returns - benchmark_returns
                tracking_error = excess_returns.std() * np.sqrt(252)
                logger.debug(f"Tracking error calculated: {tracking_error}")

                # Safe calculation of information ratio
                excess_std = excess_returns.std()
                logger.debug(f"Excess std: {excess_std}")

                if float(excess_std) > 0:
                    information_ratio = (
                        excess_returns.mean() / excess_std * np.sqrt(252)
                    )
                else:
                    information_ratio = 0
                logger.debug(f"Information ratio calculated: {information_ratio}")

                # Beta calculation with proper data alignment
                portfolio_clean = portfolio_returns.dropna()
                benchmark_clean = benchmark_returns.reindex(
                    portfolio_clean.index
                ).dropna()
                logger.debug(
                    f"Clean data shapes - portfolio: {portfolio_clean.shape}, benchmark: {benchmark_clean.shape}"
                )

                # Ensure we have matching indices
                common_index = portfolio_clean.index.intersection(benchmark_clean.index)
                logger.debug(f"Common index length: {len(common_index)}")

                if len(common_index) > 1:
                    portfolio_aligned = portfolio_clean.reindex(common_index)
                    benchmark_aligned = benchmark_clean.reindex(common_index)

                    covariance = np.cov(
                        portfolio_aligned.values, benchmark_aligned.values
                    )[0, 1]
                    benchmark_variance = benchmark_aligned.var()

                    if float(benchmark_variance) > 0:
                        beta = covariance / benchmark_variance
                    else:
                        beta = 0

                    # Correlation calculation
                    correlation = portfolio_aligned.corr(benchmark_aligned)
                    logger.debug(f"Beta: {beta}, Correlation: {correlation}")
                else:
                    beta = 0
                    correlation = 0
                    logger.debug("Not enough common data points for beta/correlation")

                # Alpha calculation (CAPM)
                risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
                alpha = portfolio_returns.mean() - (
                    risk_free_rate + beta * (benchmark_returns.mean() - risk_free_rate)
                )
                alpha_annualized = alpha * 252
                logger.debug(f"Alpha calculated: {alpha_annualized}")

                # Get portfolio total return safely
                portfolio_total_return = safe_call(self.portfolio.total_return)
                logger.debug(f"Portfolio total return: {portfolio_total_return}")

                metrics = {
                    "benchmark_total_return": float(benchmark_total_return),
                    "benchmark_volatility": float(benchmark_volatility),
                    "benchmark_sharpe": float(benchmark_sharpe),
                    "excess_return": float(
                        portfolio_total_return - benchmark_total_return
                    ),
                    "tracking_error": float(tracking_error),
                    "information_ratio": float(information_ratio),
                    "beta": float(beta),
                    "alpha_annualized": float(alpha_annualized),
                    "correlation": float(correlation),
                }

                self._cache["benchmark_metrics"] = metrics
                logger.info("Benchmark comparison metrics calculated successfully")

            except Exception as e:
                logger.error(f"Failed to calculate benchmark metrics: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

        return self._cache["benchmark_metrics"]

    def get_monthly_returns(self) -> pd.Series:
        """
        Get monthly returns for the portfolio.

        Returns:
            pd.Series: Monthly returns
        """
        try:
            returns = self.portfolio.returns
            monthly_returns = (1 + returns).resample("ME").prod() - 1
            logger.info("Monthly returns calculated successfully")
            return monthly_returns
        except Exception as e:
            logger.error(f"Failed to calculate monthly returns: {e}")
            raise

    def get_rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        """
        Get rolling performance metrics.

        Args:
            window (int): Rolling window size in periods (default: 252 for annual)

        Returns:
            pd.DataFrame: Rolling metrics
        """
        try:
            # Helper function to safely call portfolio methods
            def safe_call(method_or_prop):
                try:
                    return (
                        method_or_prop() if callable(method_or_prop) else method_or_prop
                    )
                except (AttributeError, TypeError, ValueError):
                    return method_or_prop

            returns = self.portfolio.returns

            rolling_metrics = pd.DataFrame(index=returns.index)
            rolling_metrics["rolling_return"] = (
                (1 + returns).rolling(window).apply(lambda x: x.prod() - 1)
            )
            rolling_metrics["rolling_volatility"] = returns.rolling(
                window
            ).std() * np.sqrt(252)
            rolling_metrics["rolling_sharpe"] = (
                returns.rolling(window).mean() / returns.rolling(window).std()
            ) * np.sqrt(252)

            # Handle drawdown safely
            try:
                drawdown_series = safe_call(self.portfolio.drawdown)
                rolling_metrics["rolling_max_dd"] = drawdown_series.rolling(
                    window
                ).min()
            except Exception as e:
                logger.debug(f"Could not calculate rolling drawdown: {e}")
                rolling_metrics["rolling_max_dd"] = np.nan

            logger.info(f"Rolling metrics calculated with window={window}")
            return rolling_metrics

        except Exception as e:
            logger.error(f"Failed to calculate rolling metrics: {e}")
            raise

    def export_results(self, filename: str, include_trades: bool = True) -> str:
        """
        Export performance results to CSV file.

        Args:
            filename (str): Output CSV filename
            include_trades (bool): Whether to include detailed trade data

        Returns:
            str: Path to saved file
        """
        try:
            # Helper function to safely convert values to exportable format
            def safe_convert_value(value):
                """Convert complex objects to simple exportable values."""
                try:
                    # Handle MappedArray objects from vectorbtpro
                    if hasattr(value, "values"):
                        val = value.values
                        if hasattr(val, "__len__") and len(val) > 0:
                            return (
                                float(val[0])
                                if np.isscalar(val[0])
                                else float(val.flat[0])
                            )
                        else:
                            return 0.0
                    # Handle numpy arrays
                    elif isinstance(value, np.ndarray):
                        if value.size > 0:
                            return float(value.flat[0])
                        else:
                            return 0.0
                    # Handle pandas Series
                    elif hasattr(value, "iloc") and len(value) > 0:
                        return float(value.iloc[0])
                    # Handle timedelta objects
                    elif hasattr(value, "days"):
                        return float(value.days)
                    # Handle regular numeric values
                    elif np.isscalar(value):
                        if pd.isna(value):
                            return 0.0
                        return float(value)
                    # Handle string representations of objects
                    elif isinstance(value, str) and "MappedArray" in value:
                        return 0.0  # Default for unparseable MappedArray strings
                    else:
                        return float(value) if value is not None else 0.0
                except Exception as e:
                    logger.debug(f"Could not convert value {type(value)}: {e}")
                    return 0.0

            # Combine all metrics
            all_metrics = {}

            # Add returns metrics
            returns_metrics = self.get_returns_metrics()
            for key, value in returns_metrics.items():
                all_metrics[key] = safe_convert_value(value)

            # Add trade metrics
            trade_metrics = self.get_trade_metrics()
            for key, value in trade_metrics.items():
                all_metrics[key] = safe_convert_value(value)

            # Add drawdown metrics
            drawdown_metrics = self.get_drawdown_metrics()
            for key, value in drawdown_metrics.items():
                all_metrics[key] = safe_convert_value(value)

            # Add risk metrics
            risk_metrics = self.get_risk_metrics()
            for key, value in risk_metrics.items():
                all_metrics[key] = safe_convert_value(value)

            # Add benchmark metrics if available
            benchmark_metrics = self.get_benchmark_comparison()
            if benchmark_metrics:
                for key, value in benchmark_metrics.items():
                    all_metrics[key] = safe_convert_value(value)

            # Convert to DataFrame
            metrics_df = pd.DataFrame([all_metrics])

            # Save main metrics
            filepath = Path(filename)
            metrics_df.to_csv(filepath, index=False)

            # Save detailed trade data if requested
            trade_count = self.portfolio.trades.count()
            if isinstance(trade_count, pd.Series):
                trade_count = trade_count.iloc[0] if len(trade_count) > 0 else 0
            elif hasattr(trade_count, "item"):
                trade_count = trade_count.item()
            trade_count = int(trade_count) if trade_count is not None else 0

            if include_trades and trade_count > 0:
                trades_filepath = filepath.with_suffix(".trades.csv")
                try:
                    trades_df = self.portfolio.trades.records_readable
                    trades_df = self._enhance_trades_with_strategy_levels(trades_df)
                    trades_df.to_csv(trades_filepath, index=False)
                    logger.info(f"Trade details saved to {trades_filepath}")
                except Exception as e:
                    logger.warning(f"Could not save trade details: {e}")

            # Save monthly returns
            monthly_filepath = filepath.with_suffix(".monthly.csv")
            try:
                monthly_returns = self.get_monthly_returns()
                monthly_returns.to_csv(monthly_filepath, header=["Monthly_Return"])
                logger.info(f"Monthly returns saved to {monthly_filepath}")
            except Exception as e:
                logger.warning(f"Could not save monthly returns: {e}")

            logger.info(f"Performance results exported to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise

    def generate_report(self, detailed: bool = True) -> str:
        """
        Generate formatted performance report.

        Args:
            detailed (bool): Whether to include detailed metrics

        Returns:
            str: Formatted performance report
        """
        try:
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("PORTFOLIO PERFORMANCE ANALYSIS REPORT")
            report_lines.append("=" * 60)

            # Summary metrics
            returns_metrics = self.get_returns_metrics()
            report_lines.append("\n📊 RETURNS METRICS")
            report_lines.append("-" * 30)
            report_lines.append(
                f"Total Return:        {returns_metrics['total_return']*100:>8.2f}%"
            )
            report_lines.append(
                f"Annualized Return:   {returns_metrics['annualized_return']*100:>8.2f}%"
            )
            report_lines.append(
                f"Volatility:          {returns_metrics['volatility']*100:>8.2f}%"
            )
            report_lines.append(
                f"Sharpe Ratio:        {returns_metrics['sharpe_ratio']:>8.3f}"
            )
            report_lines.append(
                f"Sortino Ratio:       {returns_metrics['sortino_ratio']:>8.3f}"
            )
            report_lines.append(
                f"Calmar Ratio:        {returns_metrics['calmar_ratio']:>8.3f}"
            )
            report_lines.append(
                f"Max Drawdown:        {returns_metrics['max_drawdown']*100:>8.2f}%"
            )

            # Trade metrics
            trade_metrics = self.get_trade_metrics()
            report_lines.append("\n📈 TRADE METRICS")
            report_lines.append("-" * 30)
            report_lines.append(
                f"Total Trades:        {trade_metrics['total_trades']:>8.0f}"
            )
            report_lines.append(
                f"Win Rate:            {trade_metrics['win_rate']*100:>8.2f}%"
            )
            report_lines.append(
                f"Profit Factor:       {trade_metrics['profit_factor']:>8.3f}"
            )
            report_lines.append(
                f"Avg Trade P&L:       {trade_metrics['avg_trade_pnl']:>8.2f}"
            )
            report_lines.append(
                f"Avg Winning Trade:   {trade_metrics['avg_winning_trade']:>8.2f}"
            )
            report_lines.append(
                f"Avg Losing Trade:    {trade_metrics['avg_losing_trade']:>8.2f}"
            )

            # Risk metrics
            if detailed:
                risk_metrics = self.get_risk_metrics()
                report_lines.append("\n⚠️  RISK METRICS")
                report_lines.append("-" * 30)
                report_lines.append(
                    f"VaR (95%):           {risk_metrics['var_95']*100:>8.2f}%"
                )
                report_lines.append(
                    f"CVaR (95%):          {risk_metrics['cvar_95']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Max Daily Loss:      {risk_metrics['max_daily_loss']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Max Daily Gain:      {risk_metrics['max_daily_gain']*100:>8.2f}%"
                )

                # Drawdown metrics
                dd_metrics = self.get_drawdown_metrics()
                report_lines.append("\n📉 DRAWDOWN METRICS")
                report_lines.append("-" * 30)
                report_lines.append(
                    f"Max Drawdown:        {dd_metrics['max_drawdown']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Avg Drawdown:        {dd_metrics['avg_drawdown']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Recovery Factor:     {dd_metrics['recovery_factor']:>8.3f}"
                )
                report_lines.append(
                    f"Drawdown Count:      {dd_metrics['drawdown_count']:>8.0f}"
                )

            # Benchmark comparison
            benchmark_metrics = self.get_benchmark_comparison()
            if benchmark_metrics:
                report_lines.append("\n📊 BENCHMARK COMPARISON")
                report_lines.append("-" * 30)
                report_lines.append(
                    f"Excess Return:       {benchmark_metrics['excess_return']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Information Ratio:   {benchmark_metrics['information_ratio']:>8.3f}"
                )
                report_lines.append(
                    f"Beta:                {benchmark_metrics['beta']:>8.3f}"
                )
                report_lines.append(
                    f"Alpha (Annual):      {benchmark_metrics['alpha_annualized']*100:>8.2f}%"
                )
                report_lines.append(
                    f"Correlation:         {benchmark_metrics['correlation']:>8.3f}"
                )

            report_lines.append("\n" + "=" * 60)

            report = "\n".join(report_lines)
            logger.info("Performance report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise

    def _calculate_max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Calculate maximum consecutive occurrences of a value."""
        if len(series) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for val in series:
            if val == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Performance analyzer cache cleared")

    def _enhance_trades_with_strategy_levels(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance trade records with stop loss and take profit levels from strategy signals.
        
        Args:
            trades_df: Standard VectorBT trade records
            
        Returns:
            Enhanced trade records with SL/TP levels
        """
        if self.signals is None:
            logger.debug("No strategy signals available for trade enhancement")
            return trades_df
        
        enhanced_df = trades_df.copy()
        
        # Initialize new columns for SL/TP levels
        enhanced_df['Strategy_SL_Price'] = np.nan
        enhanced_df['Strategy_TP_Price'] = np.nan
        enhanced_df['Strategy_SL_Pct'] = np.nan
        enhanced_df['Strategy_TP_Pct'] = np.nan
        
        try:
            # Get SL/TP data from signals if available
            sl_price_levels = self.signals.get('sl_price_levels')
            tp_price_levels = self.signals.get('tp_price_levels')
            sl_pct_levels = self.signals.get('sl_levels')
            tp_pct_levels = self.signals.get('tp_levels')
            
            logger.debug(f"Available signal data: SL prices={sl_price_levels is not None}, "
                        f"TP prices={tp_price_levels is not None}, "
                        f"SL pct={sl_pct_levels is not None}, "
                        f"TP pct={tp_pct_levels is not None}")
            
            # Match strategy levels to trades by entry timestamp
            for idx, trade in enhanced_df.iterrows():
                try:
                    # Get entry timestamp from trade record
                    entry_idx = trade.get('Entry Index')
                    if pd.isna(entry_idx):
                        continue
                    
                    # Convert to datetime if needed
                    if isinstance(entry_idx, str):
                        entry_timestamp = pd.to_datetime(entry_idx)
                    else:
                        entry_timestamp = entry_idx
                    
                    logger.debug(f"Processing trade {idx} with entry at {entry_timestamp}")
                    
                    # Extract SL price level
                    if sl_price_levels is not None and entry_timestamp in sl_price_levels.index:
                        sl_price = sl_price_levels.loc[entry_timestamp]
                        if not pd.isna(sl_price):
                            enhanced_df.loc[idx, 'Strategy_SL_Price'] = float(sl_price)
                            logger.debug(f"Added SL price ${sl_price:.4f} for trade {idx}")
                    
                    # Extract TP price level  
                    if tp_price_levels is not None and entry_timestamp in tp_price_levels.index:
                        tp_price = tp_price_levels.loc[entry_timestamp]
                        if not pd.isna(tp_price):
                            enhanced_df.loc[idx, 'Strategy_TP_Price'] = float(tp_price)
                            logger.debug(f"Added TP price ${tp_price:.4f} for trade {idx}")
                    
                    # Extract SL percentage level
                    if sl_pct_levels is not None and entry_timestamp in sl_pct_levels.index:
                        sl_pct = sl_pct_levels.loc[entry_timestamp]
                        if not pd.isna(sl_pct):
                            enhanced_df.loc[idx, 'Strategy_SL_Pct'] = float(sl_pct * 100)  # Convert to percentage
                            logger.debug(f"Added SL percentage {sl_pct*100:.2f}% for trade {idx}")
                    
                    # Extract TP percentage level
                    if tp_pct_levels is not None and entry_timestamp in tp_pct_levels.index:
                        tp_pct = tp_pct_levels.loc[entry_timestamp]
                        if not pd.isna(tp_pct):
                            enhanced_df.loc[idx, 'Strategy_TP_Pct'] = float(tp_pct * 100)  # Convert to percentage
                            logger.debug(f"Added TP percentage {tp_pct*100:.2f}% for trade {idx}")
                            
                except Exception as e:
                    logger.debug(f"Failed to enhance trade {idx}: {e}")
                    continue
            
            # Log enhancement results
            sl_price_count = enhanced_df['Strategy_SL_Price'].notna().sum()
            tp_price_count = enhanced_df['Strategy_TP_Price'].notna().sum()
            sl_pct_count = enhanced_df['Strategy_SL_Pct'].notna().sum()
            tp_pct_count = enhanced_df['Strategy_TP_Pct'].notna().sum()
            
            logger.info(f"Trade enhancement completed: {sl_price_count} SL prices, "
                       f"{tp_price_count} TP prices, {sl_pct_count} SL percentages, "
                       f"{tp_pct_count} TP percentages added")
                       
        except Exception as e:
            logger.warning(f"Failed to enhance trades with strategy levels: {e}")
        
        return enhanced_df
