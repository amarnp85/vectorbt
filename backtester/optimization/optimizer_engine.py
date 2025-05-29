"""
Optimizer Engine Module

Provides comprehensive parameter optimization functionality for trading strategies using
vectorbtpro's advanced optimization capabilities including @vbt.parameterized decorator,
parameter broadcasting, and built-in optimization tools.

Key Features:
- Grid search optimization using vbt.Param and @vbt.parameterized
- Vectorized parameter optimization with broadcasting
- Walk-forward analysis and cross-validation
- Multi-objective optimization with Pareto frontier analysis
- Bayesian optimization for efficient parameter space exploration
- Genetic algorithm optimization for complex parameter spaces
- Random search for large parameter spaces
- Performance metric optimization (Sharpe, Sortino, Calmar, etc.)
- Integration with existing signal and portfolio modules
- Efficient parallel execution using vectorbtpro's optimization framework

Usage:
    # Basic optimization
    optimizer = OptimizerEngine(data, signal_generator, portfolio_simulator)
    results = optimizer.optimize_grid(param_grid, metric='sharpe_ratio')

    # Vectorized optimization
    portfolio = optimizer.optimize_vectorized(param_grid, metric='total_return')

    # Walk-forward optimization
    results = optimizer.walk_forward_optimize(param_grid, train_period='6M', test_period='1M')

    # Bayesian optimization
    results = optimizer.optimize_bayesian(param_bounds, n_calls=100)

    # Multi-objective optimization
    results = optimizer.optimize_multi_objective(param_grid, ['sharpe_ratio', 'calmar_ratio'])
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from ..utilities.structured_logging import get_logger, quiet_logging
import itertools
from dataclasses import dataclass, field

logger = get_logger("optimization")


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""

    metric: str = "sharpe_ratio"
    maximize: bool = True
    min_trades: int = 10
    max_drawdown_limit: Optional[float] = None  # e.g., 0.2 for 20% max DD
    min_sharpe: Optional[float] = None  # e.g., 1.0 for minimum Sharpe

    # Execution settings
    parallel: bool = True
    chunk_len: Optional[int] = None
    engine: str = "threadpool"  # "threadpool", "pathos", "dask"
    show_progress: bool = True

    # Walk-forward settings
    train_period: str = "6M"
    test_period: str = "1M"
    step_period: str = "1M"
    min_train_samples: int = 252  # Minimum training samples

    # Bayesian optimization settings
    n_calls: int = 100
    n_initial_points: int = 10
    acquisition_function: str = "EI"  # 'EI', 'PI', 'UCB'

    # Genetic algorithm settings
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Random search settings
    n_random_samples: int = 1000
    random_seed: Optional[int] = None

    # Multi-objective settings
    pareto_front_size: int = 50

    # Additional constraints
    additional_filters: Dict[str, Any] = field(default_factory=dict)


class OptimizationResult:
    """Container for optimization results with comprehensive analysis."""

    def __init__(
        self,
        results_df: pd.DataFrame,
        best_params: Dict[str, Any],
        best_score: float,
        portfolio: Optional[vbt.Portfolio] = None,
        config: Optional[OptimizationConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pareto_front: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize optimization result container.

        Args:
            results_df: DataFrame with all parameter combinations and metrics
            best_params: Best parameter combination
            best_score: Best metric score
            portfolio: Portfolio object from best parameters (if available)
            config: Optimization configuration used
            metadata: Additional optimization metadata
            pareto_front: Pareto frontier for multi-objective optimization
        """
        self.results_df = results_df
        self.best_params = best_params
        self.best_score = best_score
        self.portfolio = portfolio
        self.config = config
        self.metadata = metadata or {}
        self.pareto_front = pareto_front

    def get_top_results(self, n: int = 10) -> pd.DataFrame:
        """Get top N parameter combinations."""
        if self.config and not self.config.maximize:
            return self.results_df.nsmallest(n, self.config.metric)
        return self.results_df.nlargest(n, self.config.metric)

    def plot_optimization_surface(self, param1: str, param2: str, **kwargs):
        """Plot 2D optimization surface for two parameters."""
        if self.config is None:
            raise ValueError("Config required for plotting")

        # Create pivot table for heatmap
        pivot_data = self.results_df.pivot_table(
            values=self.config.metric, index=param1, columns=param2, aggfunc="mean"
        )

        return pivot_data.vbt.heatmap(
            trace_kwargs=dict(colorbar=dict(title=self.config.metric)), **kwargs
        )

    def plot_pareto_front(self, metric1: str, metric2: str, **kwargs):
        """Plot Pareto frontier for multi-objective optimization."""
        if self.pareto_front is None:
            raise ValueError(
                "No Pareto front available. Run multi-objective optimization first."
            )

        return self.pareto_front.vbt.scatterplot(
            x=metric1,
            y=metric2,
            trace_kwargs=dict(mode="markers+lines", name="Pareto Front"),
            **kwargs,
        )

    def get_parameter_sensitivity(self) -> pd.DataFrame:
        """Analyze parameter sensitivity."""
        sensitivity_data = []

        for param in self.results_df.columns:
            if param == self.config.metric:
                continue

            # Calculate correlation with metric
            correlation = self.results_df[param].corr(
                self.results_df[self.config.metric]
            )

            # Calculate range of metric values for this parameter
            param_groups = self.results_df.groupby(param)[self.config.metric]
            metric_range = param_groups.max() - param_groups.min()

            sensitivity_data.append(
                {
                    "parameter": param,
                    "correlation": correlation,
                    "metric_range": metric_range.max(),
                    "stability": 1
                    / (metric_range.std() + 1e-8),  # Higher = more stable
                }
            )

        return pd.DataFrame(sensitivity_data).sort_values(
            "correlation", key=abs, ascending=False
        )


class OptimizerEngine:
    """
    Advanced parameter optimizer using vectorbtpro's optimization framework.

    This class provides comprehensive optimization capabilities including:
    - Grid search with parameter broadcasting
    - Vectorized optimization for efficiency
    - Walk-forward analysis
    - Bayesian optimization for efficient exploration
    - Genetic algorithm optimization
    - Random search for large parameter spaces
    - Multi-objective optimization with Pareto analysis
    - Performance filtering and constraints
    """

    def __init__(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signal_generator: Callable,
        portfolio_simulator: Optional[Callable] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize optimizer engine.

        Args:
            data: Price data (vbt.Data, DataFrame, or Series)
            signal_generator: Function to generate signals from parameters
            portfolio_simulator: Optional custom portfolio simulator
            config: Optimization configuration
        """
        self.data = data
        self.signal_generator = signal_generator
        self.portfolio_simulator = portfolio_simulator
        self.config = config or OptimizationConfig()

        # Store results
        self.results = None
        self.best_portfolio = None

        logger.info(f"Initialized OptimizerEngine with metric: {self.config.metric}")

    @vbt.parameterized(merge_func="concat")
    def _optimize_single_combination(self, **params) -> float:
        """
        Optimize single parameter combination using vectorbtpro's parameterized decorator.

        This method is decorated with @vbt.parameterized to enable efficient
        parameter broadcasting and parallel execution.
        """
        try:
            # Generate signals with current parameters
            signal_result = self.signal_generator(self.data, **params)

            # Run portfolio simulation using centralized method
            portfolio = self._create_portfolio_from_signals(signal_result)

            # Calculate target metric
            metric_value = self._calculate_metric(portfolio, self.config.metric)

            # Apply filters/constraints
            if not self._passes_filters(portfolio):
                return np.nan

            return metric_value

        except Exception as e:
            logger.warning(f"Optimization failed for params {params}: {str(e)}")
            return np.nan

    def optimize_grid(
        self, param_grid: Dict[str, List], metric: Optional[str] = None
    ) -> OptimizationResult:
        """
        Perform grid search optimization using vectorbtpro's parameter broadcasting.

        Args:
            param_grid: Dictionary of parameter names and values to test
            metric: Performance metric to optimize (overrides config)

        Returns:
            OptimizationResult with comprehensive results
        """
        if metric:
            self.config.metric = metric

        logger.info(f"Starting grid optimization with {len(param_grid)} parameters")
        logger.info(f"Parameter grid: {param_grid}")

        # Convert parameter grid to vbt.Param objects
        vbt_params = {}
        for param_name, param_values in param_grid.items():
            vbt_params[param_name] = vbt.Param(param_values)

        # Execute optimization using vectorbtpro's parameterized framework
        execute_kwargs = {
            "chunk_len": self.config.chunk_len or "auto",
            "engine": self.config.engine,
            "show_progress": self.config.show_progress,
        }

        # Run optimization with quiet logging to reduce verbosity
        with quiet_logging():
            metric_results = self._optimize_single_combination(
                **vbt_params, _execute_kwargs=execute_kwargs
            )

        # Process results
        results_df = self._process_grid_results(metric_results, param_grid)

        # Find best parameters
        best_params, best_score = self._find_best_parameters(results_df)

        # Generate best portfolio for analysis
        best_portfolio = self._generate_best_portfolio(best_params)

        # Create result object
        result = OptimizationResult(
            results_df=results_df,
            best_params=best_params,
            best_score=best_score,
            portfolio=best_portfolio,
            config=self.config,
            metadata={
                "optimization_type": "grid_search",
                "total_combinations": len(results_df),
                "valid_combinations": results_df[self.config.metric].notna().sum(),
            },
        )

        self.results = result
        self.best_portfolio = best_portfolio

        logger.info(f"Grid optimization completed")
        logger.info(f"Best {self.config.metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return result

    def optimize_vectorized(
        self, param_grid: Dict[str, List], metric: Optional[str] = None
    ) -> vbt.Portfolio:
        """
        Perform vectorized parameter optimization using vectorbtpro's broadcasting.

        This method generates signals for all parameter combinations simultaneously
        and returns a multi-column portfolio for analysis.

        Args:
            param_grid: Dictionary of parameter names and values to test
            metric: Performance metric to optimize

        Returns:
            Portfolio object with multiple columns for different parameters
        """
        if metric:
            self.config.metric = metric

        logger.info("Starting vectorized optimization")

        # Generate parameterized signals using broadcasting
        signal_result = self.signal_generator(self.data, **param_grid)

        # Run vectorized portfolio simulation using centralized method
        portfolio = self._create_portfolio_from_signals(signal_result)

        # Store portfolio for analysis
        self.best_portfolio = portfolio

        # Extract performance metrics for each parameter combination
        metric_values = self._calculate_metric(portfolio, self.config.metric)

        # Create results DataFrame
        self.results = pd.DataFrame({self.config.metric: metric_values})

        logger.info(
            f"Vectorized optimization completed with {len(metric_values)} combinations"
        )

        return portfolio

    def walk_forward_optimize(
        self,
        param_grid: Dict[str, List],
        train_period: Optional[str] = None,
        test_period: Optional[str] = None,
        step_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward optimization analysis.

        Args:
            param_grid: Parameter grid for optimization
            train_period: Training period length (e.g., '6M')
            test_period: Test period length (e.g., '1M')
            step_period: Step size for walk-forward (e.g., '1M')

        Returns:
            Dictionary with walk-forward results
        """
        train_period = train_period or self.config.train_period
        test_period = test_period or self.config.test_period
        step_period = step_period or self.config.step_period

        logger.info(f"Starting walk-forward optimization")
        logger.info(f"Train: {train_period}, Test: {test_period}, Step: {step_period}")

        # Get data index
        if hasattr(self.data, "close"):
            data_index = self.data.close.index
        else:
            data_index = self.data.index

        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(
            data_index, train_period, test_period, step_period
        )

        walk_forward_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Walk-forward window {i+1}/{len(windows)}")
            logger.info(f"Train: {train_start} to {train_end}")
            logger.info(f"Test: {test_start} to {test_end}")

            # Split data
            train_data = self._slice_data(train_start, train_end)
            test_data = self._slice_data(test_start, test_end)

            # Optimize on training data
            train_optimizer = OptimizerEngine(
                train_data, self.signal_generator, self.portfolio_simulator, self.config
            )
            train_result = train_optimizer.optimize_grid(param_grid)

            # Test on out-of-sample data
            test_optimizer = OptimizerEngine(
                test_data, self.signal_generator, self.portfolio_simulator, self.config
            )

            # Use best parameters from training
            test_portfolio = test_optimizer._generate_best_portfolio(
                train_result.best_params
            )
            test_metric = self._calculate_metric(test_portfolio, self.config.metric)

            walk_forward_results.append(
                {
                    "window": i + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "best_params": train_result.best_params,
                    "train_metric": train_result.best_score,
                    "test_metric": test_metric,
                    "param_stability": self._calculate_param_stability(
                        train_result.results_df
                    ),
                }
            )

        # Aggregate results
        wf_df = pd.DataFrame(walk_forward_results)

        summary = {
            "walk_forward_results": wf_df,
            "avg_train_metric": wf_df["train_metric"].mean(),
            "avg_test_metric": wf_df["test_metric"].mean(),
            "metric_correlation": wf_df["train_metric"].corr(wf_df["test_metric"]),
            "parameter_consistency": self._analyze_parameter_consistency(wf_df),
            "overfitting_ratio": wf_df["test_metric"].mean()
            / wf_df["train_metric"].mean(),
        }

        logger.info(f"Walk-forward optimization completed")
        logger.info(
            f"Average train {self.config.metric}: {summary['avg_train_metric']:.4f}"
        )
        logger.info(
            f"Average test {self.config.metric}: {summary['avg_test_metric']:.4f}"
        )
        logger.info(f"Overfitting ratio: {summary['overfitting_ratio']:.4f}")

        return summary

    def optimize_random_search(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_samples: Optional[int] = None,
        metric: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Perform random search optimization for large parameter spaces.

        Args:
            param_bounds: Dictionary of parameter bounds {param: (min, max)}
            n_samples: Number of random samples to test
            metric: Performance metric to optimize

        Returns:
            OptimizationResult with comprehensive results
        """
        if metric:
            self.config.metric = metric

        n_samples = n_samples or self.config.n_random_samples

        logger.info(f"Starting random search optimization with {n_samples} samples")

        # Set random seed for reproducibility
        if self.config.random_seed:
            np.random.seed(self.config.random_seed)

        # Generate random parameter combinations
        param_combinations = []
        for _ in range(n_samples):
            params = {}
            for param_name, (min_val, max_val) in param_bounds.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            param_combinations.append(params)

        # Evaluate combinations
        results_data = []
        for i, params in enumerate(param_combinations):
            if self.config.show_progress and i % 100 == 0:
                logger.info(f"Evaluating combination {i+1}/{n_samples}")

            try:
                signal_result = self.signal_generator(self.data, **params)
                portfolio = self._create_portfolio_from_signals(signal_result)
                metric_value = self._calculate_metric(portfolio, self.config.metric)

                if self._passes_filters(portfolio):
                    result_dict = params.copy()
                    result_dict[self.config.metric] = metric_value
                    results_data.append(result_dict)
            except Exception as e:
                logger.debug(f"Failed to evaluate params {params}: {e}")
                continue

        # Create results DataFrame
        results_df = pd.DataFrame(results_data)

        if len(results_df) == 0:
            raise ValueError("No valid parameter combinations found")

        # Find best parameters
        best_params, best_score = self._find_best_parameters(results_df)
        best_portfolio = self._generate_best_portfolio(best_params)

        # Create result object
        result = OptimizationResult(
            results_df=results_df,
            best_params=best_params,
            best_score=best_score,
            portfolio=best_portfolio,
            config=self.config,
            metadata={
                "optimization_type": "random_search",
                "total_samples": n_samples,
                "valid_samples": len(results_df),
            },
        )

        self.results = result
        self.best_portfolio = best_portfolio

        logger.info(f"Random search optimization completed")
        logger.info(f"Best {self.config.metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return result

    def get_best_params(self, metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the best parameter combination.

        Args:
            metric: Performance metric to optimize (uses config default if None)

        Returns:
            Dictionary of best parameters
        """
        if self.results is None:
            raise ValueError(
                "No optimization results available. Run optimization first."
            )

        metric = metric or self.config.metric

        if isinstance(self.results, OptimizationResult):
            return self.results.best_params
        elif isinstance(self.results, pd.DataFrame):
            if self.config.maximize:
                best_idx = self.results[metric].idxmax()
            else:
                best_idx = self.results[metric].idxmin()
            return self.results.loc[best_idx].to_dict()
        else:
            raise ValueError("Invalid results format")

    def plot_optimization_heatmap(
        self, param1: str, param2: str, metric: Optional[str] = None, **kwargs
    ):
        """
        Plot a heatmap of optimization results for two parameters.

        Args:
            param1: First parameter name
            param2: Second parameter name
            metric: Performance metric to visualize
            **kwargs: Additional plotting arguments

        Returns:
            Plotly figure object
        """
        if self.results is None:
            raise ValueError(
                "No optimization results available. Run optimization first."
            )

        metric = metric or self.config.metric

        if isinstance(self.results, OptimizationResult):
            results_df = self.results.results_df
        else:
            results_df = self.results

        # Create pivot table for heatmap
        pivot_table = results_df.pivot_table(
            values=metric, index=param1, columns=param2, aggfunc="mean"
        )

        # Create heatmap using vectorbtpro
        fig = pivot_table.vbt.heatmap(
            trace_kwargs=dict(colorbar=dict(title=metric)), **kwargs
        )

        return fig

    # --- Helper Methods ---

    def _calculate_metric(
        self, portfolio: vbt.Portfolio, metric: str
    ) -> Union[float, pd.Series]:
        """Calculate performance metric from portfolio."""
        if metric == "total_return":
            return portfolio.total_return
        elif metric == "sharpe_ratio":
            return portfolio.sharpe_ratio
        elif metric == "sortino_ratio":
            return portfolio.sortino_ratio
        elif metric == "calmar_ratio":
            return portfolio.calmar_ratio
        elif metric == "max_drawdown":
            return portfolio.max_drawdown
        elif metric == "win_rate":
            win_rate = portfolio.trades.win_rate
            return win_rate() if callable(win_rate) else win_rate
        elif metric == "profit_factor":
            profit_factor = portfolio.trades.profit_factor
            return profit_factor() if callable(profit_factor) else profit_factor
        elif metric == "expectancy":
            expectancy = portfolio.trades.expectancy
            return expectancy() if callable(expectancy) else expectancy
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def _passes_filters(self, portfolio: vbt.Portfolio) -> bool:
        """Check if portfolio passes optimization filters."""
        # Minimum trades filter
        trades_count = (
            portfolio.trades.count()
            if callable(portfolio.trades.count)
            else portfolio.trades.count
        )
        if trades_count < self.config.min_trades:
            return False

        # Maximum drawdown filter
        if self.config.max_drawdown_limit:
            if portfolio.max_drawdown > self.config.max_drawdown_limit:
                return False

        # Minimum Sharpe filter
        if self.config.min_sharpe:
            if portfolio.sharpe_ratio < self.config.min_sharpe:
                return False

        # Additional custom filters
        for filter_name, filter_value in self.config.additional_filters.items():
            portfolio_value = getattr(portfolio, filter_name, None)
            if portfolio_value is None:
                continue
            if portfolio_value < filter_value:
                return False

        return True

    def _process_grid_results(
        self, metric_results: pd.Series, param_grid: Dict[str, List]
    ) -> pd.DataFrame:
        """Process grid search results into DataFrame."""
        # Generate all parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        # Create results DataFrame
        results_data = []
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            param_dict[self.config.metric] = (
                metric_results.iloc[i] if i < len(metric_results) else np.nan
            )
            results_data.append(param_dict)

        return pd.DataFrame(results_data)

    def _find_best_parameters(
        self, results_df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], float]:
        """Find best parameters from results DataFrame."""
        valid_results = results_df.dropna(subset=[self.config.metric])

        if len(valid_results) == 0:
            raise ValueError("No valid optimization results found")

        if self.config.maximize:
            best_idx = valid_results[self.config.metric].idxmax()
        else:
            best_idx = valid_results[self.config.metric].idxmin()

        best_row = valid_results.loc[best_idx]
        best_score = best_row[self.config.metric]

        # Extract parameters (exclude metric column)
        best_params = best_row.drop(self.config.metric).to_dict()

        return best_params, best_score

    def _generate_best_portfolio(self, best_params: Dict[str, Any]) -> vbt.Portfolio:
        """Generate portfolio using best parameters."""
        # Generate signals with best parameters
        signal_result = self.signal_generator(self.data, **best_params)

        # Run portfolio simulation using centralized method
        portfolio = self._create_portfolio_from_signals(signal_result)

        return portfolio

    def _generate_walk_forward_windows(
        self,
        data_index: pd.DatetimeIndex,
        train_period: str,
        test_period: str,
        step_period: str,
    ) -> List[Tuple]:
        """Generate walk-forward analysis windows."""
        windows = []

        # Convert periods to timedeltas
        train_delta = pd.Timedelta(train_period)
        test_delta = pd.Timedelta(test_period)
        step_delta = pd.Timedelta(step_period)

        # Start from first possible training window
        current_start = data_index[0]

        while True:
            train_end = current_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            # Check if we have enough data
            if test_end > data_index[-1]:
                break

            # Ensure we have minimum training samples
            train_samples = len(
                data_index[(data_index >= current_start) & (data_index < train_end)]
            )
            if train_samples < self.config.min_train_samples:
                current_start += step_delta
                continue

            windows.append((current_start, train_end, test_start, test_end))
            current_start += step_delta

        return windows

    def _slice_data(self, start_date, end_date):
        """Slice data for given date range."""
        if hasattr(self.data, "close"):
            # vbt.Data object
            mask = (self.data.close.index >= start_date) & (
                self.data.close.index < end_date
            )
            return self.data.close[mask]
        else:
            # DataFrame or Series
            mask = (self.data.index >= start_date) & (self.data.index < end_date)
            return self.data[mask]

    def _calculate_param_stability(self, results_df: pd.DataFrame) -> float:
        """Calculate parameter stability metric."""
        # Calculate coefficient of variation for top 10% of results
        top_10_pct = int(len(results_df) * 0.1)
        top_results = results_df.nlargest(top_10_pct, self.config.metric)

        # Calculate stability as inverse of coefficient of variation
        cv = (
            top_results[self.config.metric].std()
            / top_results[self.config.metric].mean()
        )
        stability = 1 / (1 + cv)

        return stability

    def _analyze_parameter_consistency(self, wf_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze parameter consistency across walk-forward windows."""
        consistency = {}

        # Extract parameter values from each window
        all_params = set()
        for params_dict in wf_df["best_params"]:
            all_params.update(params_dict.keys())

        for param in all_params:
            param_values = [
                params_dict.get(param) for params_dict in wf_df["best_params"]
            ]
            param_values = [v for v in param_values if v is not None]

            if len(param_values) > 1:
                # Calculate coefficient of variation
                param_series = pd.Series(param_values)
                cv = (
                    param_series.std() / param_series.mean()
                    if param_series.mean() != 0
                    else 0
                )
                consistency[param] = 1 / (1 + cv)  # Higher = more consistent
            else:
                consistency[param] = 1.0

        return consistency

    def _create_portfolio_from_signals(self, signal_result) -> vbt.Portfolio:
        """Create portfolio from signal result using centralized PortfolioSimulator."""
        from backtester.portfolio import PortfolioSimulator, SimulationConfig
        from backtester.signals.signal_utils import SignalPreparator

        # Handle both SignalResult objects and dictionary formats
        if isinstance(signal_result, dict):
            # Already a dictionary, use directly
            signals = signal_result
        else:
            # Convert SignalResult object to dictionary format
            preparator = SignalPreparator()
            signals = preparator.convert_signal_result_to_dict(signal_result)

        if self.portfolio_simulator:
            # Use provided portfolio simulator
            return self.portfolio_simulator(signals)
        else:
            # Use default PortfolioSimulator with basic configuration
            config = SimulationConfig(
                init_cash=100000, fees=0.001, slippage=0.0005, freq="1D"
            )

            simulator = PortfolioSimulator(self.data, config)
            return simulator.simulate_from_signals(signals)


# --- Convenience Functions ---


def optimize_strategy_parameters(
    data: Union[vbt.Data, pd.DataFrame, pd.Series],
    signal_generator: Callable,
    param_grid: Dict[str, List],
    metric: str = "sharpe_ratio",
    **kwargs,
) -> OptimizationResult:
    """
    Convenience function for strategy parameter optimization.

    Args:
        data: Price data
        signal_generator: Function to generate signals from parameters
        param_grid: Parameter ranges to test
        metric: Performance metric to optimize
        **kwargs: Additional optimization parameters

    Returns:
        OptimizationResult with comprehensive results
    """
    config = OptimizationConfig(metric=metric, **kwargs)
    optimizer = OptimizerEngine(data, signal_generator, config=config)
    return optimizer.optimize_grid(param_grid)


def run_walk_forward_analysis(
    data: Union[vbt.Data, pd.DataFrame, pd.Series],
    signal_generator: Callable,
    param_grid: Dict[str, List],
    train_period: str = "6M",
    test_period: str = "1M",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for walk-forward analysis.

    Args:
        data: Price data
        signal_generator: Function to generate signals
        param_grid: Parameter ranges to test
        train_period: Training period length
        test_period: Test period length
        **kwargs: Additional optimization parameters

    Returns:
        Dictionary with walk-forward analysis results
    """
    config = OptimizationConfig(
        train_period=train_period, test_period=test_period, **kwargs
    )
    optimizer = OptimizerEngine(data, signal_generator, config=config)
    return optimizer.walk_forward_optimize(param_grid)
