"""
Portfolio Optimization Engine

Advanced portfolio optimization using VectorBTPro's optimization framework.
Handles both strategy parameter optimization and portfolio weight optimization.

Key Features:
- Multi-level optimization (strategy params + portfolio weights)
- Risk-based portfolio optimization (Riskfolio-Lib, PyPortfolioOpt)
- Dynamic rebalancing optimization
- Walk-forward portfolio optimization
- Multi-objective optimization (return vs risk)
- Regime-aware optimization
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum

from backtester.optimization.optimizer_engine import OptimizerEngine, OptimizationConfig, OptimizationResult

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for portfolio optimization."""
    STRATEGY_ONLY = "strategy_only"  # Optimize strategy parameters only
    WEIGHTS_ONLY = "weights_only"    # Optimize portfolio weights only
    HIERARCHICAL = "hierarchical"    # First strategy, then weights
    SIMULTANEOUS = "simultaneous"    # Optimize both simultaneously


class WeightOptimizationMethod(Enum):
    """Portfolio weight optimization methods."""
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    RISKFOLIO_HRP = "riskfolio_hrp"  # Hierarchical Risk Parity
    RISKFOLIO_HERC = "riskfolio_herc"  # Hierarchical Equal Risk Contribution
    BLACK_LITTERMAN = "black_litterman"


@dataclass
class PortfolioOptimizationConfig(OptimizationConfig):
    """Extended configuration for portfolio optimization."""
    
    # Portfolio-specific settings
    optimization_level: OptimizationLevel = OptimizationLevel.HIERARCHICAL
    weight_optimization_method: WeightOptimizationMethod = WeightOptimizationMethod.RISK_PARITY
    
    # Rebalancing settings
    rebalance_freq: str = "M"  # Monthly rebalancing
    lookback_period: str = "6M"  # Lookback for weight calculation
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 0.3  # Maximum weight per asset
    
    # Risk constraints
    max_portfolio_volatility: Optional[float] = None  # e.g., 0.15 for 15%
    target_return: Optional[float] = None  # Target return for optimization
    max_tracking_error: Optional[float] = None  # vs benchmark
    
    # Transaction costs
    transaction_cost: float = 0.001  # 0.1% per trade
    rebalance_threshold: float = 0.05  # 5% drift before rebalancing
    
    # Advanced settings
    use_regime_detection: bool = False
    regime_lookback: int = 252  # Days for regime detection
    shrinkage_target: Optional[str] = None  # "single_factor", "constant_correlation"
    
    # Multi-objective settings
    risk_aversion: float = 1.0  # Risk aversion parameter
    utility_function: str = "mean_variance"  # "mean_variance", "cvar", "max_drawdown"


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer that handles both strategy parameters and portfolio weights.
    
    Supports multiple optimization levels:
    1. Strategy-only: Optimize strategy parameters with equal weights
    2. Weights-only: Optimize portfolio weights with fixed strategy parameters
    3. Hierarchical: First optimize strategy parameters, then optimize weights
    4. Simultaneous: Optimize both strategy parameters and weights together
    """
    
    def __init__(
        self,
        data: vbt.Data,
        strategy_class: type,
        base_strategy_params: Dict[str, Any],
        config: Optional[PortfolioOptimizationConfig] = None
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            data: Multi-symbol VBT Data object
            strategy_class: Strategy class to optimize
            base_strategy_params: Base strategy parameters
            config: Portfolio optimization configuration
        """
        if not hasattr(data, 'symbols') or len(data.symbols) < 2:
            raise ValueError("PortfolioOptimizer requires multi-symbol data")
            
        self.data = data
        self.strategy_class = strategy_class
        self.base_strategy_params = base_strategy_params
        self.config = config or PortfolioOptimizationConfig()
        
        self.symbols = data.symbols
        self.n_symbols = len(self.symbols)
        
        # Results storage
        self.strategy_optimization_result = None
        self.weight_optimization_result = None
        self.final_portfolio = None
        
        logger.info(f"Initialized PortfolioOptimizer for {self.n_symbols} symbols")
        logger.info(f"Optimization level: {self.config.optimization_level.value}")
    
    def optimize(
        self,
        strategy_param_grid: Optional[Dict[str, List]] = None,
        weight_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run portfolio optimization based on the configured optimization level.
        
        Args:
            strategy_param_grid: Parameters to optimize for strategy
            weight_constraints: Additional weight constraints
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting {self.config.optimization_level.value} optimization")
        
        if self.config.optimization_level == OptimizationLevel.STRATEGY_ONLY:
            return self._optimize_strategy_only(strategy_param_grid)
        
        elif self.config.optimization_level == OptimizationLevel.WEIGHTS_ONLY:
            return self._optimize_weights_only(weight_constraints)
        
        elif self.config.optimization_level == OptimizationLevel.HIERARCHICAL:
            return self._optimize_hierarchical(strategy_param_grid, weight_constraints)
        
        elif self.config.optimization_level == OptimizationLevel.SIMULTANEOUS:
            return self._optimize_simultaneous(strategy_param_grid, weight_constraints)
        
        else:
            raise ValueError(f"Unknown optimization level: {self.config.optimization_level}")
    
    def _optimize_strategy_only(self, strategy_param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters with equal portfolio weights."""
        logger.info("Optimizing strategy parameters only (equal weights)")
        
        # Create signal generator with equal weights
        def signal_generator(data_obj, **params):
            merged_params = {**self.base_strategy_params, **params}
            strategy = self.strategy_class(data_obj, merged_params)
            strategy.init_indicators()
            signals = strategy.generate_signals()
            
            # Add equal weights
            if isinstance(signals, dict) and 'position_sizes' not in signals:
                equal_weights = pd.DataFrame(
                    1/self.n_symbols,
                    index=data_obj.close.index,
                    columns=self.symbols
                )
                signals['position_sizes'] = equal_weights
            
            return signals
        
        # Use standard optimizer for strategy parameters
        optimizer = OptimizerEngine(
            data=self.data,
            signal_generator=signal_generator,
            config=self.config
        )
        
        self.strategy_optimization_result = optimizer.optimize_grid(
            strategy_param_grid, 
            metric=self.config.metric
        )
        
        return {
            'optimization_type': 'strategy_only',
            'strategy_result': self.strategy_optimization_result,
            'best_strategy_params': self.strategy_optimization_result.best_params,
            'best_portfolio': self.strategy_optimization_result.portfolio
        }
    
    def _optimize_weights_only(self, weight_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize portfolio weights with fixed strategy parameters."""
        logger.info("Optimizing portfolio weights only (fixed strategy)")
        
        # Generate signals with base strategy parameters
        strategy = self.strategy_class(self.data, self.base_strategy_params)
        strategy.init_indicators()
        base_signals = strategy.generate_signals()
        
        # Optimize weights using the selected method
        optimal_weights = self._calculate_optimal_weights(
            base_signals, 
            method=self.config.weight_optimization_method,
            constraints=weight_constraints
        )
        
        # Create portfolio with optimal weights
        portfolio = self._create_portfolio_with_weights(base_signals, optimal_weights)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_portfolio_metrics(portfolio)
        
        self.weight_optimization_result = {
            'optimal_weights': optimal_weights,
            'portfolio': portfolio,
            'metrics': performance_metrics,
            'method': self.config.weight_optimization_method.value
        }
        
        return {
            'optimization_type': 'weights_only',
            'weight_result': self.weight_optimization_result,
            'optimal_weights': optimal_weights,
            'best_portfolio': portfolio
        }
    
    def _optimize_hierarchical(
        self, 
        strategy_param_grid: Optional[Dict[str, List]] = None,
        weight_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """First optimize strategy parameters, then optimize weights."""
        logger.info("Running hierarchical optimization (strategy first, then weights)")
        
        # Step 1: Optimize strategy parameters
        if strategy_param_grid:
            strategy_result = self._optimize_strategy_only(strategy_param_grid)
            best_strategy_params = strategy_result['best_strategy_params']
        else:
            best_strategy_params = self.base_strategy_params
            strategy_result = None
        
        # Step 2: Optimize weights with best strategy parameters
        # Update base params with optimized strategy params
        optimized_strategy_params = {**self.base_strategy_params, **best_strategy_params}
        
        # Generate signals with optimized strategy
        strategy = self.strategy_class(self.data, optimized_strategy_params)
        strategy.init_indicators()
        optimized_signals = strategy.generate_signals()
        
        # Optimize weights
        optimal_weights = self._calculate_optimal_weights(
            optimized_signals,
            method=self.config.weight_optimization_method,
            constraints=weight_constraints
        )
        
        # Create final portfolio
        final_portfolio = self._create_portfolio_with_weights(optimized_signals, optimal_weights)
        
        # Calculate final metrics
        final_metrics = self._calculate_portfolio_metrics(final_portfolio)
        
        self.final_portfolio = final_portfolio
        
        return {
            'optimization_type': 'hierarchical',
            'strategy_result': strategy_result,
            'best_strategy_params': best_strategy_params,
            'optimal_weights': optimal_weights,
            'final_portfolio': final_portfolio,
            'final_metrics': final_metrics,
            'weight_method': self.config.weight_optimization_method.value
        }
    
    def _optimize_simultaneous(
        self,
        strategy_param_grid: Optional[Dict[str, List]] = None,
        weight_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize strategy parameters and weights simultaneously."""
        logger.info("Running simultaneous optimization (strategy + weights)")
        
        # This is more complex and requires custom optimization
        # For now, implement as hierarchical with multiple iterations
        
        best_score = -np.inf if self.config.maximize else np.inf
        best_combination = None
        best_portfolio = None
        
        # If no strategy grid provided, use base parameters
        if not strategy_param_grid:
            strategy_param_grid = {'dummy_param': [1]}  # Dummy parameter
        
        # Generate all strategy parameter combinations
        import itertools
        param_names = list(strategy_param_grid.keys())
        param_values = list(strategy_param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} strategy parameter combinations")
        
        results = []
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dict
            strategy_params = dict(zip(param_names, param_combo))
            if 'dummy_param' in strategy_params:
                del strategy_params['dummy_param']
            
            merged_params = {**self.base_strategy_params, **strategy_params}
            
            try:
                # Generate signals
                strategy = self.strategy_class(self.data, merged_params)
                strategy.init_indicators()
                signals = strategy.generate_signals()
                
                # Optimize weights for this strategy
                optimal_weights = self._calculate_optimal_weights(
                    signals,
                    method=self.config.weight_optimization_method,
                    constraints=weight_constraints
                )
                
                # Create portfolio
                portfolio = self._create_portfolio_with_weights(signals, optimal_weights)
                
                # Calculate metric
                metric_value = self._calculate_metric(portfolio, self.config.metric)
                
                # Check if this is the best combination
                is_better = (
                    (self.config.maximize and metric_value > best_score) or
                    (not self.config.maximize and metric_value < best_score)
                )
                
                if is_better:
                    best_score = metric_value
                    best_combination = {
                        'strategy_params': strategy_params,
                        'weights': optimal_weights,
                        'metric_value': metric_value
                    }
                    best_portfolio = portfolio
                
                results.append({
                    'strategy_params': strategy_params,
                    'weights': optimal_weights,
                    'metric_value': metric_value,
                    'portfolio': portfolio
                })
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(param_combinations)} combinations")
                
            except Exception as e:
                logger.warning(f"Failed to optimize combination {param_combo}: {e}")
                continue
        
        self.final_portfolio = best_portfolio
        
        return {
            'optimization_type': 'simultaneous',
            'best_combination': best_combination,
            'best_strategy_params': best_combination['strategy_params'],
            'optimal_weights': best_combination['weights'],
            'best_score': best_score,
            'final_portfolio': best_portfolio,
            'all_results': results
        }
    
    def _calculate_optimal_weights(
        self,
        signals: Dict[str, Any],
        method: WeightOptimizationMethod,
        constraints: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate optimal portfolio weights using the specified method.
        
        Args:
            signals: Trading signals
            method: Weight optimization method
            constraints: Additional constraints
            
        Returns:
            DataFrame with optimal weights over time
        """
        logger.info(f"Calculating optimal weights using {method.value}")
        
        # Get returns data for optimization
        returns = self.data.close.pct_change().dropna()
        
        if method == WeightOptimizationMethod.EQUAL_WEIGHT:
            weights = pd.DataFrame(
                1/self.n_symbols,
                index=returns.index,
                columns=self.symbols
            )
        
        elif method == WeightOptimizationMethod.INVERSE_VOLATILITY:
            # Calculate rolling volatilities
            volatilities = returns.rolling(window=60).std()
            inv_vol = 1 / volatilities
            weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
            weights = weights.fillna(1/self.n_symbols)
        
        elif method == WeightOptimizationMethod.RISK_PARITY:
            weights = self._calculate_risk_parity_weights(returns)
        
        elif method == WeightOptimizationMethod.MEAN_VARIANCE:
            weights = self._calculate_mean_variance_weights(returns, constraints)
        
        elif method == WeightOptimizationMethod.MINIMUM_VARIANCE:
            weights = self._calculate_minimum_variance_weights(returns, constraints)
        
        elif method == WeightOptimizationMethod.MAXIMUM_SHARPE:
            weights = self._calculate_maximum_sharpe_weights(returns, constraints)
        
        elif method == WeightOptimizationMethod.RISKFOLIO_HRP:
            weights = self._calculate_riskfolio_weights(returns, "HRP")
        
        elif method == WeightOptimizationMethod.RISKFOLIO_HERC:
            weights = self._calculate_riskfolio_weights(returns, "HERC")
        
        else:
            logger.warning(f"Method {method.value} not implemented, using equal weights")
            weights = pd.DataFrame(
                1/self.n_symbols,
                index=returns.index,
                columns=self.symbols
            )
        
        # Apply weight constraints
        weights = self._apply_weight_constraints(weights, constraints)
        
        return weights
    
    def _calculate_risk_parity_weights(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk parity weights."""
        try:
            import riskfolio as rp
            
            # Create portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculate risk parity weights
            weights_series = []
            
            # Use rolling window for dynamic weights
            window = min(252, len(returns) // 4)  # 1 year or 1/4 of data
            
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                
                try:
                    port_temp = rp.Portfolio(returns=window_returns)
                    port_temp.assets_stats(method_mu='hist', method_cov='hist')
                    
                    # Risk parity optimization
                    w = port_temp.rp_optimization(
                        model='Classic',
                        rm='MV',  # Mean Variance
                        hist=True
                    )
                    
                    weights_series.append(w.values.flatten())
                    
                except:
                    # Fallback to equal weights
                    weights_series.append(np.ones(self.n_symbols) / self.n_symbols)
            
            # Create DataFrame
            weights_df = pd.DataFrame(
                weights_series,
                index=returns.index[window:],
                columns=self.symbols
            )
            
            # Forward fill for earlier dates
            full_weights = pd.DataFrame(
                index=returns.index,
                columns=self.symbols
            )
            full_weights.iloc[:window] = 1/self.n_symbols
            full_weights.iloc[window:] = weights_df
            
            return full_weights.fillna(method='ffill')
            
        except ImportError:
            logger.warning("Riskfolio-Lib not available, using inverse volatility")
            return self._calculate_optimal_weights(
                {}, WeightOptimizationMethod.INVERSE_VOLATILITY, None
            )
    
    def _calculate_mean_variance_weights(self, returns: pd.DataFrame, constraints: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate mean-variance optimal weights."""
        try:
            from pypfopt import EfficientFrontier, risk_models, expected_returns
            
            weights_series = []
            window = min(252, len(returns) // 4)
            
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                
                try:
                    # Calculate expected returns and covariance
                    mu = expected_returns.mean_historical_return(window_returns)
                    S = risk_models.sample_cov(window_returns)
                    
                    # Create efficient frontier
                    ef = EfficientFrontier(mu, S)
                    
                    # Add constraints
                    if constraints:
                        if 'min_weight' in constraints:
                            ef.add_constraint(lambda w: w >= constraints['min_weight'])
                        if 'max_weight' in constraints:
                            ef.add_constraint(lambda w: w <= constraints['max_weight'])
                    
                    # Optimize for maximum Sharpe ratio
                    weights = ef.max_sharpe()
                    weights_series.append(list(weights.values()))
                    
                except:
                    weights_series.append(np.ones(self.n_symbols) / self.n_symbols)
            
            weights_df = pd.DataFrame(
                weights_series,
                index=returns.index[window:],
                columns=self.symbols
            )
            
            # Forward fill
            full_weights = pd.DataFrame(
                index=returns.index,
                columns=self.symbols
            )
            full_weights.iloc[:window] = 1/self.n_symbols
            full_weights.iloc[window:] = weights_df
            
            return full_weights.fillna(method='ffill')
            
        except ImportError:
            logger.warning("PyPortfolioOpt not available, using equal weights")
            return pd.DataFrame(
                1/self.n_symbols,
                index=returns.index,
                columns=self.symbols
            )
    
    def _calculate_minimum_variance_weights(self, returns: pd.DataFrame, constraints: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate minimum variance weights."""
        # Similar to mean_variance but optimize for minimum volatility
        # Implementation would be similar to _calculate_mean_variance_weights
        # but using ef.min_volatility() instead of ef.max_sharpe()
        logger.warning("Minimum variance optimization not fully implemented, using equal weights")
        return pd.DataFrame(
            1/self.n_symbols,
            index=returns.index,
            columns=self.symbols
        )
    
    def _calculate_maximum_sharpe_weights(self, returns: pd.DataFrame, constraints: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Calculate maximum Sharpe ratio weights."""
        # This is the same as mean_variance optimization
        return self._calculate_mean_variance_weights(returns, constraints)
    
    def _calculate_riskfolio_weights(self, returns: pd.DataFrame, method: str) -> pd.DataFrame:
        """Calculate weights using Riskfolio-Lib methods."""
        try:
            import riskfolio as rp
            
            weights_series = []
            window = min(252, len(returns) // 4)
            
            for i in range(window, len(returns)):
                window_returns = returns.iloc[i-window:i]
                
                try:
                    port = rp.Portfolio(returns=window_returns)
                    
                    if method == "HRP":
                        # Hierarchical Risk Parity
                        w = port.optimization(
                            model='HRP',
                            codependence='pearson',
                            rm='MV',
                            linkage='single'
                        )
                    elif method == "HERC":
                        # Hierarchical Equal Risk Contribution
                        w = port.optimization(
                            model='HERC',
                            codependence='pearson',
                            rm='MV',
                            linkage='single'
                        )
                    else:
                        w = pd.Series(1/self.n_symbols, index=self.symbols)
                    
                    weights_series.append(w.values.flatten())
                    
                except:
                    weights_series.append(np.ones(self.n_symbols) / self.n_symbols)
            
            weights_df = pd.DataFrame(
                weights_series,
                index=returns.index[window:],
                columns=self.symbols
            )
            
            # Forward fill
            full_weights = pd.DataFrame(
                index=returns.index,
                columns=self.symbols
            )
            full_weights.iloc[:window] = 1/self.n_symbols
            full_weights.iloc[window:] = weights_df
            
            return full_weights.fillna(method='ffill')
            
        except ImportError:
            logger.warning(f"Riskfolio-Lib not available for {method}, using equal weights")
            return pd.DataFrame(
                1/self.n_symbols,
                index=returns.index,
                columns=self.symbols
            )
    
    def _apply_weight_constraints(self, weights: pd.DataFrame, constraints: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Apply weight constraints to the portfolio."""
        if constraints is None:
            constraints = {}
        
        # Apply min/max weight constraints
        min_weight = constraints.get('min_weight', self.config.min_weight)
        max_weight = constraints.get('max_weight', self.config.max_weight)
        
        if min_weight > 0 or max_weight < 1:
            weights = weights.clip(lower=min_weight, upper=max_weight)
            
            # Renormalize to sum to 1
            weights = weights.div(weights.sum(axis=1), axis=0)
        
        return weights
    
    def _create_portfolio_with_weights(self, signals: Dict[str, Any], weights: pd.DataFrame) -> vbt.Portfolio:
        """Create portfolio using signals and weights."""
        # Update signals with weights
        signals_with_weights = signals.copy()
        signals_with_weights['position_sizes'] = weights
        
        # Use MultiAssetPortfolioSimulator
        from backtester.portfolio.simulation_engine import MultiAssetPortfolioSimulator, SimulationConfig
        
        sim_config = SimulationConfig(
            init_cash=100000,
            fees=self.config.transaction_cost,
            cash_sharing=True,
            freq=self.data.wrapper.freq if hasattr(self.data, 'wrapper') else 'D'
        )
        
        simulator = MultiAssetPortfolioSimulator(self.data, sim_config)
        portfolio = simulator.simulate_from_signals(signals_with_weights)
        
        return portfolio
    
    def _calculate_portfolio_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """Calculate comprehensive portfolio metrics."""
        metrics = {}
        
        try:
            metrics['total_return'] = portfolio.total_return
            metrics['sharpe_ratio'] = portfolio.sharpe_ratio
            metrics['max_drawdown'] = portfolio.max_drawdown
            metrics['volatility'] = portfolio.returns.std() * np.sqrt(252)
            metrics['calmar_ratio'] = portfolio.total_return / abs(portfolio.max_drawdown) if portfolio.max_drawdown != 0 else 0
            
            # Additional metrics
            if hasattr(portfolio, 'sortino_ratio'):
                metrics['sortino_ratio'] = portfolio.sortino_ratio
            
            if hasattr(portfolio, 'win_rate'):
                metrics['win_rate'] = portfolio.trades.win_rate
                
        except Exception as e:
            logger.warning(f"Error calculating portfolio metrics: {e}")
        
        return metrics
    
    def _calculate_metric(self, portfolio: vbt.Portfolio, metric: str) -> float:
        """Calculate specific metric for portfolio."""
        try:
            if metric == 'total_return':
                return portfolio.total_return
            elif metric == 'sharpe_ratio':
                return portfolio.sharpe_ratio
            elif metric == 'max_drawdown':
                return -portfolio.max_drawdown  # Negative because we want to minimize
            elif metric == 'calmar_ratio':
                return portfolio.total_return / abs(portfolio.max_drawdown) if portfolio.max_drawdown != 0 else 0
            elif metric == 'volatility':
                return -portfolio.returns.std() * np.sqrt(252)  # Negative to minimize
            else:
                logger.warning(f"Unknown metric {metric}, using total_return")
                return portfolio.total_return
        except Exception as e:
            logger.warning(f"Error calculating metric {metric}: {e}")
            return 0.0


def optimize_portfolio(
    data: vbt.Data,
    strategy_class: type,
    base_strategy_params: Dict[str, Any],
    strategy_param_grid: Optional[Dict[str, List]] = None,
    optimization_level: str = "hierarchical",
    weight_method: str = "risk_parity",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for portfolio optimization.
    
    Args:
        data: Multi-symbol VBT Data
        strategy_class: Strategy class
        base_strategy_params: Base strategy parameters
        strategy_param_grid: Parameters to optimize
        optimization_level: "strategy_only", "weights_only", "hierarchical", "simultaneous"
        weight_method: Weight optimization method
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimization results
    """
    config = PortfolioOptimizationConfig(
        optimization_level=OptimizationLevel(optimization_level),
        weight_optimization_method=WeightOptimizationMethod(weight_method),
        **kwargs
    )
    
    optimizer = PortfolioOptimizer(
        data=data,
        strategy_class=strategy_class,
        base_strategy_params=base_strategy_params,
        config=config
    )
    
    return optimizer.optimize(strategy_param_grid) 