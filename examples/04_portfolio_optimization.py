#!/usr/bin/env python3
"""
Example 04: Portfolio Weight Optimization

This example demonstrates how to optimize portfolio weights using different methods:
- Equal weight
- Inverse volatility
- Risk parity
- Maximum Sharpe
- Mean-variance optimization

It builds upon the previous examples and uses stored optimal parameters.
"""

import os
import sys
import pandas as pd
import numpy as np
import vectorbtpro as vbt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester.data import fetch_data
from backtester.optimization.portfolio_optimizer import PortfolioOptimizer
from backtester.config.config_manager import ConfigManager
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.config.optimal_parameters_db import OptimalParametersDB  # Updated import
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.utilities.structured_logging import setup_logging, get_logger


def run_portfolio_optimization():
    """Run portfolio optimization example with different weighting methods."""
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🎯 Portfolio Weight Optimization", "Testing different allocation strategies")
    
    # Configuration
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT"]
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Initialize managers
    config_manager = ConfigManager()
    param_db = OptimalParametersDB()  # Updated to use database
    
    # 1. Load data for all symbols
    with logger.operation("Loading market data"):
        data = fetch_data(
            symbols=symbols,
            exchange="binance",
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None:
            logger.error("Failed to load data")
            return
        
        # Log data summary
        data_points = len(data.get('close'))
        logger.data_summary(symbols, timeframe, start_date, end_date, data_points)
    
    # 2. Run strategy for each symbol with optimal parameters
    logger.section("📊 Individual Symbol Analysis")
    
    portfolio_results = {}
    returns_data = {}
    
    for symbol in symbols:
        with logger.operation(f"Analyzing {symbol}"):
            try:
                # Get optimal parameters for this symbol
                optimal_params = param_db.get_optimization_summary(symbol, timeframe)
                
                if optimal_params:
                    strategy_params = optimal_params['parameters']
                    logger.info(f"Using optimal parameters (Sharpe: {optimal_params['optimization_metric']:.3f})")
                else:
                    # Load default parameters
                    config_path = Path(__file__).parent.parent / "backtester" / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"
                    config = config_manager.load_config(str(config_path))
                    strategy_params = config.get("default_parameters", {})
                    logger.info("Using default parameters")
                
                # Get symbol-specific data
                symbol_data = data.select(symbol) if hasattr(data, 'select') else data
                
                # Run strategy
                strategy = DMAATRTrendStrategy(symbol_data, **strategy_params)
                
                # Create simulation config
                sim_config = SimulationConfig(
                    initial_capital=100000,
                    commission=0.001,
                    slippage=0.0005
                )
                
                # Run backtest
                simulator = PortfolioSimulator(sim_config)
                portfolio = simulator.run_backtest(strategy)
                
                if portfolio:
                    analyzer = PerformanceAnalyzer(portfolio)
                    
                    # Store results
                    portfolio_results[symbol] = {
                        'portfolio': portfolio,
                        'analyzer': analyzer,
                        'total_return': analyzer.total_return(),
                        'sharpe_ratio': analyzer.sharpe_ratio(),
                        'volatility': analyzer.volatility(),
                        'max_drawdown': analyzer.max_drawdown()
                    }
                    
                    # Extract returns for optimization
                    returns_data[symbol] = portfolio.returns()
                    
                    logger.info(f"Return: {portfolio_results[symbol]['total_return']:.2%}, "
                               f"Sharpe: {portfolio_results[symbol]['sharpe_ratio']:.3f}")
                else:
                    logger.error(f"Failed to run backtest for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
    
    if len(portfolio_results) < 2:
        logger.error("Need at least 2 successful symbol results for portfolio optimization")
        return
    
    # 3. Portfolio optimization
    logger.section("⚖️ Portfolio Weight Optimization")
    
    # Prepare returns matrix
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 50:
        logger.warning(f"Limited data for optimization: {len(returns_df)} periods")
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer(returns_df)
    
    # Test different allocation methods
    allocation_methods = {
        'equal_weight': 'Equal Weight',
        'inverse_volatility': 'Inverse Volatility',
        'risk_parity': 'Risk Parity',
        'max_sharpe': 'Maximum Sharpe',
        'min_variance': 'Minimum Variance'
    }
    
    optimization_results = {}
    
    for method_key, method_name in allocation_methods.items():
        with logger.operation(f"Testing {method_name} allocation"):
            try:
                if method_key == 'equal_weight':
                    weights = optimizer.equal_weight()
                elif method_key == 'inverse_volatility':
                    weights = optimizer.inverse_volatility()
                elif method_key == 'risk_parity':
                    weights = optimizer.risk_parity()
                elif method_key == 'max_sharpe':
                    weights = optimizer.max_sharpe()
                elif method_key == 'min_variance':
                    weights = optimizer.min_variance()
                
                # Calculate portfolio metrics
                portfolio_return = (returns_df * weights).sum(axis=1)
                
                metrics = {
                    'weights': weights,
                    'total_return': (1 + portfolio_return).prod() - 1,
                    'annualized_return': portfolio_return.mean() * 252,
                    'volatility': portfolio_return.std() * np.sqrt(252),
                    'sharpe_ratio': (portfolio_return.mean() * 252) / (portfolio_return.std() * np.sqrt(252)),
                    'max_drawdown': (portfolio_return.cumsum() - portfolio_return.cumsum().expanding().max()).min()
                }
                
                optimization_results[method_key] = {
                    'name': method_name,
                    'metrics': metrics
                }
                
                # Log allocation
                weight_str = ", ".join([f"{symbol}: {weight:.1%}" for symbol, weight in weights.items()])
                logger.info(f"{method_name}: {weight_str}")
                logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['total_return']:.2%}")
                
            except Exception as e:
                logger.error(f"Failed to optimize {method_name}: {str(e)}")
    
    # 4. Results comparison
    logger.section("📈 Optimization Results Comparison")
    
    # Sort by Sharpe ratio
    sorted_results = sorted(optimization_results.items(), 
                           key=lambda x: x[1]['metrics']['sharpe_ratio'], 
                           reverse=True)
    
    logger.info("Ranking by Sharpe Ratio:")
    for i, (method_key, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        logger.info(f"{i}. {result['name']}: "
                   f"Sharpe {metrics['sharpe_ratio']:.3f}, "
                   f"Return {metrics['total_return']:.2%}, "
                   f"Vol {metrics['volatility']:.2%}")
    
    # Best performing method
    best_method = sorted_results[0]
    logger.success(f"🏆 Best Method: {best_method[1]['name']} "
                   f"(Sharpe: {best_method[1]['metrics']['sharpe_ratio']:.3f})")
    
    # 5. Generate visualizations
    skip_plotting = "--no-plots" in sys.argv
    if not skip_plotting:
        with logger.operation("Generating optimization visualizations"):
            try:
                output_dir = Path("results/example_04_portfolio_optimization")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create comparison charts
                optimizer.plot_efficient_frontier(
                    save_path=output_dir / "efficient_frontier.png"
                )
                
                optimizer.plot_allocation_comparison(
                    optimization_results,
                    save_path=output_dir / "allocation_comparison.png"
                )
                
                # Save detailed results
                results_df = pd.DataFrame({
                    method: {
                        'Total Return': result['metrics']['total_return'],
                        'Sharpe Ratio': result['metrics']['sharpe_ratio'],
                        'Volatility': result['metrics']['volatility'],
                        'Max Drawdown': result['metrics']['max_drawdown']
                    }
                    for method, result in optimization_results.items()
                }).T
                
                results_df.to_csv(output_dir / "optimization_results.csv")
                
                logger.success(f"Results saved to {output_dir}/")
                
            except Exception as e:
                logger.error(f"Failed to generate visualizations: {str(e)}")
    
    # 6. Implementation recommendations
    logger.section("💡 Implementation Recommendations")
    
    best_weights = best_method[1]['metrics']['weights']
    logger.info("Recommended Portfolio Allocation:")
    for symbol, weight in best_weights.items():
        logger.info(f"  {symbol}: {weight:.1%}")
    
    logger.info("\nKey Insights:")
    logger.info("• Portfolio optimization can significantly improve risk-adjusted returns")
    logger.info("• Different methods suit different market conditions and risk preferences")
    logger.info("• Regular rebalancing is important to maintain target allocations")
    logger.info("• Consider transaction costs when implementing optimized weights")
    
    return optimization_results


def main():
    """Main function to run portfolio optimization example."""
    try:
        results = run_portfolio_optimization()
        if results:
            logger = get_logger()
            logger.success("✅ Portfolio optimization completed successfully!")
        else:
            logger = get_logger()
            logger.error("❌ Portfolio optimization failed")
    except Exception as e:
        logger = get_logger()
        logger.error(f"Portfolio optimization error: {str(e)}")


if __name__ == "__main__":
    main() 