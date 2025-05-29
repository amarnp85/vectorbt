#!/usr/bin/env python3
"""
Backtester CLI - Simple command-line interface for backtesting operations

Usage:
    python -m backtester backtest --symbol BTC/USDT --timeframe 1h
    python -m backtester optimize --symbol BTC/USDT --metric sharpe_ratio
    python -m backtester portfolio --symbols BTC/USDT,ETH/USDT --method risk_parity
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules directly
from backtester.data import fetch_data
from backtester.strategies.dma_atr_trend_strategy import DMAATRTrendStrategy
from backtester.portfolio import PortfolioSimulator, SimulationConfig
from backtester.optimization.optimizer_engine import OptimizerEngine
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.plotting_engine import PlottingEngine
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters import OptimalParameterManager
from backtester.utilities.structured_logging import setup_logging, get_logger
import pandas as pd
import vectorbtpro as vbt


def get_config_path(args):
    """Get the configuration file path."""
    if args.config:
        return Path(args.config)
    else:
        # Use absolute path for default config
        return Path(__file__).parent / "config" / "strategy_params" / "production" / "dma_atr_trend_params.json"


def backtest_command(args):
    """Run a single symbol backtest"""
    print(f"\n🚀 Running backtest for {args.symbol} on {args.timeframe} timeframe...")
    
    # Setup
    logger = setup_logging("backtester")
    param_manager = OptimalParameterManager()
    
    # Load data
    print("📊 Fetching data...")
    data = fetch_data(
        symbols=[args.symbol],
        exchange_id="binance",
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=True
    )
    
    if data is None:
        print(f"❌ Failed to load data for {args.symbol}")
        return None
    
    # Load configuration
    config_path = get_config_path(args)
    config_manager = ConfigManager()
    config = config_manager.load_config(str(config_path))
    
    # Check for optimal parameters
    optimal_params = param_manager.get_optimal_params(args.symbol, args.timeframe)
    
    if optimal_params:
        print(f"🎯 Using stored optimal parameters for {args.symbol} ({args.timeframe})")
        strategy_params = optimal_params
    else:
        print(f"📄 Using default parameters from config for {args.symbol} ({args.timeframe})")
        strategy_params = config_manager.get_strategy_params()
    
    # Portfolio parameters
    portfolio_params = config_manager.get_portfolio_params()
    portfolio_params['init_cash'] = args.initial_cash
    portfolio_params['fees'] = args.commission
    
    # Output directory
    output_dir = args.output if args.output else f"backtester/results/cli_{args.symbol.replace('/', '_')}_{args.timeframe}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run backtest
    print("📈 Running strategy...")
    
    try:
        # Initialize strategy
        strategy = DMAATRTrendStrategy(data, strategy_params)
        indicators = strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Create portfolio
        sim_config = SimulationConfig(
            init_cash=portfolio_params.get("init_cash", 100000),
            fees=portfolio_params.get("fees", 0.001),
            slippage=portfolio_params.get("slippage", 0.0005),
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(portfolio, signals=signals)
        
        # Get all metrics
        metrics = {}
        metrics.update(analyzer.get_returns_metrics())
        metrics.update(analyzer.get_trade_metrics())
        metrics.update(analyzer.get_drawdown_metrics())
        metrics.update(analyzer.get_risk_metrics())
        
        # Save metrics
        analyzer.export_results(os.path.join(output_dir, "performance_metrics.csv"))
        
        # Create plots
        plotter = PlottingEngine(portfolio, data)
        
        # Generate plots
        plotter.plot_strategy_overview(
            signals=signals,
            indicators=indicators,
            save_path=os.path.join(output_dir, "strategy_overview.html")
        )
        
        plotter.plot_signal_analysis(
            signals=signals,
            save_path=os.path.join(output_dir, "signal_analysis.html")
        )
        
        # Display results
        if metrics:
            print("\n📈 Performance Metrics:")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            
            print(f"\n✅ Results saved to: {output_dir}")
            
            return {
                "portfolio": portfolio,
                "metrics": metrics,
                "signals": signals,
                "indicators": indicators
            }
        else:
            print("❌ Backtest failed")
            return None
            
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        print(f"❌ Backtest failed: {str(e)}")
        return None


def optimize_command(args):
    """Run parameter optimization for a single symbol"""
    print(f"\n🔧 Optimizing parameters for {args.symbol} on {args.timeframe} timeframe...")
    
    # Setup
    logger = setup_logging("backtester")
    param_manager = OptimalParameterManager()
    
    # Check if we should skip optimization
    if not args.force:
        existing_summary = param_manager.get_symbol_performance_summary(args.symbol, args.timeframe)
        if existing_summary:
            print(f"⏭️  {args.symbol} ({args.timeframe}) already optimized")
            print(f"   📊 Current Sharpe: {existing_summary['performance'].get('sharpe_ratio', 0):.3f}")
            print(f"   📅 Last optimized: {existing_summary['last_updated'][:10]}")
            print("   Use --force to reoptimize")
            return existing_summary
    
    # Load data
    print("📊 Fetching data...")
    data = fetch_data(
        symbols=[args.symbol],
        exchange_id="binance",
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=True
    )
    
    if data is None:
        print(f"❌ Failed to load data for {args.symbol}")
        return None
    
    # Load configuration
    config_path = get_config_path(args)
    config_manager = ConfigManager()
    config = config_manager.load_config(str(config_path))
    
    # Base strategy parameters
    base_params = config_manager.get_strategy_params()
    
    # Optimization parameters from config
    optimization_params = config_manager.get_optimization_params()
    
    # Output directory
    output_dir = args.output if args.output else f"backtester/results/cli_opt_{args.symbol.replace('/', '_')}_{args.timeframe}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimization
    print(f"🔍 Testing parameter combinations (optimizing for {args.metric})...")
    
    try:
        # Create optimizer
        optimizer = OptimizerEngine(
            data=data,
            strategy_class=DMAATRTrendStrategy,
            base_params=base_params
        )
        
        # Run optimization
        results = optimizer.optimize_grid_search(
            param_ranges=optimization_params,
            metric=args.metric
        )
        
        # Store optimal parameters
        if results and 'best_params' in results:
            best_params = results['best_params']
            best_portfolio = results.get('best_portfolio')
            
            if best_portfolio:
                # Extract performance metrics
                performance_metrics = {
                    'sharpe_ratio': float(best_portfolio.sharpe_ratio) if hasattr(best_portfolio, 'sharpe_ratio') else 0.0,
                    'total_return': float(best_portfolio.total_return) if hasattr(best_portfolio, 'total_return') else 0.0,
                    'max_drawdown': float(best_portfolio.max_drawdown) if hasattr(best_portfolio, 'max_drawdown') else 0.0,
                }
                
                # Store the results
                param_manager.store_optimization_result(
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    best_params=best_params,
                    performance_metrics=performance_metrics
                )
                
                # Save detailed results
                optimizer.save_results(output_dir)
                
                # Display results
                print(f"\n✨ Best Parameters Found:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
                
                print(f"\n📈 Best Performance:")
                print(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
                print(f"  Total Return: {performance_metrics['total_return']:.2%}")
                print(f"  Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
                
                print(f"\n✅ Results saved to: {output_dir}")
                print(f"📁 Optimal parameters stored in: {param_manager.storage_dir}")
                
                return results
        
        print("❌ Optimization failed")
        return None
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        print(f"❌ Optimization failed: {str(e)}")
        return None


def portfolio_command(args):
    """Run multi-symbol portfolio backtest"""
    symbols = args.symbols.split(',')
    print(f"\n💼 Running portfolio backtest for: {', '.join(symbols)}")
    
    # Setup
    logger = setup_logging("backtester")
    param_manager = OptimalParameterManager()
    
    # Load data
    print("📊 Fetching data for all symbols...")
    data = fetch_data(
        symbols=symbols,
        exchange_id="binance",
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=True
    )
    
    if data is None:
        print("❌ Failed to load data")
        return None
    
    # Load configuration
    config_path = get_config_path(args)
    config_manager = ConfigManager()
    config = config_manager.load_config(str(config_path))
    
    # For multi-symbol, use consensus approach or default parameters
    primary_symbol = symbols[0]
    optimal_params = param_manager.get_optimal_params(primary_symbol, args.timeframe)
    
    if optimal_params:
        print(f"🎯 Using optimal parameters from {primary_symbol} for portfolio")
        strategy_params = optimal_params
    else:
        print(f"📄 Using default parameters from config for portfolio")
        strategy_params = config_manager.get_strategy_params()
    
    # Portfolio parameters
    portfolio_params = config_manager.get_portfolio_params()
    portfolio_params['init_cash'] = args.initial_cash
    portfolio_params['fees'] = args.commission
    
    # Output directory
    output_dir = args.output if args.output else f"backtester/results/cli_portfolio_{args.timeframe}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run backtest
    print("📈 Running portfolio strategy...")
    
    try:
        # Initialize strategy
        strategy = DMAATRTrendStrategy(data, strategy_params)
        indicators = strategy.init_indicators()
        signals = strategy.generate_signals()
        
        # Create portfolio
        sim_config = SimulationConfig(
            init_cash=portfolio_params.get("init_cash", 100000),
            fees=portfolio_params.get("fees", 0.001),
            slippage=portfolio_params.get("slippage", 0.0005),
            freq=data.wrapper.freq if hasattr(data, "wrapper") else "D"
        )
        
        simulator = PortfolioSimulator(data, sim_config)
        portfolio = simulator.simulate_from_signals(signals)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(portfolio, signals=signals)
        
        # Get all metrics
        metrics = {}
        metrics.update(analyzer.get_returns_metrics())
        metrics.update(analyzer.get_trade_metrics())
        metrics.update(analyzer.get_drawdown_metrics())
        metrics.update(analyzer.get_risk_metrics())
        
        # Save metrics
        analyzer.export_results(os.path.join(output_dir, "performance_metrics.csv"))
        
        # Create plots
        plotter = PlottingEngine(portfolio, data)
        
        # Generate plots
        plotter.plot_strategy_overview(
            signals=signals,
            indicators=indicators,
            save_path=os.path.join(output_dir, "portfolio_overview.html")
        )
        
        # Display results
        if metrics:
            print("\n📈 Portfolio Performance:")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Volatility: {metrics.get('annualized_volatility', 0):.2%}")
            
            print(f"\n✅ Results saved to: {output_dir}")
            
            return {
                "portfolio": portfolio,
                "metrics": metrics,
                "signals": signals,
                "indicators": indicators
            }
        else:
            print("❌ Portfolio backtest failed")
            return None
            
    except Exception as e:
        logger.error(f"Error in portfolio backtest: {e}")
        print(f"❌ Portfolio backtest failed: {str(e)}")
        return None


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Backtester CLI - Simple interface for backtesting operations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run a single symbol backtest')
    backtest_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTC/USDT)')
    backtest_parser.add_argument('--timeframe', default='1h', help='Timeframe (1h, 4h, 1d)')
    backtest_parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', default='2023-12-31', help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-cash', type=float, default=100000, help='Initial capital')
    backtest_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    backtest_parser.add_argument('--output', help='Output directory for results')
    backtest_parser.add_argument('--config', help='Path to config file')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    optimize_parser.add_argument('--symbol', required=True, help='Trading symbol')
    optimize_parser.add_argument('--timeframe', default='1h', help='Timeframe')
    optimize_parser.add_argument('--start-date', default='2023-01-01', help='Start date')
    optimize_parser.add_argument('--end-date', default='2023-12-31', help='End date')
    optimize_parser.add_argument('--metric', default='sharpe_ratio', 
                               choices=['sharpe_ratio', 'total_return', 'calmar_ratio'],
                               help='Optimization metric')
    optimize_parser.add_argument('--force', action='store_true', help='Force reoptimization even if already optimized')
    optimize_parser.add_argument('--output', help='Output directory')
    optimize_parser.add_argument('--config', help='Path to config file')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Run portfolio backtest')
    portfolio_parser.add_argument('--symbols', required=True, 
                                help='Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)')
    portfolio_parser.add_argument('--timeframe', default='1h', help='Timeframe')
    portfolio_parser.add_argument('--start-date', default='2023-01-01', help='Start date')
    portfolio_parser.add_argument('--end-date', default='2023-12-31', help='End date')
    portfolio_parser.add_argument('--initial-cash', type=float, default=100000, help='Initial capital')
    portfolio_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    portfolio_parser.add_argument('--output', help='Output directory')
    portfolio_parser.add_argument('--config', help='Path to config file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'backtest':
            backtest_command(args)
        elif args.command == 'optimize':
            optimize_command(args)
        elif args.command == 'portfolio':
            portfolio_command(args)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 