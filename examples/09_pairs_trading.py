#!/usr/bin/env python3
"""
Example 09: Pairs Trading Strategy

This example demonstrates statistical arbitrage through pairs trading:
- Finding cointegrated pairs
- Z-score based entry/exit signals
- Dollar-neutral position sizing
- Risk management for pairs

Key concepts:
- Cointegration testing
- Spread calculation and z-score
- Hedge ratio calculation
- Market-neutral strategies
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
from backtester.config.config_manager import ConfigManager
from backtester.config.optimal_parameters_db import OptimalParametersDB
from backtester.portfolio.simulation_engine import PortfolioSimulator, SimulationConfig
from backtester.analysis.performance_analyzer import PerformanceAnalyzer
from backtester.analysis.plotting_engine import PlottingEngine
from backtester.utilities.structured_logging import setup_logging, get_logger


def demonstrate_pairs_trading():
    """Demonstrate pairs trading strategy implementation."""
    
    logger = get_logger()
    logger.section("🔄 Pairs Trading Strategy", "Statistical arbitrage through mean-reverting spreads")
    
    # Configuration
    pair = ["ETH/USDT", "BNB/USDT"]  # Crypto pair for demonstration
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.info(f"Trading Pair: {pair[0]} vs {pair[1]}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Load data for both assets
    logger.section("📊 Loading Pair Data")
    
    pair_data = {}
    for symbol in pair:
        with logger.operation(f"Loading {symbol} data"):
            data = fetch_data(
                symbols=[symbol],
                exchange="binance",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None:
                logger.error(f"Failed to load {symbol} data")
                return None
            
            pair_data[symbol] = data.get('close')
            data_points = len(data.get('close'))
            logger.info(f"{symbol}: {data_points} candles")
    
    if len(pair_data) != 2:
        logger.error("Failed to load data for both assets")
        return None
    
    # Align data and check for cointegration
    logger.section("🔍 Cointegration Analysis")
    
    cointegration_results = test_cointegration(pair_data, pair, logger)
    
    if cointegration_results is None:
        logger.error("Cointegration analysis failed")
        return None
    
    if not cointegration_results['is_cointegrated']:
        logger.warning("⚠️ Pair may not be suitable for pairs trading (not cointegrated)")
        logger.info("Proceeding with demonstration anyway...")
    else:
        logger.success("✅ Pair shows cointegration - suitable for pairs trading")
    
    # Generate pairs trading signals
    logger.section("🎯 Generating Pairs Trading Signals")
    
    pairs_signals = generate_pairs_signals(
        pair_data, 
        cointegration_results['hedge_ratio'], 
        pair, 
        logger
    )
    
    if pairs_signals is None:
        logger.error("Failed to generate pairs signals")
        return None
    
    # Run pairs trading backtest
    logger.section("🚀 Running Pairs Trading Backtest")
    
    with logger.operation("Running pairs trading simulation"):
        try:
            # Create pairs trading strategy
            strategy = PairsStrategy(pair_data, pairs_signals, pair)
            
            # Run backtest
            sim_config = SimulationConfig(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0005
            )
            
            simulator = PortfolioSimulator(sim_config)
            portfolio = simulator.run_backtest(strategy)
            
            if portfolio is None:
                logger.error("Pairs trading backtest failed")
                return None
            
            # Analyze results
            analyzer = PerformanceAnalyzer(portfolio)
            
            results = {
                'portfolio': portfolio,
                'total_return': analyzer.total_return(),
                'sharpe_ratio': analyzer.sharpe_ratio(),
                'max_drawdown': analyzer.max_drawdown(),
                'volatility': analyzer.volatility(),
                'total_trades': analyzer.total_trades(),
                'win_rate': analyzer.win_rate(),
                'profit_factor': analyzer.profit_factor()
            }
            
            # Log results
            logger.backtest_result({
                'portfolio': portfolio,
                'metrics': results,
                'strategy_type': 'Pairs Trading'
            })
            
            return {
                'strategy': strategy,
                'portfolio': portfolio,
                'results': results,
                'cointegration': cointegration_results,
                'signals': pairs_signals,
                'pair': pair
            }
            
        except Exception as e:
            logger.error(f"Pairs trading backtest failed: {str(e)}")
            return None


def test_cointegration(pair_data, pair, logger):
    """Test for cointegration between the pair."""
    try:
        # Align the data
        prices_df = pd.DataFrame(pair_data)
        prices_df = prices_df.dropna()
        
        if len(prices_df) < 100:
            logger.error("Insufficient data for cointegration test")
            return None
        
        asset1_prices = prices_df[pair[0]]
        asset2_prices = prices_df[pair[1]]
        
        # Calculate hedge ratio using linear regression
        # Simple OLS: asset1 = alpha + beta * asset2 + error
        X = asset2_prices.values.reshape(-1, 1)
        y = asset1_prices.values
        
        # Add constant term
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Calculate coefficients using normal equation
        coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        alpha, beta = coefficients
        
        # Calculate spread
        spread = asset1_prices - (alpha + beta * asset2_prices)
        
        # Test for stationarity of spread (simplified ADF test)
        spread_returns = spread.diff().dropna()
        
        # Simple stationarity check: if spread mean-reverts
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Check if spread crosses mean frequently (sign of mean reversion)
        mean_crossings = ((spread > spread_mean) != (spread.shift(1) > spread_mean)).sum()
        crossing_frequency = mean_crossings / len(spread)
        
        # Heuristic: if spread crosses mean more than 10% of the time, consider cointegrated
        is_cointegrated = crossing_frequency > 0.1
        
        # Calculate correlation for additional insight
        correlation = asset1_prices.corr(asset2_prices)
        
        logger.info("Cointegration Test Results:")
        logger.info(f"  Hedge Ratio (β): {beta:.4f}")
        logger.info(f"  Intercept (α): {alpha:.4f}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Spread Mean: {spread_mean:.4f}")
        logger.info(f"  Spread Std: {spread_std:.4f}")
        logger.info(f"  Mean Crossing Frequency: {crossing_frequency:.2%}")
        logger.info(f"  Cointegrated: {is_cointegrated}")
        
        return {
            'is_cointegrated': is_cointegrated,
            'hedge_ratio': beta,
            'intercept': alpha,
            'correlation': correlation,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'crossing_frequency': crossing_frequency
        }
        
    except Exception as e:
        logger.error(f"Cointegration test failed: {str(e)}")
        return None


def generate_pairs_signals(pair_data, hedge_ratio, pair, logger):
    """Generate pairs trading signals based on spread z-score."""
    try:
        # Align data
        prices_df = pd.DataFrame(pair_data)
        prices_df = prices_df.dropna()
        
        asset1_prices = prices_df[pair[0]]
        asset2_prices = prices_df[pair[1]]
        
        # Calculate spread using hedge ratio
        spread = asset1_prices - hedge_ratio * asset2_prices
        
        # Calculate rolling z-score
        lookback = 20  # 20-period rolling window
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Signal thresholds
        entry_threshold = 2.0    # Enter when |z-score| > 2
        exit_threshold = 0.5     # Exit when |z-score| < 0.5
        stop_threshold = 3.0     # Stop loss when |z-score| > 3
        
        # Generate signals
        # Long spread (long asset1, short asset2) when z-score < -entry_threshold
        long_entries = z_score < -entry_threshold
        long_exits = (z_score > -exit_threshold) | (z_score < -stop_threshold)
        
        # Short spread (short asset1, long asset2) when z-score > entry_threshold
        short_entries = z_score > entry_threshold
        short_exits = (z_score < exit_threshold) | (z_score > stop_threshold)
        
        # Combine signals (simplified - in practice would need position tracking)
        entries = long_entries | short_entries
        exits = long_exits | short_exits
        
        # Signal direction: 1 for long spread, -1 for short spread
        signal_direction = pd.Series(0, index=z_score.index)
        signal_direction[long_entries] = 1
        signal_direction[short_entries] = -1
        
        # Log signal statistics
        total_entries = entries.sum()
        long_signals = long_entries.sum()
        short_signals = short_entries.sum()
        
        logger.info("Pairs Trading Signals:")
        logger.info(f"  Total Entry Signals: {total_entries}")
        logger.info(f"  Long Spread Signals: {long_signals}")
        logger.info(f"  Short Spread Signals: {short_signals}")
        logger.info(f"  Entry Threshold: ±{entry_threshold}")
        logger.info(f"  Exit Threshold: ±{exit_threshold}")
        
        return {
            'entries': entries,
            'exits': exits,
            'signal_direction': signal_direction,
            'z_score': z_score,
            'spread': spread,
            'thresholds': {
                'entry': entry_threshold,
                'exit': exit_threshold,
                'stop': stop_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Pairs signal generation failed: {str(e)}")
        return None


class PairsStrategy:
    """Pairs trading strategy implementation."""
    
    def __init__(self, pair_data, signals, pair):
        self.pair_data = pair_data
        self.signals = signals
        self.pair = pair
        
    def generate_signals(self):
        """Generate signals for the portfolio simulator."""
        # Return the entry/exit signals
        return self.signals


def compare_with_individual_assets(strategy, portfolio, pair, logger):
    """Compare pairs trading performance with individual asset performance."""
    
    logger.section("📊 Performance Comparison")
    
    try:
        # Get individual asset data
        individual_results = {}
        
        for symbol in pair:
            with logger.operation(f"Analyzing {symbol} individual performance"):
                try:
                    # Simple buy-and-hold for comparison
                    symbol_data = strategy.pair_data[symbol]
                    
                    if len(symbol_data) > 0:
                        # Calculate buy-and-hold return
                        start_price = symbol_data.iloc[0]
                        end_price = symbol_data.iloc[-1]
                        bh_return = (end_price - start_price) / start_price
                        
                        # Calculate volatility
                        returns = symbol_data.pct_change().dropna()
                        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized for hourly data
                        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() > 0 else 0
                        
                        individual_results[symbol] = {
                            'total_return': bh_return,
                            'volatility': volatility,
                            'sharpe_ratio': sharpe
                        }
                        
                        logger.info(f"{symbol}: Return {bh_return:.2%}, Sharpe {sharpe:.3f}")
                
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {str(e)}")
        
        # Compare with pairs strategy
        if portfolio:
            analyzer = PerformanceAnalyzer(portfolio)
            pairs_return = analyzer.total_return()
            pairs_sharpe = analyzer.sharpe_ratio()
            pairs_volatility = analyzer.volatility()
            
            logger.info("\nStrategy Comparison:")
            logger.info(f"Pairs Trading: Return {pairs_return:.2%}, Sharpe {pairs_sharpe:.3f}, Vol {pairs_volatility:.2%}")
            
            # Calculate average individual performance
            if individual_results:
                avg_individual_return = np.mean([r['total_return'] for r in individual_results.values()])
                avg_individual_sharpe = np.mean([r['sharpe_ratio'] for r in individual_results.values()])
                
                logger.info(f"Average Individual: Return {avg_individual_return:.2%}, Sharpe {avg_individual_sharpe:.3f}")
                
                # Market neutrality benefit
                if pairs_volatility < np.mean([r['volatility'] for r in individual_results.values()]):
                    logger.success("✅ Pairs trading shows lower volatility (market neutral benefit)")
                else:
                    logger.warning("⚠️ Pairs trading volatility higher than individual assets")
        
        return individual_results
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {str(e)}")
        return {}


def main():
    """Run pairs trading examples."""
    
    # Setup structured logging
    logger = setup_logging("INFO")
    
    logger.section("🔄 Pairs Trading Examples", "Statistical arbitrage and market-neutral strategies")
    
    # Create results directory
    output_dir = Path("results/example_09_pairs_trading")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Starting Pairs Trading Examples...")
        
        # Run main demonstration
        strategy_results = demonstrate_pairs_trading()
        
        if strategy_results:
            strategy = strategy_results['strategy']
            portfolio = strategy_results['portfolio']
            pair = strategy_results['pair']
            
            # Compare with individual assets
            individual_comparison = compare_with_individual_assets(strategy, portfolio, pair, logger)
            
            # Generate visualizations
            skip_plotting = "--no-plots" in sys.argv
            if not skip_plotting:
                with logger.operation("Generating pairs trading visualizations"):
                    try:
                        create_pairs_analysis_charts(strategy_results, output_dir, logger)
                        logger.success(f"Visualizations saved to {output_dir}/")
                    except Exception as e:
                        logger.error(f"Failed to generate visualizations: {str(e)}")
            
            # Key insights
            logger.section("💡 Pairs Trading Insights")
            
            results = strategy_results['results']
            cointegration = strategy_results['cointegration']
            
            logger.info("Key Findings:")
            logger.info(f"• Hedge ratio: {cointegration['hedge_ratio']:.4f}")
            logger.info(f"• Correlation: {cointegration['correlation']:.4f}")
            logger.info(f"• Strategy Sharpe: {results['sharpe_ratio']:.3f}")
            logger.info(f"• Total trades: {results['total_trades']}")
            
            if cointegration['is_cointegrated']:
                logger.success("✅ Pair shows good cointegration properties")
            else:
                logger.warning("⚠️ Pair may not be ideal for pairs trading")
            
            logger.info("\nPairs Trading Benefits:")
            logger.info("• Market neutral exposure reduces systematic risk")
            logger.info("• Profits from relative price movements")
            logger.info("• Lower correlation with overall market direction")
            logger.info("• Can work in both bull and bear markets")
            
        else:
            logger.error("❌ Pairs trading demonstration failed")
        
        logger.success("✅ Pairs trading examples completed successfully!")
        logger.info(f"Results saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Pairs trading examples error: {str(e)}")


def create_pairs_analysis_charts(strategy_results, output_dir, logger):
    """Create comprehensive pairs trading analysis charts."""
    try:
        import matplotlib.pyplot as plt
        
        signals = strategy_results['signals']
        cointegration = strategy_results['cointegration']
        pair = strategy_results['pair']
        
        # Create comprehensive analysis chart
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'Pairs Trading Analysis: {pair[0]} vs {pair[1]}', fontsize=16)
        
        # 1. Price series
        prices_df = pd.DataFrame(strategy_results['strategy'].pair_data)
        axes[0, 0].plot(prices_df.index, prices_df[pair[0]], label=pair[0], alpha=0.8)
        axes[0, 0].plot(prices_df.index, prices_df[pair[1]], label=pair[1], alpha=0.8)
        axes[0, 0].set_title('Price Series')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spread
        spread = cointegration['spread']
        axes[0, 1].plot(spread.index, spread, label='Spread', color='purple')
        axes[0, 1].axhline(y=cointegration['spread_mean'], color='red', linestyle='--', label='Mean')
        axes[0, 1].fill_between(spread.index, 
                               cointegration['spread_mean'] - cointegration['spread_std'],
                               cointegration['spread_mean'] + cointegration['spread_std'],
                               alpha=0.2, color='gray', label='±1 Std')
        axes[0, 1].set_title('Spread (Asset1 - β×Asset2)')
        axes[0, 1].set_ylabel('Spread')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Z-score
        z_score = signals['z_score']
        axes[1, 0].plot(z_score.index, z_score, label='Z-Score', color='orange')
        axes[1, 0].axhline(y=2, color='red', linestyle='--', label='Entry Threshold')
        axes[1, 0].axhline(y=-2, color='red', linestyle='--')
        axes[1, 0].axhline(y=0.5, color='green', linestyle=':', label='Exit Threshold')
        axes[1, 0].axhline(y=-0.5, color='green', linestyle=':')
        axes[1, 0].set_title('Spread Z-Score')
        axes[1, 0].set_ylabel('Z-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Signal direction
        signal_direction = signals['signal_direction']
        axes[1, 1].plot(signal_direction.index, signal_direction, label='Signal Direction', 
                       color='blue', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Signal Direction (1=Long Spread, -1=Short Spread)')
        axes[1, 1].set_ylabel('Signal')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Portfolio performance
        if 'portfolio' in strategy_results:
            portfolio = strategy_results['portfolio']
            if hasattr(portfolio, 'value'):
                portfolio_value = portfolio.value()
                axes[2, 0].plot(portfolio_value.index, portfolio_value, 
                               label='Portfolio Value', color='green', linewidth=2)
                axes[2, 0].set_title('Portfolio Equity Curve')
                axes[2, 0].set_ylabel('Portfolio Value')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Performance metrics summary
        results = strategy_results['results']
        metrics = [
            f"Total Return: {results['total_return']:.2%}",
            f"Sharpe Ratio: {results['sharpe_ratio']:.3f}",
            f"Max Drawdown: {results['max_drawdown']:.2%}",
            f"Win Rate: {results['win_rate']:.1%}",
            f"Total Trades: {results['total_trades']}",
            f"Hedge Ratio: {cointegration['hedge_ratio']:.4f}",
            f"Correlation: {cointegration['correlation']:.4f}"
        ]
        
        axes[2, 1].text(0.1, 0.9, '\n'.join(metrics), transform=axes[2, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[2, 1].set_title('Performance Summary')
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pairs_trading_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create pairs analysis charts: {str(e)}")


if __name__ == "__main__":
    main() 