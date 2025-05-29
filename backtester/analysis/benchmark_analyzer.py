"""
Benchmark Analysis Module for Backtesting Framework

This module provides comprehensive benchmark comparison and sizing calibration
using vectorbtpro's built-in capabilities for alpha/beta calculations and data fetching.
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class BenchmarkAnalyzer:
    """
    Comprehensive benchmark analysis using vectorbtpro's built-in capabilities.
    
    Features:
    - Alpha/beta calculations using vectorbtpro's Portfolio methods
    - Multiple benchmark support (SPY, QQQ, BTC-USD, etc.)
    - Sizing calibration based on benchmark characteristics
    - Risk-adjusted performance metrics
    """
    
    SUPPORTED_BENCHMARKS = {
        'SPY': 'SPDR S&P 500 ETF Trust',
        'QQQ': 'Invesco QQQ Trust',
        'BTC-USD': 'Bitcoin USD',
        'ETH-USD': 'Ethereum USD', 
        'GLD': 'SPDR Gold Trust',
        'TLT': 'iShares 20+ Year Treasury Bond ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'IWM': 'iShares Russell 2000 ETF'
    }
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize benchmark analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_data = {}
        self.analysis_results = {}
        
    def fetch_benchmark_data(self, 
                           benchmark_symbol: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           period: str = "2y") -> pd.DataFrame:
        """
        Fetch benchmark data using vectorbtpro's YFData.
        
        Args:
            benchmark_symbol: Benchmark symbol (e.g., 'SPY', 'QQQ')
            start_date: Start date for data fetch
            end_date: End date for data fetch  
            period: Period for data fetch if dates not specified
            
        Returns:
            DataFrame with benchmark OHLCV data
        """
        try:
            logger.info(f"Fetching benchmark data for {benchmark_symbol}")
            
            # Use vectorbtpro's YFData for fetching
            if start_date and end_date:
                data = vbt.YFData.fetch(
                    benchmark_symbol,
                    start=start_date,
                    end=end_date
                )
            else:
                data = vbt.YFData.fetch(
                    benchmark_symbol,
                    period=period
                )
            
            # Store the data
            self.benchmark_data[benchmark_symbol] = data
            
            logger.info(f"Successfully fetched {len(data.data)} records for {benchmark_symbol}")
            return data.data
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {benchmark_symbol}: {e}")
            raise
    
    def calculate_benchmark_metrics(self,
                                  portfolio: vbt.Portfolio,
                                  benchmark_symbol: str,
                                  benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive benchmark comparison metrics using vectorbtpro.
        
        Args:
            portfolio: vectorbtpro Portfolio object
            benchmark_symbol: Benchmark symbol
            benchmark_data: Optional benchmark data (will fetch if not provided)
            
        Returns:
            Dictionary with benchmark analysis results
        """
        try:
            # Fetch benchmark data if not provided
            if benchmark_data is None:
                if benchmark_symbol not in self.benchmark_data:
                    self.fetch_benchmark_data(benchmark_symbol)
                benchmark_data = self.benchmark_data[benchmark_symbol].data
            
            # Get benchmark returns
            benchmark_close = benchmark_data['Close'] if 'Close' in benchmark_data.columns else benchmark_data.iloc[:, -1]
            benchmark_returns = benchmark_close.pct_change().dropna()
            
            # Align dates with portfolio
            portfolio_returns = portfolio.returns()
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) == 0:
                raise ValueError("No overlapping dates between portfolio and benchmark")
            
            portfolio_returns_aligned = portfolio_returns.loc[common_dates]
            benchmark_returns_aligned = benchmark_returns.loc[common_dates]
            
            # Use vectorbtpro's built-in alpha/beta calculations
            try:
                alpha = portfolio.get_alpha(bm_returns=benchmark_returns_aligned, risk_free=self.risk_free_rate)
                beta = portfolio.get_beta(bm_returns=benchmark_returns_aligned)
            except Exception as e:
                logger.warning(f"VectorBTPro alpha/beta calculation failed: {e}")
                # Fallback to manual calculation
                portfolio_excess = portfolio_returns_aligned - self.risk_free_rate / 252
                benchmark_excess = benchmark_returns_aligned - self.risk_free_rate / 252
                
                # Calculate beta manually
                covariance = np.cov(portfolio_excess, benchmark_excess)[0, 1]
                benchmark_variance = np.var(benchmark_excess)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                # Calculate alpha manually
                alpha = portfolio_excess.mean() - beta * benchmark_excess.mean()
            
            # Calculate additional metrics
            correlation = self._safe_extract_value(portfolio_returns_aligned.corr(benchmark_returns_aligned))
            
            # Portfolio metrics
            portfolio_total_return = self._safe_extract_value(portfolio.total_return())
            portfolio_sharpe = self._safe_extract_value(portfolio.sharpe_ratio(risk_free=self.risk_free_rate))
            portfolio_volatility = self._safe_extract_value(portfolio.returns().std() * np.sqrt(252))
            portfolio_max_dd = self._safe_extract_value(portfolio.max_drawdown())
            
            # Benchmark metrics
            benchmark_total_return = (benchmark_close.iloc[-1] / benchmark_close.iloc[0]) - 1
            benchmark_volatility = benchmark_returns_aligned.std() * np.sqrt(252)
            benchmark_sharpe = (benchmark_returns_aligned.mean() * 252 - self.risk_free_rate) / benchmark_volatility
            
            # Calculate benchmark max drawdown
            benchmark_cumulative = (1 + benchmark_returns_aligned).cumprod()
            benchmark_running_max = benchmark_cumulative.expanding().max()
            benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
            benchmark_max_dd = benchmark_drawdown.min()
            
            # Risk-adjusted metrics
            information_ratio = (portfolio_returns_aligned.mean() - benchmark_returns_aligned.mean()) / (portfolio_returns_aligned - benchmark_returns_aligned).std() * np.sqrt(252)
            treynor_ratio = (portfolio_returns_aligned.mean() * 252 - self.risk_free_rate) / self._safe_extract_value(beta) if self._safe_extract_value(beta) != 0 else np.nan
            
            # Capture ratios
            up_capture = self._calculate_capture_ratio(portfolio_returns_aligned, benchmark_returns_aligned, up=True)
            down_capture = self._calculate_capture_ratio(portfolio_returns_aligned, benchmark_returns_aligned, up=False)
            
            results = {
                'benchmark_symbol': benchmark_symbol,
                'benchmark_name': self.SUPPORTED_BENCHMARKS.get(benchmark_symbol, benchmark_symbol),
                'analysis_period': {
                    'start': common_dates[0].strftime('%Y-%m-%d'),
                    'end': common_dates[-1].strftime('%Y-%m-%d'),
                    'days': len(common_dates)
                },
                'alpha_beta': {
                    'alpha': self._safe_extract_value(alpha),
                    'beta': self._safe_extract_value(beta),
                    'correlation': correlation
                },
                'performance_comparison': {
                    'portfolio_total_return': portfolio_total_return,
                    'benchmark_total_return': benchmark_total_return,
                    'excess_return': portfolio_total_return - benchmark_total_return,
                    'portfolio_sharpe': portfolio_sharpe,
                    'benchmark_sharpe': benchmark_sharpe,
                    'portfolio_volatility': portfolio_volatility,
                    'benchmark_volatility': benchmark_volatility,
                    'portfolio_max_drawdown': portfolio_max_dd,
                    'benchmark_max_drawdown': benchmark_max_dd
                },
                'risk_adjusted_metrics': {
                    'information_ratio': information_ratio,
                    'treynor_ratio': treynor_ratio,
                    'up_capture_ratio': up_capture,
                    'down_capture_ratio': down_capture
                },
                'raw_data': {
                    'portfolio_returns': portfolio_returns_aligned,
                    'benchmark_returns': benchmark_returns_aligned,
                    'benchmark_close': benchmark_close
                }
            }
            
            self.analysis_results[benchmark_symbol] = results
            return results
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
            raise
    
    def _safe_extract_value(self, value):
        """Safely extract scalar value from vectorbtpro return types."""
        if hasattr(value, 'values'):
            if hasattr(value.values, 'item'):
                return value.values.item()
            elif len(value.values) == 1:
                return value.values[0]
            else:
                return value.values
        elif hasattr(value, 'item'):
            return value.item()
        else:
            return value
    
    def _calculate_capture_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, up: bool = True) -> float:
        """Calculate up/down capture ratios."""
        if up:
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0
            
        if mask.sum() == 0:
            return np.nan
            
        portfolio_avg = portfolio_returns[mask].mean()
        benchmark_avg = benchmark_returns[mask].mean()
        
        return portfolio_avg / benchmark_avg if benchmark_avg != 0 else np.nan
    
    def calibrate_sizing(self,
                        portfolio: vbt.Portfolio,
                        benchmark_symbol: str,
                        target_volatility: Optional[float] = None,
                        sizing_method: str = 'volatility_target') -> Dict[str, Any]:
        """
        Calibrate position sizing based on benchmark characteristics.
        
        Args:
            portfolio: Current portfolio
            benchmark_symbol: Benchmark for calibration
            target_volatility: Target volatility (if None, uses benchmark volatility)
            sizing_method: Method for sizing ('volatility_target', 'beta_target', 'risk_parity', 'kelly')
            
        Returns:
            Dictionary with sizing recommendations
        """
        try:
            # Get benchmark analysis
            if benchmark_symbol not in self.analysis_results:
                self.calculate_benchmark_metrics(portfolio, benchmark_symbol)
            
            analysis = self.analysis_results[benchmark_symbol]
            
            # Current portfolio metrics
            current_volatility = analysis['performance_comparison']['portfolio_volatility']
            benchmark_volatility = analysis['performance_comparison']['benchmark_volatility']
            beta = analysis['alpha_beta']['beta']
            
            # Set target volatility
            if target_volatility is None:
                target_volatility = benchmark_volatility
            
            sizing_recommendations = {
                'current_volatility': current_volatility,
                'target_volatility': target_volatility,
                'benchmark_volatility': benchmark_volatility,
                'beta': beta,
                'sizing_method': sizing_method
            }
            
            if sizing_method == 'volatility_target':
                # Scale position size to match target volatility
                volatility_ratio = target_volatility / current_volatility if current_volatility > 0 else 1.0
                sizing_recommendations['recommended_size_multiplier'] = volatility_ratio
                sizing_recommendations['explanation'] = f"Scale position size by {volatility_ratio:.3f} to match target volatility of {target_volatility:.1%}"
                
            elif sizing_method == 'beta_target':
                # Scale to achieve target beta (default 1.0)
                target_beta = 1.0
                beta_ratio = target_beta / beta if beta != 0 else 1.0
                sizing_recommendations['recommended_size_multiplier'] = beta_ratio
                sizing_recommendations['target_beta'] = target_beta
                sizing_recommendations['explanation'] = f"Scale position size by {beta_ratio:.3f} to achieve target beta of {target_beta}"
                
            elif sizing_method == 'risk_parity':
                # Risk parity sizing based on inverse volatility
                risk_parity_weight = benchmark_volatility / current_volatility if current_volatility > 0 else 1.0
                sizing_recommendations['recommended_size_multiplier'] = risk_parity_weight
                sizing_recommendations['explanation'] = f"Risk parity sizing: {risk_parity_weight:.3f} based on inverse volatility"
                
            elif sizing_method == 'kelly':
                # Kelly criterion sizing
                portfolio_returns = analysis['raw_data']['portfolio_returns']
                win_rate = (portfolio_returns > 0).mean()
                avg_win = portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).sum() > 0 else 0
                avg_loss = abs(portfolio_returns[portfolio_returns < 0].mean()) if (portfolio_returns < 0).sum() > 0 else 0.01
                
                if avg_loss > 0:
                    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                else:
                    kelly_fraction = 0.1
                    
                sizing_recommendations['recommended_size_multiplier'] = kelly_fraction
                sizing_recommendations['kelly_fraction'] = kelly_fraction
                sizing_recommendations['win_rate'] = win_rate
                sizing_recommendations['avg_win'] = avg_win
                sizing_recommendations['avg_loss'] = avg_loss
                sizing_recommendations['explanation'] = f"Kelly criterion suggests {kelly_fraction:.1%} position sizing"
            
            return sizing_recommendations
            
        except Exception as e:
            logger.error(f"Error calibrating sizing: {e}")
            raise
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark analysis report.
        
        Args:
            output_path: Optional path to save report as JSON
            
        Returns:
            Complete analysis report
        """
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'risk_free_rate': self.risk_free_rate,
            'benchmarks_analyzed': list(self.analysis_results.keys()),
            'analysis_results': {}
        }
        
        # Clean results for JSON serialization
        for benchmark, results in self.analysis_results.items():
            clean_results = results.copy()
            # Remove raw data for JSON serialization
            if 'raw_data' in clean_results:
                del clean_results['raw_data']
            report['analysis_results'][benchmark] = clean_results
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Benchmark analysis report saved to {output_path}")
        
        return report
    
    def compare_multiple_benchmarks(self,
                                  portfolio: vbt.Portfolio,
                                  benchmark_symbols: List[str]) -> pd.DataFrame:
        """
        Compare portfolio against multiple benchmarks.
        
        Args:
            portfolio: Portfolio to analyze
            benchmark_symbols: List of benchmark symbols
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for symbol in benchmark_symbols:
            try:
                metrics = self.calculate_benchmark_metrics(portfolio, symbol)
                
                comparison_data.append({
                    'Benchmark': symbol,
                    'Alpha': metrics['alpha_beta']['alpha'],
                    'Beta': metrics['alpha_beta']['beta'],
                    'Correlation': metrics['alpha_beta']['correlation'],
                    'Excess Return': metrics['performance_comparison']['excess_return'],
                    'Information Ratio': metrics['risk_adjusted_metrics']['information_ratio'],
                    'Up Capture': metrics['risk_adjusted_metrics']['up_capture_ratio'],
                    'Down Capture': metrics['risk_adjusted_metrics']['down_capture_ratio']
                })
                
            except Exception as e:
                logger.warning(f"Failed to analyze benchmark {symbol}: {e}")
                continue
        
        return pd.DataFrame(comparison_data) 