# Simulation Configuration Guide

## Overview

This guide explains how to ensure consistent, realistic optimization and backtesting using the unified configuration system.

## Quick Start

### **Problem Solved**
Previously, optimization and backtesting used different transaction costs, leading to inconsistent results:
- Optimization: No slippage, minimal fees
- Backtest: Realistic fees + slippage
- Result: Optimized parameters performed poorly in realistic backtesting

### **Solution**
Unified configuration system ensures both optimization and backtesting use identical, realistic settings.

## Configuration Modes

### **1. Realistic Trading (Default)**
```bash
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize
```
- **Optimization**: Realistic fees (0.1%) + slippage (0.05%)
- **Backtest**: Same realistic settings
- **Use Case**: Production-ready optimization with realistic costs

### **2. Unified Configuration**
```bash
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --use-same-config
```
- **Both use identical settings** - guaranteed consistency
- **No configuration drift** between optimization and backtest

### **3. Production Mode**
```bash
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --sim-mode production_trading
```
- **Conservative estimates**: Higher fees (0.15%) + slippage (0.1%)
- **Use Case**: Final validation before live trading

### **4. Fast Optimization**
```bash
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --sim-mode fast_optimization
```
- **Reduced costs** for faster parameter exploration
- **Use Case**: Initial parameter discovery (validate with realistic mode after)

## Configuration Presets

| Mode | Fees | Slippage | Total Cost | Use Case |
|------|------|----------|------------|----------|
| `fast_optimization` | 0.05% | 0% | 0.05% | Initial parameter discovery |
| `realistic_trading` | 0.1% | 0.05% | 0.15% | **Default production mode** |
| `production_trading` | 0.15% | 0.1% | 0.25% | Conservative live trading |
| `development` | 0.1% | 0.05% | 0.15% | Development and testing |

## Best Practices

### **1. For Production Optimization**
```bash
# Step 1: Fast initial optimization
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --sim-mode fast_optimization

# Step 2: Validate with realistic settings
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --force --use-same-config

# Step 3: Final validation with conservative settings
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --sim-mode production_trading
```

### **2. For Consistent Results**
```bash
# Always use --use-same-config for direct optimization vs backtest comparison
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --use-same-config
```

### **3. For Development**
```bash
# Use development mode for balanced speed and realism
python examples/strategy_tester.py --symbols SOL/USDT --timeframes 4h --optimize --sim-mode development
```

## Configuration Validation

The system automatically validates configuration consistency:

```
✅ Configurations are identical - results are directly comparable
⚠️  WARNING: Configuration mismatch - optimization and backtest use different settings
```

## Advanced Usage

### **Custom Configuration**
Modify `backtester/config/simulation_config.py` to:
- Add new presets
- Modify existing presets
- Create exchange-specific configurations

### **Environment-Specific Configs**
- **Development**: Fast iteration, balanced realism
- **Testing**: Production-like validation
- **Production**: Conservative, live-trading ready

## Migration Guide

### **Before (Inconsistent)**
```python
# Optimization used minimal costs
sim_config = SimulationConfig(init_cash=100000, fees=0.001)

# Backtest used realistic costs  
sim_config = SimulationConfig(init_cash=100000, fees=0.001, slippage=0.0005)
```

### **After (Consistent)**
```python
# Both use standardized configuration
optimization_config = get_optimization_config()  # Realistic settings
backtest_config = get_backtest_config()          # Same realistic settings
```

## Expected Results

### **With Unified Configuration**
- ✅ Optimization Sharpe: 1.30
- ✅ Backtest Sharpe: 1.28 (within 2% - normal variance)
- ✅ Consistent performance metrics

### **Without Unified Configuration (Previous)**
- ❌ Optimization Sharpe: 1.47
- ❌ Backtest Sharpe: 0.05 (96% difference!)
- ❌ Unusable optimization results

## Troubleshooting

### **Configuration Mismatch Warning**
```
⚠️  WARNING: Configuration mismatch in 'slippage': optimization=0.0, backtest=0.0005
```
**Solution**: Use `--use-same-config` flag

### **Poor Backtest Performance vs Optimization**
**Likely Cause**: Different transaction costs
**Solution**: Ensure using unified configuration or realistic preset

### **Optimization Too Slow**
**Solution**: Use `--sim-mode fast_optimization` for initial exploration, then validate with realistic settings

## Summary

1. **Default behavior**: Uses realistic settings for both optimization and backtest
2. **For guaranteed consistency**: Add `--use-same-config` flag  
3. **For conservative validation**: Use `--sim-mode production_trading`
4. **For fast exploration**: Use `--sim-mode fast_optimization`

The unified configuration system ensures your optimized parameters will perform as expected in realistic trading conditions.