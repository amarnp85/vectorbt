# Indicators¶

## Hurst exponent¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="12 months ago")
>>> hurst = vbt.HURST.run(data.close, method=["standard", "logrs", "rs", "dma", "dsod"])
>>> fig = vbt.make_subplots(specs=[[dict(secondary_y=True)]])
>>> data.plot(plot_volume=False, ohlc_trace_kwargs=dict(opacity=0.3), fig=fig)
>>> fig = hurst.hurst.vbt.plot(fig=fig, add_trace_kwargs=dict(secondary_y=True))
>>> fig = fig.select_range(start=hurst.param_defaults["window"])
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="12 months ago")
>>> hurst = vbt.HURST.run(data.close, method=["standard", "logrs", "rs", "dma", "dsod"])
>>> fig = vbt.make_subplots(specs=[[dict(secondary_y=True)]])
>>> data.plot(plot_volume=False, ohlc_trace_kwargs=dict(opacity=0.3), fig=fig)
>>> fig = hurst.hurst.vbt.plot(fig=fig, add_trace_kwargs=dict(secondary_y=True))
>>> fig = fig.select_range(start=hurst.param_defaults["window"])
>>> fig.show()
```

## Smart Money Concepts¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="6 months ago")
>>> phl = vbt.smc("previous_high_low").run(  # (1)!
...     data.open,
...     data.high,
...     data.low,
...     data.close,
...     data.volume,
...     time_frame=vbt.Default("7D")
... )
>>> fig = data.plot()
>>> phl.previous_high.rename("previous_high").vbt.plot(fig=fig)
>>> phl.previous_low.rename("previous_low").vbt.plot(fig=fig)
>>> (phl.broken_high == 1).rename("broken_high").vbt.signals.plot_as_markers(
...     y=phl.previous_high, 
...     trace_kwargs=dict(marker=dict(color="limegreen")),
...     fig=fig
... )
>>> (phl.broken_low == 1).rename("broken_low").vbt.signals.plot_as_markers(
...     y=phl.previous_low, 
...     trace_kwargs=dict(marker=dict(color="orangered")),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="6 months ago")
>>> phl = vbt.smc("previous_high_low").run(  # (1)!
...     data.open,
...     data.high,
...     data.low,
...     data.close,
...     data.volume,
...     time_frame=vbt.Default("7D")
... )
>>> fig = data.plot()
>>> phl.previous_high.rename("previous_high").vbt.plot(fig=fig)
>>> phl.previous_low.rename("previous_low").vbt.plot(fig=fig)
>>> (phl.broken_high == 1).rename("broken_high").vbt.signals.plot_as_markers(
...     y=phl.previous_high, 
...     trace_kwargs=dict(marker=dict(color="limegreen")),
...     fig=fig
... )
>>> (phl.broken_low == 1).rename("broken_low").vbt.signals.plot_as_markers(
...     y=phl.previous_low, 
...     trace_kwargs=dict(marker=dict(color="orangered")),
...     fig=fig
... )
>>> fig.show()
```

## Signal unraveling¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_sma = data.run("talib_func:sma", timeperiod=20)  # (1)!
>>> slow_sma = data.run("talib_func:sma", timeperiod=50)
>>> entries = fast_sma.vbt.crossed_above(slow_sma)
>>> exits = fast_sma.vbt.crossed_below(slow_sma)
>>> entries, exits = entries.vbt.signals.unravel_between(exits, relation="anychain")  # (2)!
>>> pf = vbt.PF.from_signals(
...     data, 
...     long_entries=entries, 
...     short_entries=exits, 
...     size=100,  # (3)!
...     size_type="value",
...     init_cash="auto",  # (4)!
...     tp_stop=0.2, 
...     sl_stop=0.1, 
...     group_by=vbt.ExceptLevel("signal"),  # (5)!
...     cash_sharing=True
... )
>>> pf.positions.returns.to_pd(ignore_index=True).vbt.barplot(
...     trace_kwargs=dict(marker=dict(colorscale="Spectral"))
... ).show()  # (6)!
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fast_sma = data.run("talib_func:sma", timeperiod=20)  # (1)!
>>> slow_sma = data.run("talib_func:sma", timeperiod=50)
>>> entries = fast_sma.vbt.crossed_above(slow_sma)
>>> exits = fast_sma.vbt.crossed_below(slow_sma)
>>> entries, exits = entries.vbt.signals.unravel_between(exits, relation="anychain")  # (2)!
>>> pf = vbt.PF.from_signals(
...     data, 
...     long_entries=entries, 
...     short_entries=exits, 
...     size=100,  # (3)!
...     size_type="value",
...     init_cash="auto",  # (4)!
...     tp_stop=0.2, 
...     sl_stop=0.1, 
...     group_by=vbt.ExceptLevel("signal"),  # (5)!
...     cash_sharing=True
... )
>>> pf.positions.returns.to_pd(ignore_index=True).vbt.barplot(
...     trace_kwargs=dict(marker=dict(colorscale="Spectral"))
... ).show()  # (6)!
```

```python
"talib_func:sma"
```

```python
"talib:sma"
```

## Lightweight TA-Lib¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> run_rsi = vbt.talib_func("rsi")
>>> rsi = run_rsi(data.close, timeperiod=12, timeframe="M")  # (1)!
>>> rsi
Date
2014-09-17 00:00:00+00:00          NaN
2014-09-18 00:00:00+00:00          NaN
2014-09-19 00:00:00+00:00          NaN
2014-09-20 00:00:00+00:00          NaN
2014-09-21 00:00:00+00:00          NaN
                                   ...
2024-01-18 00:00:00+00:00    64.210811
2024-01-19 00:00:00+00:00    64.210811
2024-01-20 00:00:00+00:00    64.210811
2024-01-21 00:00:00+00:00    64.210811
2024-01-22 00:00:00+00:00    64.210811
Freq: D, Name: Close, Length: 3415, dtype: float64

>>> plot_rsi = vbt.talib_plot_func("rsi")
>>> plot_rsi(rsi).show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> run_rsi = vbt.talib_func("rsi")
>>> rsi = run_rsi(data.close, timeperiod=12, timeframe="M")  # (1)!
>>> rsi
Date
2014-09-17 00:00:00+00:00          NaN
2014-09-18 00:00:00+00:00          NaN
2014-09-19 00:00:00+00:00          NaN
2014-09-20 00:00:00+00:00          NaN
2014-09-21 00:00:00+00:00          NaN
                                   ...
2024-01-18 00:00:00+00:00    64.210811
2024-01-19 00:00:00+00:00    64.210811
2024-01-20 00:00:00+00:00    64.210811
2024-01-21 00:00:00+00:00    64.210811
2024-01-22 00:00:00+00:00    64.210811
Freq: D, Name: Close, Length: 3415, dtype: float64

>>> plot_rsi = vbt.talib_plot_func("rsi")
>>> plot_rsi(rsi).show()
```

## Indicator search¶

```python
>>> vbt.IF.list_indicators("*ma")
[
    'vbt:MA',
    'talib:DEMA',
    'talib:EMA',
    'talib:KAMA',
    'talib:MA',
    ...
    'technical:ZEMA',
    'technical:ZLEMA',
    'technical:ZLHMA',
    'technical:ZLMA'
]

>>> vbt.indicator("technical:ZLMA")  # (1)!
vectorbtpro.indicators.factory.technical.ZLMA
```

```python
>>> vbt.IF.list_indicators("*ma")
[
    'vbt:MA',
    'talib:DEMA',
    'talib:EMA',
    'talib:KAMA',
    'talib:MA',
    ...
    'technical:ZEMA',
    'technical:ZLEMA',
    'technical:ZLHMA',
    'technical:ZLMA'
]

>>> vbt.indicator("technical:ZLMA")  # (1)!
vectorbtpro.indicators.factory.technical.ZLMA
```

```python
vbt.IF.get_indicator
```

## Indicators for ML¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> features = data.run("talib", mavp=vbt.run_arg_dict(periods=14))
>>> features.shape
(3046, 175)
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> features = data.run("talib", mavp=vbt.run_arg_dict(periods=14))
>>> features.shape
(3046, 175)
```

## Signal detection¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True)
>>> bbands = data.run("bbands")
>>> bbands.loc["2022"].plot(add_trace_kwargs=dict(row=1, col=1), fig=fig)
>>> sigdet = vbt.SIGDET.run(bbands.bandwidth, factor=5)
>>> sigdet.loc["2022"].plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> fig = vbt.make_subplots(rows=2, cols=1, shared_xaxes=True)
>>> bbands = data.run("bbands")
>>> bbands.loc["2022"].plot(add_trace_kwargs=dict(row=1, col=1), fig=fig)
>>> sigdet = vbt.SIGDET.run(bbands.bandwidth, factor=5)
>>> sigdet.loc["2022"].plot(add_trace_kwargs=dict(row=2, col=1), fig=fig)
>>> fig.show()
```

## Pivot detection¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2023")
>>> fig = data.plot(plot_volume=False)
>>> pivot_info = data.run("pivotinfo", up_th=1.0, down_th=0.5)
>>> pivot_info.plot(fig=fig, conf_value_trace_kwargs=dict(visible=False))
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2023")
>>> fig = data.plot(plot_volume=False)
>>> pivot_info = data.run("pivotinfo", up_th=1.0, down_th=0.5)
>>> pivot_info.plot(fig=fig, conf_value_trace_kwargs=dict(visible=False))
>>> fig.show()
```

## Technical indicators¶

```python
>>> vbt.YFData.pull("BTC-USD").run("sumcon", smooth=100).plot().show()
```

```python
>>> vbt.YFData.pull("BTC-USD").run("sumcon", smooth=100).plot().show()
```

## Renko chart¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2021", end="2022")
>>> renko_ohlc = data.close.vbt.to_renko_ohlc(1000, reset_index=True)  # (1)!
>>> renko_ohlc.vbt.ohlcv.plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2021", end="2022")
>>> renko_ohlc = data.close.vbt.to_renko_ohlc(1000, reset_index=True)  # (1)!
>>> renko_ohlc.vbt.ohlcv.plot().show()
```

## Rolling OLS¶

```python
>>> data = vbt.YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2022", 
...     end="2023",
...     missing_index="drop"
... )
>>> ols = vbt.OLS.run(
...     data.get("Close", "BTC-USD"), 
...     data.get("Close", "ETH-USD")
... )
>>> ols.plot_zscore().show()
```

```python
>>> data = vbt.YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2022", 
...     end="2023",
...     missing_index="drop"
... )
>>> ols = vbt.OLS.run(
...     data.get("Close", "BTC-USD"), 
...     data.get("Close", "ETH-USD")
... )
>>> ols.plot_zscore().show()
```

## TA-Lib time frames¶

Tutorial

Learn more in the MTF analysis tutorial.

```python
>>> h1_data = vbt.BinanceData.pull(
...     "BTCUSDT", 
...     start="3 months ago UTC", 
...     timeframe="1h"
... )
>>> mtf_sma = vbt.talib("SMA").run(
...     h1_data.close, 
...     timeperiod=14, 
...     timeframe=["1d", "4h", "1h"], 
...     skipna=True
... )
>>> mtf_sma.real.vbt.ts_heatmap().show()
```

```python
>>> h1_data = vbt.BinanceData.pull(
...     "BTCUSDT", 
...     start="3 months ago UTC", 
...     timeframe="1h"
... )
>>> mtf_sma = vbt.talib("SMA").run(
...     h1_data.close, 
...     timeperiod=14, 
...     timeframe=["1d", "4h", "1h"], 
...     skipna=True
... )
>>> mtf_sma.real.vbt.ts_heatmap().show()
```

## 1D-native indicators¶

```python
>>> import talib

>>> params = dict(
...     rsi_period=14, 
...     fastk_period=5, 
...     slowk_period=3, 
...     slowk_matype=0, 
...     slowd_period=3, 
...     slowd_matype=0
... )

>>> def stochrsi_1d(close, *args):
...     rsi = talib.RSI(close, args[0])
...     k, d = talib.STOCH(rsi, rsi, rsi, *args[1:])
...     return rsi, k, d

>>> STOCHRSI = vbt.IF(
...     input_names=["close"], 
...     param_names=list(params.keys()),
...     output_names=["rsi", "k", "d"]
... ).with_apply_func(stochrsi_1d, takes_1d=True, **params)

>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-06")
>>> stochrsi = STOCHRSI.run(data.close)
>>> fig = stochrsi.k.rename("%K").vbt.plot()
>>> stochrsi.d.rename("%D").vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> import talib

>>> params = dict(
...     rsi_period=14, 
...     fastk_period=5, 
...     slowk_period=3, 
...     slowk_matype=0, 
...     slowd_period=3, 
...     slowd_matype=0
... )

>>> def stochrsi_1d(close, *args):
...     rsi = talib.RSI(close, args[0])
...     k, d = talib.STOCH(rsi, rsi, rsi, *args[1:])
...     return rsi, k, d

>>> STOCHRSI = vbt.IF(
...     input_names=["close"], 
...     param_names=list(params.keys()),
...     output_names=["rsi", "k", "d"]
... ).with_apply_func(stochrsi_1d, takes_1d=True, **params)

>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-06")
>>> stochrsi = STOCHRSI.run(data.close)
>>> fig = stochrsi.k.rename("%K").vbt.plot()
>>> stochrsi.d.rename("%D").vbt.plot(fig=fig)
>>> fig.show()
```

## Parallelizable indicators¶

```python
>>> @njit
... def minmax_nb(close, window):
...     return (
...         vbt.nb.rolling_min_nb(close, window),
...         vbt.nb.rolling_max_nb(close, window)
...     )

>>> MINMAX = vbt.IF(
...     class_name="MINMAX",
...     input_names=["close"], 
...     param_names=["window"], 
...     output_names=["min", "max"]
... ).with_apply_func(minmax_nb, window=14)

>>> data = vbt.YFData.pull("BTC-USD")
```

```python
>>> @njit
... def minmax_nb(close, window):
...     return (
...         vbt.nb.rolling_min_nb(close, window),
...         vbt.nb.rolling_max_nb(close, window)
...     )

>>> MINMAX = vbt.IF(
...     class_name="MINMAX",
...     input_names=["close"], 
...     param_names=["window"], 
...     output_names=["min", "max"]
... ).with_apply_func(minmax_nb, window=14)

>>> data = vbt.YFData.pull("BTC-USD")
```

```python
>>> %%timeit
>>> minmax = MINMAX.run(
...     data.close, 
...     np.arange(2, 200),
...     jitted_loop=True
... )
420 ms ± 2.05 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> minmax = MINMAX.run(
...     data.close, 
...     np.arange(2, 200),
...     jitted_loop=True
... )
420 ms ± 2.05 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> minmax = MINMAX.run(
...     data.close, 
...     np.arange(2, 200),
...     jitted_loop=True,
...     jitted_warmup=True,  # (1)!
...     execute_kwargs=dict(engine="threadpool", n_chunks="auto")  # (2)!
... )
120 ms ± 355 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
>>> %%timeit
>>> minmax = MINMAX.run(
...     data.close, 
...     np.arange(2, 200),
...     jitted_loop=True,
...     jitted_warmup=True,  # (1)!
...     execute_kwargs=dict(engine="threadpool", n_chunks="auto")  # (2)!
... )
120 ms ± 355 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## TA-Lib plotting¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> vbt.talib("MACD").run(data.close).plot().show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> vbt.talib("MACD").run(data.close).plot().show()
```

## Indicator expressions¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> expr = """
... MACD:
... fast_ema = @talib_ema(close, @p_fast_w)
... slow_ema = @talib_ema(close, @p_slow_w)
... macd = fast_ema - slow_ema
... signal = @talib_ema(macd, @p_signal_w)
... macd, signal
... """
>>> MACD = vbt.IF.from_expr(expr, fast_w=12, slow_w=26, signal_w=9)  # (1)!
>>> macd = MACD.run(data.close)
>>> fig = macd.macd.rename("MACD").vbt.plot()
>>> macd.signal.rename("Signal").vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")

>>> expr = """
... MACD:
... fast_ema = @talib_ema(close, @p_fast_w)
... slow_ema = @talib_ema(close, @p_slow_w)
... macd = fast_ema - slow_ema
... signal = @talib_ema(macd, @p_signal_w)
... macd, signal
... """
>>> MACD = vbt.IF.from_expr(expr, fast_w=12, slow_w=26, signal_w=9)  # (1)!
>>> macd = MACD.run(data.close)
>>> fig = macd.macd.rename("MACD").vbt.plot()
>>> macd.signal.rename("Signal").vbt.plot(fig=fig)
>>> fig.show()
```

```python
input_names
```

```python
param_names
```

## WorldQuant Alphas¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD", "XRP-USD"], missing_index="drop")

>>> vbt.wqa101(1).run(data.close).out
symbol                      BTC-USD   ETH-USD   XRP-USD
Date                                                   
2017-11-09 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-10 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-11 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-12 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-13 00:00:00+00:00  0.166667  0.166667  0.166667
...                             ...       ...       ...
2023-01-31 00:00:00+00:00  0.166667  0.166667  0.166667
2023-02-01 00:00:00+00:00  0.000000  0.000000  0.500000
2023-02-02 00:00:00+00:00  0.000000  0.000000  0.500000
2023-02-03 00:00:00+00:00  0.000000  0.500000  0.000000
2023-02-04 00:00:00+00:00 -0.166667  0.333333  0.333333

[1914 rows x 3 columns]
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD", "XRP-USD"], missing_index="drop")

>>> vbt.wqa101(1).run(data.close).out
symbol                      BTC-USD   ETH-USD   XRP-USD
Date                                                   
2017-11-09 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-10 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-11 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-12 00:00:00+00:00  0.166667  0.166667  0.166667
2017-11-13 00:00:00+00:00  0.166667  0.166667  0.166667
...                             ...       ...       ...
2023-01-31 00:00:00+00:00  0.166667  0.166667  0.166667
2023-02-01 00:00:00+00:00  0.000000  0.000000  0.500000
2023-02-02 00:00:00+00:00  0.000000  0.000000  0.500000
2023-02-03 00:00:00+00:00  0.000000  0.500000  0.000000
2023-02-04 00:00:00+00:00 -0.166667  0.333333  0.333333

[1914 rows x 3 columns]
```

## Robust crossovers¶

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-03")
>>> fast_sma = vbt.talib("SMA").run(data.close, vbt.Default(5)).real
>>> slow_sma = vbt.talib("SMA").run(data.close, vbt.Default(10)).real
>>> fast_sma.iloc[np.random.choice(np.arange(len(fast_sma)), 5)] = np.nan
>>> slow_sma.iloc[np.random.choice(np.arange(len(slow_sma)), 5)] = np.nan
>>> crossed_above = fast_sma.vbt.crossed_above(slow_sma, dropna=True)
>>> crossed_below = fast_sma.vbt.crossed_below(slow_sma, dropna=True)

>>> fig = fast_sma.rename("Fast SMA").vbt.lineplot()
>>> slow_sma.rename("Slow SMA").vbt.lineplot(fig=fig)
>>> crossed_above.vbt.signals.plot_as_entries(fast_sma, fig=fig)
>>> crossed_below.vbt.signals.plot_as_exits(fast_sma, fig=fig)
>>> fig.show()
```

```python
>>> data = vbt.YFData.pull("BTC-USD", start="2022-01", end="2022-03")
>>> fast_sma = vbt.talib("SMA").run(data.close, vbt.Default(5)).real
>>> slow_sma = vbt.talib("SMA").run(data.close, vbt.Default(10)).real
>>> fast_sma.iloc[np.random.choice(np.arange(len(fast_sma)), 5)] = np.nan
>>> slow_sma.iloc[np.random.choice(np.arange(len(slow_sma)), 5)] = np.nan
>>> crossed_above = fast_sma.vbt.crossed_above(slow_sma, dropna=True)
>>> crossed_below = fast_sma.vbt.crossed_below(slow_sma, dropna=True)

>>> fig = fast_sma.rename("Fast SMA").vbt.lineplot()
>>> slow_sma.rename("Slow SMA").vbt.lineplot(fig=fig)
>>> crossed_above.vbt.signals.plot_as_entries(fast_sma, fig=fig)
>>> crossed_below.vbt.signals.plot_as_exits(fast_sma, fig=fig)
>>> fig.show()
```

Python code

