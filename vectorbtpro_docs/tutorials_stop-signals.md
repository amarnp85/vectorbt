# Stop signals¶

Our goal is to utilize large-scale backtesting to compare the performance of trading with and without stop loss (SL), trailing stop (TS), and take profit (TP) signals. To make this attempt representative, we will run a huge number of experiments across three different dimensions: instruments, time, and parameters.

First, we will pick 10 cryptocurrencies by market capitalization (except stablecoins such as USDT) and fetch 3 years of their daily pricing data. In particular, we aim at backtesting the time period from 2018 to 2021 as it contains periods of sharp price drops (e.g., corrections due to ATH in December 2017 and coronavirus in March 2020) as well as surges (ATH in December 2020) — this keeps things balanced. For each instrument, we will split this time period into 400 smaller (and overlapping) time windows, each 6 months long. We will run our tests on each of these windows to account for different market regimes. For each instrument and time window, we will then generate an entry signal at the very first bar and find an exit signal according to the stop configuration. We will test 100 stop values with a 1% increment and compare the performance of each one to that of trading randomly and holding within this particular time window. In total, we will conduct 2,000,000 tests.

Important

Make sure that you have at least 16 GB of free RAM available, or memory swapping enabled.

## Parameters¶

The first step is to define the parameters of the analysis pipeline. As discussed above, we will backtest 3 years of pricing data, 400 time windows, 10 cryptocurrencies, and 100 stop values. We will also set fees and slippage both to 0.25% and initial capital to $100 (the amount per se doesn't matter, but it must be the same for all assets to be comparable). Feel free to change any parameter of interest.

```python
>>> from vectorbtpro import *
>>> import ipywidgets

>>> seed = 42
>>> symbols = [
...     "BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD", 
...     "BNB-USD", "EOS-USD", "XLM-USD", "XMR-USD", "ADA-USD"
... ]
>>> start_date = vbt.utc_timestamp("2018-01-01")
>>> end_date = vbt.utc_timestamp("2021-01-01")
>>> time_delta = end_date - start_date
>>> window_len = vbt.timedelta("180d")
>>> window_cnt = 400
>>> exit_types = ["SL", "TS", "TP", "Random", "Holding"]
>>> step = 0.01  # (1)!
>>> stops = np.arange(step, 1 + step, step)

>>> vbt.settings.wrapping["freq"] = "d"
>>> vbt.settings.plotting["layout"]["template"] = "vbt_dark"
>>> vbt.settings.portfolio["init_cash"] = 100.  # (2)!
>>> vbt.settings.portfolio["fees"] = 0.0025  # (3)!
>>> vbt.settings.portfolio["slippage"] = 0.0025  # (4)!

>>> pd.Series({
...     "Start date": start_date,
...     "End date": end_date,
...     "Time period (days)": time_delta.days,
...     "Assets": len(symbols),
...     "Window length": window_len,
...     "Windows": window_cnt,
...     "Exit types": len(exit_types),
...     "Stop values": len(stops),
...     "Tests per asset": window_cnt * len(stops) * len(exit_types),
...     "Tests per window": len(symbols) * len(stops) * len(exit_types),
...     "Tests per exit type": len(symbols) * window_cnt * len(stops),
...     "Tests per stop type and value": len(symbols) * window_cnt,
...     "Tests total": len(symbols) * window_cnt * len(stops) * len(exit_types)
... })
Start date                       2018-01-01 00:00:00+00:00
End date                         2021-01-01 00:00:00+00:00
Time period (days)                                    1096
Assets                                                  10
Window length                            180 days 00:00:00
Windows                                                400
Exit types                                               5
Stop values                                            100
Tests per asset                                     200000
Tests per window                                      5000
Tests per exit type                                 400000
Tests per stop type and value                         4000
Tests total                                        2000000
dtype: object
```

```python
>>> from vectorbtpro import *
>>> import ipywidgets

>>> seed = 42
>>> symbols = [
...     "BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD", 
...     "BNB-USD", "EOS-USD", "XLM-USD", "XMR-USD", "ADA-USD"
... ]
>>> start_date = vbt.utc_timestamp("2018-01-01")
>>> end_date = vbt.utc_timestamp("2021-01-01")
>>> time_delta = end_date - start_date
>>> window_len = vbt.timedelta("180d")
>>> window_cnt = 400
>>> exit_types = ["SL", "TS", "TP", "Random", "Holding"]
>>> step = 0.01  # (1)!
>>> stops = np.arange(step, 1 + step, step)

>>> vbt.settings.wrapping["freq"] = "d"
>>> vbt.settings.plotting["layout"]["template"] = "vbt_dark"
>>> vbt.settings.portfolio["init_cash"] = 100.  # (2)!
>>> vbt.settings.portfolio["fees"] = 0.0025  # (3)!
>>> vbt.settings.portfolio["slippage"] = 0.0025  # (4)!

>>> pd.Series({
...     "Start date": start_date,
...     "End date": end_date,
...     "Time period (days)": time_delta.days,
...     "Assets": len(symbols),
...     "Window length": window_len,
...     "Windows": window_cnt,
...     "Exit types": len(exit_types),
...     "Stop values": len(stops),
...     "Tests per asset": window_cnt * len(stops) * len(exit_types),
...     "Tests per window": len(symbols) * len(stops) * len(exit_types),
...     "Tests per exit type": len(symbols) * window_cnt * len(stops),
...     "Tests per stop type and value": len(symbols) * window_cnt,
...     "Tests total": len(symbols) * window_cnt * len(stops) * len(exit_types)
... })
Start date                       2018-01-01 00:00:00+00:00
End date                         2021-01-01 00:00:00+00:00
Time period (days)                                    1096
Assets                                                  10
Window length                            180 days 00:00:00
Windows                                                400
Exit types                                               5
Stop values                                            100
Tests per asset                                     200000
Tests per window                                      5000
Tests per exit type                                 400000
Tests per stop type and value                         4000
Tests total                                        2000000
dtype: object
```

Our configuration yields sample sizes with enough statistical power to analyze four variables: assets (200k tests per asset), time (5k tests per time window), exit types (400k tests per exit type), and stop values (4k tests per stop type and value). Similar to how Tableau handles dimensions and measures, we will be able to group our performance by each of these variables, but we will mainly focus on 5 exit types: SL exits, TS exits, TP exits, random exits, and holding exits (placed at the last bar).

```python
>>> cols = ["Open", "Low", "High", "Close", "Volume"]
>>> yfdata = vbt.YFData.pull(symbols, start=start_date, end=end_date)
```

```python
>>> cols = ["Open", "Low", "High", "Close", "Volume"]
>>> yfdata = vbt.YFData.pull(symbols, start=start_date, end=end_date)
```

Symbol 10/10

Symbol 10/10

```python
>>> yfdata.data.keys()
dict_keys(['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD', 
           'BNB-USD', 'EOS-USD', 'XLM-USD', 'XMR-USD', 'ADA-USD'])

>>> yfdata.data["BTC-USD"].shape
(1096, 7)
```

```python
>>> yfdata.data.keys()
dict_keys(['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD', 
           'BNB-USD', 'EOS-USD', 'XLM-USD', 'XMR-USD', 'ADA-USD'])

>>> yfdata.data["BTC-USD"].shape
(1096, 7)
```

The data instance yfdata contains a dictionary with the OHLCV data by cryptocurrency name. Each DataFrame has 1096 rows (days) and 5 columns (O, H, L, C, and V). You can plot the DataFrame as follows:

```python
yfdata
```

```python
>>> yfdata.plot(symbol="BTC-USD").show()  # (1)!
```

```python
>>> yfdata.plot(symbol="BTC-USD").show()  # (1)!
```

Since assets are one of the dimensions we want to analyze, vectorbt expects us to pack them as columns into a single DataFrame and label them accordingly. To do so, we simply swap assets and features to get a dictionary of DataFrames (with assets now as columns) keyed by feature name, such as "Open".

```python
>>> ohlcv = yfdata.concat()

>>> ohlcv.keys()
dict_keys(['Open', 'High', 'Low', 'Close', 
           'Volume', 'Dividends', 'Stock Splits'])

>>> ohlcv['Open'].shape
(1096, 10)
```

```python
>>> ohlcv = yfdata.concat()

>>> ohlcv.keys()
dict_keys(['Open', 'High', 'Low', 'Close', 
           'Volume', 'Dividends', 'Stock Splits'])

>>> ohlcv['Open'].shape
(1096, 10)
```

## Time windows¶

Next, we will move a 6-month sliding window over the whole time period and take 400 "snapshots" of each price DataFrame within this window. Each snapshot will correspond to a subset of data that should be independently backtested. As with assets and other variables, snapshots also need to be stacked horizontally as columns. As a result, we will get 180 rows (window length in days) and 4000 columns (10 assets x 400 windows); that is, one column will correspond to the price of one asset within one particular time window.

```python
>>> splitter = vbt.Splitter.from_n_rolling(  # (1)!
...     ohlcv["Open"].index, 
...     n=window_cnt,
...     length=window_len.days
... )

>>> split_ohlcv = {}
>>> for k, v in ohlcv.items():  # (2)!
...     split_ohlcv[k] = splitter.take(v, into="reset_stacked")  # (3)!
>>> print(split_ohlcv["Open"].shape)
(180, 4000)

>>> split_indexes = splitter.take(ohlcv["Open"].index)  # (4)!
>>> print(split_indexes)
split
0      DatetimeIndex(['2018-01-01 00:00:00+00:00', '2...
1      DatetimeIndex(['2018-01-03 00:00:00+00:00', '2...
2      DatetimeIndex(['2018-01-06 00:00:00+00:00', '2...
3      DatetimeIndex(['2018-01-08 00:00:00+00:00', '2...
4      DatetimeIndex(['2018-01-10 00:00:00+00:00', '2...
                             ...                        
395    DatetimeIndex(['2020-06-26 00:00:00+00:00', '2...
396    DatetimeIndex(['2020-06-28 00:00:00+00:00', '2...
397    DatetimeIndex(['2020-06-30 00:00:00+00:00', '2...
398    DatetimeIndex(['2020-07-03 00:00:00+00:00', '2...
399    DatetimeIndex(['2020-07-05 00:00:00+00:00', '2...
Length: 400, dtype: object

>>> print(split_indexes[10])  # (5)!
DatetimeIndex(['2018-01-24 00:00:00+00:00', '2018-01-25 00:00:00+00:00',
               '2018-01-26 00:00:00+00:00', '2018-01-27 00:00:00+00:00',
               '2018-01-28 00:00:00+00:00', '2018-01-29 00:00:00+00:00',
               ...
               '2018-07-17 00:00:00+00:00', '2018-07-18 00:00:00+00:00',
               '2018-07-19 00:00:00+00:00', '2018-07-20 00:00:00+00:00',
               '2018-07-21 00:00:00+00:00', '2018-07-22 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Date', length=180, freq='D')
```

```python
>>> splitter = vbt.Splitter.from_n_rolling(  # (1)!
...     ohlcv["Open"].index, 
...     n=window_cnt,
...     length=window_len.days
... )

>>> split_ohlcv = {}
>>> for k, v in ohlcv.items():  # (2)!
...     split_ohlcv[k] = splitter.take(v, into="reset_stacked")  # (3)!
>>> print(split_ohlcv["Open"].shape)
(180, 4000)

>>> split_indexes = splitter.take(ohlcv["Open"].index)  # (4)!
>>> print(split_indexes)
split
0      DatetimeIndex(['2018-01-01 00:00:00+00:00', '2...
1      DatetimeIndex(['2018-01-03 00:00:00+00:00', '2...
2      DatetimeIndex(['2018-01-06 00:00:00+00:00', '2...
3      DatetimeIndex(['2018-01-08 00:00:00+00:00', '2...
4      DatetimeIndex(['2018-01-10 00:00:00+00:00', '2...
                             ...                        
395    DatetimeIndex(['2020-06-26 00:00:00+00:00', '2...
396    DatetimeIndex(['2020-06-28 00:00:00+00:00', '2...
397    DatetimeIndex(['2020-06-30 00:00:00+00:00', '2...
398    DatetimeIndex(['2020-07-03 00:00:00+00:00', '2...
399    DatetimeIndex(['2020-07-05 00:00:00+00:00', '2...
Length: 400, dtype: object

>>> print(split_indexes[10])  # (5)!
DatetimeIndex(['2018-01-24 00:00:00+00:00', '2018-01-25 00:00:00+00:00',
               '2018-01-26 00:00:00+00:00', '2018-01-27 00:00:00+00:00',
               '2018-01-28 00:00:00+00:00', '2018-01-29 00:00:00+00:00',
               ...
               '2018-07-17 00:00:00+00:00', '2018-07-18 00:00:00+00:00',
               '2018-07-19 00:00:00+00:00', '2018-07-20 00:00:00+00:00',
               '2018-07-21 00:00:00+00:00', '2018-07-22 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Date', length=180, freq='D')
```

```python
ohlcv
```

```python
10
```

A nice feature of vectorbt is that it makes use of hierarchical indexing to store valuable information on each backtest. It also ensures that this column hierarchy is preserved across the whole backtesting pipeline — from signal generation to performance modeling — and can be extended easily. Currently, our columns have the following hierarchy:

```python
>>> split_ohlcv["Open"].columns
MultiIndex([(  0, 'BTC-USD'),
            (  0, 'ETH-USD'),
            (  0, 'XRP-USD'),
            ...
            (399, 'XLM-USD'),
            (399, 'XMR-USD'),
            (399, 'ADA-USD')],
           names=['split', 'symbol'], length=4000)
```

```python
>>> split_ohlcv["Open"].columns
MultiIndex([(  0, 'BTC-USD'),
            (  0, 'ETH-USD'),
            (  0, 'XRP-USD'),
            ...
            (399, 'XLM-USD'),
            (399, 'XMR-USD'),
            (399, 'ADA-USD')],
           names=['split', 'symbol'], length=4000)
```

This multi-index captures three parameters: the symbol, the start date of the time window, and its end date. Later, we will extend this multi-index with exit types and stop values such that each of the 2 million tests has its own price series.

## Entry signals¶

In contrast to most other backtesting libraries, signals are not stored as a signed integer array, but they are split into two boolean arrays: entries and exits, which makes manipulation a lot easier. At the beginning of each time window, let's generate an entry signal indicating a buy order. The data frame will have the same shape, index, and columns as that of price so that vectorbt can link their elements together.

```python
>>> entries = pd.DataFrame.vbt.signals.empty_like(split_ohlcv["Open"])
>>> entries.iloc[0, :] = True

>>> entries.shape
(180, 4000)
```

```python
>>> entries = pd.DataFrame.vbt.signals.empty_like(split_ohlcv["Open"])
>>> entries.iloc[0, :] = True

>>> entries.shape
(180, 4000)
```

## Exit signals¶

For each of the entry signals we generated, we will find an exit signal according to our 5 exit types: SL, TS, TP, random, and holding. We will also concatenate their DataFrames into a single (huge) DataFrame with 180 rows and 2,000,000 columns, each representing a separate backtest. Since exit signals are boolean, their memory footprint is tolerable.

Let's generate exit signals according to stop conditions first. We want to test 100 different stop values with a 1% increment, starting from 1% and ending with 100% (i.e., find a timestamp where the price exceeds the entry price by 100%). When OHLC data is checked against such conditions, the position is closed at (or shortly after) the time of hitting the particular stop.

Hint

We use OHLCSTX instead of built-in stop-loss in Portfolio.from_signals because we want to analyze signals before simulation. Also, it's easier to construct param grids. For reality check, run the same setup using Portfolio.from_signals alone.

```python
>>> sl_ohlcstx = vbt.OHLCSTX.run(
...     entries,  # (1)!
...     entry_price=split_ohlcv["Close"],  # (2)!
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     sl_stop=list(stops),  # (3)!
...     stop_type=None  # (4)!
... )
>>> sl_exits = sl_ohlcstx.exits.copy()  # (5)!
>>> sl_price = sl_ohlcstx.close.copy()  # (6)!
>>> sl_price[sl_exits] = sl_ohlcstx.stop_price
>>> del sl_ohlcstx  # (7)!

>>> sl_exits.shape
(180, 400000)

>>> tsl_ohlcstx = vbt.OHLCSTX.run(
...     entries, 
...     entry_price=split_ohlcv["Close"], 
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     tsl_stop=list(stops),
...     stop_type=None
... )
>>> tsl_exits = tsl_ohlcstx.exits.copy()
>>> tsl_price = tsl_ohlcstx.close.copy()
>>> tsl_price[tsl_exits] = tsl_ohlcstx.stop_price
>>> del tsl_ohlcstx

>>> tsl_exits.shape
(180, 400000)

>>> tp_ohlcstx = vbt.OHLCSTX.run(
...     entries, 
...     entry_price=split_ohlcv["Close"], 
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     tp_stop=list(stops),
...     stop_type=None
... )
>>> tp_exits = tp_ohlcstx.exits.copy()
>>> tp_price = tp_ohlcstx.close.copy()
>>> tp_price[tp_exits] = tp_ohlcstx.stop_price
>>> del tp_ohlcstx

>>> tp_exits.shape
(180, 400000)
```

```python
>>> sl_ohlcstx = vbt.OHLCSTX.run(
...     entries,  # (1)!
...     entry_price=split_ohlcv["Close"],  # (2)!
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     sl_stop=list(stops),  # (3)!
...     stop_type=None  # (4)!
... )
>>> sl_exits = sl_ohlcstx.exits.copy()  # (5)!
>>> sl_price = sl_ohlcstx.close.copy()  # (6)!
>>> sl_price[sl_exits] = sl_ohlcstx.stop_price
>>> del sl_ohlcstx  # (7)!

>>> sl_exits.shape
(180, 400000)

>>> tsl_ohlcstx = vbt.OHLCSTX.run(
...     entries, 
...     entry_price=split_ohlcv["Close"], 
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     tsl_stop=list(stops),
...     stop_type=None
... )
>>> tsl_exits = tsl_ohlcstx.exits.copy()
>>> tsl_price = tsl_ohlcstx.close.copy()
>>> tsl_price[tsl_exits] = tsl_ohlcstx.stop_price
>>> del tsl_ohlcstx

>>> tsl_exits.shape
(180, 400000)

>>> tp_ohlcstx = vbt.OHLCSTX.run(
...     entries, 
...     entry_price=split_ohlcv["Close"], 
...     open=split_ohlcv["Open"], 
...     high=split_ohlcv["High"], 
...     low=split_ohlcv["Low"], 
...     close=split_ohlcv["Close"], 
...     tp_stop=list(stops),
...     stop_type=None
... )
>>> tp_exits = tp_ohlcstx.exits.copy()
>>> tp_price = tp_ohlcstx.close.copy()
>>> tp_price[tp_exits] = tp_ohlcstx.stop_price
>>> del tp_ohlcstx

>>> tp_exits.shape
(180, 400000)
```

```python
stop_type
```

```python
None
```

This also extended our column hierarchy with a new column level indicating the stop value, we only have to make it consistent across all DataFrames:

```python
>>> def rename_stop_level(df):
...     return df.vbt.rename_levels({
...         "ohlcstx_sl_stop": "stop_value",
...         "ohlcstx_tsl_stop": "stop_value",
...         "ohlcstx_tp_stop": "stop_value"
...     }, strict=False)

>>> sl_exits = rename_stop_level(sl_exits)
>>> tsl_exits = rename_stop_level(tsl_exits)
>>> tp_exits = rename_stop_level(tp_exits)

>>> sl_price = rename_stop_level(sl_price)
>>> tsl_price = rename_stop_level(tsl_price)
>>> tp_price = rename_stop_level(tp_price)

>>> sl_exits.columns
MultiIndex([(0.01,   0, 'BTC-USD'),
            (0.01,   0, 'ETH-USD'),
            (0.01,   0, 'XRP-USD'),
            ...
            ( 1.0, 399, 'XLM-USD'),
            ( 1.0, 399, 'XMR-USD'),
            ( 1.0, 399, 'ADA-USD')],
           names=['stop_value', 'split', 'symbol'], length=400000)
```

```python
>>> def rename_stop_level(df):
...     return df.vbt.rename_levels({
...         "ohlcstx_sl_stop": "stop_value",
...         "ohlcstx_tsl_stop": "stop_value",
...         "ohlcstx_tp_stop": "stop_value"
...     }, strict=False)

>>> sl_exits = rename_stop_level(sl_exits)
>>> tsl_exits = rename_stop_level(tsl_exits)
>>> tp_exits = rename_stop_level(tp_exits)

>>> sl_price = rename_stop_level(sl_price)
>>> tsl_price = rename_stop_level(tsl_price)
>>> tp_price = rename_stop_level(tp_price)

>>> sl_exits.columns
MultiIndex([(0.01,   0, 'BTC-USD'),
            (0.01,   0, 'ETH-USD'),
            (0.01,   0, 'XRP-USD'),
            ...
            ( 1.0, 399, 'XLM-USD'),
            ( 1.0, 399, 'XMR-USD'),
            ( 1.0, 399, 'ADA-USD')],
           names=['stop_value', 'split', 'symbol'], length=400000)
```

One major feature of vectorbt is that it places a strong focus on data science, and so it allows us to apply popular analysis tools to almost any part of the backtesting pipeline. For example, let's explore how the number of exit signals depends upon the stop type and value:

```python
>>> pd.Series({
...     "SL": sl_exits.vbt.signals.total().mean(),
...     "TS": tsl_exits.vbt.signals.total().mean(),
...     "TP": tp_exits.vbt.signals.total().mean()
... }, name="avg_num_signals")
SL    0.428585
TS    0.587100
TP    0.520042
Name: avg_num_signals, dtype: float64

>>> def groupby_stop_value(df):
...     return df.vbt.signals.total().groupby("stop_value").mean()

>>> pd.DataFrame({
...     "Stop Loss": groupby_stop_value(sl_exits),
...     "Trailing Stop": groupby_stop_value(tsl_exits),
...     "Take Profit": groupby_stop_value(tp_exits)
... }).vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Avg number of signals"
... ).show()
```

```python
>>> pd.Series({
...     "SL": sl_exits.vbt.signals.total().mean(),
...     "TS": tsl_exits.vbt.signals.total().mean(),
...     "TP": tp_exits.vbt.signals.total().mean()
... }, name="avg_num_signals")
SL    0.428585
TS    0.587100
TP    0.520042
Name: avg_num_signals, dtype: float64

>>> def groupby_stop_value(df):
...     return df.vbt.signals.total().groupby("stop_value").mean()

>>> pd.DataFrame({
...     "Stop Loss": groupby_stop_value(sl_exits),
...     "Trailing Stop": groupby_stop_value(tsl_exits),
...     "Take Profit": groupby_stop_value(tp_exits)
... }).vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Avg number of signals"
... ).show()
```

We see that TS is by far the most occurring exit signal. The SL and TP curves come hand in hand up to the stop value of 50% and then diverge in favor of TP. While it might seem that bulls are mostly in charge, especially for bigger price movements, remember that it is much easier to post a 50% profit than a 50% loss because the latter requires a 100% profit to recover; thus, negative downward spikes seem to dominate small to medium price movements (and shake out weak hands potentially). These are well-known cryptocurrency dynamics.

To simplify the analysis that follows, we should ensure that each column has at least one exit signal to close the position, which means that if a column has no exit signal now, it should get one at the last timestamp. This is done by combining the stop exits with the last-bar exit using the OR rule and selecting the one that comes first:

```python
>>> sl_exits.iloc[-1, :] = True
>>> tsl_exits.iloc[-1, :] = True
>>> tp_exits.iloc[-1, :] = True

>>> sl_exits = sl_exits.vbt.signals.first_after(entries)  # (1)!
>>> tsl_exits = tsl_exits.vbt.signals.first_after(entries)
>>> tp_exits = tp_exits.vbt.signals.first_after(entries)

>>> pd.Series({
...     "SL": sl_exits.vbt.signals.total().mean(),
...     "TS": tsl_exits.vbt.signals.total().mean(),
...     "TP": tp_exits.vbt.signals.total().mean()
... }, name="avg_num_signals")
SL    1.0
TS    1.0
TP    1.0
Name: avg_num_signals, dtype: float64
```

```python
>>> sl_exits.iloc[-1, :] = True
>>> tsl_exits.iloc[-1, :] = True
>>> tp_exits.iloc[-1, :] = True

>>> sl_exits = sl_exits.vbt.signals.first_after(entries)  # (1)!
>>> tsl_exits = tsl_exits.vbt.signals.first_after(entries)
>>> tp_exits = tp_exits.vbt.signals.first_after(entries)

>>> pd.Series({
...     "SL": sl_exits.vbt.signals.total().mean(),
...     "TS": tsl_exits.vbt.signals.total().mean(),
...     "TP": tp_exits.vbt.signals.total().mean()
... }, name="avg_num_signals")
SL    1.0
TS    1.0
TP    1.0
Name: avg_num_signals, dtype: float64
```

Next, we will generate signals of the two remaining exit types: random and holding — they will act as benchmarks to compare SL, TS, and TP against.

"Holding" exit signals are signals placed at the very last bar of each time series. On most occasions, we shouldn't bother ourselves with placing them, since we can simply assess open positions. The reason we do it anyway is consistency — we want to ensure that each column has (exactly) one signal. The other consideration is shape and columns: they should match that of stop signals, so we can concatenate all DataFrames later.

```python
>>> hold_exits = pd.DataFrame.vbt.signals.empty_like(sl_exits)
>>> hold_exits.iloc[-1, :] = True
>>> hold_price = vbt.broadcast_to(split_ohlcv["Close"], sl_price)

>>> hold_exits.shape
(180, 400000)
```

```python
>>> hold_exits = pd.DataFrame.vbt.signals.empty_like(sl_exits)
>>> hold_exits.iloc[-1, :] = True
>>> hold_price = vbt.broadcast_to(split_ohlcv["Close"], sl_price)

>>> hold_exits.shape
(180, 400000)
```

To generate random exit signals, just shuffle any signal array. The only requirement is that each column should contain exactly one signal.

```python
>>> rand_exits = hold_exits.vbt.shuffle(seed=seed)  # (1)!
>>> rand_price = hold_price

>>> rand_exits.shape
(180, 400000)
```

```python
>>> rand_exits = hold_exits.vbt.shuffle(seed=seed)  # (1)!
>>> rand_price = hold_price

>>> rand_exits.shape
(180, 400000)
```

The last step is the concatenation of all DataFrames along the column axis and labeling them using a new column level exit_type:

```python
exit_type
```

```python
>>> exits = pd.DataFrame.vbt.concat(
...     sl_exits, 
...     tsl_exits, 
...     tp_exits, 
...     rand_exits, 
...     hold_exits, 
...     keys=pd.Index(exit_types, name="exit_type")
... )
>>> del sl_exits  # (1)!
>>> del tsl_exits
>>> del tp_exits
>>> del rand_exits
>>> del hold_exits

>>> exits.shape
(180, 2000000)

>>> price = pd.DataFrame.vbt.concat(
...     sl_price, 
...     tsl_price, 
...     tp_price, 
...     rand_price, 
...     hold_price, 
...     keys=pd.Index(exit_types, name="exit_type")
... )
>>> del sl_price
>>> del tsl_price
>>> del tp_price
>>> del rand_price
>>> del hold_price

>>> price.shape
```

```python
>>> exits = pd.DataFrame.vbt.concat(
...     sl_exits, 
...     tsl_exits, 
...     tp_exits, 
...     rand_exits, 
...     hold_exits, 
...     keys=pd.Index(exit_types, name="exit_type")
... )
>>> del sl_exits  # (1)!
>>> del tsl_exits
>>> del tp_exits
>>> del rand_exits
>>> del hold_exits

>>> exits.shape
(180, 2000000)

>>> price = pd.DataFrame.vbt.concat(
...     sl_price, 
...     tsl_price, 
...     tp_price, 
...     rand_price, 
...     hold_price, 
...     keys=pd.Index(exit_types, name="exit_type")
... )
>>> del sl_price
>>> del tsl_price
>>> del tp_price
>>> del rand_price
>>> del hold_price

>>> price.shape
```

The exits array now contains 2,000,000 columns — one per backtest. The column hierarchy is also complete — one tuple of parameters per backtest.

```python
exits
```

```python
>>> exits.columns
MultiIndex([(     'SL', 0.01,   0, 'BTC-USD'),
            (     'SL', 0.01,   0, 'ETH-USD'),
            (     'SL', 0.01,   0, 'XRP-USD'),
            ...
            ('Holding',  1.0, 399, 'XLM-USD'),
            ('Holding',  1.0, 399, 'XMR-USD'),
            ('Holding',  1.0, 399, 'ADA-USD')],
           names=['exit_type', 'stop_value', 'split', 'symbol'], 
           length=2000000)
```

```python
>>> exits.columns
MultiIndex([(     'SL', 0.01,   0, 'BTC-USD'),
            (     'SL', 0.01,   0, 'ETH-USD'),
            (     'SL', 0.01,   0, 'XRP-USD'),
            ...
            ('Holding',  1.0, 399, 'XLM-USD'),
            ('Holding',  1.0, 399, 'XMR-USD'),
            ('Holding',  1.0, 399, 'ADA-USD')],
           names=['exit_type', 'stop_value', 'split', 'symbol'], 
           length=2000000)
```

Warning

One boolean array takes roughly 400 MB of RAM:

```python
>>> print(exits.vbt.getsize())
390.0 MB
```

```python
>>> print(exits.vbt.getsize())
390.0 MB
```

One floating array takes roughly 3 GB of RAM:

```python
>>> print(price.vbt.getsize())
2.9 GB
```

```python
>>> print(price.vbt.getsize())
2.9 GB
```

This allows us to group signals by one or multiple levels and conveniently analyze them in one go. For example, let's compare different exit types and stop values by an average distance of exit signal to entry signal (in days):

```python
>>> avg_distance = entries.vbt.signals.between_ranges(target=exits)\  # (1)!
...     .duration.mean()\
...     .groupby(["exit_type", "stop_value"])\
...     .mean()\
...     .unstack(level="exit_type")

>>> avg_distance.mean()
exit_type
Holding    179.000000
Random      89.432010
SL         124.686960
TP         113.887502
TS         104.159855
dtype: float64

>>> avg_distance[exit_types].vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Avg distance to entry"
... ).show()
```

```python
>>> avg_distance = entries.vbt.signals.between_ranges(target=exits)\  # (1)!
...     .duration.mean()\
...     .groupby(["exit_type", "stop_value"])\
...     .mean()\
...     .unstack(level="exit_type")

>>> avg_distance.mean()
exit_type
Holding    179.000000
Random      89.432010
SL         124.686960
TP         113.887502
TS         104.159855
dtype: float64

>>> avg_distance[exit_types].vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Avg distance to entry"
... ).show()
```

This scatterplot gives us a more detailed view of the distribution of exit signals. As expected, exit signals of plain holding have an exact distance of 179 days after entry (maximum possible), while random exit signals are evenly distributed over the time window and are not dependent upon any stop value. But we are more interested in stop curves, which are flat and thus hint at high volatility of price movements within our timeframe — the lower the curve, the higher is the chance of hitting a stop. To give an example, a TS of 20% is hit after just 30 days on average, while it would take 72 days for SL and 81 days for TP. But does an early exit any good?

## Simulation¶

Here comes the actual backtesting part:

```python
>>> %%time
>>> pf = vbt.Portfolio.from_signals(
...     split_ohlcv["Close"],  # (1)!
...     entries, 
...     exits, 
...     price=price
... )

>>> len(pf.orders)
3995570
CPU times: user 21.2 s, sys: 9.11 s, total: 30.3 s
Wall time: 51.5 s
```

```python
>>> %%time
>>> pf = vbt.Portfolio.from_signals(
...     split_ohlcv["Close"],  # (1)!
...     entries, 
...     exits, 
...     price=price
... )

>>> len(pf.orders)
3995570
CPU times: user 21.2 s, sys: 9.11 s, total: 30.3 s
Wall time: 51.5 s
```

Fairly easy, right?

The simulation took roughly 50 seconds on my Apple M1 to finish and generated in total 3,995,570 orders that are ready to be analyzed (should be 4 million, but some price data points seem to be missing). Notice, however, that any floating array produced by the portfolio object of the same shape as our exit signals, such as portfolio value or returns, requires 8 * 180 * 2000000 bytes or almost 3GB of RAM. We can analyze anything from trades to Sharpe ratio, but given the amount of data, we will stick to a fast-to-calculate metric — total return.

```python
>>> total_return = pf.total_return
>>> del pf  # (1)!

>>> total_return.shape
(2000000,)
```

```python
>>> total_return = pf.total_return
>>> del pf  # (1)!

>>> total_return.shape
(2000000,)
```

If your computer takes a substantial amount of time to simulate, you have several options:

```python
np.float32
```

```python
>>> total_returns = []
>>> for i in vbt.ProgressBar(range(len(exit_types))):  # (1)!
...     exit_type_columns = exits.columns.get_level_values("exit_type")
...     chunk_mask = exit_type_columns == exit_types[i]
...     chunk_pf = vbt.Portfolio.from_signals(
...         split_ohlcv["Close"], 
...         entries, 
...         exits.loc[:, chunk_mask],  # (2)!
...         price=price.loc[:, chunk_mask]
...     )
...     total_returns.append(chunk_pf.total_return)
...     
...     del chunk_pf
...     vbt.flush()  # (3)!

>>> total_return = pd.concat(total_returns)

>>> total_return.shape
```

```python
>>> total_returns = []
>>> for i in vbt.ProgressBar(range(len(exit_types))):  # (1)!
...     exit_type_columns = exits.columns.get_level_values("exit_type")
...     chunk_mask = exit_type_columns == exit_types[i]
...     chunk_pf = vbt.Portfolio.from_signals(
...         split_ohlcv["Close"], 
...         entries, 
...         exits.loc[:, chunk_mask],  # (2)!
...         price=price.loc[:, chunk_mask]
...     )
...     total_returns.append(chunk_pf.total_return)
...     
...     del chunk_pf
...     vbt.flush()  # (3)!

>>> total_return = pd.concat(total_returns)

>>> total_return.shape
```

Chunk 5/5

Chunk 5/5

```python
>>> total_return = pd.concat(total_returns)

>>> total_return.shape
(2000000,)
```

```python
>>> total_return = pd.concat(total_returns)

>>> total_return.shape
(2000000,)
```

This approach has a similar execution time but is much healthier for memory.

## Performance¶

The first step is always taking a look at the distribution of the baseline:

```python
>>> return_by_type = total_return.unstack(level="exit_type")[exit_types]

>>> return_by_type["Holding"].describe(percentiles=[])
count    400000.000000
mean          0.086025
std           0.824799
min          -0.910154
50%          -0.139127
max           6.490103
Name: Holding, dtype: float64

>>> purple_color = vbt.settings["plotting"]["color_schema"]["purple"]
>>> return_by_type["Holding"].vbt.histplot(
...     xaxis_title="Total return",
...     xaxis_tickformat=".2%",
...     yaxis_title="Count",
...     trace_kwargs=dict(marker_color=purple_color)
... ).show()
```

```python
>>> return_by_type = total_return.unstack(level="exit_type")[exit_types]

>>> return_by_type["Holding"].describe(percentiles=[])
count    400000.000000
mean          0.086025
std           0.824799
min          -0.910154
50%          -0.139127
max           6.490103
Name: Holding, dtype: float64

>>> purple_color = vbt.settings["plotting"]["color_schema"]["purple"]
>>> return_by_type["Holding"].vbt.histplot(
...     xaxis_title="Total return",
...     xaxis_tickformat=".2%",
...     yaxis_title="Count",
...     trace_kwargs=dict(marker_color=purple_color)
... ).show()
```

The distribution of holding performance across time windows is highly left-skewed. On the one hand, this indicates prolonged sideways and bearish regimes within our timeframe. On the other hand, the price of any asset can climb to infinity but is limited by 0 — making the distribution denser on the left and more sparse on the right by nature. Every second return is a loss of more than 6%, but thanks to bull runs the strategy still manages to post an average profit of 9%.

Let's include other strategies into the analysis:

```python
>>> pd.DataFrame({
...     "Mean": return_by_type.mean(),
...     "Median": return_by_type.median(),
...     "Std": return_by_type.std(),
... })
               Mean    Median       Std
exit_type                              
SL         0.054361 -0.158458  0.764171
TS         0.057613 -0.093185  0.692137
TP         0.036844  0.077450  0.465555
Random     0.025284 -0.073612  0.575394
Holding    0.086025 -0.139127  0.824799

>>> return_by_type.vbt.boxplot(
...     trace_kwargs=dict(boxpoints=False),  # (1)!
...     yaxis_title="Total return",
...     yaxis_tickformat=".2%"
... ).show()
```

```python
>>> pd.DataFrame({
...     "Mean": return_by_type.mean(),
...     "Median": return_by_type.median(),
...     "Std": return_by_type.std(),
... })
               Mean    Median       Std
exit_type                              
SL         0.054361 -0.158458  0.764171
TS         0.057613 -0.093185  0.692137
TP         0.036844  0.077450  0.465555
Random     0.025284 -0.073612  0.575394
Holding    0.086025 -0.139127  0.824799

>>> return_by_type.vbt.boxplot(
...     trace_kwargs=dict(boxpoints=False),  # (1)!
...     yaxis_title="Total return",
...     yaxis_tickformat=".2%"
... ).show()
```

None of the strategies beat the average return of the baseline. The TP strategy is the most consistent one though — although it introduces an upper bound that limits huge profits (see missing outliers), its trade returns are less volatile and mostly positive. The reason why SL and TS are unbounded at the top is that some stops haven't been hit, and so their columns fall back to plain holding. The random strategy is also interesting: while it's inferior in terms of average return, it finishes second after TP in terms of median return and returns volatility.

To confirm the picture above, let's calculate the win rate of each strategy:

```python
>>> (return_by_type > 0).mean().rename("win_rate")
exit_type
SL         0.305702
TS         0.364098
TP         0.585405
Random     0.405520
Holding    0.402250
Name: win_rate, dtype: float64
```

```python
>>> (return_by_type > 0).mean().rename("win_rate")
exit_type
SL         0.305702
TS         0.364098
TP         0.585405
Random     0.405520
Holding    0.402250
Name: win_rate, dtype: float64
```

Almost 60% of trades with TP are profitable — a high contrast to other strategies. But having a high win ratio doesn't necessarily guarantee longer-term trading success if your winning trades are often much smaller than your losing trades. Thus, let's aggregate by stop type and value and compute the expectancy:

```python
>>> init_cash = vbt.settings.portfolio["init_cash"]

>>> def get_expectancy(return_by_type, level_name):
...     grouped = return_by_type.groupby(level_name, axis=0)
...     win_rate = grouped.apply(lambda x: (x > 0).mean())
...     avg_win = grouped.apply(lambda x: init_cash * x[x > 0].mean())
...     avg_win = avg_win.fillna(0)
...     avg_loss = grouped.apply(lambda x: init_cash * x[x < 0].mean())
...     avg_loss = avg_loss.fillna(0)
...     return win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)

>>> expectancy_by_stop = get_expectancy(return_by_type, "stop_value")

>>> expectancy_by_stop.mean()
exit_type
SL         5.436087
TS         5.761281
TP         3.684371
Random     2.362847
Holding    8.602499
dtype: float64

>>> expectancy_by_stop.vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Expectancy"
... ).show()
```

```python
>>> init_cash = vbt.settings.portfolio["init_cash"]

>>> def get_expectancy(return_by_type, level_name):
...     grouped = return_by_type.groupby(level_name, axis=0)
...     win_rate = grouped.apply(lambda x: (x > 0).mean())
...     avg_win = grouped.apply(lambda x: init_cash * x[x > 0].mean())
...     avg_win = avg_win.fillna(0)
...     avg_loss = grouped.apply(lambda x: init_cash * x[x < 0].mean())
...     avg_loss = avg_loss.fillna(0)
...     return win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)

>>> expectancy_by_stop = get_expectancy(return_by_type, "stop_value")

>>> expectancy_by_stop.mean()
exit_type
SL         5.436087
TS         5.761281
TP         3.684371
Random     2.362847
Holding    8.602499
dtype: float64

>>> expectancy_by_stop.vbt.plot(
...     xaxis_title="Stop value", 
...     yaxis_title="Expectancy"
... ).show()
```

Each strategy is able to add gradually to our account in the long run, with the holding strategy being the clear winner here — we can expect to add to our account an average of almost $9 out of $100 invested after every 6 months of holding. The only configuration that beats the baseline is TS with stop values ranging from 20% to 40%. The worst-performing configuration is SL and TS with stop values around 45% and 60% respectively; both seem to get triggered once most corrections find the bottom, which is even worse than exiting randomly. The TP strategy, on the other hand, beats the random exit strategy after the stop value of 30%. Generally, waiting seems to pay off for cryptocurrencies.

Finally, let’s take a look at how our strategies perform under different market conditions. We will consider a simplified form of regime classification that divides holding returns into 20 bins and calculates the expectancy of each strategy within the boundaries of each bin (we leave out the latest bin for the sake of chart readability). Note that due to the highly skewed distribution of holding returns, we need to take into account the density of observations and make bins equally-sized.

```python
>>> return_values = np.sort(return_by_type["Holding"].values)
>>> idxs = np.ceil(np.linspace(0, len(return_values) - 1, 21)).astype(int)
>>> bins = return_values[idxs][:-1]

>>> def bin_return(return_by_type):
...     classes = pd.cut(return_by_type["Holding"], bins=bins, right=True)
...     new_level = np.array(classes.apply(lambda x: x.right))
...     new_level = pd.Index(new_level, name="bin_right")
...     return return_by_type.vbt.add_levels(new_level, axis=0)

>>> binned_return_by_type = bin_return(return_by_type)

>>> expectancy_by_bin = get_expectancy(binned_return_by_type, "bin_right")

>>> expectancy_by_bin.vbt.plot(
...     trace_kwargs=dict(mode="lines"),
...     xaxis_title="Total return of holding",
...     xaxis_tickformat=".2%",
...     yaxis_title="Expectancy"
... ).show()
```

```python
>>> return_values = np.sort(return_by_type["Holding"].values)
>>> idxs = np.ceil(np.linspace(0, len(return_values) - 1, 21)).astype(int)
>>> bins = return_values[idxs][:-1]

>>> def bin_return(return_by_type):
...     classes = pd.cut(return_by_type["Holding"], bins=bins, right=True)
...     new_level = np.array(classes.apply(lambda x: x.right))
...     new_level = pd.Index(new_level, name="bin_right")
...     return return_by_type.vbt.add_levels(new_level, axis=0)

>>> binned_return_by_type = bin_return(return_by_type)

>>> expectancy_by_bin = get_expectancy(binned_return_by_type, "bin_right")

>>> expectancy_by_bin.vbt.plot(
...     trace_kwargs=dict(mode="lines"),
...     xaxis_title="Total return of holding",
...     xaxis_tickformat=".2%",
...     yaxis_title="Expectancy"
... ).show()
```

The chart above confirms the general intuition behind the behavior of stop orders: SL and TS limit the trader’s loss during downtrends, TP is beneficial for short-term traders interested in profiting from a quick bump in price, and holding performs best in top-growth markets. Surprisingly, while random exits perform poorly in sideways and bull markets, they match and often outperform stop exits in bear markets.

## Bonus: Dashboard¶

Dashboards can be a really powerful way of interacting with the data.

First, let’s define the components of our dashboard. We have two types of components: controls, such as asset dropdown, and graphs. Controls define parameters and trigger updates for graphs.

```python
>>> range_starts = pd.DatetimeIndex(list(map(lambda x: x[0], split_indexes)))
>>> range_ends = pd.DatetimeIndex(list(map(lambda x: x[-1], split_indexes)))

>>> symbol_lvl = return_by_type.index.get_level_values("symbol")
>>> split_lvl = return_by_type.index.get_level_values("split")
>>> range_start_lvl = range_starts[split_lvl]
>>> range_end_lvl = range_ends[split_lvl]

>>> asset_multi_select = ipywidgets.SelectMultiple(
...     options=symbols,
...     value=symbols,
...     rows=len(symbols),
...     description="Symbols"
... )
>>> dates = np.unique(yfdata.wrapper.index)
>>> date_range_slider = ipywidgets.SelectionRangeSlider(
...     options=dates,
...     index=(0, len(dates)-1),
...     orientation="horizontal",
...     readout=False,
...     continuous_update=False
... )
>>> range_start_label = ipywidgets.Label()
>>> range_end_label = ipywidgets.Label()
>>> metric_dropdown = ipywidgets.Dropdown(
...     options=["Mean", "Median", "Win Rate", "Expectancy"],
...     value="Expectancy"
... )
>>> stop_scatter = vbt.Scatter(
...     trace_names=exit_types,
...     x_labels=stops, 
...     xaxis_title="Stop value", 
...     yaxis_title="Expectancy"
... )
>>> stop_scatter_img = ipywidgets.Image(
...     format="png",
...     width=stop_scatter.fig.layout.width,
...     height=stop_scatter.fig.layout.height
... )
>>> bin_scatter = vbt.Scatter(
...     trace_names=exit_types,
...     x_labels=expectancy_by_bin.index, 
...     trace_kwargs=dict(mode="lines"),
...     xaxis_title="Total return of holding",
...     xaxis_tickformat="%",
...     yaxis_title="Expectancy"
... )
>>> bin_scatter_img = ipywidgets.Image(
...     format="png",
...     width=bin_scatter.fig.layout.width,
...     height=bin_scatter.fig.layout.height
... )
```

```python
>>> range_starts = pd.DatetimeIndex(list(map(lambda x: x[0], split_indexes)))
>>> range_ends = pd.DatetimeIndex(list(map(lambda x: x[-1], split_indexes)))

>>> symbol_lvl = return_by_type.index.get_level_values("symbol")
>>> split_lvl = return_by_type.index.get_level_values("split")
>>> range_start_lvl = range_starts[split_lvl]
>>> range_end_lvl = range_ends[split_lvl]

>>> asset_multi_select = ipywidgets.SelectMultiple(
...     options=symbols,
...     value=symbols,
...     rows=len(symbols),
...     description="Symbols"
... )
>>> dates = np.unique(yfdata.wrapper.index)
>>> date_range_slider = ipywidgets.SelectionRangeSlider(
...     options=dates,
...     index=(0, len(dates)-1),
...     orientation="horizontal",
...     readout=False,
...     continuous_update=False
... )
>>> range_start_label = ipywidgets.Label()
>>> range_end_label = ipywidgets.Label()
>>> metric_dropdown = ipywidgets.Dropdown(
...     options=["Mean", "Median", "Win Rate", "Expectancy"],
...     value="Expectancy"
... )
>>> stop_scatter = vbt.Scatter(
...     trace_names=exit_types,
...     x_labels=stops, 
...     xaxis_title="Stop value", 
...     yaxis_title="Expectancy"
... )
>>> stop_scatter_img = ipywidgets.Image(
...     format="png",
...     width=stop_scatter.fig.layout.width,
...     height=stop_scatter.fig.layout.height
... )
>>> bin_scatter = vbt.Scatter(
...     trace_names=exit_types,
...     x_labels=expectancy_by_bin.index, 
...     trace_kwargs=dict(mode="lines"),
...     xaxis_title="Total return of holding",
...     xaxis_tickformat="%",
...     yaxis_title="Expectancy"
... )
>>> bin_scatter_img = ipywidgets.Image(
...     format="png",
...     width=bin_scatter.fig.layout.width,
...     height=bin_scatter.fig.layout.height
... )
```

The second step is the definition of the update function, which is triggered once any control has been changed. We also manually call this function to initialize the graphs with default parameters.

```python
>>> def update_scatter(*args, **kwargs):
...     _symbols = asset_multi_select.value
...     _from = date_range_slider.value[0]
...     _to = date_range_slider.value[1]
...     _metric_name = metric_dropdown.value
...     
...     range_mask = (range_start_lvl >= _from) & (range_end_lvl <= _to)
...     asset_mask = symbol_lvl.isin(_symbols)
...     filt = return_by_type[range_mask & asset_mask]
...     
...     filt_binned = bin_return(filt)
...     if _metric_name == "Mean":
...         filt_metric = filt.groupby("stop_value").mean()
...         filt_bin_metric = filt_binned.groupby("bin_right").mean()
...     elif _metric_name == "Median":
...         filt_metric = filt.groupby("stop_value").median()
...         filt_bin_metric = filt_binned.groupby("bin_right").median()
...     elif _metric_name == "Win Rate":
...         filt_metric = (filt > 0).groupby("stop_value").mean()
...         filt_bin_metric = (filt_binned > 0).groupby("bin_right").mean()
...     elif _metric_name == "Expectancy":
...         filt_metric = get_expectancy(filt, "stop_value")
...         filt_bin_metric = get_expectancy(filt_binned, "bin_right")
...         
...     stop_scatter.fig.update_layout(yaxis_title=_metric_name)
...     stop_scatter.update(filt_metric)
...     stop_scatter_img.value = stop_scatter.fig.to_image(format="png")
...     
...     bin_scatter.fig.update_layout(yaxis_title=_metric_name)
...     bin_scatter.update(filt_bin_metric)
...     bin_scatter_img.value = bin_scatter.fig.to_image(format="png")
...     
...     range_start_label.value = np.datetime_as_string(
...         _from.to_datetime64(), unit="D")
...     range_end_label.value = np.datetime_as_string(
...         _to.to_datetime64(), unit="D")

>>> asset_multi_select.observe(update_scatter, names="value")
>>> date_range_slider.observe(update_scatter, names="value")
>>> metric_dropdown.observe(update_scatter, names="value")
>>> update_scatter()
```

```python
>>> def update_scatter(*args, **kwargs):
...     _symbols = asset_multi_select.value
...     _from = date_range_slider.value[0]
...     _to = date_range_slider.value[1]
...     _metric_name = metric_dropdown.value
...     
...     range_mask = (range_start_lvl >= _from) & (range_end_lvl <= _to)
...     asset_mask = symbol_lvl.isin(_symbols)
...     filt = return_by_type[range_mask & asset_mask]
...     
...     filt_binned = bin_return(filt)
...     if _metric_name == "Mean":
...         filt_metric = filt.groupby("stop_value").mean()
...         filt_bin_metric = filt_binned.groupby("bin_right").mean()
...     elif _metric_name == "Median":
...         filt_metric = filt.groupby("stop_value").median()
...         filt_bin_metric = filt_binned.groupby("bin_right").median()
...     elif _metric_name == "Win Rate":
...         filt_metric = (filt > 0).groupby("stop_value").mean()
...         filt_bin_metric = (filt_binned > 0).groupby("bin_right").mean()
...     elif _metric_name == "Expectancy":
...         filt_metric = get_expectancy(filt, "stop_value")
...         filt_bin_metric = get_expectancy(filt_binned, "bin_right")
...         
...     stop_scatter.fig.update_layout(yaxis_title=_metric_name)
...     stop_scatter.update(filt_metric)
...     stop_scatter_img.value = stop_scatter.fig.to_image(format="png")
...     
...     bin_scatter.fig.update_layout(yaxis_title=_metric_name)
...     bin_scatter.update(filt_bin_metric)
...     bin_scatter_img.value = bin_scatter.fig.to_image(format="png")
...     
...     range_start_label.value = np.datetime_as_string(
...         _from.to_datetime64(), unit="D")
...     range_end_label.value = np.datetime_as_string(
...         _to.to_datetime64(), unit="D")

>>> asset_multi_select.observe(update_scatter, names="value")
>>> date_range_slider.observe(update_scatter, names="value")
>>> metric_dropdown.observe(update_scatter, names="value")
>>> update_scatter()
```

In the last step, we will define the layout of the dashboard and finally run it:

```python
>>> dashboard = ipywidgets.VBox([
...     asset_multi_select,
...     ipywidgets.HBox([
...         range_start_label,
...         date_range_slider,
...         range_end_label
...     ]),
...     metric_dropdown,
...     stop_scatter_img,
...     bin_scatter_img
... ])
>>> dashboard
```

```python
>>> dashboard = ipywidgets.VBox([
...     asset_multi_select,
...     ipywidgets.HBox([
...         range_start_label,
...         date_range_slider,
...         range_end_label
...     ]),
...     metric_dropdown,
...     stop_scatter_img,
...     bin_scatter_img
... ])
>>> dashboard
```

Dashboard

Run the notebook to view the dashboard!

## Summary¶

The use of large-scale backtesting is not limited to hyperparameter optimization, but when properly utilized, it gives us a vehicle to explore complex phenomena related to trading. Especially utilization of multidimensional arrays, dynamic compilation, and integration with pandas, as done by vectorbt, allows us to quickly get new insights by applying popular data science tools to each component of a backtesting pipeline.

In this particular example, we conducted 2 million tests to observe how different stop values impact the performance of stop signals and how different stop signals compare to holding and trading randomly. On the one hand, the findings confirm what we already know about the behavior of stop signals under various market conditions. On the other hand, they reveal optimal configurations that might have worked well for the last couple of years of trading cryptocurrencies.

Python code  Notebook

