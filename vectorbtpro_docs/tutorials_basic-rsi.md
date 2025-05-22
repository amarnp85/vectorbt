# Basic RSI strategy¶

One of the main powers of vectorbt (PRO) is the ability to create and backtest numerous strategy configurations in the blink of an eye. In this introductory example, we will explore how profitable is the following RSI strategy commonly used by beginners:

If the RSI is less than 30, it indicates a stock is reaching oversold conditions and may see a trend reversal, or bounceback, towards a higher share price. Once the reversal is confirmed, a buy trade is placed. Conversely, if the RSI is more than 70, it indicates that a stock is reaching an overbought condition and may see a trend reversal, or pullback, in price. After a confirmation of the reversal, a sell trade is placed.

As a bonus, we will gradually expand the analysis towards multiple parameter combinations. Sounds fun? Let's start.

## Single backtest¶

First, we will take care of data. Using a one-liner, we will download all available daily data for the pair BTC/USDT from Binance:

```python
>>> from vectorbtpro import *  # (1)!

>>> data = vbt.BinanceData.pull('BTCUSDT')
>>> data
<vectorbtpro.data.custom.binance.BinanceData at 0x7f9c40c59550>
```

```python
>>> from vectorbtpro import *  # (1)!

>>> data = vbt.BinanceData.pull('BTCUSDT')
>>> data
<vectorbtpro.data.custom.binance.BinanceData at 0x7f9c40c59550>
```

```python
whats_imported()
```

100%

100%

The returned object is of type BinanceData, which extends Data to communicate with the Binance API. The class Data is a vectorbt's in-house container for retrieving, storing, and managing data. Upon receiving a DataFrame, it post-processes and stores the DataFrame inside the dictionary Data.data keyed by pair (also referred to as a "symbol" in vectorbt). We can get our DataFrame either from this dictionary, or by using the convenient method Data.get, which also allows for specifying one or more columns instead of returning the entire DataFrame at once.

Let's plot the data with Data.plot:

```python
>>> data.plot().show()  # (1)!
```

```python
>>> data.plot().show()  # (1)!
```

```python
symbol
```

```python
feature
```

Another way to describe the data is by using the Pandas' info method. The tabular format is especially useful for counting null values (which our data apparently doesn't have - good!)

```python
>>> data.data['BTCUSDT'].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1813 entries, 2017-08-17 00:00:00+00:00 to 2022-08-03 00:00:00+00:00
Freq: D
Data columns (total 9 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Open                1813 non-null   float64
 1   High                1813 non-null   float64
 2   Low                 1813 non-null   float64
 3   Close               1813 non-null   float64
 4   Volume              1813 non-null   float64
 5   Quote volume        1813 non-null   float64
 6   Trade count         1813 non-null   int64  
 7   Taker base volume   1813 non-null   float64
 8   Taker quote volume  1813 non-null   float64
dtypes: float64(8), int64(1)
memory usage: 141.6 KB
```

```python
>>> data.data['BTCUSDT'].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1813 entries, 2017-08-17 00:00:00+00:00 to 2022-08-03 00:00:00+00:00
Freq: D
Data columns (total 9 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Open                1813 non-null   float64
 1   High                1813 non-null   float64
 2   Low                 1813 non-null   float64
 3   Close               1813 non-null   float64
 4   Volume              1813 non-null   float64
 5   Quote volume        1813 non-null   float64
 6   Trade count         1813 non-null   int64  
 7   Taker base volume   1813 non-null   float64
 8   Taker quote volume  1813 non-null   float64
dtypes: float64(8), int64(1)
memory usage: 141.6 KB
```

In our example, we will generate signals based on the opening price and execute them based on the closing price. We can also place orders a soon as the signal is generated, or at any later time, but we will illustrate how to separate generation of signals from their execution.

```python
>>> open_price = data.get('Open')
>>> close_price = data.get('Close')  # (1)!
```

```python
>>> open_price = data.get('Open')
>>> close_price = data.get('Close')  # (1)!
```

```python
data.close
```

It's time to run the indicator!

VectorBT® PRO supports 5 (!) different implementations of RSI: one implemented using Numba, and the other four ported from three different technical analysis libraries. Each indicator has been wrapped with the almighty IndicatorFactory

To list all the available indicators or to search for a specific indicator, we can use IndicatorFactory.list_indicators:

```python
>>> vbt.IF.list_indicators("RSI*")
['vbt:RSI', 'talib:RSI', 'pandas_ta:RSI', 'ta:RSIIndicator', 'technical:RSI']
```

```python
>>> vbt.IF.list_indicators("RSI*")
['vbt:RSI', 'talib:RSI', 'pandas_ta:RSI', 'ta:RSIIndicator', 'technical:RSI']
```

We can then retrieve the actual indicator class as follows:

```python
>>> vbt.indicator("talib:RSI")
vectorbtpro.indicators.factory.talib.RSI
```

```python
>>> vbt.indicator("talib:RSI")
vectorbtpro.indicators.factory.talib.RSI
```

Or manually:

```python
>>> vbt.RSI  # (1)!
vectorbtpro.indicators.custom.RSI

>>> vbt.talib('RSI')  # (2)!
vectorbtpro.indicators.factory.talib.RSI

>>> vbt.ta('RSIIndicator')  # (3)!
vectorbtpro.indicators.factory.ta.RSIIndicator

>>> vbt.pandas_ta('RSI')  # (4)!
vectorbtpro.indicators.factory.pandas_ta.RSI

>>> vbt.technical('RSI')  # (5)!
vectorbtpro.indicators.factory.technical.RSI
```

```python
>>> vbt.RSI  # (1)!
vectorbtpro.indicators.custom.RSI

>>> vbt.talib('RSI')  # (2)!
vectorbtpro.indicators.factory.talib.RSI

>>> vbt.ta('RSIIndicator')  # (3)!
vectorbtpro.indicators.factory.ta.RSIIndicator

>>> vbt.pandas_ta('RSI')  # (4)!
vectorbtpro.indicators.factory.pandas_ta.RSI

>>> vbt.technical('RSI')  # (5)!
vectorbtpro.indicators.factory.technical.RSI
```

Here's a rule of thumb on which implementation to choose:

To run any indicator, use the method run. To see what arguments the method accepts, pass it to phelp:

```python
run
```

```python
>>> vbt.phelp(vbt.RSI.run)
RSI.run(
    close,
    window=Default(value=14),
    wtype=Default(value='wilder'),
    short_name='rsi',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `RSI` indicator.

    * Inputs: `close`
    * Parameters: `window`, `wtype`
    * Outputs: `rsi`

    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.

    Other keyword arguments are passed to `RSI.run_pipeline`.
```

```python
>>> vbt.phelp(vbt.RSI.run)
RSI.run(
    close,
    window=Default(value=14),
    wtype=Default(value='wilder'),
    short_name='rsi',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `RSI` indicator.

    * Inputs: `close`
    * Parameters: `window`, `wtype`
    * Outputs: `rsi`

    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.

    Other keyword arguments are passed to `RSI.run_pipeline`.
```

As we can see above, we need to at least provide close, which can be any numeric time series. Also, by default, the rolling window is 14 bars long and uses the Wilder's smoothed moving average. Since we want to make decisions based on the opening price, we will pass open_price as close:

```python
close
```

```python
open_price
```

```python
close
```

```python
>>> rsi = vbt.RSI.run(open_price)
>>> rsi
<vectorbtpro.indicators.custom.RSI at 0x7f9c20921ac8>
```

```python
>>> rsi = vbt.RSI.run(open_price)
>>> rsi
<vectorbtpro.indicators.custom.RSI at 0x7f9c20921ac8>
```

That's all! By executing the method RSI.run, we calculated the RSI values and have received an instance with various methods and properties for their analysis. To retrieve the resulting Pandas object, we need to query the rsi attribute (see "Outputs" in the output of phelp).

```python
rsi
```

```python
phelp
```

```python
>>> rsi.rsi
Open time
2017-08-17 00:00:00+00:00          NaN
2017-08-18 00:00:00+00:00          NaN
2017-08-19 00:00:00+00:00          NaN
2017-08-20 00:00:00+00:00          NaN
2017-08-21 00:00:00+00:00          NaN
...                                ...
2022-07-30 00:00:00+00:00    60.541637
2022-07-31 00:00:00+00:00    59.503179
2022-08-01 00:00:00+00:00    56.750576
2022-08-02 00:00:00+00:00    56.512434
2022-08-03 00:00:00+00:00    54.177385
Freq: D, Name: Open, Length: 1813, dtype: float64
```

```python
>>> rsi.rsi
Open time
2017-08-17 00:00:00+00:00          NaN
2017-08-18 00:00:00+00:00          NaN
2017-08-19 00:00:00+00:00          NaN
2017-08-20 00:00:00+00:00          NaN
2017-08-21 00:00:00+00:00          NaN
...                                ...
2022-07-30 00:00:00+00:00    60.541637
2022-07-31 00:00:00+00:00    59.503179
2022-08-01 00:00:00+00:00    56.750576
2022-08-02 00:00:00+00:00    56.512434
2022-08-03 00:00:00+00:00    54.177385
Freq: D, Name: Open, Length: 1813, dtype: float64
```

Having the RSI array, we now want to generate an entry signal whenever any RSI value crosses below 30 and an exit signal whenever any RSI value crosses above 70:

```python
>>> entries = rsi.rsi.vbt.crossed_below(30)  # (1)!
>>> entries
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool

>>> exits = rsi.rsi.vbt.crossed_above(70)  # (2)!
>>> exits
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool
```

```python
>>> entries = rsi.rsi.vbt.crossed_below(30)  # (1)!
>>> entries
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool

>>> exits = rsi.rsi.vbt.crossed_above(70)  # (2)!
>>> exits
Open time
2017-08-17 00:00:00+00:00    False
2017-08-18 00:00:00+00:00    False
2017-08-19 00:00:00+00:00    False
2017-08-20 00:00:00+00:00    False
2017-08-21 00:00:00+00:00    False
...                            ...
2022-07-30 00:00:00+00:00    False
2022-07-31 00:00:00+00:00    False
2022-08-01 00:00:00+00:00    False
2022-08-02 00:00:00+00:00    False
2022-08-03 00:00:00+00:00    False
Freq: D, Name: Open, Length: 1813, dtype: bool
```

The same can be done using the methods RSI.rsi_crossed_below and RSI.rsi_crossed_above that were auto-generated for the output rsi by IndicatorFactory:

```python
rsi
```

```python
>>> entries = rsi.rsi_crossed_below(30)
>>> exits = rsi.rsi_crossed_above(70)
```

```python
>>> entries = rsi.rsi_crossed_below(30)
>>> exits = rsi.rsi_crossed_above(70)
```

Hint

If you are curious what else has been generated, print dir(rsi) or look into the API generated for the class.

```python
dir(rsi)
```

Before we proceed with the portfolio modeling, let's plot the RSI and signals to ensure that we did everything right:

```python
>>> def plot_rsi(rsi, entries, exits):
...     fig = rsi.plot()  # (1)!
...     entries.vbt.signals.plot_as_entries(rsi.rsi, fig=fig)  # (2)!
...     exits.vbt.signals.plot_as_exits(rsi.rsi, fig=fig)  # (3)!
...     return fig

>>> plot_rsi(rsi, entries, exits).show()
```

```python
>>> def plot_rsi(rsi, entries, exits):
...     fig = rsi.plot()  # (1)!
...     entries.vbt.signals.plot_as_entries(rsi.rsi, fig=fig)  # (2)!
...     exits.vbt.signals.plot_as_exits(rsi.rsi, fig=fig)  # (3)!
...     return fig

>>> plot_rsi(rsi, entries, exits).show()
```

The graph looks legit. But notice how there are multiple entries between two exits and vice versa? How does vectorbt handle it? When using Portfolio.from_signals, vectorbt will automatically filter out all entry signals if the position has already been entered, and exit signals if the position has already been exited. But to make our analysis cleaner, let's keep each first signal:

```python
>>> clean_entries, clean_exits = entries.vbt.signals.clean(exits)  # (1)!

>>> plot_rsi(rsi, clean_entries, clean_exits).show()
```

```python
>>> clean_entries, clean_exits = entries.vbt.signals.clean(exits)  # (1)!

>>> plot_rsi(rsi, clean_entries, clean_exits).show()
```

We can immediately see the difference. But what other methods exist to analyze the distribution of signals? How to quantify such analysis? That's what vectorbt is all about. Let's compute various statistics of clean_entries and clean_exits using SignalsAccessor:

```python
clean_entries
```

```python
clean_exits
```

```python
>>> clean_entries.vbt.signals.total()  # (1)!
8

>>> clean_exits.vbt.signals.total()  # (2)!
7

>>> ranges = clean_entries.vbt.signals.between_ranges(target=clean_exits)  # (3)!
>>> ranges.duration.mean(wrap_kwargs=dict(to_timedelta=True))  # (4)!
Timedelta('86 days 10:17:08.571428572')
```

```python
>>> clean_entries.vbt.signals.total()  # (1)!
8

>>> clean_exits.vbt.signals.total()  # (2)!
7

>>> ranges = clean_entries.vbt.signals.between_ranges(target=clean_exits)  # (3)!
>>> ranges.duration.mean(wrap_kwargs=dict(to_timedelta=True))  # (4)!
Timedelta('86 days 10:17:08.571428572')
```

We are ready for modeling! We will be using the class method Portfolio.from_signals, which will receive the signal arrays, process each signal one by one, and generate orders. It will then create an instance of Portfolio that can be used to assess the performance of the strategy.

Our experiment is simple: buy $100 of Bitcoin upon an entry signal and close the position upon an exit signal. Start with an infinite capital to not limit our buying power at any time.

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=clean_entries, 
...     exits=clean_exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c40eea438>
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=clean_entries, 
...     exits=clean_exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c40eea438>
```

Info

Running the method above for the first time may take some time as it must be compiled first. Compilation will take place each time a new combination of data types is discovered. But don't worry: Numba caches most compiled functions and re-uses them in each new runtime.

Hint

If you look into the API of Portfolio.from_signals, you will find many arguments to be set to None. The value None has a special meaning that instructs vectorbt to pull the default value from the global settings. You can discover all the default values for the Portfolio class here.

```python
None
```

```python
Portfolio
```

Let's print the statistics of our portfolio:

```python
>>> pf.stats()
Start                         2017-08-17 00:00:00+00:00
End                           2022-08-03 00:00:00+00:00
Period                               1813 days 00:00:00
Start Value                                       100.0
Min Value                                     97.185676
Max Value                                    203.182943
End Value                                    171.335425
Total Return [%]                              71.335425
Benchmark Return [%]                         446.481746
Total Time Exposure [%]                       38.113624
Max Gross Exposure [%]                            100.0
Max Drawdown [%]                              46.385941
Max Drawdown Duration                1613 days 00:00:00
Total Orders                                         15
Total Fees Paid                                     0.0
Total Trades                                          8
Win Rate [%]                                  71.428571
Best Trade [%]                                54.519055
Worst Trade [%]                              -32.078597
Avg Winning Trade [%]                         26.905709
Avg Losing Trade [%]                         -19.345383
Avg Winning Trade Duration             87 days 09:36:00
Avg Losing Trade Duration              84 days 00:00:00
Profit Factor                                  3.477019
Expectancy                                    13.691111
Sharpe Ratio                                   0.505486
Calmar Ratio                                   0.246836
Omega Ratio                                    1.132505
Sortino Ratio                                  0.796701
dtype: object
```

```python
>>> pf.stats()
Start                         2017-08-17 00:00:00+00:00
End                           2022-08-03 00:00:00+00:00
Period                               1813 days 00:00:00
Start Value                                       100.0
Min Value                                     97.185676
Max Value                                    203.182943
End Value                                    171.335425
Total Return [%]                              71.335425
Benchmark Return [%]                         446.481746
Total Time Exposure [%]                       38.113624
Max Gross Exposure [%]                            100.0
Max Drawdown [%]                              46.385941
Max Drawdown Duration                1613 days 00:00:00
Total Orders                                         15
Total Fees Paid                                     0.0
Total Trades                                          8
Win Rate [%]                                  71.428571
Best Trade [%]                                54.519055
Worst Trade [%]                              -32.078597
Avg Winning Trade [%]                         26.905709
Avg Losing Trade [%]                         -19.345383
Avg Winning Trade Duration             87 days 09:36:00
Avg Losing Trade Duration              84 days 00:00:00
Profit Factor                                  3.477019
Expectancy                                    13.691111
Sharpe Ratio                                   0.505486
Calmar Ratio                                   0.246836
Omega Ratio                                    1.132505
Sortino Ratio                                  0.796701
dtype: object
```

Hint

That are lots of statistics, right? If you're looking for the way they are implemented, print pf.metrics and look for the calc_func argument of the metric of interest. If some function is a lambda, look into the source code to reveal its contents.

```python
pf.metrics
```

```python
calc_func
```

Our strategy is not too bad: the portfolio has gained over 71% in profit over the last years, but holding Bitcoin is still better - staggering 450%. Despite the Bitcoin's high volatility, the minimum recorded portfolio value sits at $97 from $100 initially invested. The total time exposure of 38% means that we were in the market 38% of the time. The maximum gross exposure of 100% means that we invested 100% of our available cash balance, each single trade. The maximum drawdown (MDD) of 46% is the maximum distance our portfolio value fell after recording a new high (stop loss to the rescue?).

The total number of orders matches the total number of (cleaned) signals, but why is the total number of trades suddenly 8 instead of 15? By default, a trade in the vectorbt's universe is a sell order; as soon as an exit order has been filled (by reducing or closing the current position), the profit and loss (PnL) based on the weighted average entry and exit price is calculated. The win rate of 70% means that 70% of the trades (sell orders) generated a profit, with the best trade bringing 54% in profit and the worst one bringing 32% in loss. Since the average winning trade generating more profit than the average losing trade generating loss, we can see various metrics being positive, such as the profit factor and the expectancy.

```python
>>> pf.plot(settings=dict(bm_returns=False)).show()
```

```python
>>> pf.plot(settings=dict(bm_returns=False)).show()
```

Hint

A benefit of an interactive plot like above is that you can use tools from the Plotly toolbar to draw a vertical line that connects orders, their P&L, and how they affect the cumulative returns. Try it out!

So, how do we improve from here?

## Multiple backtests¶

### Using for-loop¶

Even such a basic strategy as ours has many potential parameters:

```python
lower_th
```

```python
upper_th
```

```python
window
```

```python
ewm
```

To make our analysis as flexible as possible, we will write a function that lets us specify all of that information, and return a subset of statistics:

```python
>>> def test_rsi(window=14, wtype="wilder", lower_th=30, upper_th=70):
...     rsi = vbt.RSI.run(open_price, window=window, wtype=wtype)
...     entries = rsi.rsi_crossed_below(lower_th)
...     exits = rsi.rsi_crossed_above(upper_th)
...     pf = vbt.Portfolio.from_signals(
...         close=close_price, 
...         entries=entries, 
...         exits=exits,
...         size=100,
...         size_type='value',
...         init_cash='auto')
...     return pf.stats([
...         'total_return', 
...         'total_trades', 
...         'win_rate', 
...         'expectancy'
...     ])

>>> test_rsi()
Total Return [%]    71.335425
Total Trades                8
Win Rate [%]        71.428571
Expectancy          13.691111
dtype: object

>>> test_rsi(lower_th=20, upper_th=80)
Total Return [%]    6.652287
Total Trades               2
Win Rate [%]            50.0
Expectancy          3.737274
dtype: object
```

```python
>>> def test_rsi(window=14, wtype="wilder", lower_th=30, upper_th=70):
...     rsi = vbt.RSI.run(open_price, window=window, wtype=wtype)
...     entries = rsi.rsi_crossed_below(lower_th)
...     exits = rsi.rsi_crossed_above(upper_th)
...     pf = vbt.Portfolio.from_signals(
...         close=close_price, 
...         entries=entries, 
...         exits=exits,
...         size=100,
...         size_type='value',
...         init_cash='auto')
...     return pf.stats([
...         'total_return', 
...         'total_trades', 
...         'win_rate', 
...         'expectancy'
...     ])

>>> test_rsi()
Total Return [%]    71.335425
Total Trades                8
Win Rate [%]        71.428571
Expectancy          13.691111
dtype: object

>>> test_rsi(lower_th=20, upper_th=80)
Total Return [%]    6.652287
Total Trades               2
Win Rate [%]            50.0
Expectancy          3.737274
dtype: object
```

Note

We removed the signal cleaning step because it makes no difference when signals are passed to Portfolio.from_signals (which cleans the signals automatically anyway).

By raising the upper threshold to 80% and lowering the lower threshold to 20%, the number of trades has decreased to just 2 because it becomes more difficult to cross the thresholds. We can also observe how the total return fell to roughly 7% - not a good sign. But how do we actually know whether this negative result indicates that our strategy is trash and not because of a pure luck? Testing one parameter combination from a huge space usually means making a wild guess.

Let's generate multiple parameter combinations for thresholds, simulate them, and concatenate their statistics for further analysis:

```python
>>> lower_ths = range(20, 31)  # (1)!
>>> upper_ths = range(70, 81)  # (2)!
>>> th_combs = list(product(lower_ths, upper_ths))  # (3)!
>>> len(th_combs)
121

>>> comb_stats = [
...     test_rsi(lower_th=lower_th, upper_th=upper_th)
...     for lower_th, upper_th in th_combs
... ]  # (4)!
```

```python
>>> lower_ths = range(20, 31)  # (1)!
>>> upper_ths = range(70, 81)  # (2)!
>>> th_combs = list(product(lower_ths, upper_ths))  # (3)!
>>> len(th_combs)
121

>>> comb_stats = [
...     test_rsi(lower_th=lower_th, upper_th=upper_th)
...     for lower_th, upper_th in th_combs
... ]  # (4)!
```

```python
lower_ths
```

```python
upper_ths
```

We just simulated 121 different combinations of the upper and lower threshold and stored their statistics inside a list. In order to analyze this list, we need to convert it to a DataFrame first, with metrics arranged as columns:

```python
>>> comb_stats_df = pd.DataFrame(comb_stats)
>>> comb_stats_df
     Total Return [%]  Total Trades  Win Rate [%]  Expectancy
0           24.369550             3     66.666667   10.606342
1           37.380341             3     66.666667   16.203667
2           34.560194             3     66.666667   14.981187
3           31.090080             3     66.666667   13.833710
4           31.090080             3     66.666667   13.833710
..                ...           ...           ...         ...
116         51.074571             6     80.000000   18.978193
117         62.853840             6     80.000000   21.334047
118         40.685579             5     75.000000   21.125494
119         -5.990835             4     66.666667   13.119897
120        -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

```python
>>> comb_stats_df = pd.DataFrame(comb_stats)
>>> comb_stats_df
     Total Return [%]  Total Trades  Win Rate [%]  Expectancy
0           24.369550             3     66.666667   10.606342
1           37.380341             3     66.666667   16.203667
2           34.560194             3     66.666667   14.981187
3           31.090080             3     66.666667   13.833710
4           31.090080             3     66.666667   13.833710
..                ...           ...           ...         ...
116         51.074571             6     80.000000   18.978193
117         62.853840             6     80.000000   21.334047
118         40.685579             5     75.000000   21.125494
119         -5.990835             4     66.666667   13.119897
120        -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

But how do we know which row corresponds to which parameter combination? We will build a MultiIndex with two levels, lower_th and upper_th, and make it the index of comb_stats_df:

```python
lower_th
```

```python
upper_th
```

```python
comb_stats_df
```

```python
>>> comb_stats_df.index = pd.MultiIndex.from_tuples(
...     th_combs, 
...     names=['lower_th', 'upper_th'])
>>> comb_stats_df
                   Total Return [%]  Total Trades  Win Rate [%]  Expectancy
lower_th upper_th                                                          
20       70               24.369550             3     66.666667   10.606342
         71               37.380341             3     66.666667   16.203667
         72               34.560194             3     66.666667   14.981187
         73               31.090080             3     66.666667   13.833710
         74               31.090080             3     66.666667   13.833710
...                             ...           ...           ...         ...
30       76               51.074571             6     80.000000   18.978193
         77               62.853840             6     80.000000   21.334047
         78               40.685579             5     75.000000   21.125494
         79               -5.990835             4     66.666667   13.119897
         80              -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

```python
>>> comb_stats_df.index = pd.MultiIndex.from_tuples(
...     th_combs, 
...     names=['lower_th', 'upper_th'])
>>> comb_stats_df
                   Total Return [%]  Total Trades  Win Rate [%]  Expectancy
lower_th upper_th                                                          
20       70               24.369550             3     66.666667   10.606342
         71               37.380341             3     66.666667   16.203667
         72               34.560194             3     66.666667   14.981187
         73               31.090080             3     66.666667   13.833710
         74               31.090080             3     66.666667   13.833710
...                             ...           ...           ...         ...
30       76               51.074571             6     80.000000   18.978193
         77               62.853840             6     80.000000   21.334047
         78               40.685579             5     75.000000   21.125494
         79               -5.990835             4     66.666667   13.119897
         80              -10.315159             4     66.666667   11.678455

[121 rows x 4 columns]
```

Much better! We can now analyze every piece of the retrieved information from different angles. Since we have the same number of lower and upper thresholds, let's create a heatmap with the X axis reflecting the lower thresholds, the Y axis reflecting the upper thresholds, and the color bar reflecting the expectancy:

```python
>>> comb_stats_df['Expectancy'].vbt.heatmap().show()
```

```python
>>> comb_stats_df['Expectancy'].vbt.heatmap().show()
```

We can explore entire regions of parameter combinations that yield positive or negative results.

### Using columns¶

As you might have read in the documentation, vectorbt loves processing multidimensional data. In particular, it's built around the idea that you can represent each asset, period, parameter combination, and a backtest in general, as a column in a two-dimensional array.

Instead of computing everything in a loop (which isn't too bad but usually executes magnitudes slower than a vectorized solution) we can change our code to accept parameters as arrays. A function that takes such array will automatically convert multiple parameters into multiple columns. A big benefit of this approach is that we don't have to collect our results, put them in a list, and convert into a DataFrame - it's all done by vectorbt!

First, define the parameters that we would like to test:

```python
>>> windows = list(range(8, 21))
>>> wtypes = ["simple", "exp", "wilder"]
>>> lower_ths = list(range(20, 31))
>>> upper_ths = list(range(70, 81))
```

```python
>>> windows = list(range(8, 21))
>>> wtypes = ["simple", "exp", "wilder"]
>>> lower_ths = list(range(20, 31))
>>> upper_ths = list(range(70, 81))
```

Instead of applying itertools.product, we will instruct various parts of our pipeline to build a product instead, so we can observe how each part affects the column hierarchy.

```python
itertools.product
```

The RSI part is easy: we can pass param_product=True to build a product of windows and wtypes and run the calculation over each column in open_price:

```python
param_product=True
```

```python
windows
```

```python
wtypes
```

```python
open_price
```

```python
>>> rsi = vbt.RSI.run(
...     open_price, 
...     window=windows, 
...     wtype=wtypes, 
...     param_product=True)
>>> rsi.rsi.columns
MultiIndex([( 8, 'simple'),
            ( 8,    'exp'),
            ( 8, 'wilder'),
            ...
            (20, 'simple'),
            (20,    'exp'),
            (20, 'wilder')],
           names=['rsi_window', 'rsi_wtype'])
```

```python
>>> rsi = vbt.RSI.run(
...     open_price, 
...     window=windows, 
...     wtype=wtypes, 
...     param_product=True)
>>> rsi.rsi.columns
MultiIndex([( 8, 'simple'),
            ( 8,    'exp'),
            ( 8, 'wilder'),
            ...
            (20, 'simple'),
            (20,    'exp'),
            (20, 'wilder')],
           names=['rsi_window', 'rsi_wtype'])
```

We see that RSI appended two levels to the column hierarchy: rsi_window and rsi_wtype. Those are similar to the ones we created manually for thresholds in Using for-loop. There are now 39 columns in total, which is just len(open_price.columns) x len(windows) x len(wtypes).

```python
rsi_window
```

```python
rsi_wtype
```

```python
len(open_price.columns)
```

```python
len(windows)
```

```python
len(wtypes)
```

The next part are crossovers. In contrast to indicators, they are regular functions that take any array-like object, broadcast it to the rsi array, and search for crossovers. The broadcasting step is done using broadcast, which is a very powerful function for bringing multiple arrays to a single shape (learn more about broadcasting in the documentation).

```python
rsi
```

In our case, we want to build a product of lower_ths, upper_th_index, and all columns in rsi. Since both rsi_crossed_below and rsi_crossed_above are two different functions, we need to build a product of the threshold values manually and then instruct each crossover function to combine them with every column in rsi:

```python
lower_ths
```

```python
upper_th_index
```

```python
rsi
```

```python
rsi_crossed_below
```

```python
rsi_crossed_above
```

```python
rsi
```

```python
>>> lower_ths_prod, upper_ths_prod = zip(*product(lower_ths, upper_ths))
>>> len(lower_ths_prod)  # (1)!
121
>>> len(upper_ths_prod)
121

>>> lower_th_index = vbt.Param(lower_ths_prod, name='lower_th')  # (2)!
>>> entries = rsi.rsi_crossed_below(lower_th_index)
>>> entries.columns
MultiIndex([(20,  8, 'simple'),
            (20,  8,    'exp'),
            (20,  8, 'wilder'),
            ...
            (30, 20, 'simple'),
            (30, 20,    'exp'),
            (30, 20, 'wilder')],
           names=['lower_th', 'rsi_window', 'rsi_wtype'], length=4719)

>>> upper_th_index = vbt.Param(upper_ths_prod, name='upper_th')
>>> exits = rsi.rsi_crossed_above(upper_th_index)
>>> exits.columns
MultiIndex([(70,  8, 'simple'),
            (70,  8,    'exp'),
            (70,  8, 'wilder'),
            ...
            (80, 20, 'simple'),
            (80, 20,    'exp'),
            (80, 20, 'wilder')],
           names=['upper_th', 'rsi_window', 'rsi_wtype'], length=4719)
```

```python
>>> lower_ths_prod, upper_ths_prod = zip(*product(lower_ths, upper_ths))
>>> len(lower_ths_prod)  # (1)!
121
>>> len(upper_ths_prod)
121

>>> lower_th_index = vbt.Param(lower_ths_prod, name='lower_th')  # (2)!
>>> entries = rsi.rsi_crossed_below(lower_th_index)
>>> entries.columns
MultiIndex([(20,  8, 'simple'),
            (20,  8,    'exp'),
            (20,  8, 'wilder'),
            ...
            (30, 20, 'simple'),
            (30, 20,    'exp'),
            (30, 20, 'wilder')],
           names=['lower_th', 'rsi_window', 'rsi_wtype'], length=4719)

>>> upper_th_index = vbt.Param(upper_ths_prod, name='upper_th')
>>> exits = rsi.rsi_crossed_above(upper_th_index)
>>> exits.columns
MultiIndex([(70,  8, 'simple'),
            (70,  8,    'exp'),
            (70,  8, 'wilder'),
            ...
            (80, 20, 'simple'),
            (80, 20,    'exp'),
            (80, 20, 'wilder')],
           names=['upper_th', 'rsi_window', 'rsi_wtype'], length=4719)
```

```python
lower_ths_prod
```

```python
upper_ths_prod
```

```python
vbt.Param
```

```python
rsi
```

We have produced over 4719 columns - madness! But did you notice that entries and exits have different columns now? The first one has lower_th as one of the column levels, the second one has upper_th. How are we supposed to pass differently labeled arrays (including close_price with one column) to Portfolio.from_signals?

```python
entries
```

```python
exits
```

```python
lower_th
```

```python
upper_th
```

```python
close_price
```

No worries, vectorbt knows exactly how to merge this information. Let's see:

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=entries, 
...     exits=exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c415ed5c0>

>>> stats_df = pf.stats([
...     'total_return', 
...     'total_trades', 
...     'win_rate', 
...     'expectancy'
... ], agg_func=None)  # (1)!
>>> stats_df
                                        Total Return [%]  Total Trades  \
lower_th upper_th rsi_window rsi_wtype                                   
20       70       8          simple           -25.285842            31   
                             exp               -7.939736            29   
                             wilder            61.979801            11   
...                                                  ...           ...   
                  20         simple           -59.159157             4   
                             exp               -3.331163             8   
                             wilder            31.479482             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
20       70       8          simple        51.612903   -1.224523  
                             exp           58.620690   -0.307862  
                             wilder        72.727273    5.634527  
...                                              ...         ...  
                  20         simple        33.333333  -16.159733  
                             exp           57.142857    7.032204  
                             wilder        50.000000   38.861607  

[4719 rows x 4 columns]
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close_price, 
...     entries=entries, 
...     exits=exits,
...     size=100,
...     size_type='value',
...     init_cash='auto'
... )
>>> pf
<vectorbtpro.portfolio.base.Portfolio at 0x7f9c415ed5c0>

>>> stats_df = pf.stats([
...     'total_return', 
...     'total_trades', 
...     'win_rate', 
...     'expectancy'
... ], agg_func=None)  # (1)!
>>> stats_df
                                        Total Return [%]  Total Trades  \
lower_th upper_th rsi_window rsi_wtype                                   
20       70       8          simple           -25.285842            31   
                             exp               -7.939736            29   
                             wilder            61.979801            11   
...                                                  ...           ...   
                  20         simple           -59.159157             4   
                             exp               -3.331163             8   
                             wilder            31.479482             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
20       70       8          simple        51.612903   -1.224523  
                             exp           58.620690   -0.307862  
                             wilder        72.727273    5.634527  
...                                              ...         ...  
                  20         simple        33.333333  -16.159733  
                             exp           57.142857    7.032204  
                             wilder        50.000000   38.861607  

[4719 rows x 4 columns]
```

Congrats! We just backtested 4719 parameter combinations in less than a second

Important

Even though we gained some unreal performance, we need to be careful to not occupy the entire RAM with our wide arrays. We can check the size of any Pickleable instance using Pickleable.getsize. For example, to print the total size of our portfolio in a human-readable format:

```python
>>> print(pf.getsize())
9.4 MB
```

```python
>>> print(pf.getsize())
9.4 MB
```

Even though the portfolio holds about 10 MB of compressed data, it must generate many arrays, such as the portfolio value, that have the same shape as the number of timestamps x parameter combinations:

```python
>>> np.product(pf.wrapper.shape) * 8 / 1024 / 1024
65.27364349365234
```

```python
>>> np.product(pf.wrapper.shape) * 8 / 1024 / 1024
65.27364349365234
```

We can see that each floating array occupies 65 MB of memory. By creating a dozen of such arrays (which is often the worst case), the memory consumption may jump to 1 GB very quickly.

One option is to use Pandas itself to analyze the produced statistics. For example, calculate the mean expectancy of each rsi_window:

```python
rsi_window
```

```python
>>> stats_df['Expectancy'].groupby('rsi_window').mean()
rsi_window
8      0.154425
9      0.064130
10    -0.915478
11    -0.523294
12     0.742266
13     3.898482
14     4.414367
15     6.916872
16     8.915225
17    12.204188
18    12.897135
19    14.508950
20    16.429515
Name: Expectancy, dtype: float64
```

```python
>>> stats_df['Expectancy'].groupby('rsi_window').mean()
rsi_window
8      0.154425
9      0.064130
10    -0.915478
11    -0.523294
12     0.742266
13     3.898482
14     4.414367
15     6.916872
16     8.915225
17    12.204188
18    12.897135
19    14.508950
20    16.429515
Name: Expectancy, dtype: float64
```

The longer is the RSI window, the higher is the mean expectancy.

Display the top 5 parameter combinations:

```python
>>> stats_df.sort_values(by='Expectancy', ascending=False).head()
                                        Total Return [%]  Total Trades  \
lower_th upper_th rsi_window rsi_wtype                                   
22       80       20         wilder           187.478208             2   
21       80       20         wilder           187.478208             2   
26       80       20         wilder           152.087039             3   
23       80       20         wilder           187.478208             2   
25       80       20         wilder           201.297495             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
22       80       20         wilder            100.0   93.739104  
21       80       20         wilder            100.0   93.739104  
26       80       20         wilder            100.0   93.739104  
23       80       20         wilder            100.0   93.739104  
25       80       20         wilder            100.0   93.739104
```

```python
>>> stats_df.sort_values(by='Expectancy', ascending=False).head()
                                        Total Return [%]  Total Trades  \
lower_th upper_th rsi_window rsi_wtype                                   
22       80       20         wilder           187.478208             2   
21       80       20         wilder           187.478208             2   
26       80       20         wilder           152.087039             3   
23       80       20         wilder           187.478208             2   
25       80       20         wilder           201.297495             3   

                                        Win Rate [%]  Expectancy  
lower_th upper_th rsi_window rsi_wtype                            
22       80       20         wilder            100.0   93.739104  
21       80       20         wilder            100.0   93.739104  
26       80       20         wilder            100.0   93.739104  
23       80       20         wilder            100.0   93.739104  
25       80       20         wilder            100.0   93.739104
```

To analyze any particular combination using vectorbt, we can select it from the portfolio the same way as we selected a column in a regular Pandas DataFrame. Let's plot the equity of the most successful combination:

```python
>>> pf[(22, 80, 20, "wilder")].plot_value().show()
```

```python
>>> pf[(22, 80, 20, "wilder")].plot_value().show()
```

Hint

Instead of selecting a column from a portfolio, which will create a new portfolio with only that column, you can also check whether the method you want to call supports the argument column and pass your column using this argument. For instance, we could have also used pf.plot_value(column=(22, 80, 20, "wilder")).

```python
column
```

```python
pf.plot_value(column=(22, 80, 20, "wilder"))
```

Even though, in theory, the best found setting doubles our money, it's still inferior to simply holding Bitcoin - our basic RSI strategy cannot beat the market

But even if it did, there is much more to just searching for right parameters: we need at least to (cross-) validate the strategy. We can also observe how the strategy behaves on other assets. Curious how to do it? Just expand open_price and close_price to contain multiple assets, and each example would work out-of-the-box!

```python
open_price
```

```python
close_price
```

```python
>>> data = vbt.BinanceData.pull(['BTCUSDT', 'ETHUSDT'])
```

```python
>>> data = vbt.BinanceData.pull(['BTCUSDT', 'ETHUSDT'])
```

100%

100%

Your homework is to run the examples on this data.

The final columns should become as follows:

```python
MultiIndex([(20, 70,  8, 'simple', 'BTCUSDT'),
            (20, 70,  8, 'simple', 'ETHUSDT'),
            (20, 70,  8,    'exp', 'BTCUSDT'),
            ...
            (30, 80, 20,    'exp', 'ETHUSDT'),
            (30, 80, 20, 'wilder', 'BTCUSDT'),
            (30, 80, 20, 'wilder', 'ETHUSDT')],
           names=['lower_th', 'upper_th', 'rsi_window', 'rsi_wtype', 'symbol'], length=9438)
```

```python
MultiIndex([(20, 70,  8, 'simple', 'BTCUSDT'),
            (20, 70,  8, 'simple', 'ETHUSDT'),
            (20, 70,  8,    'exp', 'BTCUSDT'),
            ...
            (30, 80, 20,    'exp', 'ETHUSDT'),
            (30, 80, 20, 'wilder', 'BTCUSDT'),
            (30, 80, 20, 'wilder', 'ETHUSDT')],
           names=['lower_th', 'upper_th', 'rsi_window', 'rsi_wtype', 'symbol'], length=9438)
```

We see that the column hierarchy now contains another level - symbol - denoting the asset. Let's visualize the distribution of the expectancy across both assets:

```python
symbol
```

```python
>>> eth_mask = stats_df.index.get_level_values('symbol') == 'ETHUSDT'
>>> btc_mask = stats_df.index.get_level_values('symbol') == 'BTCUSDT'
>>> pd.DataFrame({
...     'ETHUSDT': stats_df[eth_mask]['Expectancy'].values,
...     'BTCUSDT': stats_df[btc_mask]['Expectancy'].values
... }).vbt.histplot(xaxis=dict(title="Expectancy")).show()  # (1)!
```

```python
>>> eth_mask = stats_df.index.get_level_values('symbol') == 'ETHUSDT'
>>> btc_mask = stats_df.index.get_level_values('symbol') == 'BTCUSDT'
>>> pd.DataFrame({
...     'ETHUSDT': stats_df[eth_mask]['Expectancy'].values,
...     'BTCUSDT': stats_df[btc_mask]['Expectancy'].values
... }).vbt.histplot(xaxis=dict(title="Expectancy")).show()  # (1)!
```

ETH seems to react more aggressively to our strategy on average than BTC, maybe due to the market's higher volatility, a different structure, or just pure randomness.

And here's one of the main takeaways of such analysis: using strategies with simple and explainable mechanics, we can try to explain the mechanics of the market itself. Not only can we use this to improve ourselves and design better indicators, but use this information as an input to ML models, which are better at connecting dots than humans. Possibilities are endless!

## Summary¶

VectorBT® PRO is a powerful vehicle that enables us to discover uncharted territories faster and analyze them in more detail. Instead of using overused and outdated charts and indicators from books and YouTube videos, we can build our own tools that go hand in hand with the market. We can backtest thousands of strategy configurations to learn how the market reacts to each one of them - in a matter of milliseconds. All it takes is creativity

Python code  Notebook

