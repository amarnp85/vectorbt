# Pairs trading¶

A pairs trading strategy is a statistical arbitrage and convergence strategy that is based on the historical correlation of two instruments and involves matching a long position with a short position. The two offsetting positions form the basis for a hedging strategy that seeks to benefit from either a positive or negative trend. A high positive correlation (mostly a minimum of 0.8) of both instruments is the primary driver behind the strategy's profits. Whenever the correlation eventually deviates, we would seek to buy the underperforming instrument and sell short the outperforming instrument. If the securities return to their historical correlation (which is what we bet on!), a profit is made from the convergence of the prices. Thus, pairs trading is used to generate profits regardless of any market condition: uptrend, downtrend, or sideways.

## Selection¶

When designing a pairs trading strategy, it is more important that the pairs are selected based on cointegration rather than just correlation. Correlated instruments tend to move in a similar way, but over time, the price ratio (or spread) between the two instruments might diverge considerably. Cointegrated instruments, on the other hand, don't necessarily always move in the same direction: the spread between them can on some days increase but the prices usually find themselves being "pulled back together" to the mean, which provides optimal conditions for pairs arbitrage trading.

The two workhorses of finding the cointegration are the Engle-Granger test and the Johansen test. We'll go with the former since its augmented version is implemented in statsmodels. The idea of Engle-Granger test is simple. We perform a linear regression between the two asset prices and check if the residual is stationary using the Augmented Dick-Fuller (ADF) test. If the residual is stationary, then the two asset prices are cointegrated.

```python
statsmodels
```

But first, let's build a universe of instruments to select our pairs from. For this, we will search for all the available USDT symbols on Binance and download their daily history. Instead of bulk-fetching all of them at once, we will fetch each symbol individually and append it to an HDF file. The reason behind such a separation is that most symbols have a limited history, and we don't want to waste much RAM by prolonging it with NaNs. We will also skip the entire procedure if the file already exists.

Note

Make sure to delete the HDF file if you want to re-fetch.

```python
>>> from vectorbtpro import *

>>> SYMBOLS = vbt.BinanceData.list_symbols("*USDT")  # (1)!
>>> POOL_FILE = "temp/data_pool.h5"
>>> START = "2018"
>>> END = "2023"

>>> # vbt.remove_dir("temp", with_contents=True, missing_ok=True)
>>> vbt.make_dir("temp")  # (2)!

>>> if not vbt.file_exists(POOL_FILE):
...     with vbt.ProgressBar(total=len(SYMBOLS)) as pbar:  # (3)!
...         collected = 0
...         for symbol in SYMBOLS:
...             try:
...                 data = vbt.BinanceData.pull(
...                     symbol, 
...                     start=START,
...                     end=END,
...                     show_progress=False,
...                     silence_warnings=True
...                 )
...                 data.to_hdf(POOL_FILE)  # (4)!
...                 collected += 1
...             except Exception:
...                 pass
...             pbar.set_prefix(f"{symbol} ({collected})")  # (5)!
...             pbar.update()
```

```python
>>> from vectorbtpro import *

>>> SYMBOLS = vbt.BinanceData.list_symbols("*USDT")  # (1)!
>>> POOL_FILE = "temp/data_pool.h5"
>>> START = "2018"
>>> END = "2023"

>>> # vbt.remove_dir("temp", with_contents=True, missing_ok=True)
>>> vbt.make_dir("temp")  # (2)!

>>> if not vbt.file_exists(POOL_FILE):
...     with vbt.ProgressBar(total=len(SYMBOLS)) as pbar:  # (3)!
...         collected = 0
...         for symbol in SYMBOLS:
...             try:
...                 data = vbt.BinanceData.pull(
...                     symbol, 
...                     start=START,
...                     end=END,
...                     show_progress=False,
...                     silence_warnings=True
...                 )
...                 data.to_hdf(POOL_FILE)  # (4)!
...                 collected += 1
...             except Exception:
...                 pass
...             pbar.set_prefix(f"{symbol} ({collected})")  # (5)!
...             pbar.update()
```

```python
tqdm
```

```python
POOL_FILE
```

Symbol 423/423

Symbol 423/423

The procedure took a while, but now we've got a file with the data distributed across keys. The great thing about working with HDF files (and VBT in particular!) is that we can import an entire file and join all the contained keys with a single command!

But we need to do one more decision: which period should we analyze to select the optimal pair? What is really important here is to ensure that we don't use the same date range for both pairs selection and strategy backtesting since we would make ourselves vulnerable to a survivorship bias, thus let's reserve a more recent period of time for the backtesting part.

```python
>>> SELECT_START = "2020"
>>> SELECT_END = "2021"

>>> data = vbt.HDFData.pull(
...     POOL_FILE, 
...     start=SELECT_START, 
...     end=SELECT_END, 
...     silence_warnings=True
... )

>>> print(len(data.symbols))
245
```

```python
>>> SELECT_START = "2020"
>>> SELECT_END = "2021"

>>> data = vbt.HDFData.pull(
...     POOL_FILE, 
...     start=SELECT_START, 
...     end=SELECT_END, 
...     silence_warnings=True
... )

>>> print(len(data.symbols))
245
```

We've imported 245 datasets, but some of these datasets are probably incomplete. In order for our analysis to be as seamless as possible, we should remove them.

```python
>>> data = data.select([
...     k 
...     for k, v in data.data.items() 
...     if not v.isnull().any().any()
... ])

>>> print(len(data.symbols))
82
```

```python
>>> data = data.select([
...     k 
...     for k, v in data.data.items() 
...     if not v.isnull().any().any()
... ])

>>> print(len(data.symbols))
82
```

A big chunk of our data has gone for good!

The next step is finding the pairs that satisfy our cointegration test. The methods for finding viable pairs all live on a spectrum. At the one end, we have some extra knowledge that leads us to believe that the pair is cointegrated, so we go out and test for the presence of cointegration. At the other end of the spectrum, we perform a search through hundreds of different instruments for any viable pairs according to our test. In this case, we may incur a multiple comparisons bias, which is an increased chance to incorrectly generate a significant p-value when many tests are run. For example, if 100 tests are run on a random data, we should expect to see 5 p-values below 0.05. In practice a second verification step would be needed if looking for pairs this way, which we will do later.

The test itself is performed with statsmodels.tsa.stattools.coint. This function returns of tuple, the second element of which is the p-value of interest. But since testing each single pair would require going through 82 * 82 or 6,724 pairs and the coint function is not the fastest function of all, we'll parallelize the entire thing using pathos

```python
statsmodels.tsa.stattools.coint
```

```python
82 * 82
```

```python
coint
```

```python
pathos
```

```python
>>> @vbt.parameterized(  # (1)!
...     merge_func="concat", 
...     engine="pathos",
...     distribute="chunks",  # (2)!
...     n_chunks="auto"  # (3)!
... )
... def coint_pvalue(close, s1, s2):
...     import statsmodels.tsa.stattools as ts  # (4)!
...     import numpy as np
...     return ts.coint(np.log(close[s1]), np.log(close[s2]))[1]

>>> COINT_FILE = "temp/coint_pvalues.pickle"

>>> # vbt.remove_file(COINT_FILE, missing_ok=True)
>>> if not vbt.file_exists(COINT_FILE):
...     coint_pvalues = coint_pvalue(  # (5)!
...         data.close,
...         vbt.Param(data.symbols, condition="s1 != s2"),  # (6)!
...         vbt.Param(data.symbols)
...     )
...     vbt.save(coint_pvalues, COINT_FILE)
... else:
...     coint_pvalues = vbt.load(COINT_FILE)
```

```python
>>> @vbt.parameterized(  # (1)!
...     merge_func="concat", 
...     engine="pathos",
...     distribute="chunks",  # (2)!
...     n_chunks="auto"  # (3)!
... )
... def coint_pvalue(close, s1, s2):
...     import statsmodels.tsa.stattools as ts  # (4)!
...     import numpy as np
...     return ts.coint(np.log(close[s1]), np.log(close[s2]))[1]

>>> COINT_FILE = "temp/coint_pvalues.pickle"

>>> # vbt.remove_file(COINT_FILE, missing_ok=True)
>>> if not vbt.file_exists(COINT_FILE):
...     coint_pvalues = coint_pvalue(  # (5)!
...         data.close,
...         vbt.Param(data.symbols, condition="s1 != s2"),  # (6)!
...         vbt.Param(data.symbols)
...     )
...     vbt.save(coint_pvalues, COINT_FILE)
... else:
...     coint_pvalues = vbt.load(COINT_FILE)
```

Hint

It's OK to analyze raw prices, but log prices are preferable.

The result is a Pandas Series where each pair of two symbols in the index is pointing to its p-value. If the p-value is small, below a critical size (< 0.05), then we can reject the hypothesis that there is no cointegrating relationship. Thus, let's arrange the p-values in increasing order:

```python
>>> coint_pvalues = coint_pvalues.sort_values()

>>> print(coint_pvalues)
s1        s2      
TUSDUSDT  BUSDUSDT    6.179128e-17
USDCUSDT  BUSDUSDT    7.703666e-14
BUSDUSDT  USDCUSDT    2.687508e-13
TUSDUSDT  USDCUSDT    2.906244e-12
BUSDUSDT  TUSDUSDT    1.853641e-11
                               ...
BTCUSDT   XTZUSDT     1.000000e+00
          EOSUSDT     1.000000e+00
          ENJUSDT     1.000000e+00
          NKNUSDT     1.000000e+00
ZILUSDT   HBARUSDT    1.000000e+00
Length: 6642, dtype: float64
```

```python
>>> coint_pvalues = coint_pvalues.sort_values()

>>> print(coint_pvalues)
s1        s2      
TUSDUSDT  BUSDUSDT    6.179128e-17
USDCUSDT  BUSDUSDT    7.703666e-14
BUSDUSDT  USDCUSDT    2.687508e-13
TUSDUSDT  USDCUSDT    2.906244e-12
BUSDUSDT  TUSDUSDT    1.853641e-11
                               ...
BTCUSDT   XTZUSDT     1.000000e+00
          EOSUSDT     1.000000e+00
          ENJUSDT     1.000000e+00
          NKNUSDT     1.000000e+00
ZILUSDT   HBARUSDT    1.000000e+00
Length: 6642, dtype: float64
```

Coincidence that the most cointegrated pairs are stablecoins? Don't think so.

Remember about the multiple comparisons bias? Let's test the first bunch of pairs by plotting the charts below and ensuring that the difference between each pair of symbols bounces back and forth around its mean. For example, here's the analysis for the pair (ALGOUSDT, QTUMUSDT):

```python
(ALGOUSDT, QTUMUSDT)
```

```python
>>> S1, S2 = "ALGOUSDT", "QTUMUSDT"

>>> data.plot(column="Close", symbol=[S1, S2], base=1).show()
```

```python
>>> S1, S2 = "ALGOUSDT", "QTUMUSDT"

>>> data.plot(column="Close", symbol=[S1, S2], base=1).show()
```

The prices move together.

```python
>>> S1_log = np.log(data.get("Close", S1))  # (1)!
>>> S2_log = np.log(data.get("Close", S2))
>>> log_diff = (S2_log - S1_log).rename("Log diff")
>>> fig = log_diff.vbt.plot()
>>> fig.add_hline(y=log_diff.mean(), line_color="yellow", line_dash="dot")
>>> fig.show()
```

```python
>>> S1_log = np.log(data.get("Close", S1))  # (1)!
>>> S2_log = np.log(data.get("Close", S2))
>>> log_diff = (S2_log - S1_log).rename("Log diff")
>>> fig = log_diff.vbt.plot()
>>> fig.add_hline(y=log_diff.mean(), line_color="yellow", line_dash="dot")
>>> fig.show()
```

```python
data.close[S1]
```

The linear combination between them varies around the mean.

## Testing¶

The more data the better: let's re-fetch our pair's history but with a higher granularity.

```python
>>> DATA_FILE = "temp/data.pickle"

>>> # vbt.remove_file(DATA_FILE, missing_ok=True)
>>> if not vbt.file_exists(DATA_FILE):
...     data = vbt.BinanceData.pull(
...         [S1, S2], 
...         start=SELECT_END,  # (1)!
...         end=END, 
...         timeframe="hourly"
...     )
...     vbt.save(data, DATA_FILE)
... else:
...     data = vbt.load(DATA_FILE)

>>> print(len(data.index))
17507
```

```python
>>> DATA_FILE = "temp/data.pickle"

>>> # vbt.remove_file(DATA_FILE, missing_ok=True)
>>> if not vbt.file_exists(DATA_FILE):
...     data = vbt.BinanceData.pull(
...         [S1, S2], 
...         start=SELECT_END,  # (1)!
...         end=END, 
...         timeframe="hourly"
...     )
...     vbt.save(data, DATA_FILE)
... else:
...     data = vbt.load(DATA_FILE)

>>> print(len(data.index))
17507
```

Note

Make sure that none of the tickers were delisted.

We've got 17,507 data points for each symbol.

### Level: Researcher ¶

Spread is the relative performance of both instruments. Whenever both instruments drift apart, the spread increases and may reach a certain threshold where we take a long position in the underperformer and a short position in the overachiever. Such a threshold is usually set to a number of standard deviations from the mean. All of that happens in a rolling fashion since the linear combination between both instruments is constantly changing. We'll use the prediction error of the ordinary least squares (OLS), that is, the difference between the true and predicted value. Gladly, we have the OLS indicator, which can not only calculate the prediction error but also the z-score of that error:

```python
>>> import scipy.stats as st

>>> WINDOW = 24 * 30  # (1)!
>>> UPPER = st.norm.ppf(1 - 0.05 / 2)  # (2)!
>>> LOWER = -st.norm.ppf(1 - 0.05 / 2)

>>> S1_close = data.get("Close", S1)
>>> S2_close = data.get("Close", S2)
>>> ols = vbt.OLS.run(S1_close, S2_close, window=vbt.Default(WINDOW))
>>> spread = ols.error.rename("Spread")
>>> zscore = ols.zscore.rename("Z-score")
>>> print(pd.concat((spread, zscore), axis=1))
                             Spread   Z-score
Open time                                    
2021-01-01 00:00:00+00:00       NaN       NaN
2021-01-01 01:00:00+00:00       NaN       NaN
2021-01-01 02:00:00+00:00       NaN       NaN
2021-01-01 03:00:00+00:00       NaN       NaN
2021-01-01 04:00:00+00:00       NaN       NaN
...                             ...       ...
2022-12-31 19:00:00+00:00 -0.121450 -1.066809
2022-12-31 20:00:00+00:00 -0.123244 -1.078957
2022-12-31 21:00:00+00:00 -0.122595 -1.070667
2022-12-31 22:00:00+00:00 -0.125066 -1.088617
2022-12-31 23:00:00+00:00 -0.130532 -1.131498

[17507 rows x 2 columns]
```

```python
>>> import scipy.stats as st

>>> WINDOW = 24 * 30  # (1)!
>>> UPPER = st.norm.ppf(1 - 0.05 / 2)  # (2)!
>>> LOWER = -st.norm.ppf(1 - 0.05 / 2)

>>> S1_close = data.get("Close", S1)
>>> S2_close = data.get("Close", S2)
>>> ols = vbt.OLS.run(S1_close, S2_close, window=vbt.Default(WINDOW))
>>> spread = ols.error.rename("Spread")
>>> zscore = ols.zscore.rename("Z-score")
>>> print(pd.concat((spread, zscore), axis=1))
                             Spread   Z-score
Open time                                    
2021-01-01 00:00:00+00:00       NaN       NaN
2021-01-01 01:00:00+00:00       NaN       NaN
2021-01-01 02:00:00+00:00       NaN       NaN
2021-01-01 03:00:00+00:00       NaN       NaN
2021-01-01 04:00:00+00:00       NaN       NaN
...                             ...       ...
2022-12-31 19:00:00+00:00 -0.121450 -1.066809
2022-12-31 20:00:00+00:00 -0.123244 -1.078957
2022-12-31 21:00:00+00:00 -0.122595 -1.070667
2022-12-31 22:00:00+00:00 -0.125066 -1.088617
2022-12-31 23:00:00+00:00 -0.130532 -1.131498

[17507 rows x 2 columns]
```

Let's plot the z-score, the two thresholds, and the points where the former crosses the latter:

```python
>>> upper_crossed = zscore.vbt.crossed_above(UPPER)
>>> lower_crossed = zscore.vbt.crossed_below(LOWER)

>>> fig = zscore.vbt.plot()
>>> fig.add_hline(y=UPPER, line_color="orangered", line_dash="dot")
>>> fig.add_hline(y=0, line_color="yellow", line_dash="dot")
>>> fig.add_hline(y=LOWER, line_color="limegreen", line_dash="dot")
>>> upper_crossed.vbt.signals.plot_as_exits(zscore, fig=fig)
>>> lower_crossed.vbt.signals.plot_as_entries(zscore, fig=fig)
>>> fig.show()
```

```python
>>> upper_crossed = zscore.vbt.crossed_above(UPPER)
>>> lower_crossed = zscore.vbt.crossed_below(LOWER)

>>> fig = zscore.vbt.plot()
>>> fig.add_hline(y=UPPER, line_color="orangered", line_dash="dot")
>>> fig.add_hline(y=0, line_color="yellow", line_dash="dot")
>>> fig.add_hline(y=LOWER, line_color="limegreen", line_dash="dot")
>>> upper_crossed.vbt.signals.plot_as_exits(zscore, fig=fig)
>>> lower_crossed.vbt.signals.plot_as_entries(zscore, fig=fig)
>>> fig.show()
```

If we look closely at the chart above, we'll notice many signals of the same type happening one after another. This is because of the price fluctuations that cause the price to repeatedly cross each threshold. This won't cause us any troubles if we use Portfolio.from_signals as our simulation method of choice since it ignores any duplicate signals by default.

What's left is the construction of proper signal arrays. Remember that pairs trading involves two opposite positions that should be part of a single portfolio, that is, we need to transform our crossover signals into two arrays, long entries and short entries, with two columns each. Whenever there is an upper-threshold crossover signal, we will issue a short entry signal for the first asset and a long entry signal for the second one. Conversely, whenever there is a lower-threshold crossover signal, we will issue a long entry signal for the first asset and a short entry signal for the second one.

```python
>>> long_entries = data.symbol_wrapper.fill(False)
>>> short_entries = data.symbol_wrapper.fill(False)

>>> short_entries.loc[upper_crossed, S1] = True
>>> long_entries.loc[upper_crossed, S2] = True
>>> long_entries.loc[lower_crossed, S1] = True
>>> short_entries.loc[lower_crossed, S2] = True

>>> print(long_entries.sum())
symbol
ALGOUSDT    52
QTUMUSDT    73
dtype: int64
>>> print(short_entries.sum())
symbol
ALGOUSDT    73
QTUMUSDT    52
dtype: int64
```

```python
>>> long_entries = data.symbol_wrapper.fill(False)
>>> short_entries = data.symbol_wrapper.fill(False)

>>> short_entries.loc[upper_crossed, S1] = True
>>> long_entries.loc[upper_crossed, S2] = True
>>> long_entries.loc[lower_crossed, S1] = True
>>> short_entries.loc[lower_crossed, S2] = True

>>> print(long_entries.sum())
symbol
ALGOUSDT    52
QTUMUSDT    73
dtype: int64
>>> print(short_entries.sum())
symbol
ALGOUSDT    73
QTUMUSDT    52
dtype: int64
```

It's time to simulate our configuration! Position size of the pair should be matched by dollar value rather than by the number of shares; this way a 5% move in one equals a 5% move in the other. To avoid running out of cash, we'll make the position size of each order dependent on the current equity. By the way, you might wonder why we are allowed to use Portfolio.from_signals even though it doesn't support target size types? In pairs trading, a position is either opened or reversed (that is, closed out and opened again but with the opposite sign), such that we don't need to use target size types at all - regular size types would suffice.

```python
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=long_entries,
...     short_entries=short_entries,
...     size=10,  # (1)!
...     size_type="valuepercent100",  # (2)!
...     group_by=True,  # (3)!
...     cash_sharing=True,
...     call_seq="auto"
... )
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=long_entries,
...     short_entries=short_entries,
...     size=10,  # (1)!
...     size_type="valuepercent100",  # (2)!
...     group_by=True,  # (3)!
...     cash_sharing=True,
...     call_seq="auto"
... )
```

The simulation is completed. When working with grouped portfolios that involve some kind of rebalancing, we should always throw a look at the allocations first, for validation:

```python
>>> fig = pf.plot_allocations()
>>> rebalancing_dates = data.index[np.unique(pf.orders.idx.values)]
>>> for date in rebalancing_dates:
...     fig.add_vline(x=date, line_color="teal", line_dash="dot")
>>> fig.show()
```

```python
>>> fig = pf.plot_allocations()
>>> rebalancing_dates = data.index[np.unique(pf.orders.idx.values)]
>>> for date in rebalancing_dates:
...     fig.add_vline(x=date, line_color="teal", line_dash="dot")
>>> fig.show()
```

The chart makes sense: positions of both symbols are continuously switching as if we're looking at a chess board. Also, the long and short allocations rarely go beyond the specified position size of 10%. Next, we should calculate the portfolio statistics to assess the profitability of our strategy:

```python
>>> pf.stats()
Start                          2021-01-01 00:00:00+00:00
End                            2022-12-31 23:00:00+00:00
Period                                 729 days 11:00:00
Start Value                                        100.0
Min Value                                      96.401924
Max Value                                     127.670782
End Value                                     119.930329
Total Return [%]                               19.930329
Benchmark Return [%]                          -34.051206
Total Time Exposure [%]                        89.946878
Max Gross Exposure [%]                          12.17734
Max Drawdown [%]                                8.592299
Max Drawdown Duration                  318 days 00:00:00
Total Orders                                          34
Total Fees Paid                                      0.0
Total Trades                                          34
Win Rate [%]                                       43.75
Best Trade [%]                                160.511614
Worst Trade [%]                               -54.796964
Avg Winning Trade [%]                          41.851493
Avg Losing Trade [%]                          -20.499826
Avg Winning Trade Duration    42 days 22:04:17.142857143
Avg Losing Trade Duration               33 days 14:43:20
Profit Factor                                   1.553538
Expectancy                                      0.713595
Sharpe Ratio                                    0.782316
Calmar Ratio                                     1.10712
Omega Ratio                                     1.034258
Sortino Ratio                                   1.221721
Name: group, dtype: object
```

```python
>>> pf.stats()
Start                          2021-01-01 00:00:00+00:00
End                            2022-12-31 23:00:00+00:00
Period                                 729 days 11:00:00
Start Value                                        100.0
Min Value                                      96.401924
Max Value                                     127.670782
End Value                                     119.930329
Total Return [%]                               19.930329
Benchmark Return [%]                          -34.051206
Total Time Exposure [%]                        89.946878
Max Gross Exposure [%]                          12.17734
Max Drawdown [%]                                8.592299
Max Drawdown Duration                  318 days 00:00:00
Total Orders                                          34
Total Fees Paid                                      0.0
Total Trades                                          34
Win Rate [%]                                       43.75
Best Trade [%]                                160.511614
Worst Trade [%]                               -54.796964
Avg Winning Trade [%]                          41.851493
Avg Losing Trade [%]                          -20.499826
Avg Winning Trade Duration    42 days 22:04:17.142857143
Avg Losing Trade Duration               33 days 14:43:20
Profit Factor                                   1.553538
Expectancy                                      0.713595
Sharpe Ratio                                    0.782316
Calmar Ratio                                     1.10712
Omega Ratio                                     1.034258
Sortino Ratio                                   1.221721
Name: group, dtype: object
```

We're up by almost 20% from the initial portfolio value - a comfortable win considering that the benchmark is down by almost 35%. As expected, the amount of time our portfolio was in any position is 90%; the only time we were not in a position is the initial period before the first signal. Also, even though the win rate is below 50%, we made a profit because the average trade brings 50% more profit than loss, which leads us to a long-term profit of $0.70 per trade.

But can we somehow prove that the simulation closely resembles the signals it's based upon? For this, we can reframe our problem into a portfolio optimization problem: we can mark the points where the z-score crosses any of the thresholds as allocation points, and appoint the corresponding weights at these points. All of this is a child's play using PortfolioOptimizer:

```python
>>> allocations = data.symbol_wrapper.fill()  # (1)!
>>> allocations.loc[upper_crossed, S1] = -0.1
>>> allocations.loc[upper_crossed, S2] = 0.1
>>> allocations.loc[lower_crossed, S1] = 0.1
>>> allocations.loc[lower_crossed, S2] = -0.1
>>> pfo = vbt.PortfolioOptimizer.from_filled_allocations(allocations)

>>> print(pfo.allocations)  # (2)!
symbol                     ALGOUSDT  QTUMUSDT
Open time                                    
2021-03-15 10:00:00+00:00       0.1      -0.1
2021-03-23 03:00:00+00:00      -0.1       0.1
2021-04-17 14:00:00+00:00       0.1      -0.1
2021-04-19 00:00:00+00:00      -0.1       0.1
2021-06-03 16:00:00+00:00       0.1      -0.1
2021-06-30 22:00:00+00:00      -0.1       0.1
2021-08-19 06:00:00+00:00       0.1      -0.1
2021-10-02 21:00:00+00:00      -0.1       0.1
2021-11-12 03:00:00+00:00       0.1      -0.1
2022-02-02 09:00:00+00:00      -0.1       0.1
2022-04-28 21:00:00+00:00       0.1      -0.1
2022-07-22 02:00:00+00:00      -0.1       0.1
2022-09-04 01:00:00+00:00       0.1      -0.1
2022-09-27 03:00:00+00:00      -0.1       0.1
2022-10-13 10:00:00+00:00       0.1      -0.1
2022-10-23 19:00:00+00:00      -0.1       0.1
2022-11-08 20:00:00+00:00       0.1      -0.1

>>> pfo.plot().show()
```

```python
>>> allocations = data.symbol_wrapper.fill()  # (1)!
>>> allocations.loc[upper_crossed, S1] = -0.1
>>> allocations.loc[upper_crossed, S2] = 0.1
>>> allocations.loc[lower_crossed, S1] = 0.1
>>> allocations.loc[lower_crossed, S2] = -0.1
>>> pfo = vbt.PortfolioOptimizer.from_filled_allocations(allocations)

>>> print(pfo.allocations)  # (2)!
symbol                     ALGOUSDT  QTUMUSDT
Open time                                    
2021-03-15 10:00:00+00:00       0.1      -0.1
2021-03-23 03:00:00+00:00      -0.1       0.1
2021-04-17 14:00:00+00:00       0.1      -0.1
2021-04-19 00:00:00+00:00      -0.1       0.1
2021-06-03 16:00:00+00:00       0.1      -0.1
2021-06-30 22:00:00+00:00      -0.1       0.1
2021-08-19 06:00:00+00:00       0.1      -0.1
2021-10-02 21:00:00+00:00      -0.1       0.1
2021-11-12 03:00:00+00:00       0.1      -0.1
2022-02-02 09:00:00+00:00      -0.1       0.1
2022-04-28 21:00:00+00:00       0.1      -0.1
2022-07-22 02:00:00+00:00      -0.1       0.1
2022-09-04 01:00:00+00:00       0.1      -0.1
2022-09-27 03:00:00+00:00      -0.1       0.1
2022-10-13 10:00:00+00:00       0.1      -0.1
2022-10-23 19:00:00+00:00      -0.1       0.1
2022-11-08 20:00:00+00:00       0.1      -0.1

>>> pfo.plot().show()
```

There's also a handy method to simulate the optimizer:

```python
>>> pf = pfo.simulate(data, pf_method="from_signals")
>>> pf.total_return
0.19930328736504038
```

```python
>>> pf = pfo.simulate(data, pf_method="from_signals")
>>> pf.total_return
0.19930328736504038
```

Info

This method is based on a dynamic signal function that translates target percentages into signals, thus the compilation may take up to a minute (once compiled, it will be ultrafast though). You can also remove the pf_method argument to use the cacheable Portfolio.from_orders with similar results.

```python
pf_method
```

What about parameter optimization? There are several places in the code that can be parameterized. The signal generation part is affected by the parameters WINDOW, UPPER, and LOWER. The simulation part can be tweaked more freely; for example, we can add stops to limit our exposure to unfortunate price movements. Let's do the former part first.

```python
WINDOW
```

```python
UPPER
```

```python
LOWER
```

The main issue that we may come across is the combination of the two parameters UPPER and LOWER: we cannot just pass them to their respective crossover functions and hope for the best; we need to unify the columns that they produce to combine them later. One trick is to build a meta-indicator that encapsulates other indicators and deals with multiple parameter combinations out of the box:

```python
UPPER
```

```python
LOWER
```

```python
>>> PTS_expr = """
...     PTS:
...     x = @in_close.iloc[:, 0]
...     y = @in_close.iloc[:, 1]
...     ols = vbt.OLS.run(x, y, window=@p_window, hide_params=True)
...     upper = st.norm.ppf(1 - @p_upper_alpha / 2)
...     lower = -st.norm.ppf(1 - @p_lower_alpha / 2)
...     upper_crossed = ols.zscore.vbt.crossed_above(upper)
...     lower_crossed = ols.zscore.vbt.crossed_below(lower)
...     long_entries = wrapper.fill(False)
...     short_entries = wrapper.fill(False)
...     short_entries.loc[upper_crossed, x.name] = True
...     long_entries.loc[upper_crossed, y.name] = True
...     long_entries.loc[lower_crossed, x.name] = True
...     short_entries.loc[lower_crossed, y.name] = True
...     long_entries, short_entries
... """

>>> PTS = vbt.IF.from_expr(PTS_expr, keep_pd=True, st=st)  # (1)!
>>> vbt.phelp(PTS.run)  # (2)!
PTS.run(
    close,
    window,
    upper_alpha,
    lower_alpha,
    short_name='pts',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `PTS` indicator.

    * Inputs: `close`
    * Parameters: `window`, `upper_alpha`, `lower_alpha`
    * Outputs: `long_entries`, `short_entries`
```

```python
>>> PTS_expr = """
...     PTS:
...     x = @in_close.iloc[:, 0]
...     y = @in_close.iloc[:, 1]
...     ols = vbt.OLS.run(x, y, window=@p_window, hide_params=True)
...     upper = st.norm.ppf(1 - @p_upper_alpha / 2)
...     lower = -st.norm.ppf(1 - @p_lower_alpha / 2)
...     upper_crossed = ols.zscore.vbt.crossed_above(upper)
...     lower_crossed = ols.zscore.vbt.crossed_below(lower)
...     long_entries = wrapper.fill(False)
...     short_entries = wrapper.fill(False)
...     short_entries.loc[upper_crossed, x.name] = True
...     long_entries.loc[upper_crossed, y.name] = True
...     long_entries.loc[lower_crossed, x.name] = True
...     short_entries.loc[lower_crossed, y.name] = True
...     long_entries, short_entries
... """

>>> PTS = vbt.IF.from_expr(PTS_expr, keep_pd=True, st=st)  # (1)!
>>> vbt.phelp(PTS.run)  # (2)!
PTS.run(
    close,
    window,
    upper_alpha,
    lower_alpha,
    short_name='pts',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `PTS` indicator.

    * Inputs: `close`
    * Parameters: `window`, `upper_alpha`, `lower_alpha`
    * Outputs: `long_entries`, `short_entries`
```

```python
keep_pd
```

Our goal was to create an indicator that takes an input array with two columns (i.e., assets), executes our pipeline, and returns two signals arrays with two columns each. For this, we have constructed an expression that is a regular Python code but with useful enhancements. For example, we have specified that the variable close is an input array by prepending to it the prefix @in_. Also, we have annotated the three parameters window, upper_alpha, and lower_alpha by prepending to them the prefix @p_. Later, when the expression is run, the indicator factory will replace @in_close by our close price and @p_window, @p_upper_alpha, and @p_lower_alpha by our parameter values. Moreover, the factory will recognize the widely-understood variables vbt and wrapper and replace them by the vectorbt module and the current Pandas wrapper respectively. The last line consists of the variables that we want to return, their names will be used as output names automatically.

```python
close
```

```python
@in_
```

```python
window
```

```python
upper_alpha
```

```python
lower_alpha
```

```python
@p_
```

```python
@in_close
```

```python
@p_window
```

```python
@p_upper_alpha
```

```python
@p_lower_alpha
```

```python
vbt
```

```python
wrapper
```

Let's run this indicator on a grid of parameter combinations we want to test. But beware of wide parameter grids and potential out-of-memory errors: each parameter combination will build multiple arrays of the same shape as the data, thus use random search to effectively reduce the number of parameter combinations.

```python
>>> WINDOW_SPACE = np.arange(5, 50).tolist()  # (1)!
>>> ALPHA_SPACE = (np.arange(1, 100) / 1000).tolist()  # (2)!

>>> long_entries, short_entries = data.run(  # (3)!
...     PTS, 
...     window=WINDOW_SPACE,
...     upper_alpha=ALPHA_SPACE,
...     lower_alpha=ALPHA_SPACE,
...     param_product=True,
...     random_subset=1000,  # (4)!
...     seed=42,  # (5)!
...     unpack=True  # (6)!
... )
>>> print(long_entries.columns)
MultiIndex([( 5, 0.007,  0.09, 'ALGOUSDT'),
            ( 5, 0.007,  0.09, 'QTUMUSDT'),
            ( 5, 0.009, 0.086, 'ALGOUSDT'),
            ( 5, 0.009, 0.086, 'QTUMUSDT'),
            ( 5, 0.015, 0.082, 'ALGOUSDT'),
            ...
            (49, 0.091, 0.094, 'QTUMUSDT'),
            (49, 0.094, 0.054, 'ALGOUSDT'),
            (49, 0.094, 0.054, 'QTUMUSDT'),
            (49, 0.095, 0.074, 'ALGOUSDT'),
            (49, 0.095, 0.074, 'QTUMUSDT')],
           names=['pts_window', 'pts_upper_alpha', 'pts_lower_alpha', 'symbol'], length=2000)
```

```python
>>> WINDOW_SPACE = np.arange(5, 50).tolist()  # (1)!
>>> ALPHA_SPACE = (np.arange(1, 100) / 1000).tolist()  # (2)!

>>> long_entries, short_entries = data.run(  # (3)!
...     PTS, 
...     window=WINDOW_SPACE,
...     upper_alpha=ALPHA_SPACE,
...     lower_alpha=ALPHA_SPACE,
...     param_product=True,
...     random_subset=1000,  # (4)!
...     seed=42,  # (5)!
...     unpack=True  # (6)!
... )
>>> print(long_entries.columns)
MultiIndex([( 5, 0.007,  0.09, 'ALGOUSDT'),
            ( 5, 0.007,  0.09, 'QTUMUSDT'),
            ( 5, 0.009, 0.086, 'ALGOUSDT'),
            ( 5, 0.009, 0.086, 'QTUMUSDT'),
            ( 5, 0.015, 0.082, 'ALGOUSDT'),
            ...
            (49, 0.091, 0.094, 'QTUMUSDT'),
            (49, 0.094, 0.054, 'ALGOUSDT'),
            (49, 0.094, 0.054, 'QTUMUSDT'),
            (49, 0.095, 0.074, 'ALGOUSDT'),
            (49, 0.095, 0.074, 'QTUMUSDT')],
           names=['pts_window', 'pts_upper_alpha', 'pts_lower_alpha', 'symbol'], length=2000)
```

We took a data instance and told it to introspect the indicator PTS, find its input names among the arrays stored in the data instance, and pass the arrays along with the parameters to the indicator. This way, we don't have to deal with input and output names at all

```python
PTS
```

So, which parameter combinations are the most profitable?

```python
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=long_entries,
...     short_entries=short_entries,
...     size=10,
...     size_type="valuepercent100",
...     group_by=vbt.ExceptLevel("symbol"),  # (1)!
...     cash_sharing=True,
...     call_seq="auto"
... )

>>> opt_results = pd.concat((
...     pf.total_return,
...     pf.trades.expectancy,
... ), axis=1)
>>> print(opt_results.sort_values(by="total_return", ascending=False))
                                            total_return  expectancy
pts_window pts_upper_alpha pts_lower_alpha                          
41         0.076           0.001                0.503014    0.399218
15         0.079           0.001                0.489249    2.718049
16         0.023           0.016                0.474538    0.104986
6          0.078           0.048                0.445623    0.057574
41         0.028           0.001                0.441388    0.387182
...                                                  ...         ...
43         0.003           0.004               -0.263967   -0.131984
15         0.002           0.049               -0.273170   -0.182113
42         0.002           0.036               -0.316947   -0.110821
35         0.001           0.008               -0.330056   -0.196462
41         0.001           0.015               -0.363547   -0.191341

[1000 rows x 2 columns]
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=long_entries,
...     short_entries=short_entries,
...     size=10,
...     size_type="valuepercent100",
...     group_by=vbt.ExceptLevel("symbol"),  # (1)!
...     cash_sharing=True,
...     call_seq="auto"
... )

>>> opt_results = pd.concat((
...     pf.total_return,
...     pf.trades.expectancy,
... ), axis=1)
>>> print(opt_results.sort_values(by="total_return", ascending=False))
                                            total_return  expectancy
pts_window pts_upper_alpha pts_lower_alpha                          
41         0.076           0.001                0.503014    0.399218
15         0.079           0.001                0.489249    2.718049
16         0.023           0.016                0.474538    0.104986
6          0.078           0.048                0.445623    0.057574
41         0.028           0.001                0.441388    0.387182
...                                                  ...         ...
43         0.003           0.004               -0.263967   -0.131984
15         0.002           0.049               -0.273170   -0.182113
42         0.002           0.036               -0.316947   -0.110821
35         0.001           0.008               -0.330056   -0.196462
41         0.001           0.015               -0.363547   -0.191341

[1000 rows x 2 columns]
```

Now, let's do the second optimization part by picking one parameter combination from above and testing various stop configurations using Param:

```python
>>> best_index = opt_results.idxmax()["expectancy"]  # (1)!
>>> best_long_entries = long_entries[best_index]
>>> best_short_entries = short_entries[best_index]
>>> STOP_SPACE = [np.nan] + np.arange(1, 100).tolist()  # (2)!

>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=best_long_entries,
...     short_entries=best_short_entries,
...     size=10,
...     size_type="valuepercent100",
...     group_by=vbt.ExceptLevel("symbol"),
...     cash_sharing=True,
...     call_seq="auto",
...     sl_stop=vbt.Param(STOP_SPACE),  # (3)!
...     tsl_stop=vbt.Param(STOP_SPACE),
...     tp_stop=vbt.Param(STOP_SPACE),
...     delta_format="percent100",  # (4)!
...     stop_exit_price="close",  # (5)!
...     broadcast_kwargs=dict(random_subset=1000, seed=42)  # (6)!
... )

>>> opt_results = pd.concat((
...     pf.total_return,
...     pf.trades.expectancy,
... ), axis=1)
>>> print(opt_results.sort_values(by="total_return", ascending=False))
                          total_return  expectancy
sl_stop tsl_stop tp_stop                          
86.0    98.0     NaN          0.602834    2.740152
47.0    62.0     NaN          0.587525    1.632014
43.0    90.0     NaN          0.579859    1.757150
16.0    62.0     54.0         0.412477    0.448345
2.0     95.0     71.0         0.406624    0.125115
...                                ...         ...
27.0    41.0     20.0        -0.063945   -0.046337
52.0    46.0     20.0        -0.065675   -0.067706
23.0    61.0     22.0        -0.071294   -0.057495
6.0     57.0     31.0        -0.080679   -0.029232
23.0    45.0     22.0        -0.090643   -0.073099

[1000 rows x 2 columns]
```

```python
>>> best_index = opt_results.idxmax()["expectancy"]  # (1)!
>>> best_long_entries = long_entries[best_index]
>>> best_short_entries = short_entries[best_index]
>>> STOP_SPACE = [np.nan] + np.arange(1, 100).tolist()  # (2)!

>>> pf = vbt.Portfolio.from_signals(
...     data,
...     entries=best_long_entries,
...     short_entries=best_short_entries,
...     size=10,
...     size_type="valuepercent100",
...     group_by=vbt.ExceptLevel("symbol"),
...     cash_sharing=True,
...     call_seq="auto",
...     sl_stop=vbt.Param(STOP_SPACE),  # (3)!
...     tsl_stop=vbt.Param(STOP_SPACE),
...     tp_stop=vbt.Param(STOP_SPACE),
...     delta_format="percent100",  # (4)!
...     stop_exit_price="close",  # (5)!
...     broadcast_kwargs=dict(random_subset=1000, seed=42)  # (6)!
... )

>>> opt_results = pd.concat((
...     pf.total_return,
...     pf.trades.expectancy,
... ), axis=1)
>>> print(opt_results.sort_values(by="total_return", ascending=False))
                          total_return  expectancy
sl_stop tsl_stop tp_stop                          
86.0    98.0     NaN          0.602834    2.740152
47.0    62.0     NaN          0.587525    1.632014
43.0    90.0     NaN          0.579859    1.757150
16.0    62.0     54.0         0.412477    0.448345
2.0     95.0     71.0         0.406624    0.125115
...                                ...         ...
27.0    41.0     20.0        -0.063945   -0.046337
52.0    46.0     20.0        -0.065675   -0.067706
23.0    61.0     22.0        -0.071294   -0.057495
6.0     57.0     31.0        -0.080679   -0.029232
23.0    45.0     22.0        -0.090643   -0.073099

[1000 rows x 2 columns]
```

```python
param_product=True
```

```python
call_seq="auto"
```

We can also observe how the performance degrades with lower SL and TSL, and how the optimizer wants to discourage us from using TP at all. Let's take a closer look how a metric of interest depends on the values of each stop type:

```python
>>> def plot_metric_by_stop(stop_name, metric_name, stat_name, smooth):
...     from scipy.signal import savgol_filter
...
...     values = pf.deep_getattr(metric_name)  # (1)!
...     values = values.vbt.select_levels(stop_name)  # (2)!
...     values = getattr(values.groupby(values.index), stat_name)()  # (3)!
...     smooth_values = savgol_filter(values, smooth, 1)  # (4)!
...     smooth_values = values.vbt.wrapper.wrap(smooth_values)  # (5)!
...     fig = values.rename(metric_name).vbt.plot()
...     smooth_values.rename(f"{metric_name} (smoothed)").vbt.plot(
...         trace_kwargs=dict(line=dict(dash="dot", color="yellow")),
...         fig=fig, 
...     )
...     return fig
```

```python
>>> def plot_metric_by_stop(stop_name, metric_name, stat_name, smooth):
...     from scipy.signal import savgol_filter
...
...     values = pf.deep_getattr(metric_name)  # (1)!
...     values = values.vbt.select_levels(stop_name)  # (2)!
...     values = getattr(values.groupby(values.index), stat_name)()  # (3)!
...     smooth_values = savgol_filter(values, smooth, 1)  # (4)!
...     smooth_values = values.vbt.wrapper.wrap(smooth_values)  # (5)!
...     fig = values.rename(metric_name).vbt.plot()
...     smooth_values.rename(f"{metric_name} (smoothed)").vbt.plot(
...         trace_kwargs=dict(line=dict(dash="dot", color="yellow")),
...         fig=fig, 
...     )
...     return fig
```

```python
trades.winning.pnl.mean
```

```python
pd.MultiIndex
```

```python
>>> plot_metric_by_stop(
...     "sl_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

```python
>>> plot_metric_by_stop(
...     "sl_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

```python
>>> plot_metric_by_stop(
...     "tsl_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

```python
>>> plot_metric_by_stop(
...     "tsl_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

```python
>>> plot_metric_by_stop(
...     "tp_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

```python
>>> plot_metric_by_stop(
...     "tp_stop", 
...     "trades.expectancy", 
...     "median",
...     10
... ).show()
```

We've got an abstract picture of how stop orders affect the strategy performance.

### Level: Engineer ¶

The approach we followed above works great if we're still in the phase of strategy discovery; Pandas and the vectorbt's high-level API allow us to experiment rapidly and with ease. But as soon as we have finished developing a general framework for our strategy, we should start focusing on optimizing the strategy for best CPU and memory performance in order to be able to test the strategy on more assets, periods, and parameter combinations. Previously, we could test multiple parameter combinations by putting them all into memory; the only reason why we hadn't any issues running them is because we made use of random search. But let's do the parameter search more practical by increasing the speed of our backtests on the one side, and decreasing the memory consumption on the other.

Let's start by rewriting our indicator strictly with Numba:

```python
>>> @njit(nogil=True)  # (1)!
... def pt_signals_nb(close, window=WINDOW, upper=UPPER, lower=LOWER):
...     x = np.expand_dims(close[:, 0], 1)  # (2)!
...     y = np.expand_dims(close[:, 1], 1)
...     _, _, zscore = vbt.ind_nb.ols_nb(x, y, window)  # (3)!
...     zscore_1d = zscore[:, 0]  # (4)!
...     upper_ts = np.full_like(zscore_1d, upper, dtype=float_)  # (5)!
...     lower_ts = np.full_like(zscore_1d, lower, dtype=float_)
...     upper_crossed = vbt.nb.crossed_above_1d_nb(zscore_1d, upper_ts)  # (6)!
...     lower_crossed = vbt.nb.crossed_above_1d_nb(lower_ts, zscore_1d)  # (7)!
...     long_entries = np.full_like(close, False, dtype=np.bool_)  # (8)!
...     short_entries = np.full_like(close, False, dtype=np.bool_)
...     short_entries[upper_crossed, 0] = True  # (9)!
...     long_entries[upper_crossed, 1] = True
...     long_entries[lower_crossed, 0] = True
...     short_entries[lower_crossed, 1] = True
...     return long_entries, short_entries
```

```python
>>> @njit(nogil=True)  # (1)!
... def pt_signals_nb(close, window=WINDOW, upper=UPPER, lower=LOWER):
...     x = np.expand_dims(close[:, 0], 1)  # (2)!
...     y = np.expand_dims(close[:, 1], 1)
...     _, _, zscore = vbt.ind_nb.ols_nb(x, y, window)  # (3)!
...     zscore_1d = zscore[:, 0]  # (4)!
...     upper_ts = np.full_like(zscore_1d, upper, dtype=float_)  # (5)!
...     lower_ts = np.full_like(zscore_1d, lower, dtype=float_)
...     upper_crossed = vbt.nb.crossed_above_1d_nb(zscore_1d, upper_ts)  # (6)!
...     lower_crossed = vbt.nb.crossed_above_1d_nb(lower_ts, zscore_1d)  # (7)!
...     long_entries = np.full_like(close, False, dtype=np.bool_)  # (8)!
...     short_entries = np.full_like(close, False, dtype=np.bool_)
...     short_entries[upper_crossed, 0] = True  # (9)!
...     long_entries[upper_crossed, 1] = True
...     long_entries[lower_crossed, 0] = True
...     short_entries[lower_crossed, 1] = True
...     return long_entries, short_entries
```

```python
_1d
```

```python
_1d
```

Below we are ensuring that the indicator produces the same number of signals as previously:

```python
>>> long_entries, short_entries = pt_signals_nb(data.close.values)  # (1)!
>>> long_entries = data.symbol_wrapper.wrap(long_entries)  # (2)!
>>> short_entries = data.symbol_wrapper.wrap(short_entries)

>>> print(long_entries.sum())
symbol
ALGOUSDT    52
QTUMUSDT    73
dtype: int64
>>> print(short_entries.sum())
symbol
ALGOUSDT    73
QTUMUSDT    52
dtype: int64
```

```python
>>> long_entries, short_entries = pt_signals_nb(data.close.values)  # (1)!
>>> long_entries = data.symbol_wrapper.wrap(long_entries)  # (2)!
>>> short_entries = data.symbol_wrapper.wrap(short_entries)

>>> print(long_entries.sum())
symbol
ALGOUSDT    52
QTUMUSDT    73
dtype: int64
>>> print(short_entries.sum())
symbol
ALGOUSDT    73
QTUMUSDT    52
dtype: int64
```

Perfect alignment.

Even though this function is faster than the expression-based one, it doesn't address the memory issue because the output arrays it generates must still reside in memory to be later passed to the simulator, especially when multiple parameter combinations should be run. What we can do though is to wrap both the signal generation part and the simulation part into the same pipeline, and make it return lightweight arrays such as the total return. By using a proper chunking approach, we could then run an almost infinite number of parameter combinations!

The next step is rewriting the simulation part with Numba:

```python
>>> @njit(nogil=True)
... def pt_portfolio_nb(
...     open, 
...     high, 
...     low, 
...     close,
...     long_entries,
...     short_entries,
...     sl_stop=np.nan,
...     tsl_stop=np.nan,
...     tp_stop=np.nan,
... ):
...     target_shape = close.shape  # (1)!
...     group_lens = np.array([2])  # (2)!
...     sim_out = vbt.pf_nb.from_signals_nb(  # (3)!
...         target_shape=target_shape,
...         group_lens=group_lens,
...         auto_call_seq=True,  # (4)!
...         open=open,
...         high=high,
...         low=low,
...         close=close,
...         long_entries=long_entries,
...         short_entries=short_entries,
...         size=10,
...         size_type=vbt.pf_enums.SizeType.ValuePercent100,  # (5)!
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop,
...         delta_format=vbt.pf_enums.DeltaFormat.Percent100,
...         stop_exit_price=vbt.pf_enums.StopExitPrice.Close
...     )
...     return sim_out
```

```python
>>> @njit(nogil=True)
... def pt_portfolio_nb(
...     open, 
...     high, 
...     low, 
...     close,
...     long_entries,
...     short_entries,
...     sl_stop=np.nan,
...     tsl_stop=np.nan,
...     tp_stop=np.nan,
... ):
...     target_shape = close.shape  # (1)!
...     group_lens = np.array([2])  # (2)!
...     sim_out = vbt.pf_nb.from_signals_nb(  # (3)!
...         target_shape=target_shape,
...         group_lens=group_lens,
...         auto_call_seq=True,  # (4)!
...         open=open,
...         high=high,
...         low=low,
...         close=close,
...         long_entries=long_entries,
...         short_entries=short_entries,
...         size=10,
...         size_type=vbt.pf_enums.SizeType.ValuePercent100,  # (5)!
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop,
...         delta_format=vbt.pf_enums.DeltaFormat.Percent100,
...         stop_exit_price=vbt.pf_enums.StopExitPrice.Close
...     )
...     return sim_out
```

```python
call_seq="auto"
```

Let's run it:

```python
>>> sim_out = pt_portfolio_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values,
...     long_entries.values,
...     short_entries.values
... )
```

```python
>>> sim_out = pt_portfolio_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values,
...     long_entries.values,
...     short_entries.values
... )
```

The output of this function is an instance of the type SimulationOutput, which can be used to construct a new Portfolio instance for analysis:

```python
>>> pf = vbt.Portfolio(
...     data.symbol_wrapper.regroup(group_by=True),  # (1)!
...     sim_out,  # (2)!
...     open=data.open,  # (3)!
...     high=data.high,
...     low=data.low,
...     close=data.close,
...     cash_sharing=True,
...     init_cash=100  # (4)!
... )

>>> print(pf.total_return)
0.19930328736504038
```

```python
>>> pf = vbt.Portfolio(
...     data.symbol_wrapper.regroup(group_by=True),  # (1)!
...     sim_out,  # (2)!
...     open=data.open,  # (3)!
...     high=data.high,
...     low=data.low,
...     close=data.close,
...     cash_sharing=True,
...     init_cash=100  # (4)!
... )

>>> print(pf.total_return)
0.19930328736504038
```

```python
group_by
```

What's missing is a Numba-compiled version of the analysis part:

```python
>>> @njit(nogil=True)
... def pt_metrics_nb(close, sim_out):
...     target_shape = close.shape
...     group_lens = np.array([2])
...     filled_close = vbt.nb.fbfill_nb(close)  # (1)!
...     col_map = vbt.rec_nb.col_map_nb(  # (2)!
...         col_arr=sim_out.order_records["col"], 
...         n_cols=target_shape[1]
...     )
...     total_profit = vbt.pf_nb.total_profit_nb(  # (3)!
...         target_shape=target_shape,
...         close=filled_close,
...         order_records=sim_out.order_records,
...         col_map=col_map
...     )
...     total_profit_grouped = vbt.pf_nb.total_profit_grouped_nb(  # (4)!
...         total_profit=total_profit,
...         group_lens=group_lens,
...     )[0]  # (5)!
...     total_return = total_profit_grouped / 100  # (6)!
...     trade_records = vbt.pf_nb.get_exit_trades_nb(  # (7)!
...         order_records=sim_out.order_records, 
...         close=filled_close, 
...         col_map=col_map
...     )
...     trade_records = trade_records[  # (8)!
...         trade_records["status"] == vbt.pf_enums.TradeStatus.Closed
...     ]
...     expectancy = vbt.pf_nb.expectancy_reduce_nb(  # (9)!
...         pnl_arr=trade_records["pnl"]
...     )
...     return total_return, expectancy
```

```python
>>> @njit(nogil=True)
... def pt_metrics_nb(close, sim_out):
...     target_shape = close.shape
...     group_lens = np.array([2])
...     filled_close = vbt.nb.fbfill_nb(close)  # (1)!
...     col_map = vbt.rec_nb.col_map_nb(  # (2)!
...         col_arr=sim_out.order_records["col"], 
...         n_cols=target_shape[1]
...     )
...     total_profit = vbt.pf_nb.total_profit_nb(  # (3)!
...         target_shape=target_shape,
...         close=filled_close,
...         order_records=sim_out.order_records,
...         col_map=col_map
...     )
...     total_profit_grouped = vbt.pf_nb.total_profit_grouped_nb(  # (4)!
...         total_profit=total_profit,
...         group_lens=group_lens,
...     )[0]  # (5)!
...     total_return = total_profit_grouped / 100  # (6)!
...     trade_records = vbt.pf_nb.get_exit_trades_nb(  # (7)!
...         order_records=sim_out.order_records, 
...         close=filled_close, 
...         col_map=col_map
...     )
...     trade_records = trade_records[  # (8)!
...         trade_records["status"] == vbt.pf_enums.TradeStatus.Closed
...     ]
...     expectancy = vbt.pf_nb.expectancy_reduce_nb(  # (9)!
...         pnl_arr=trade_records["pnl"]
...     )
...     return total_return, expectancy
```

Don't let this function frighten you! We were able to calculate our metrics based on the close price and order records alone - the process which is also called a "reconstruction". The principle behind it is simple: start with information that you want to gain (a metric, for example) and see which information it requires. Then, find a function that provides that information, and repeat.

Let's run it for validation:

```python
>>> pt_metrics_nb(data.close.values, sim_out)
(0.19930328736504038, 0.7135952049405152)
```

```python
>>> pt_metrics_nb(data.close.values, sim_out)
(0.19930328736504038, 0.7135952049405152)
```

100% accuracy.

Finally, we'll put all parts into the same pipeline and benchmark it:

```python
>>> @njit(nogil=True)
... def pt_pipeline_nb(
...     open, 
...     high, 
...     low, 
...     close,
...     window=WINDOW,  # (1)!
...     upper=UPPER,
...     lower=LOWER,
...     sl_stop=np.nan,
...     tsl_stop=np.nan,
...     tp_stop=np.nan,
... ):
...     long_entries, short_entries = pt_signals_nb(
...         close, 
...         window=window, 
...         upper=upper, 
...         lower=lower
...     )
...     sim_out = pt_portfolio_nb(
...         open,
...         high,
...         low,
...         close,
...         long_entries,
...         short_entries,
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop
...     )
...     return pt_metrics_nb(close, sim_out)

>>> pt_pipeline_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values
... )
(0.19930328736504038, 0.7135952049405152)

>>> %%timeit
... pt_pipeline_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values
... )
5.4 ms ± 13.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

```python
>>> @njit(nogil=True)
... def pt_pipeline_nb(
...     open, 
...     high, 
...     low, 
...     close,
...     window=WINDOW,  # (1)!
...     upper=UPPER,
...     lower=LOWER,
...     sl_stop=np.nan,
...     tsl_stop=np.nan,
...     tp_stop=np.nan,
... ):
...     long_entries, short_entries = pt_signals_nb(
...         close, 
...         window=window, 
...         upper=upper, 
...         lower=lower
...     )
...     sim_out = pt_portfolio_nb(
...         open,
...         high,
...         low,
...         close,
...         long_entries,
...         short_entries,
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop
...     )
...     return pt_metrics_nb(close, sim_out)

>>> pt_pipeline_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values
... )
(0.19930328736504038, 0.7135952049405152)

>>> %%timeit
... pt_pipeline_nb(
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values
... )
5.4 ms ± 13.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Just 5 milliseconds per complete backtest

This kind of performance invites us to try some hard-core parameter optimization. Luckily for us, there are multiple ways we can implement that. The first and the most convenient way is to wrap the pipeline with the @parameterized decorator, which takes care of building the parameter grid and calling the pipeline function on each parameter combination from that grid. Also, because each of our Numba functions releases the GIL, we can finally utilize multithreading! Let's test all parameter combinations related to signal generation by dividing them into chunks and distributing the combinations within each chunk, which took me 5 minutes on Apple Silicon:

Important

Remember that @parameterized builds the entire parameter grid even if a random subset was specified, which may take a significant amount of time. For example, 6 parameters with 100 values each will build a grid of 100 ** 6, or one trillion combinations - too many to combine.

```python
@parameterized
```

```python
100 ** 6
```

```python
>>> param_pt_pipeline = vbt.parameterized(  # (1)!
...     pt_pipeline_nb, 
...     merge_func="concat",  # (2)!
...     seed=42,
...     engine="threadpool",  # (3)!
...     chunk_len="auto"
... )

>>> UPPER_SPACE = [st.norm.ppf(1 - x / 2) for x in ALPHA_SPACE]  # (4)!
>>> LOWER_SPACE = [-st.norm.ppf(1 - x / 2) for x in ALPHA_SPACE]
>>> POPT_FILE = "temp/param_opt.pickle"

>>> # vbt.remove_file(POPT_FILE, missing_ok=True)
>>> if not vbt.file_exists(POPT_FILE):
...     param_opt = param_pt_pipeline(
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=vbt.Param(WINDOW_SPACE),
...         upper=vbt.Param(UPPER_SPACE),
...         lower=vbt.Param(LOWER_SPACE)
...     )
...     vbt.save(param_opt, POPT_FILE)
... else:
...     param_opt = vbt.load(POPT_FILE)

>>> total_return, expectancy = param_opt  # (5)!
```

```python
>>> param_pt_pipeline = vbt.parameterized(  # (1)!
...     pt_pipeline_nb, 
...     merge_func="concat",  # (2)!
...     seed=42,
...     engine="threadpool",  # (3)!
...     chunk_len="auto"
... )

>>> UPPER_SPACE = [st.norm.ppf(1 - x / 2) for x in ALPHA_SPACE]  # (4)!
>>> LOWER_SPACE = [-st.norm.ppf(1 - x / 2) for x in ALPHA_SPACE]
>>> POPT_FILE = "temp/param_opt.pickle"

>>> # vbt.remove_file(POPT_FILE, missing_ok=True)
>>> if not vbt.file_exists(POPT_FILE):
...     param_opt = param_pt_pipeline(
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=vbt.Param(WINDOW_SPACE),
...         upper=vbt.Param(UPPER_SPACE),
...         lower=vbt.Param(LOWER_SPACE)
...     )
...     vbt.save(param_opt, POPT_FILE)
... else:
...     param_opt = vbt.load(POPT_FILE)

>>> total_return, expectancy = param_opt  # (5)!
```

```python
@
```

Chunk 55131/55131

Chunk 55131/55131

Let's analyze the total return:

```python
>>> print(total_return)
window  upper     lower    
5       3.290527  -3.290527    0.000000
                  -3.090232    0.000000
                  -2.967738    0.000000
                  -2.878162    0.000000
                  -2.807034    0.000000
                                    ...
49      1.649721  -1.669593    0.196197
                  -1.664563    0.192152
                  -1.659575    0.190713
                  -1.654628    0.201239
                  -1.649721    0.204764
Length: 441045, dtype: float64

>>> grouped_metric = total_return.groupby(level=["upper", "lower"]).mean()
>>> grouped_metric.vbt.heatmap(
...     trace_kwargs=dict(colorscale="RdBu", zmid=0),
...     yaxis=dict(autorange="reversed")
... ).show()
```

```python
>>> print(total_return)
window  upper     lower    
5       3.290527  -3.290527    0.000000
                  -3.090232    0.000000
                  -2.967738    0.000000
                  -2.878162    0.000000
                  -2.807034    0.000000
                                    ...
49      1.649721  -1.669593    0.196197
                  -1.664563    0.192152
                  -1.659575    0.190713
                  -1.654628    0.201239
                  -1.649721    0.204764
Length: 441045, dtype: float64

>>> grouped_metric = total_return.groupby(level=["upper", "lower"]).mean()
>>> grouped_metric.vbt.heatmap(
...     trace_kwargs=dict(colorscale="RdBu", zmid=0),
...     yaxis=dict(autorange="reversed")
... ).show()
```

The effect of the thresholds seems to be asymmetric: we can register the highest total return for the lowest upper threshold and the lowest lower threshold, such that the neutral range between the two thresholds is basically shifted downwards to the maximum extent.

The @parameterized decorator has two major limitations though: it runs only one parameter combination at a time, and it needs to build the parameter grid fully, even when querying a random subset. Why is the former a limitation at all? Because we can parallelize only a bunch of pipeline calls at a time: even though there's a special argument distribute="chunks" using which we can parallelize entire chunks, the calls within those chunks happen serially using a regular Python loop - bad for multithreading, which requires the GIL to be released for the entire procedure. And if you come to the idea of using multiprocessing: the method will have to serialize all the arguments (including the data) each time the pipeline is called, which is a huge overhead, but it may still result in some speedup compared to the approach above.

```python
@parameterized
```

```python
distribute="chunks"
```

To process multiple parameter combinations within each thread, we need to split them into chunks and write a parent Numba function that processes the combinations of each chunk in a loop. We can then parallelize this parent function using multithreading. So, we need to: 1) construct the parameter grid manually, 2) split it into chunks, and 3) iterate over the chunks and pass each one to the parent function for execution. The first step can be done using combine_params, which is the  of the @parameterized decorator. The second and third steps can be done using another decorator - @chunked, which is specialized in argument chunking. Let's do that!

```python
@parameterized
```

```python
>>> @njit(nogil=True)
... def pt_pipeline_mult_nb(
...     n_params: int,  # (1)!
...     open:     tp.Array2d,  # (2)!
...     high:     tp.Array2d, 
...     low:      tp.Array2d, 
...     close:    tp.Array2d,
...     window:   tp.FlexArray1dLike = WINDOW,  # (3)!
...     upper:    tp.FlexArray1dLike = UPPER,
...     lower:    tp.FlexArray1dLike = LOWER,
...     sl_stop:  tp.FlexArray1dLike = np.nan,
...     tsl_stop: tp.FlexArray1dLike = np.nan,
...     tp_stop:  tp.FlexArray1dLike = np.nan,
... ):
...     window_ = vbt.to_1d_array_nb(np.asarray(window))  # (4)!
...     upper_ = vbt.to_1d_array_nb(np.asarray(upper))
...     lower_ = vbt.to_1d_array_nb(np.asarray(lower))
...     sl_stop_ = vbt.to_1d_array_nb(np.asarray(sl_stop))
...     tsl_stop_ = vbt.to_1d_array_nb(np.asarray(tsl_stop))
...     tp_stop_ = vbt.to_1d_array_nb(np.asarray(tp_stop))
...
...     total_return = np.empty(n_params, dtype=float_)  # (5)!
...     expectancy = np.empty(n_params, dtype=float_)
...
...     for i in range(n_params):  # (6)!
...         total_return[i], expectancy[i] = pt_pipeline_nb(
...             open,
...             high,
...             low,
...             close,
...             window=vbt.flex_select_1d_nb(window_, i),  # (7)!
...             upper=vbt.flex_select_1d_nb(upper_, i),
...             lower=vbt.flex_select_1d_nb(lower_, i),
...             sl_stop=vbt.flex_select_1d_nb(sl_stop_, i),
...             tsl_stop=vbt.flex_select_1d_nb(tsl_stop_, i),
...             tp_stop=vbt.flex_select_1d_nb(tp_stop_, i),
...         )
...     return total_return, expectancy
```

```python
>>> @njit(nogil=True)
... def pt_pipeline_mult_nb(
...     n_params: int,  # (1)!
...     open:     tp.Array2d,  # (2)!
...     high:     tp.Array2d, 
...     low:      tp.Array2d, 
...     close:    tp.Array2d,
...     window:   tp.FlexArray1dLike = WINDOW,  # (3)!
...     upper:    tp.FlexArray1dLike = UPPER,
...     lower:    tp.FlexArray1dLike = LOWER,
...     sl_stop:  tp.FlexArray1dLike = np.nan,
...     tsl_stop: tp.FlexArray1dLike = np.nan,
...     tp_stop:  tp.FlexArray1dLike = np.nan,
... ):
...     window_ = vbt.to_1d_array_nb(np.asarray(window))  # (4)!
...     upper_ = vbt.to_1d_array_nb(np.asarray(upper))
...     lower_ = vbt.to_1d_array_nb(np.asarray(lower))
...     sl_stop_ = vbt.to_1d_array_nb(np.asarray(sl_stop))
...     tsl_stop_ = vbt.to_1d_array_nb(np.asarray(tsl_stop))
...     tp_stop_ = vbt.to_1d_array_nb(np.asarray(tp_stop))
...
...     total_return = np.empty(n_params, dtype=float_)  # (5)!
...     expectancy = np.empty(n_params, dtype=float_)
...
...     for i in range(n_params):  # (6)!
...         total_return[i], expectancy[i] = pt_pipeline_nb(
...             open,
...             high,
...             low,
...             close,
...             window=vbt.flex_select_1d_nb(window_, i),  # (7)!
...             upper=vbt.flex_select_1d_nb(upper_, i),
...             lower=vbt.flex_select_1d_nb(lower_, i),
...             sl_stop=vbt.flex_select_1d_nb(sl_stop_, i),
...             tsl_stop=vbt.flex_select_1d_nb(tsl_stop_, i),
...             tp_stop=vbt.flex_select_1d_nb(tp_stop_, i),
...         )
...     return total_return, expectancy
```

```python
Array2d
```

```python
FlexArray1dLike
```

```python
n_params
```

The magic of the function above is that we don't need to make arrays out of all parameters: thanks to flexible indexing, we can pass some parameters as arrays, and keep some at their defaults. For example, let's test three window combinations:

```python
>>> pt_pipeline_mult_nb(
...     3,
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values,
...     window=np.array([10, 20, 30])
... )
(array([ 0.11131525, -0.04819178,  0.13124959]),
 array([ 0.01039436, -0.00483853,  0.01756337]))
```

```python
>>> pt_pipeline_mult_nb(
...     3,
...     data.open.values,
...     data.high.values,
...     data.low.values,
...     data.close.values,
...     window=np.array([10, 20, 30])
... )
(array([ 0.11131525, -0.04819178,  0.13124959]),
 array([ 0.01039436, -0.00483853,  0.01756337]))
```

Next, we'll wrap the function with the @chunked decorator. Not only we need to specify the chunks as we did in @parameterized, but we also need to specify where to get the total number of parameter combinations from, and how to split each single argument. To assist us in this matter, vectorbt has its own collection of annotation classes. For instance, we can instruct @chunked to take the total number of combinations from the argument n_params by annotating this argument with the class ArgSizer. Then, we can annotate each parameter as a flexible one-dimensional array using the class FlexArraySlicer: whenever @chunked builds a new chunk, it will "slice" the corresponding subset of values from each parameter array.

```python
@chunked
```

```python
@parameterized
```

```python
@chunked
```

```python
n_params
```

```python
@chunked
```

```python
>>> chunked_pt_pipeline = vbt.chunked(
...     pt_pipeline_mult_nb,
...     size=vbt.ArgSizer(arg_query="n_params"),
...     arg_take_spec=dict(
...         n_params=vbt.CountAdapter(),
...         open=None,  # (1)!
...         high=None,
...         low=None,
...         close=None,
...         window=vbt.FlexArraySlicer(),
...         upper=vbt.FlexArraySlicer(),
...         lower=vbt.FlexArraySlicer(),
...         sl_stop=vbt.FlexArraySlicer(),
...         tsl_stop=vbt.FlexArraySlicer(),
...         tp_stop=vbt.FlexArraySlicer()
...     ),
...     chunk_len=1000,  # (2)!
...     merge_func="concat",  # (3)!
...     execute_kwargs=dict(  # (4)!
...         chunk_len="auto",
...         engine="threadpool"
...     )
... )
```

```python
>>> chunked_pt_pipeline = vbt.chunked(
...     pt_pipeline_mult_nb,
...     size=vbt.ArgSizer(arg_query="n_params"),
...     arg_take_spec=dict(
...         n_params=vbt.CountAdapter(),
...         open=None,  # (1)!
...         high=None,
...         low=None,
...         close=None,
...         window=vbt.FlexArraySlicer(),
...         upper=vbt.FlexArraySlicer(),
...         lower=vbt.FlexArraySlicer(),
...         sl_stop=vbt.FlexArraySlicer(),
...         tsl_stop=vbt.FlexArraySlicer(),
...         tp_stop=vbt.FlexArraySlicer()
...     ),
...     chunk_len=1000,  # (2)!
...     merge_func="concat",  # (3)!
...     execute_kwargs=dict(  # (4)!
...         chunk_len="auto",
...         engine="threadpool"
...     )
... )
```

```python
pt_pipeline_mult_nb
```

Here's what happens: whenever we call chunked_pt_pipeline, it takes the total number of parameter combinations from the argument n_params. It then builds chunks of the length 1,000 and slices each chunk from the parameter arguments; one chunk corresponds to one call of pt_pipeline_mult_nb. Then, to parallelize the execution of chunks, we put them into super chunks. Chunks within a super chunk are executed in parallel, while super chunks themselves are executed serially; that's why the progress bar shows the progress of super chunks.

```python
chunked_pt_pipeline
```

```python
n_params
```

```python
pt_pipeline_mult_nb
```

Let's build the full parameter grid and run our function:

```python
>>> param_product, param_index = vbt.combine_params(  # (1)!
...     dict(
...         window=vbt.Param(WINDOW_SPACE),  # (2)!
...         upper=vbt.Param(UPPER_SPACE),
...         lower=vbt.Param(LOWER_SPACE)
...     )
... )

>>> COPT_FILE = "temp/chunked_opt.pickle"

>>> # vbt.remove_file(COPT_FILE, missing_ok=True)
>>> if not vbt.file_exists(COPT_FILE):
...     chunked_opt = chunked_pt_pipeline(
...         len(param_index),  # (3)!
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=param_product["window"],
...         upper=param_product["upper"],
...         lower=param_product["lower"]
...     )
...     vbt.save(chunked_opt, COPT_FILE)
... else:
...     chunked_opt = vbt.load(COPT_FILE)
```

```python
>>> param_product, param_index = vbt.combine_params(  # (1)!
...     dict(
...         window=vbt.Param(WINDOW_SPACE),  # (2)!
...         upper=vbt.Param(UPPER_SPACE),
...         lower=vbt.Param(LOWER_SPACE)
...     )
... )

>>> COPT_FILE = "temp/chunked_opt.pickle"

>>> # vbt.remove_file(COPT_FILE, missing_ok=True)
>>> if not vbt.file_exists(COPT_FILE):
...     chunked_opt = chunked_pt_pipeline(
...         len(param_index),  # (3)!
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=param_product["window"],
...         upper=param_product["upper"],
...         lower=param_product["lower"]
...     )
...     vbt.save(chunked_opt, COPT_FILE)
... else:
...     chunked_opt = vbt.load(COPT_FILE)
```

```python
@parameterized
```

Super chunk 56/56

Super chunk 56/56

This runs in 40% less time than previously because the overhead of spawning a thread now weights much less compared to the higher workload in that thread. Also, we don't sacrifice any more RAM for this speedup since still only one parameter combination is processed at a time in pt_pipeline_mult_nb.

```python
pt_pipeline_mult_nb
```

```python
>>> total_return, expectancy = chunked_opt

>>> total_return = pd.Series(total_return, index=param_index)  # (1)!
>>> expectancy = pd.Series(expectancy, index=param_index)
```

```python
>>> total_return, expectancy = chunked_opt

>>> total_return = pd.Series(total_return, index=param_index)  # (1)!
>>> expectancy = pd.Series(expectancy, index=param_index)
```

We have just two limitations left: generating parameter combinations takes a considerable amount of time, and also, if the execution interrupts, all the optimization results will be lost. These issues can be mitigated by creating a while-loop that at each iteration generates a subset of parameter combinations, executes them, and then caches the results to disk. It runs until all the combinations are successfully processed. Another advantage: we can continue from the point where the execution stopped the last time, and even run the optimization procedure until some satisfactory set of parameters is found!

Let's say we want to include the stop parameters as well, but execute just a subset of random parameter combinations. Generating them with combine_params wouldn't be possible:

```python
>>> GRID_LEN = len(WINDOW_SPACE) * \
...     len(UPPER_SPACE) * \
...     len(LOWER_SPACE) * \
...     len(STOP_SPACE) ** 3
>>> print(GRID_LEN)
441045000000
```

```python
>>> GRID_LEN = len(WINDOW_SPACE) * \
...     len(UPPER_SPACE) * \
...     len(LOWER_SPACE) * \
...     len(STOP_SPACE) ** 3
>>> print(GRID_LEN)
441045000000
```

We've got a half of a trillion parameter combinations

Instead, let's pick our parameter combinations in a smart way: use a function pick_from_param_grid that takes a dictionary of parameter spaces (order matters!) and the position of the parameter combination of interest, and returns the actual parameter combination that corresponds to that position. For example, pick the combination under the index 123,456,789:

```python
>>> GRID = dict(
...     window=WINDOW_SPACE,
...     upper=UPPER_SPACE,
...     lower=LOWER_SPACE,
...     sl_stop=STOP_SPACE,
...     tsl_stop=STOP_SPACE,
...     tp_stop=STOP_SPACE,
... )
>>> vbt.pprint(vbt.pick_from_param_grid(GRID, 123_456_789))
dict(
    window=5,
    upper=3.090232306167813,
    lower=-2.241402727604947,
    sl_stop=45.0,
    tsl_stop=67.0,
    tp_stop=89.0
)
```

```python
>>> GRID = dict(
...     window=WINDOW_SPACE,
...     upper=UPPER_SPACE,
...     lower=LOWER_SPACE,
...     sl_stop=STOP_SPACE,
...     tsl_stop=STOP_SPACE,
...     tp_stop=STOP_SPACE,
... )
>>> vbt.pprint(vbt.pick_from_param_grid(GRID, 123_456_789))
dict(
    window=5,
    upper=3.090232306167813,
    lower=-2.241402727604947,
    sl_stop=45.0,
    tsl_stop=67.0,
    tp_stop=89.0
)
```

It's exactly the same parameter combination as if we combined all the parameters and did param_index[123_456_789], but with almost zero performance and memory overhead!

```python
param_index[123_456_789]
```

We can now construct our while-loop. Let's do a random parameter search until we get at least 100 values with the expectancy of 1 or more!

```python
>>> FOUND_FILE = "temp/found.pickle"
>>> BEST_N = 100  # (1)!
>>> BEST_TH = 1.0  # (2)!
>>> CHUNK_LEN = 10_000  # (3)!

>>> # vbt.remove_file(FOUND_FILE, missing_ok=True)
>>> if vbt.file_exists(FOUND_FILE):
...     found = vbt.load(FOUND_FILE)  # (4)!
>>> else:
...     found = None
>>> with (  # (5)!
...     vbt.ProgressBar(
...         desc="Found", 
...         initial=0 if found is None else len(found),
...         total=BEST_N
...     ) as pbar1,
...     vbt.ProgressBar(
...         desc="Processed"
...     ) as pbar2
... ):
...     while found is None or len(found) < BEST_N:  # (6)!
...         param_df = pd.DataFrame([
...             vbt.pick_from_param_grid(GRID)  # (7)!
...             for _ in range(CHUNK_LEN)
...         ])
...         param_index = pd.MultiIndex.from_frame(param_df)
...         _, expectancy = chunked_pt_pipeline(  # (8)!
...             CHUNK_LEN,
...             data.open.values,
...             data.high.values,
...             data.low.values,
...             data.close.values,
...             window=param_df["window"],
...             upper=param_df["upper"],
...             lower=param_df["lower"],
...             sl_stop=param_df["sl_stop"],
...             tsl_stop=param_df["tsl_stop"],
...             tp_stop=param_df["tp_stop"],
...             _chunk_len=None,
...             _execute_kwargs=dict(
...                 chunk_len=None
...             )
...         )
...         expectancy = pd.Series(expectancy, index=param_index)
...         best_mask = expectancy >= BEST_TH
...         if best_mask.any():  # (9)!
...             best = expectancy[best_mask]
...             if found is None:
...                 found = best
...             else:
...                 found = pd.concat((found, best))
...                 found = found[~found.index.duplicated(keep="first")]
...             vbt.save(found, FOUND_FILE)
...             pbar1.update_to(len(found))
...             pbar1.refresh()
...         pbar2.update(len(expectancy))
```

```python
>>> FOUND_FILE = "temp/found.pickle"
>>> BEST_N = 100  # (1)!
>>> BEST_TH = 1.0  # (2)!
>>> CHUNK_LEN = 10_000  # (3)!

>>> # vbt.remove_file(FOUND_FILE, missing_ok=True)
>>> if vbt.file_exists(FOUND_FILE):
...     found = vbt.load(FOUND_FILE)  # (4)!
>>> else:
...     found = None
>>> with (  # (5)!
...     vbt.ProgressBar(
...         desc="Found", 
...         initial=0 if found is None else len(found),
...         total=BEST_N
...     ) as pbar1,
...     vbt.ProgressBar(
...         desc="Processed"
...     ) as pbar2
... ):
...     while found is None or len(found) < BEST_N:  # (6)!
...         param_df = pd.DataFrame([
...             vbt.pick_from_param_grid(GRID)  # (7)!
...             for _ in range(CHUNK_LEN)
...         ])
...         param_index = pd.MultiIndex.from_frame(param_df)
...         _, expectancy = chunked_pt_pipeline(  # (8)!
...             CHUNK_LEN,
...             data.open.values,
...             data.high.values,
...             data.low.values,
...             data.close.values,
...             window=param_df["window"],
...             upper=param_df["upper"],
...             lower=param_df["lower"],
...             sl_stop=param_df["sl_stop"],
...             tsl_stop=param_df["tsl_stop"],
...             tp_stop=param_df["tp_stop"],
...             _chunk_len=None,
...             _execute_kwargs=dict(
...                 chunk_len=None
...             )
...         )
...         expectancy = pd.Series(expectancy, index=param_index)
...         best_mask = expectancy >= BEST_TH
...         if best_mask.any():  # (9)!
...             best = expectancy[best_mask]
...             if found is None:
...                 found = best
...             else:
...                 found = pd.concat((found, best))
...                 found = found[~found.index.duplicated(keep="first")]
...             vbt.save(found, FOUND_FILE)
...             pbar1.update_to(len(found))
...             pbar1.refresh()
...         pbar2.update(len(expectancy))
```

```python
CHUNK_LEN
```

Found 100/100

Found 100/100

We can now run the cell above, interrupt it, and continue with the execution at a later time

By having put similar parameter combinations into the same bucket, we can also aggregate them to derive a single optimal parameter combination:

```python
>>> def get_param_median(param):  # (1)!
...     return found.index.get_level_values(param).to_series().median()

>>> pt_pipeline_nb(
...     data.open.values, 
...     data.high.values, 
...     data.low.values, 
...     data.close.values,
...     window=int(get_param_median("window")),
...     upper=get_param_median("upper"),
...     lower=get_param_median("lower"),
...     sl_stop=get_param_median("sl_stop"),
...     tsl_stop=get_param_median("tsl_stop"),
...     tp_stop=get_param_median("tp_stop")
... )
(0.24251123364060986, 1.7316489316495804)
```

```python
>>> def get_param_median(param):  # (1)!
...     return found.index.get_level_values(param).to_series().median()

>>> pt_pipeline_nb(
...     data.open.values, 
...     data.high.values, 
...     data.low.values, 
...     data.close.values,
...     window=int(get_param_median("window")),
...     upper=get_param_median("upper"),
...     lower=get_param_median("lower"),
...     sl_stop=get_param_median("sl_stop"),
...     tsl_stop=get_param_median("tsl_stop"),
...     tp_stop=get_param_median("tp_stop")
... )
(0.24251123364060986, 1.7316489316495804)
```

We can see that the median parameter combination satisfies our expectancy condition as well.

But sometimes, there's no need to test for so many parameter combinations, we can just use an established parameter optimization framework like Optuna. The advantages of this approach are on hand: we can use the original pt_pipeline_nb function without any decorators, we don't need to handle large parameter grids, and we can use various statistical approaches to both increase the effectiveness of the search and decrease the number of parameter combinations that we need to test.

```python
pt_pipeline_nb
```

Info

Make sure to install Optuna before running the following cell.

```python
>>> import optuna

>>> optuna.logging.disable_default_handler()
>>> optuna.logging.set_verbosity(optuna.logging.WARNING)

>>> def objective(trial):
...     window = trial.suggest_categorical("window", WINDOW_SPACE)  # (1)!
...     upper = trial.suggest_categorical("upper", UPPER_SPACE)  # (2)!
...     lower = trial.suggest_categorical("lower", LOWER_SPACE)
...     sl_stop = trial.suggest_categorical("sl_stop", STOP_SPACE)
...     tsl_stop = trial.suggest_categorical("tsl_stop", STOP_SPACE)
...     tp_stop = trial.suggest_categorical("tp_stop", STOP_SPACE)
...     total_return, expectancy = pt_pipeline_nb(
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=window,
...         upper=upper,
...         lower=lower,
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop
...     )
...     if np.isnan(total_return):
...         raise optuna.TrialPruned()  # (3)!
...     if np.isnan(expectancy):
...         raise optuna.TrialPruned()
...     return total_return, expectancy

>>> study = optuna.create_study(directions=["maximize", "maximize"])  # (4)!
>>> study.optimize(objective, n_trials=1000)  # (5)!

>>> trials_df = study.trials_dataframe(attrs=["params", "values"])
>>> trials_df.set_index([
...     "params_window", 
...     "params_upper", 
...     "params_lower",
...     "params_sl_stop",
...     "params_tsl_stop",
...     "params_tp_stop"
... ], inplace=True)
>>> trials_df.index.rename([
...     "window", 
...     "upper", 
...     "lower",
...     "sl_stop",
...     "tsl_stop",
...     "tp_stop"
... ], inplace=True)
>>> trials_df.columns = ["total_return", "expectancy"]
>>> trials_df = trials_df[~trials_df.index.duplicated(keep="first")]
>>> print(trials_df.sort_values(by="total_return", ascending=False))
                                                    total_return  expectancy
window upper    lower     sl_stop tsl_stop tp_stop                          
44     1.746485 -3.072346 9.0     67.0     55.0         0.558865    0.184924
42     1.871392 -3.286737 53.0    98.0     55.0         0.500062    0.330489
                                           77.0         0.496029    0.334432
5      1.746485 -1.759648 76.0    94.0     45.0         0.492721    0.043832
43     1.807618 -3.192280 87.0    36.0     60.0         0.475732    0.229682
...                                                          ...         ...
7      2.639845 -3.072346 80.0    95.0     55.0              NaN         NaN
5      3.117886 -3.072346 78.0    90.0     47.0              NaN         NaN
       2.769169 -3.072346 78.0    90.0     55.0              NaN         NaN
7      3.098951 -3.072346 78.0    95.0     55.0              NaN         NaN
       2.607536 -3.072346 78.0    95.0     77.0              NaN         NaN

[892 rows x 2 columns]
```

```python
>>> import optuna

>>> optuna.logging.disable_default_handler()
>>> optuna.logging.set_verbosity(optuna.logging.WARNING)

>>> def objective(trial):
...     window = trial.suggest_categorical("window", WINDOW_SPACE)  # (1)!
...     upper = trial.suggest_categorical("upper", UPPER_SPACE)  # (2)!
...     lower = trial.suggest_categorical("lower", LOWER_SPACE)
...     sl_stop = trial.suggest_categorical("sl_stop", STOP_SPACE)
...     tsl_stop = trial.suggest_categorical("tsl_stop", STOP_SPACE)
...     tp_stop = trial.suggest_categorical("tp_stop", STOP_SPACE)
...     total_return, expectancy = pt_pipeline_nb(
...         data.open.values,
...         data.high.values,
...         data.low.values,
...         data.close.values,
...         window=window,
...         upper=upper,
...         lower=lower,
...         sl_stop=sl_stop,
...         tsl_stop=tsl_stop,
...         tp_stop=tp_stop
...     )
...     if np.isnan(total_return):
...         raise optuna.TrialPruned()  # (3)!
...     if np.isnan(expectancy):
...         raise optuna.TrialPruned()
...     return total_return, expectancy

>>> study = optuna.create_study(directions=["maximize", "maximize"])  # (4)!
>>> study.optimize(objective, n_trials=1000)  # (5)!

>>> trials_df = study.trials_dataframe(attrs=["params", "values"])
>>> trials_df.set_index([
...     "params_window", 
...     "params_upper", 
...     "params_lower",
...     "params_sl_stop",
...     "params_tsl_stop",
...     "params_tp_stop"
... ], inplace=True)
>>> trials_df.index.rename([
...     "window", 
...     "upper", 
...     "lower",
...     "sl_stop",
...     "tsl_stop",
...     "tp_stop"
... ], inplace=True)
>>> trials_df.columns = ["total_return", "expectancy"]
>>> trials_df = trials_df[~trials_df.index.duplicated(keep="first")]
>>> print(trials_df.sort_values(by="total_return", ascending=False))
                                                    total_return  expectancy
window upper    lower     sl_stop tsl_stop tp_stop                          
44     1.746485 -3.072346 9.0     67.0     55.0         0.558865    0.184924
42     1.871392 -3.286737 53.0    98.0     55.0         0.500062    0.330489
                                           77.0         0.496029    0.334432
5      1.746485 -1.759648 76.0    94.0     45.0         0.492721    0.043832
43     1.807618 -3.192280 87.0    36.0     60.0         0.475732    0.229682
...                                                          ...         ...
7      2.639845 -3.072346 80.0    95.0     55.0              NaN         NaN
5      3.117886 -3.072346 78.0    90.0     47.0              NaN         NaN
       2.769169 -3.072346 78.0    90.0     55.0              NaN         NaN
7      3.098951 -3.072346 78.0    95.0     55.0              NaN         NaN
       2.607536 -3.072346 78.0    95.0     77.0              NaN         NaN

[892 rows x 2 columns]
```

```python
suggest_int
```

```python
suggest_float
```

The only downside to this approach is that we are limited by the picked results and are not able to explore the entire parameter landscape - the ability that vectorbt truly stands for.

### Level: Architect ¶

Let's say that we want to have the full control over the execution, for example, to execute at most one rebalancing operation in N days. Furthermore, we want to restrain from pre-calculating arrays and do everything in an event-driven fashion. For the sake of simplicity, let's switch our signaling algorithm from the cointegration with OLS to a basic distance measure: log prices.

We'll implement the strategy as a custom signal function in Portfolio.from_signals, which is the trickiest approach because the signal function is run per column while we want to make a decision per segment (i.e., just once for both columns). A worthy idea would be doing the calculations under the first column that is being processed at this bar, writing the results to some temporary arrays, and then accessing them under each column to return the signals. A perfect place for storing such arrays is a built-in named tuple in_outputs, which can be accessed both during the simulation phase and during the analysis phase.

```python
in_outputs
```

```python
>>> InOutputs = namedtuple("InOutputs", ["spread", "zscore"])  # (1)!

>>> @njit(nogil=True, boundscheck=True)  # (2)!
... def can_execute_nb(c, wait_days):  # (3)!
...     if c.order_counts[c.col] == 0:  # (4)!
...         return True
...     last_order = c.order_records[c.order_counts[c.col] - 1, c.col]  # (5)!
...     ns_delta = c.index[c.i] - c.index[last_order.idx]  # (6)!
...     if ns_delta >= wait_days * vbt.dt_nb.d_ns:  # (7)!
...         return True
...     return False

>>> @njit(nogil=True, boundscheck=True)
... def create_signals_nb(c, upper, lower, wait_days):  # (8)!
...     _upper = vbt.pf_nb.select_nb(c, upper)  # (9)!
...     _lower = vbt.pf_nb.select_nb(c, lower)
...     _wait_days = vbt.pf_nb.select_nb(c, wait_days)
...
...     if c.i > 0:  # (10)!
...         prev_zscore = c.in_outputs.zscore[c.i - 1, c.group]
...         zscore = c.in_outputs.zscore[c.i, c.group]
...         if prev_zscore < _upper and zscore > _upper:  # (11)!
...             if can_execute_nb(c, _wait_days):
...                 if c.col % 2 == 0:
...                     return False, False, True, False  # (12)!
...                 return True, False, False, False  # (13)!
...         if prev_zscore > _lower and zscore < _lower:
...             if can_execute_nb(c, _wait_days):
...                 if c.col % 2 == 0:
...                     return True, False, False, False
...                 return False, False, True, False
...     return False, False, False, False  # (14)!

>>> @njit(nogil=True, boundscheck=True)
... def signal_func_nb(c, window, upper, lower, wait_days):  # (15)!
...     _window = vbt.pf_nb.select_nb(c, window)
...         
...     if c.col % 2 == 0:  # (16)!
...         x = vbt.pf_nb.select_nb(c, c.close, col=c.col)  # (17)!
...         y = vbt.pf_nb.select_nb(c, c.close, col=c.col + 1)
...         c.in_outputs.spread[c.i, c.group] = np.log(y) - np.log(x)  # (18)!
...         
...         window_start = c.i - _window + 1  # (19)!
...         window_end = c.i + 1  # (20)!
...         if window_start >= 0:  # (21)!
...             s = c.in_outputs.spread[window_start : window_end, c.group]
...             s_mean = np.nanmean(s)
...             s_std = np.nanstd(s)
...             c.in_outputs.zscore[c.i, c.group] = (s[-1] - s_mean) / s_std
...     return create_signals_nb(c, upper, lower, wait_days)
```

```python
>>> InOutputs = namedtuple("InOutputs", ["spread", "zscore"])  # (1)!

>>> @njit(nogil=True, boundscheck=True)  # (2)!
... def can_execute_nb(c, wait_days):  # (3)!
...     if c.order_counts[c.col] == 0:  # (4)!
...         return True
...     last_order = c.order_records[c.order_counts[c.col] - 1, c.col]  # (5)!
...     ns_delta = c.index[c.i] - c.index[last_order.idx]  # (6)!
...     if ns_delta >= wait_days * vbt.dt_nb.d_ns:  # (7)!
...         return True
...     return False

>>> @njit(nogil=True, boundscheck=True)
... def create_signals_nb(c, upper, lower, wait_days):  # (8)!
...     _upper = vbt.pf_nb.select_nb(c, upper)  # (9)!
...     _lower = vbt.pf_nb.select_nb(c, lower)
...     _wait_days = vbt.pf_nb.select_nb(c, wait_days)
...
...     if c.i > 0:  # (10)!
...         prev_zscore = c.in_outputs.zscore[c.i - 1, c.group]
...         zscore = c.in_outputs.zscore[c.i, c.group]
...         if prev_zscore < _upper and zscore > _upper:  # (11)!
...             if can_execute_nb(c, _wait_days):
...                 if c.col % 2 == 0:
...                     return False, False, True, False  # (12)!
...                 return True, False, False, False  # (13)!
...         if prev_zscore > _lower and zscore < _lower:
...             if can_execute_nb(c, _wait_days):
...                 if c.col % 2 == 0:
...                     return True, False, False, False
...                 return False, False, True, False
...     return False, False, False, False  # (14)!

>>> @njit(nogil=True, boundscheck=True)
... def signal_func_nb(c, window, upper, lower, wait_days):  # (15)!
...     _window = vbt.pf_nb.select_nb(c, window)
...         
...     if c.col % 2 == 0:  # (16)!
...         x = vbt.pf_nb.select_nb(c, c.close, col=c.col)  # (17)!
...         y = vbt.pf_nb.select_nb(c, c.close, col=c.col + 1)
...         c.in_outputs.spread[c.i, c.group] = np.log(y) - np.log(x)  # (18)!
...         
...         window_start = c.i - _window + 1  # (19)!
...         window_end = c.i + 1  # (20)!
...         if window_start >= 0:  # (21)!
...             s = c.in_outputs.spread[window_start : window_end, c.group]
...             s_mean = np.nanmean(s)
...             s_std = np.nanstd(s)
...             c.in_outputs.zscore[c.i, c.group] = (s[-1] - s_mean) / s_std
...     return create_signals_nb(c, upper, lower, wait_days)
```

```python
wait_days
```

```python
wait_days
```

```python
wait_days
```

Next, create a pipeline that runs the simulation from the signal function above:

```python
>>> WAIT_DAYS = 30

>>> def iter_pt_portfolio(
...     window=WINDOW, 
...     upper=UPPER, 
...     lower=LOWER, 
...     wait_days=WAIT_DAYS,
...     signal_func_nb=signal_func_nb,  # (1)!
...     more_signal_args=(),
...     **kwargs
... ):
...     return vbt.Portfolio.from_signals(
...         data,
...         broadcast_named_args=dict(  # (2)!
...             window=window,
...             upper=upper,
...             lower=lower,
...             wait_days=wait_days
...         ),
...         in_outputs=vbt.RepEval("""
...             InOutputs(
...                 np.full((target_shape[0], target_shape[1] // 2), np.nan), 
...                 np.full((target_shape[0], target_shape[1] // 2), np.nan)
...             )
...         """, context=dict(InOutputs=InOutputs)),  # (3)!
...         signal_func_nb=signal_func_nb,  # (4)!
...         signal_args=(  # (5)!
...             vbt.Rep("window"),  # (6)!
...             vbt.Rep("upper"),
...             vbt.Rep("lower"),
...             vbt.Rep("wait_days"),
...             *more_signal_args
...         ),
...         size=10,
...         size_type="valuepercent100",
...         group_by=vbt.ExceptLevel("symbol"),
...         cash_sharing=True,
...         call_seq="auto",
...         delta_format="percent100",
...         stop_exit_price="close",
...         **kwargs
...     )

>>> pf = iter_pt_portfolio()
```

```python
>>> WAIT_DAYS = 30

>>> def iter_pt_portfolio(
...     window=WINDOW, 
...     upper=UPPER, 
...     lower=LOWER, 
...     wait_days=WAIT_DAYS,
...     signal_func_nb=signal_func_nb,  # (1)!
...     more_signal_args=(),
...     **kwargs
... ):
...     return vbt.Portfolio.from_signals(
...         data,
...         broadcast_named_args=dict(  # (2)!
...             window=window,
...             upper=upper,
...             lower=lower,
...             wait_days=wait_days
...         ),
...         in_outputs=vbt.RepEval("""
...             InOutputs(
...                 np.full((target_shape[0], target_shape[1] // 2), np.nan), 
...                 np.full((target_shape[0], target_shape[1] // 2), np.nan)
...             )
...         """, context=dict(InOutputs=InOutputs)),  # (3)!
...         signal_func_nb=signal_func_nb,  # (4)!
...         signal_args=(  # (5)!
...             vbt.Rep("window"),  # (6)!
...             vbt.Rep("upper"),
...             vbt.Rep("lower"),
...             vbt.Rep("wait_days"),
...             *more_signal_args
...         ),
...         size=10,
...         size_type="valuepercent100",
...         group_by=vbt.ExceptLevel("symbol"),
...         cash_sharing=True,
...         call_seq="auto",
...         delta_format="percent100",
...         stop_exit_price="close",
...         **kwargs
...     )

>>> pf = iter_pt_portfolio()
```

```python
c
```

```python
signal_func_nb
```

```python
broadcast_named_args
```

Let's visually validate our implementation:

```python
>>> fig = vbt.make_subplots(  # (1)!
...     rows=2, 
...     cols=1, 
...     vertical_spacing=0,
...     shared_xaxes=True
... )
>>> zscore = pf.get_in_output("zscore").rename("Z-score")  # (2)!
>>> zscore.vbt.plot(  # (3)!
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> fig.add_hline(row=1, y=UPPER, line_color="orangered", line_dash="dot")
>>> fig.add_hline(row=1, y=0, line_color="yellow", line_dash="dot")
>>> fig.add_hline(row=1, y=LOWER, line_color="limegreen", line_dash="dot")
>>> orders = pf.orders.regroup(group_by=False).iloc[:, 0]  # (4)!
>>> exit_mask = orders.side_sell.get_pd_mask(idx_arr="signal_idx")  # (5)!
>>> entry_mask = orders.side_buy.get_pd_mask(idx_arr="signal_idx")  # (6)!
>>> upper_crossed = zscore.vbt.crossed_above(UPPER)
>>> lower_crossed = zscore.vbt.crossed_below(LOWER)
>>> (upper_crossed & ~exit_mask).vbt.signals.plot_as_exits(  # (7)!
...     pf.get_in_output("zscore"),
...     trace_kwargs=dict(
...         name="Exits (ignored)", 
...         marker=dict(color="lightgray"), 
...         opacity=0.5
...     ),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> (lower_crossed & ~entry_mask).vbt.signals.plot_as_entries(
...     pf.get_in_output("zscore"),
...     trace_kwargs=dict(
...         name="Entries (ignored)", 
...         marker=dict(color="lightgray"), 
...         opacity=0.5
...     ),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> exit_mask.vbt.signals.plot_as_exits(  # (8)!
...     pf.get_in_output("zscore"),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> entry_mask.vbt.signals.plot_as_entries(
...     pf.get_in_output("zscore"),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> pf.plot_allocations(  # (9)!
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> rebalancing_dates = data.index[np.unique(orders.idx.values)]
>>> for date in rebalancing_dates:
...     fig.add_vline(row=2, x=date, line_color="teal", line_dash="dot")
>>> fig.update_layout(height=600)
>>> fig.show()
```

```python
>>> fig = vbt.make_subplots(  # (1)!
...     rows=2, 
...     cols=1, 
...     vertical_spacing=0,
...     shared_xaxes=True
... )
>>> zscore = pf.get_in_output("zscore").rename("Z-score")  # (2)!
>>> zscore.vbt.plot(  # (3)!
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> fig.add_hline(row=1, y=UPPER, line_color="orangered", line_dash="dot")
>>> fig.add_hline(row=1, y=0, line_color="yellow", line_dash="dot")
>>> fig.add_hline(row=1, y=LOWER, line_color="limegreen", line_dash="dot")
>>> orders = pf.orders.regroup(group_by=False).iloc[:, 0]  # (4)!
>>> exit_mask = orders.side_sell.get_pd_mask(idx_arr="signal_idx")  # (5)!
>>> entry_mask = orders.side_buy.get_pd_mask(idx_arr="signal_idx")  # (6)!
>>> upper_crossed = zscore.vbt.crossed_above(UPPER)
>>> lower_crossed = zscore.vbt.crossed_below(LOWER)
>>> (upper_crossed & ~exit_mask).vbt.signals.plot_as_exits(  # (7)!
...     pf.get_in_output("zscore"),
...     trace_kwargs=dict(
...         name="Exits (ignored)", 
...         marker=dict(color="lightgray"), 
...         opacity=0.5
...     ),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> (lower_crossed & ~entry_mask).vbt.signals.plot_as_entries(
...     pf.get_in_output("zscore"),
...     trace_kwargs=dict(
...         name="Entries (ignored)", 
...         marker=dict(color="lightgray"), 
...         opacity=0.5
...     ),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> exit_mask.vbt.signals.plot_as_exits(  # (8)!
...     pf.get_in_output("zscore"),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> entry_mask.vbt.signals.plot_as_entries(
...     pf.get_in_output("zscore"),
...     add_trace_kwargs=dict(row=1, col=1),
...     fig=fig
... )
>>> pf.plot_allocations(  # (9)!
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> rebalancing_dates = data.index[np.unique(orders.idx.values)]
>>> for date in rebalancing_dates:
...     fig.add_vline(row=2, x=date, line_color="teal", line_dash="dot")
>>> fig.update_layout(height=600)
>>> fig.show()
```

Great, but what about parameter optimization? Thanks to the fact that we have defined our parameters as flexible arrays, we can pass them in a variety of formats, including Param! Let's discover how the waiting time affects the number of orders:

```python
>>> WAIT_SPACE = np.arange(30, 370, 5).tolist()

>>> pf = iter_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))
>>> pf.orders.count().vbt.scatterplot(
...     xaxis_title="Wait days",
...     yaxis_title="Order count"
... ).show()
```

```python
>>> WAIT_SPACE = np.arange(30, 370, 5).tolist()

>>> pf = iter_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))
>>> pf.orders.count().vbt.scatterplot(
...     xaxis_title="Wait days",
...     yaxis_title="Order count"
... ).show()
```

Note, however, that the optimization approach above is associated with a high RAM usage since the OHLC data will have to be tiled the same number of times as there are parameter combinations; but this is the exactly same consideration as for optimizing sl_stop and other built-in parameters. Also, you might have noticed that both the compilation and execution take (much) longer than before: signal functions cannot be cached, thus, not only the entire simulator must be compiled from scratch with each new runtime, but we also must use an entirely different simulation path than the faster path based on signal arrays that we used before. Furthermore, our z-score implementation is quite slow because the mean and standard deviation must be re-computed for each single bar (remember that the previous OLS indicator was based on one of the fastest algorithms out there).

```python
sl_stop
```

```python
>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     iter_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
8.62 seconds

>>> print(tracer.peak_usage())
306.2 MB
```

```python
>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     iter_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
8.62 seconds

>>> print(tracer.peak_usage())
306.2 MB
```

While the compilation time is hard to manipulate, the execution time can be reduced significantly by replacing both the mean and standard deviation operations with a z-score accumulator, which can compute z-scores incrementally without the need for expensive aggregations. Luckily, there's such an accumulator already implemented in vectorbt - rolling_zscore_acc_nb. The workings are as follows: provide an input state that holds the information required to process the value for this bar, pass it to the accumulator, and in return, get the output state that contains both the calculated z-score and the new information required by the next bar.

The first question is: which information should we store? Generally, we should store the information that the accumulator changes and then expects to be provided at the next bar. The next question is: how to store such an information? A state is usually comprised of a bunch of single values, but named tuples aren't mutable  The trick is to create a structured NumPy array; you can imagine it being a regular NumPy array that holds mutable named tuples. We'll create a one-dimensional array with one tuple per group:

```python
>>> zscore_state_dt = np.dtype(  # (1)!
...     [
...         ("cumsum", float_),
...         ("cumsum_sq", float_),
...         ("nancnt", int_)
...     ],
...     align=True,
... )

>>> @njit(nogil=True, boundscheck=True)
... def stream_signal_func_nb(
...     c, 
...     window, 
...     upper, 
...     lower, 
...     wait_days, 
...     zscore_state  # (2)!
... ):
...     _window = vbt.pf_nb.select_nb(c, window)
...         
...     if c.col % 2 == 0:
...         x = vbt.pf_nb.select_nb(c, c.close, col=c.col)
...         y = vbt.pf_nb.select_nb(c, c.close, col=c.col + 1)
...         c.in_outputs.spread[c.i, c.group] = np.log(y) - np.log(x)
...         
...         value = c.in_outputs.spread[c.i, c.group]  # (3)!
...         pre_i = c.i - _window
...         if pre_i >= 0:
...             pre_window_value = c.in_outputs.spread[pre_i, c.group]  # (4)!
...         else:
...             pre_window_value = np.nan
...         zscore_in_state = vbt.enums.RollZScoreAIS(  # (5)!
...             i=c.i,
...             value=value,
...             pre_window_value=pre_window_value,
...             cumsum=zscore_state["cumsum"][c.group],
...             cumsum_sq=zscore_state["cumsum_sq"][c.group],
...             nancnt=zscore_state["nancnt"][c.group],
...             window=_window,
...             minp=_window,
...             ddof=0
...         )
...         zscore_out_state = vbt.nb.rolling_zscore_acc_nb(zscore_in_state)  # (6)!
...         c.in_outputs.zscore[c.i, c.group] = zscore_out_state.value  # (7)!
...         zscore_state["cumsum"][c.group] = zscore_out_state.cumsum  # (8)!
...         zscore_state["cumsum_sq"][c.group] = zscore_out_state.cumsum_sq
...         zscore_state["nancnt"][c.group] = zscore_out_state.nancnt
...         
...     return create_signals_nb(c, upper, lower, wait_days)
```

```python
>>> zscore_state_dt = np.dtype(  # (1)!
...     [
...         ("cumsum", float_),
...         ("cumsum_sq", float_),
...         ("nancnt", int_)
...     ],
...     align=True,
... )

>>> @njit(nogil=True, boundscheck=True)
... def stream_signal_func_nb(
...     c, 
...     window, 
...     upper, 
...     lower, 
...     wait_days, 
...     zscore_state  # (2)!
... ):
...     _window = vbt.pf_nb.select_nb(c, window)
...         
...     if c.col % 2 == 0:
...         x = vbt.pf_nb.select_nb(c, c.close, col=c.col)
...         y = vbt.pf_nb.select_nb(c, c.close, col=c.col + 1)
...         c.in_outputs.spread[c.i, c.group] = np.log(y) - np.log(x)
...         
...         value = c.in_outputs.spread[c.i, c.group]  # (3)!
...         pre_i = c.i - _window
...         if pre_i >= 0:
...             pre_window_value = c.in_outputs.spread[pre_i, c.group]  # (4)!
...         else:
...             pre_window_value = np.nan
...         zscore_in_state = vbt.enums.RollZScoreAIS(  # (5)!
...             i=c.i,
...             value=value,
...             pre_window_value=pre_window_value,
...             cumsum=zscore_state["cumsum"][c.group],
...             cumsum_sq=zscore_state["cumsum_sq"][c.group],
...             nancnt=zscore_state["nancnt"][c.group],
...             window=_window,
...             minp=_window,
...             ddof=0
...         )
...         zscore_out_state = vbt.nb.rolling_zscore_acc_nb(zscore_in_state)  # (6)!
...         c.in_outputs.zscore[c.i, c.group] = zscore_out_state.value  # (7)!
...         zscore_state["cumsum"][c.group] = zscore_out_state.cumsum  # (8)!
...         zscore_state["cumsum_sq"][c.group] = zscore_out_state.cumsum_sq
...         zscore_state["nancnt"][c.group] = zscore_out_state.nancnt
...         
...     return create_signals_nb(c, upper, lower, wait_days)
```

```python
window
```

Next, we'll adapt the portfolio function to use our new signal function:

```python
>>> stream_pt_portfolio = partial(  # (1)!
...     iter_pt_portfolio,
...     signal_func_nb=stream_signal_func_nb,
...     more_signal_args=(  # (2)!
...         vbt.RepEval(  # (3)!
...             """
...             n_groups = target_shape[1] // 2
...             zscore_state = np.empty(n_groups, dtype=zscore_state_dt)
...             zscore_state["cumsum"] = 0.0
...             zscore_state["cumsum_sq"] = 0.0
...             zscore_state["nancnt"] = 0
...             zscore_state
...             """, 
...             context=dict(zscore_state_dt=zscore_state_dt)
...         ),
...     )
... )
```

```python
>>> stream_pt_portfolio = partial(  # (1)!
...     iter_pt_portfolio,
...     signal_func_nb=stream_signal_func_nb,
...     more_signal_args=(  # (2)!
...         vbt.RepEval(  # (3)!
...             """
...             n_groups = target_shape[1] // 2
...             zscore_state = np.empty(n_groups, dtype=zscore_state_dt)
...             zscore_state["cumsum"] = 0.0
...             zscore_state["cumsum_sq"] = 0.0
...             zscore_state["nancnt"] = 0
...             zscore_state
...             """, 
...             context=dict(zscore_state_dt=zscore_state_dt)
...         ),
...     )
... )
```

```python
iter_pt_portfolio
```

We're ready! Let's build the portfolio and compare to the previous one for validation:

```python
>>> stream_pf = stream_pt_portfolio()
>>> print(stream_pf.total_return)
0.15210165047643728

>>> pf = iter_pt_portfolio()
>>> print(pf.total_return)
0.15210165047643728
```

```python
>>> stream_pf = stream_pt_portfolio()
>>> print(stream_pf.total_return)
0.15210165047643728

>>> pf = iter_pt_portfolio()
>>> print(pf.total_return)
0.15210165047643728
```

Finally, let's benchmark the new portfolio on WAIT_SPACE:

```python
WAIT_SPACE
```

```python
>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     stream_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
1.52 seconds

>>> print(tracer.peak_usage())
306.2 MB
```

```python
>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     stream_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
1.52 seconds

>>> print(tracer.peak_usage())
306.2 MB
```

Info

Run this cell at least two times since the simulation may need to compile the first time. This is because the compilation is required whenever a new unique set of argument types is discovered. One parameter combination and multiple parameter combinations produce two different sets of argument types!

The final optimization to speed up the simulation of multiple parameter combinations is enabling the in-house chunking. There's one catch though: Portfolio.from_signals doesn't know how to chunk any user-defined arrays, thus we need to manually provide the chunking specification for all the arguments in both signal_args and in_outputs:

```python
signal_args
```

```python
in_outputs
```

```python
>>> chunked_stream_pt_portfolio = partial(
...     stream_pt_portfolio,
...     chunked=dict(  # (1)!
...         engine="threadpool",
...         arg_take_spec=dict(
...             signal_args=vbt.ArgsTaker(
...                 vbt.flex_array_gl_slicer,  # (2)!
...                 vbt.flex_array_gl_slicer,
...                 vbt.flex_array_gl_slicer,
...                 vbt.flex_array_gl_slicer,
...                 vbt.ArraySlicer(axis=0)  # (3)!
...             ),
...             in_outputs=vbt.SequenceTaker([
...                 vbt.ArraySlicer(axis=1),  # (4)!
...                 vbt.ArraySlicer(axis=1)
...             ])
...         )
...     )
... )

>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     chunked_stream_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
520.08 milliseconds

>>> print(tracer.peak_usage())
306.4 MB
```

```python
>>> chunked_stream_pt_portfolio = partial(
...     stream_pt_portfolio,
...     chunked=dict(  # (1)!
...         engine="threadpool",
...         arg_take_spec=dict(
...             signal_args=vbt.ArgsTaker(
...                 vbt.flex_array_gl_slicer,  # (2)!
...                 vbt.flex_array_gl_slicer,
...                 vbt.flex_array_gl_slicer,
...                 vbt.flex_array_gl_slicer,
...                 vbt.ArraySlicer(axis=0)  # (3)!
...             ),
...             in_outputs=vbt.SequenceTaker([
...                 vbt.ArraySlicer(axis=1),  # (4)!
...                 vbt.ArraySlicer(axis=1)
...             ])
...         )
...     )
... )

>>> with (vbt.Timer() as timer, vbt.MemTracer() as tracer):
...     chunked_stream_pt_portfolio(wait_days=vbt.Param(WAIT_SPACE))

>>> print(timer.elapsed())
520.08 milliseconds

>>> print(tracer.peak_usage())
306.4 MB
```

```python
@chunked
```

```python
zscore_state
```

```python
spread
```

```python
zscore
```

The optimization is now an order of magnitude faster than before

Hint

To make the execution consume less memory, use the "serial" engine or build super chunks.

## Summary¶

Pairs trading involves two columns with opposite position signs - an almost perfect use case because of the vectorbt's integrated grouping mechanics. And it gets even better: we can implement most pairs trading strategies by using semi-vectorized, iterative, and even streaming approaches alike. The focus of this tutorial is to showcase how a strategy can be incrementally developed and then optimized for both a high strategy performance and low resource consumption.

We started the journey with the discovery phase where we designed and implemented the strategy from the ground up using various high-level tools and with no special regard for its performance. Once the framework of the strategy has been established, we moved over to make the execution faster to discover more lucrative configurations in a shorter span of time. Finally, we decided to flip the table and make the strategy iterative to gain complete control over its execution. But even this doesn't have to be the end of the story: if you're curious enough, you can build your simulator and gain unmatched power that others can just dream of

Python code  Notebook

