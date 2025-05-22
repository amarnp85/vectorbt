# SuperFast SuperTrend¶

While Python is slower than many compiled languages, it's easy to use and extremely diverse. For many, especially in the data science domain, the practicality of the language beats the speed considerations - it's like a Swiss army knife for programmers and researchers alike.

Unfortunately for quants, Python becomes a real bottleneck when iterating over (a large amount of) data. For this reason, there is an entire ecosystem of scientific packages such as NumPy and Pandas, which are highly optimized for performance, with critical code paths often written in Cython or C. Those packages mostly work on arrays, giving us a common interface for processing data in an efficient manner.

This ability is highly appreciated when constructing indicators that can be translated into a set of vectorized operations, such as OBV. But even non-vectorized operations, such as the exponential weighted moving average (EMA) powering numerous indicators such as MACD, were implemented in a compiled language and are offered as a ready-to-use Python function. But sometimes, an indicator is difficult or even impossible to develop solely using standard array operations because the indicator introduces a path dependency, where a decision today depends upon a decision made yesterday. One member of such a family of indicators is SuperTrend.

In this example, you will learn how to design and implement a SuperTrend indicator, and gradually optimize it towards a never-seen performance using TA-Lib and Numba. We will also backtest the newly created indicator on a range of parameters using vectorbt (PRO).

## Data¶

The first step is always getting the (right) data. In particular, we need a sufficient amount of data to benchmark different SuperTrend implementations. Let's pull 2 years of hourly Bitcoin and Ethereum data from Binance using the vectorbt's BinanceData class:

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull(
...     ['BTCUSDT', 'ETHUSDT'], 
...     start='2020-01-01 UTC',
...     end='2022-01-01 UTC',
...     timeframe='1h'
... )
```

```python
>>> from vectorbtpro import *

>>> data = vbt.BinanceData.pull(
...     ['BTCUSDT', 'ETHUSDT'], 
...     start='2020-01-01 UTC',
...     end='2022-01-01 UTC',
...     timeframe='1h'
... )
```

Symbol 2/2

Symbol 2/2

Period 36/36

Period 36/36

The fetching operation for both symbols took us around 80 seconds to complete. Since Binance, as any other exchange, will never return the whole data at once, vectorbt first requested the maximum amount of data starting on January 1st, 2020 and then gradually collected the remaining data by also respecting the Binance's API rate limits. In total, this resulted in 36 requests per symbol. Finally, vectorbt aligned both symbols in case their indexes or columns were different and made the final index timezone-aware (in UTC).

To avoid repeatedly hitting the Binance servers each time we start a new Python session, we should save the downloaded data locally using either the vectorbt's Data.to_csv or Data.to_hdf:

```python
>>> data.to_hdf('my_data.h5')
```

```python
>>> data.to_hdf('my_data.h5')
```

We can then access the saved data easily using HDFData:

```python
>>> data = vbt.HDFData.pull('my_data.h5')
```

```python
>>> data = vbt.HDFData.pull('my_data.h5')
```

Symbol 2/2

Symbol 2/2

Hint

We can access any of the symbols in an HDF file using regular path expressions. For example, the same as above: vbt.HDFData.pull(['my_data.h5/BTCUSDT', 'my_data.h5/ETHUSDT']).

```python
vbt.HDFData.pull(['my_data.h5/BTCUSDT', 'my_data.h5/ETHUSDT'])
```

Once we have the data, let's take a quick look at what's inside. To get any of the stored DataFrames, use the Data.data dictionary with each DataFrame keyed by symbol:

```python
>>> data.data['BTCUSDT'].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 17514 entries, 2019-12-31 23:00:00+00:00 to 2021-12-31 22:00:00+00:00
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype              
---  ------              --------------  -----              
 0   Open                17514 non-null  float64            
 1   High                17514 non-null  float64            
 2   Low                 17514 non-null  float64            
 3   Close               17514 non-null  float64            
 4   Volume              17514 non-null  float64            
 5   Close time          17514 non-null  datetime64[ns, UTC]
 6   Quote volume        17514 non-null  float64            
 7   Number of trades    17514 non-null  int64              
 8   Taker base volume   17514 non-null  float64            
 9   Taker quote volume  17514 non-null  float64            
dtypes: datetime64[ns, UTC](1), float64(8), int64(1)
memory usage: 2.0 MB
```

```python
>>> data.data['BTCUSDT'].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 17514 entries, 2019-12-31 23:00:00+00:00 to 2021-12-31 22:00:00+00:00
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype              
---  ------              --------------  -----              
 0   Open                17514 non-null  float64            
 1   High                17514 non-null  float64            
 2   Low                 17514 non-null  float64            
 3   Close               17514 non-null  float64            
 4   Volume              17514 non-null  float64            
 5   Close time          17514 non-null  datetime64[ns, UTC]
 6   Quote volume        17514 non-null  float64            
 7   Number of trades    17514 non-null  int64              
 8   Taker base volume   17514 non-null  float64            
 9   Taker quote volume  17514 non-null  float64            
dtypes: datetime64[ns, UTC](1), float64(8), int64(1)
memory usage: 2.0 MB
```

We can also get an overview of all the symbols captured:

```python
>>> data.stats()
Start                   2020-01-01 00:00:00+00:00
End                     2021-12-31 23:00:00+00:00
Period                                      17513
Total Symbols                                   2
Last Index: BTCUSDT     2021-12-31 23:00:00+00:00
Last Index: ETHUSDT     2021-12-31 23:00:00+00:00
Null Counts: BTCUSDT                            0
Null Counts: ETHUSDT                            0
Name: agg_stats, dtype: object
```

```python
>>> data.stats()
Start                   2020-01-01 00:00:00+00:00
End                     2021-12-31 23:00:00+00:00
Period                                      17513
Total Symbols                                   2
Last Index: BTCUSDT     2021-12-31 23:00:00+00:00
Last Index: ETHUSDT     2021-12-31 23:00:00+00:00
Null Counts: BTCUSDT                            0
Null Counts: ETHUSDT                            0
Name: agg_stats, dtype: object
```

Each symbol has 17513 data points with no NaNs - good!

If you ever worked with vectorbt, you would know that vectorbt loves the data to be supplied with symbols as columns - one per backtest - rather than features as columns. Since SuperTrend depends upon the high, low, and close price, let's get those three features as separate DataFrames using Data.get:

```python
>>> high = data.get('High')
>>> low = data.get('Low')
>>> close = data.get('Close')

>>> close
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2020-01-01 00:00:00+00:00   7177.02   128.87
2020-01-01 01:00:00+00:00   7216.27   130.64
2020-01-01 02:00:00+00:00   7242.85   130.85
2020-01-01 03:00:00+00:00   7225.01   130.20
2020-01-01 04:00:00+00:00   7217.27   130.20
...                             ...      ...
2021-12-31 19:00:00+00:00  45728.28  3626.27
2021-12-31 20:00:00+00:00  45879.24  3645.04
2021-12-31 21:00:00+00:00  46333.86  3688.41
2021-12-31 22:00:00+00:00  46303.99  3681.80
2021-12-31 23:00:00+00:00  46216.93  3676.23

[17513 rows x 2 columns]
```

```python
>>> high = data.get('High')
>>> low = data.get('Low')
>>> close = data.get('Close')

>>> close
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2020-01-01 00:00:00+00:00   7177.02   128.87
2020-01-01 01:00:00+00:00   7216.27   130.64
2020-01-01 02:00:00+00:00   7242.85   130.85
2020-01-01 03:00:00+00:00   7225.01   130.20
2020-01-01 04:00:00+00:00   7217.27   130.20
...                             ...      ...
2021-12-31 19:00:00+00:00  45728.28  3626.27
2021-12-31 20:00:00+00:00  45879.24  3645.04
2021-12-31 21:00:00+00:00  46333.86  3688.41
2021-12-31 22:00:00+00:00  46303.99  3681.80
2021-12-31 23:00:00+00:00  46216.93  3676.23

[17513 rows x 2 columns]
```

Hint

To get a column of a particular symbol as a Series, use data.get('Close', 'BTCUSDT').

```python
data.get('Close', 'BTCUSDT')
```

We're all set to design our first SuperTrend indicator!

## Design¶

SuperTrend is a trend-following indicator that uses Average True Range (ATR) and median price to define a set of upper and lower bands. The idea is rather simple: when the close price crosses above the upper band, the asset is considered to be entering an uptrend, hence a buy signal. When the close price crosses below the lower band, the asset is considered to have exited the uptrend, hence a sell signal.

Unlike the idea, the calculation procedure is anything but simple:

```python
BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

FINAL UPPERBAND = IF (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND)
                  THEN Current BASIC UPPERBAND
                  ELSE Previous FINAL UPPERBAND
FINAL LOWERBAND = IF (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)
                  THEN Current BASIC LOWERBAND 
                  ELSE Previous FINAL LOWERBAND

SUPERTREND      = IF (Previous SUPERTREND == Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) 
                  THEN Current FINAL UPPERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND) 
                  THEN Current FINAL LOWERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND) 
                  THEN Current FINAL LOWERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND) 
                  THEN Current FINAL UPPERBAND
```

```python
BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

FINAL UPPERBAND = IF (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND)
                  THEN Current BASIC UPPERBAND
                  ELSE Previous FINAL UPPERBAND
FINAL LOWERBAND = IF (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)
                  THEN Current BASIC LOWERBAND 
                  ELSE Previous FINAL LOWERBAND

SUPERTREND      = IF (Previous SUPERTREND == Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) 
                  THEN Current FINAL UPPERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND) 
                  THEN Current FINAL LOWERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND) 
                  THEN Current FINAL LOWERBAND
                  ELIF (Previous SUPERTREND == Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND) 
                  THEN Current FINAL UPPERBAND
```

Even though the basic bands can be well computed using the standard tools, you'll certainly get a headache when attempting to do this for the final bands. The consensus among most open-source solutions is to use a basic Python for-loop and write the array elements one at a time. But is this scalable? We're here to find out!

### Pandas¶

Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool. Since it's a go-to library for processing data in Python, let's write our first implementation using Pandas alone. It will take one column and one combination of parameters, and return four arrays: one for the SuperTrend (trend), one for the direction (dir_), one for the uptrend (long), and one for the downtrend (short). We'll also split the implementation into 5 parts for readability and to be able to optimize any component at any time:

```python
trend
```

```python
dir_
```

```python
long
```

```python
short
```

```python
get_med_price
```

```python
get_atr
```

```python
get_basic_bands
```

```python
get_final_bands
```

```python
supertrend
```

```python
>>> def get_med_price(high, low):
...     return (high + low) / 2

>>> def get_atr(high, low, close, period):
...     tr0 = abs(high - low)
...     tr1 = abs(high - close.shift())
...     tr2 = abs(low - close.shift())
...     tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)  # (1)!
...     atr = tr.ewm(
...         alpha=1 / period, 
...         adjust=False, 
...         min_periods=period).mean()  # (2)!
...     return atr

>>> def get_basic_bands(med_price, atr, multiplier):
...     matr = multiplier * atr
...     upper = med_price + matr
...     lower = med_price - matr
...     return upper, lower

>>> def get_final_bands(close, upper, lower):  # (3)!
...     trend = pd.Series(np.full(close.shape, np.nan), index=close.index)
...     dir_ = pd.Series(np.full(close.shape, 1), index=close.index)
...     long = pd.Series(np.full(close.shape, np.nan), index=close.index)
...     short = pd.Series(np.full(close.shape, np.nan), index=close.index)
... 
...     for i in range(1, close.shape[0]):  # (4)!
...         if close.iloc[i] > upper.iloc[i - 1]:
...             dir_.iloc[i] = 1
...         elif close.iloc[i] < lower.iloc[i - 1]:
...             dir_.iloc[i] = -1
...         else:
...             dir_.iloc[i] = dir_.iloc[i - 1]
...             if dir_.iloc[i] > 0 and lower.iloc[i] < lower.iloc[i - 1]:
...                 lower.iloc[i] = lower.iloc[i - 1]
...             if dir_.iloc[i] < 0 and upper.iloc[i] > upper.iloc[i - 1]:
...                 upper.iloc[i] = upper.iloc[i - 1]
... 
...         if dir_.iloc[i] > 0:
...             trend.iloc[i] = long.iloc[i] = lower.iloc[i]
...         else:
...             trend.iloc[i] = short.iloc[i] = upper.iloc[i]
...             
...     return trend, dir_, long, short

>>> def supertrend(high, low, close, period=7, multiplier=3):
...     med_price = get_med_price(high, low)
...     atr = get_atr(high, low, close, period)
...     upper, lower = get_basic_bands(med_price, atr, multiplier)
...     return get_final_bands(close, upper, lower)
```

```python
>>> def get_med_price(high, low):
...     return (high + low) / 2

>>> def get_atr(high, low, close, period):
...     tr0 = abs(high - low)
...     tr1 = abs(high - close.shift())
...     tr2 = abs(low - close.shift())
...     tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)  # (1)!
...     atr = tr.ewm(
...         alpha=1 / period, 
...         adjust=False, 
...         min_periods=period).mean()  # (2)!
...     return atr

>>> def get_basic_bands(med_price, atr, multiplier):
...     matr = multiplier * atr
...     upper = med_price + matr
...     lower = med_price - matr
...     return upper, lower

>>> def get_final_bands(close, upper, lower):  # (3)!
...     trend = pd.Series(np.full(close.shape, np.nan), index=close.index)
...     dir_ = pd.Series(np.full(close.shape, 1), index=close.index)
...     long = pd.Series(np.full(close.shape, np.nan), index=close.index)
...     short = pd.Series(np.full(close.shape, np.nan), index=close.index)
... 
...     for i in range(1, close.shape[0]):  # (4)!
...         if close.iloc[i] > upper.iloc[i - 1]:
...             dir_.iloc[i] = 1
...         elif close.iloc[i] < lower.iloc[i - 1]:
...             dir_.iloc[i] = -1
...         else:
...             dir_.iloc[i] = dir_.iloc[i - 1]
...             if dir_.iloc[i] > 0 and lower.iloc[i] < lower.iloc[i - 1]:
...                 lower.iloc[i] = lower.iloc[i - 1]
...             if dir_.iloc[i] < 0 and upper.iloc[i] > upper.iloc[i - 1]:
...                 upper.iloc[i] = upper.iloc[i - 1]
... 
...         if dir_.iloc[i] > 0:
...             trend.iloc[i] = long.iloc[i] = lower.iloc[i]
...         else:
...             trend.iloc[i] = short.iloc[i] = upper.iloc[i]
...             
...     return trend, dir_, long, short

>>> def supertrend(high, low, close, period=7, multiplier=3):
...     med_price = get_med_price(high, low)
...     atr = get_atr(high, low, close, period)
...     upper, lower = get_basic_bands(med_price, atr, multiplier)
...     return get_final_bands(close, upper, lower)
```

Let's run the supertrend function on the BTCUSDT symbol:

```python
supertrend
```

```python
BTCUSDT
```

```python
>>> supert, superd, superl, supers = supertrend(
...     high['BTCUSDT'], 
...     low['BTCUSDT'], 
...     close['BTCUSDT']
... )

>>> supert
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64

>>> superd  # (1)!
Open time
2020-01-01 00:00:00+00:00    1
2020-01-01 01:00:00+00:00    1
2020-01-01 02:00:00+00:00    1
                           ...
2021-12-31 21:00:00+00:00   -1
2021-12-31 22:00:00+00:00   -1
2021-12-31 23:00:00+00:00   -1
Length: 17513, dtype: int64

>>> superl  # (2)!
Open time
2020-01-01 00:00:00+00:00   NaN
2020-01-01 01:00:00+00:00   NaN
2020-01-01 02:00:00+00:00   NaN
                            ...
2021-12-31 21:00:00+00:00   NaN
2021-12-31 22:00:00+00:00   NaN
2021-12-31 23:00:00+00:00   NaN
Length: 17513, dtype: float64

>>> supers  # (3)!
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64
```

```python
>>> supert, superd, superl, supers = supertrend(
...     high['BTCUSDT'], 
...     low['BTCUSDT'], 
...     close['BTCUSDT']
... )

>>> supert
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64

>>> superd  # (1)!
Open time
2020-01-01 00:00:00+00:00    1
2020-01-01 01:00:00+00:00    1
2020-01-01 02:00:00+00:00    1
                           ...
2021-12-31 21:00:00+00:00   -1
2021-12-31 22:00:00+00:00   -1
2021-12-31 23:00:00+00:00   -1
Length: 17513, dtype: int64

>>> superl  # (2)!
Open time
2020-01-01 00:00:00+00:00   NaN
2020-01-01 01:00:00+00:00   NaN
2020-01-01 02:00:00+00:00   NaN
                            ...
2021-12-31 21:00:00+00:00   NaN
2021-12-31 22:00:00+00:00   NaN
2021-12-31 23:00:00+00:00   NaN
Length: 17513, dtype: float64

>>> supers  # (3)!
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64
```

If you print out the head of the supert Series using supert.head(10), you'll notice that the first 6 data points are all NaN. This is because the ATR's rolling period is 7, so the first 6 computed windows contained incomplete data.

```python
supert
```

```python
supert.head(10)
```

A graph is worth 1,000 words. Let's plot the first month of data (January 2020):

```python
>>> date_range = slice('2020-01-01', '2020-02-01')
>>> fig = close.loc[date_range, 'BTCUSDT'].rename('Close').vbt.plot()  # (1)!
>>> supers.loc[date_range].rename('Short').vbt.plot(fig=fig)
>>> superl.loc[date_range].rename('Long').vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> date_range = slice('2020-01-01', '2020-02-01')
>>> fig = close.loc[date_range, 'BTCUSDT'].rename('Close').vbt.plot()  # (1)!
>>> supers.loc[date_range].rename('Short').vbt.plot(fig=fig)
>>> superl.loc[date_range].rename('Long').vbt.plot(fig=fig)
>>> fig.show()
```

We've generated and visualized the SuperTrend values, but what about performance? Can we already make our overfitting machine with thousands of parameter combinations rolling? Not so fast. As you might have guessed, the supertrend function takes some time to compute:

```python
supertrend
```

```python
>>> %%timeit
>>> supertrend(high['BTCUSDT'], low['BTCUSDT'], close['BTCUSDT'])
2.15 s ± 19.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> %%timeit
>>> supertrend(high['BTCUSDT'], low['BTCUSDT'], close['BTCUSDT'])
2.15 s ± 19.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Ouch! Doing 1000 backtests would take us roughly 33 minutes.

Let's hear what Pandas TA has to say about this:

```python
>>> SUPERTREND = vbt.pandas_ta('SUPERTREND')  # (1)!

>>> %%timeit
>>> SUPERTREND.run(high['BTCUSDT'], low['BTCUSDT'], close['BTCUSDT'])
784 ms ± 14.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> SUPERTREND = vbt.pandas_ta('SUPERTREND')  # (1)!

>>> %%timeit
>>> SUPERTREND.run(high['BTCUSDT'], low['BTCUSDT'], close['BTCUSDT'])
784 ms ± 14.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
vbt.pandas_ta
```

That's a 3x speedup, mostly due to the fact that Pandas TA uses ATR from TA-Lib.

Is it now acceptable? Of course not  Can we get better than this? Hell yeah!

### NumPy + Numba = ¶

Pandas shines whenever it comes to manipulating heterogeneous tabular data, but is this really applicable to indicators? You might have noticed that even though we used Pandas, none of the operations in any of our newly defined functions makes use of index or column labels. Moreover, most indicators take, manipulate, and return arrays of the same dimensions and shape, which makes indicator development a purely algebraic challenge that can be well decomposed into multiple vectorized steps or solved on the per-element basis (or both!). Given that Pandas just extends NumPy and the latter is considered as a faster (although lower level) package, let's adapt our logic to NumPy arrays instead.

Both functions get_med_price and get_basic_bands are based on basic arithmetic computations such as addition and multiplication, which are applicable to both Pandas and NumPy arrays and require no further changes. But what about get_atr and get_final_bands? The former can be re-implemented using NumPy and vectorbt's own arsenal of Numba-compiled functions:

```python
get_med_price
```

```python
get_basic_bands
```

```python
get_atr
```

```python
get_final_bands
```

```python
>>> def get_atr_np(high, low, close, period):
...     shifted_close = vbt.nb.fshift_1d_nb(close)  # (1)!
...     tr0 = np.abs(high - low)
...     tr1 = np.abs(high - shifted_close)
...     tr2 = np.abs(low - shifted_close)
...     tr = np.column_stack((tr0, tr1, tr2)).max(axis=1)  # (2)!
...     atr = vbt.nb.wwm_mean_1d_nb(tr, period)  # (3)!
...     return atr
```

```python
>>> def get_atr_np(high, low, close, period):
...     shifted_close = vbt.nb.fshift_1d_nb(close)  # (1)!
...     tr0 = np.abs(high - low)
...     tr1 = np.abs(high - shifted_close)
...     tr2 = np.abs(low - shifted_close)
...     tr = np.column_stack((tr0, tr1, tr2)).max(axis=1)  # (2)!
...     atr = vbt.nb.wwm_mean_1d_nb(tr, period)  # (3)!
...     return atr
```

```python
n
```

The latter, on the other hand, is an iterative algorithm - it's rather a poor fit for NumPy and an ideal fit for Numba, which can easily run for-loops at a machine code speed:

```python
>>> @njit
... def get_final_bands_nb(close, upper, lower):  # (1)!
...     trend = np.full(close.shape, np.nan)  # (2)!
...     dir_ = np.full(close.shape, 1)
...     long = np.full(close.shape, np.nan)
...     short = np.full(close.shape, np.nan)
... 
...     for i in range(1, close.shape[0]):
...         if close[i] > upper[i - 1]:  # (3)!
...             dir_[i] = 1
...         elif close[i] < lower[i - 1]:
...             dir_[i] = -1
...         else:
...             dir_[i] = dir_[i - 1]
...             if dir_[i] > 0 and lower[i] < lower[i - 1]:
...                 lower[i] = lower[i - 1]
...             if dir_[i] < 0 and upper[i] > upper[i - 1]:
...                 upper[i] = upper[i - 1]
... 
...         if dir_[i] > 0:
...             trend[i] = long[i] = lower[i]
...         else:
...             trend[i] = short[i] = upper[i]
...             
...     return trend, dir_, long, short
```

```python
>>> @njit
... def get_final_bands_nb(close, upper, lower):  # (1)!
...     trend = np.full(close.shape, np.nan)  # (2)!
...     dir_ = np.full(close.shape, 1)
...     long = np.full(close.shape, np.nan)
...     short = np.full(close.shape, np.nan)
... 
...     for i in range(1, close.shape[0]):
...         if close[i] > upper[i - 1]:  # (3)!
...             dir_[i] = 1
...         elif close[i] < lower[i - 1]:
...             dir_[i] = -1
...         else:
...             dir_[i] = dir_[i - 1]
...             if dir_[i] > 0 and lower[i] < lower[i - 1]:
...                 lower[i] = lower[i - 1]
...             if dir_[i] < 0 and upper[i] > upper[i - 1]:
...                 upper[i] = upper[i - 1]
... 
...         if dir_[i] > 0:
...             trend[i] = long[i] = lower[i]
...         else:
...             trend[i] = short[i] = upper[i]
...             
...     return trend, dir_, long, short
```

```python
nb
```

```python
pd.Series
```

```python
iloc
```

If you look at the function above, you'll notice that 1) it's a regular Python code that can run even without being decorated with @njit, and 2) it's almost identical to the implementation with Pandas - the main difference is in each iloc[...] being replaced by [...]. We can write a simple Python function that operates on constants and NumPy arrays, and Numba will try to make it much faster, fully automatically. Isn't that impressive?

```python
@njit
```

```python
iloc[...]
```

```python
[...]
```

Let's look at the result of this refactoring:

```python
>>> def faster_supertrend(high, low, close, period=7, multiplier=3):
...     med_price = get_med_price(high, low)
...     atr = get_atr_np(high, low, close, period)
...     upper, lower = get_basic_bands(med_price, atr, multiplier)
...     return get_final_bands_nb(close, upper, lower)

>>> supert, superd, superl, supers = faster_supertrend(
...     high['BTCUSDT'].values,  # (1)!
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )

>>> supert
array([          nan,           nan,           nan, ..., 47608.3465635,
       47608.3465635, 47608.3465635])
>>> superd
array([ 1,  1,  1, ..., -1, -1, -1])

>>> superl
array([nan, nan, nan, ..., nan, nan, nan])

>>> supers
array([          nan,           nan,           nan, ..., 47608.3465635,
       47608.3465635, 47608.3465635])
```

```python
>>> def faster_supertrend(high, low, close, period=7, multiplier=3):
...     med_price = get_med_price(high, low)
...     atr = get_atr_np(high, low, close, period)
...     upper, lower = get_basic_bands(med_price, atr, multiplier)
...     return get_final_bands_nb(close, upper, lower)

>>> supert, superd, superl, supers = faster_supertrend(
...     high['BTCUSDT'].values,  # (1)!
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )

>>> supert
array([          nan,           nan,           nan, ..., 47608.3465635,
       47608.3465635, 47608.3465635])
>>> superd
array([ 1,  1,  1, ..., -1, -1, -1])

>>> superl
array([nan, nan, nan, ..., nan, nan, nan])

>>> supers
array([          nan,           nan,           nan, ..., 47608.3465635,
       47608.3465635, 47608.3465635])
```

```python
values
```

Info

When executing a Numba-decorated function for the first time, it may take longer due to compilation.

As expected, those are arrays similar to the ones returned by the supertrend function, just without any labels. To attach labels, we can simply do:

```python
supertrend
```

```python
>>> pd.Series(supert, index=close.index)
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64
```

```python
>>> pd.Series(supert, index=close.index)
Open time
2020-01-01 00:00:00+00:00             NaN
2020-01-01 01:00:00+00:00             NaN
2020-01-01 02:00:00+00:00             NaN
                                      ...
2021-12-31 21:00:00+00:00    47608.346563
2021-12-31 22:00:00+00:00    47608.346563
2021-12-31 23:00:00+00:00    47608.346563
Length: 17513, dtype: float64
```

Wondering how much our code has gained in performance? Wonder no more:

```python
%%timeit
>>> faster_supertrend(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values,
...     close['BTCUSDT'].values
... )
1.11 ms ± 7.05 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

```python
%%timeit
>>> faster_supertrend(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values,
...     close['BTCUSDT'].values
... )
1.11 ms ± 7.05 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

That's a 780x speedup over an average Pandas TA run

### NumPy + Numba + TA-Lib = ¶

If you think that this result cannot be topped, then apparently you haven't worked with TA-Lib. Even though there is no SuperTrend indicator available in TA-Lib, we can still use its highly-optimized indicator functions for intermediate calculations. In particular, instead of reinventing the wheel and implementing the median price and ATR functionality from scratch, we can use the MEDPRICE and ATR TA-Lib functions respectively. They have two major advantages over our custom implementation:

```python
MEDPRICE
```

```python
ATR
```

```python
>>> import talib

>>> def faster_supertrend_talib(high, low, close, period=7, multiplier=3):
...     avg_price = talib.MEDPRICE(high, low)  # (1)!
...     atr = talib.ATR(high, low, close, period)
...     upper, lower = get_basic_bands(avg_price, atr, multiplier)
...     return get_final_bands_nb(close, upper, lower)

>>> faster_supertrend_talib(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )
(array([          nan,           nan,           nan, ..., 47608.3465635,
        47608.3465635, 47608.3465635]),
 array([ 1,  1,  1, ..., -1, -1, -1]),
 array([nan, nan, nan, ..., nan, nan, nan]),
 array([          nan,           nan,           nan, ..., 47608.3465635,
        47608.3465635, 47608.3465635]))

>>> %%timeit
>>> faster_supertrend_talib(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )
253 µs ± 815 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

```python
>>> import talib

>>> def faster_supertrend_talib(high, low, close, period=7, multiplier=3):
...     avg_price = talib.MEDPRICE(high, low)  # (1)!
...     atr = talib.ATR(high, low, close, period)
...     upper, lower = get_basic_bands(avg_price, atr, multiplier)
...     return get_final_bands_nb(close, upper, lower)

>>> faster_supertrend_talib(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )
(array([          nan,           nan,           nan, ..., 47608.3465635,
        47608.3465635, 47608.3465635]),
 array([ 1,  1,  1, ..., -1, -1, -1]),
 array([nan, nan, nan, ..., nan, nan, nan]),
 array([          nan,           nan,           nan, ..., 47608.3465635,
        47608.3465635, 47608.3465635]))

>>> %%timeit
>>> faster_supertrend_talib(
...     high['BTCUSDT'].values, 
...     low['BTCUSDT'].values, 
...     close['BTCUSDT'].values
... )
253 µs ± 815 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Another 4x improvement - by the time another trader processed a single column of data, we would have processed around 3 thousand columns. Agreed, the speed of our indicator is slowly getting ridiculously high

## Indicator factory¶

Let's stop here and ask ourselves: why do we even need such a crazy performance?

That's when parameter optimization comes into play. The two parameters that we have - period and multiplier - are the default values commonly used in technical analysis. But what makes those values universal and how do we know whether there aren't any better values for the markets we're participating in? Imagine having a pipeline that can backtest hundreds or even thousands of parameters and reveal configurations and market regimes that correlate better on average?

```python
period
```

```python
multiplier
```

IndicatorFactory is a vectorbt's own powerhouse that can make any indicator function parametrizable. To get a better idea of what this means, let's supercharge the faster_supertrend_talib function:

```python
faster_supertrend_talib
```

```python
>>> SuperTrend = vbt.IF(
...     class_name='SuperTrend',
...     short_name='st',
...     input_names=['high', 'low', 'close'],
...     param_names=['period', 'multiplier'],
...     output_names=['supert', 'superd', 'superl', 'supers']
... ).with_apply_func(
...     faster_supertrend_talib, 
...     takes_1d=True,  # (1)!
...     period=7,  # (2)!
...     multiplier=3
... )
```

```python
>>> SuperTrend = vbt.IF(
...     class_name='SuperTrend',
...     short_name='st',
...     input_names=['high', 'low', 'close'],
...     param_names=['period', 'multiplier'],
...     output_names=['supert', 'superd', 'superl', 'supers']
... ).with_apply_func(
...     faster_supertrend_talib, 
...     takes_1d=True,  # (1)!
...     period=7,  # (2)!
...     multiplier=3
... )
```

The indicator factory is a class that can generate so-called indicator classes. You can imagine it being a conveyor belt that can take a specification of your indicator function and produce a stand-alone Python class for running that function in a very flexible way. In our example, when we called vbt.IF(...), it has internally created an indicator class SuperTrend, and once we supplied faster_supertrend_talib to IndicatorFactory.with_apply_func, it attached a method SuperTrend.run for running the indicator. Let's try it out!

```python
vbt.IF(...)
```

```python
SuperTrend
```

```python
faster_supertrend_talib
```

```python
SuperTrend.run
```

```python
>>> vbt.phelp(SuperTrend.run)  # (1)!
SuperTrend.run(
    high,
    low,
    close,
    period=Default(value=7),
    multiplier=Default(value=3),
    short_name='st',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `SuperTrend` indicator.

    * Inputs: `high`, `low`, `close`
    * Parameters: `period`, `multiplier`
    * Outputs: `supert`, `superd`, `superl`, `supers`

    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.

    Other keyword arguments are passed to `SuperTrend.run_pipeline`.

>>> st = SuperTrend.run(high, low, close)
>>> st.supert
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2020-01-01 00:00:00+00:00           NaN          NaN
2020-01-01 01:00:00+00:00           NaN          NaN
2020-01-01 02:00:00+00:00           NaN          NaN
...                                 ...          ...
2021-12-31 21:00:00+00:00  47608.346563  3770.258246
2021-12-31 22:00:00+00:00  47608.346563  3770.258246
2021-12-31 23:00:00+00:00  47608.346563  3770.258246

[17513 rows x 2 columns]
```

```python
>>> vbt.phelp(SuperTrend.run)  # (1)!
SuperTrend.run(
    high,
    low,
    close,
    period=Default(value=7),
    multiplier=Default(value=3),
    short_name='st',
    hide_params=None,
    hide_default=True,
    **kwargs
):
    Run `SuperTrend` indicator.

    * Inputs: `high`, `low`, `close`
    * Parameters: `period`, `multiplier`
    * Outputs: `supert`, `superd`, `superl`, `supers`

    Pass a list of parameter names as `hide_params` to hide their column levels.
    Set `hide_default` to False to show the column levels of the parameters with a default value.

    Other keyword arguments are passed to `SuperTrend.run_pipeline`.

>>> st = SuperTrend.run(high, low, close)
>>> st.supert
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2020-01-01 00:00:00+00:00           NaN          NaN
2020-01-01 01:00:00+00:00           NaN          NaN
2020-01-01 02:00:00+00:00           NaN          NaN
...                                 ...          ...
2021-12-31 21:00:00+00:00  47608.346563  3770.258246
2021-12-31 22:00:00+00:00  47608.346563  3770.258246
2021-12-31 23:00:00+00:00  47608.346563  3770.258246

[17513 rows x 2 columns]
```

```python
SuperTrend.run
```

Notice how our SuperTrend indicator magically accepted two-dimensional Pandas arrays, even though the function itself can only work on one-dimensional NumPy arrays. Not only it computed the SuperTrend on each column, but it also converted the resulting arrays back into the Pandas format for pure convenience. So, how does all of this impact the performance?

```python
>>> %%timeit
>>> SuperTrend.run(high, low, close)
2 ms ± 130 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

```python
>>> %%timeit
>>> SuperTrend.run(high, low, close)
2 ms ± 130 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Not that much! With all the pre- and postprocessing taking place, the indicator needs roughly one millisecond to process one column (that is, 17k data points).

### Expressions¶

If you think that calling vbt.IF(...) and providing input_names, param_names, and other information manually is too much work, well, vectorbt has something for you. Our faster_supertrend_talib is effectively a black box to the indicator factory - that's why the factory cannot introspect it and derive the required information programmatically. But it easily could if we converted faster_supertrend_talib into an expression!

```python
vbt.IF(...)
```

```python
input_names
```

```python
param_names
```

```python
faster_supertrend_talib
```

```python
faster_supertrend_talib
```

Expressions are regular strings that can be evaluated into Python code. By giving such a string to IndicatorFactory.from_expr, the factory will be able to see what's inside, parse the specification, and generate a full-blown indicator class.

Hint

Instance methods with the prefix with (such as with_apply_func) require the specification to be provided manually, while class methods with the prefix from (such as from_expr) can parse this information automatically.

```python
with
```

```python
with_apply_func
```

```python
from
```

```python
from_expr
```

Here's an expression for faster_supertrend_talib:

```python
faster_supertrend_talib
```

```python
>>> expr = """
... SuperTrend[st]:
... medprice = @talib_medprice(high, low)
... atr = @talib_atr(high, low, close, @p_period)
... upper, lower = get_basic_bands(medprice, atr, @p_multiplier)
... supert, superd, superl, supers = get_final_bands(close, upper, lower)
... supert, superd, superl, supers
... """
```

```python
>>> expr = """
... SuperTrend[st]:
... medprice = @talib_medprice(high, low)
... atr = @talib_atr(high, low, close, @p_period)
... upper, lower = get_basic_bands(medprice, atr, @p_multiplier)
... supert, superd, superl, supers = get_final_bands(close, upper, lower)
... supert, superd, superl, supers
... """
```

Using annotations with @ we tell the factory how to treat specific variables. For instance, any variable with the prefix @talib gets replaced by the respective TA-Lib function that has been upgraded with broadcasting and multidimensionality. You can also see that parameters were annotated with @p, while inputs and outputs weren't annotated at all - the factory knows exactly that high is the high price, while the latest line apparently returns 4 output objects.

```python
@
```

```python
@talib
```

```python
@p
```

```python
high
```

For more examples, see the documentation on expression parsing.

```python
>>> SuperTrend = vbt.IF.from_expr(
...     expr, 
...     takes_1d=True,
...     get_basic_bands=get_basic_bands,  # (1)!
...     get_final_bands=get_final_bands_nb,
...     period=7, 
...     multiplier=3
... )

>>> st = SuperTrend.run(high, low, close)
>>> st.supert
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2020-01-01 00:00:00+00:00           NaN          NaN
2020-01-01 01:00:00+00:00           NaN          NaN
2020-01-01 02:00:00+00:00           NaN          NaN
...                                 ...          ...
2021-12-31 21:00:00+00:00  47608.346563  3770.258246
2021-12-31 22:00:00+00:00  47608.346563  3770.258246
2021-12-31 23:00:00+00:00  47608.346563  3770.258246

[17513 rows x 2 columns]

>>> %%timeit
>>> SuperTrend.run(high, low, close)
2.35 ms ± 81.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

```python
>>> SuperTrend = vbt.IF.from_expr(
...     expr, 
...     takes_1d=True,
...     get_basic_bands=get_basic_bands,  # (1)!
...     get_final_bands=get_final_bands_nb,
...     period=7, 
...     multiplier=3
... )

>>> st = SuperTrend.run(high, low, close)
>>> st.supert
symbol                          BTCUSDT      ETHUSDT
Open time                                           
2020-01-01 00:00:00+00:00           NaN          NaN
2020-01-01 01:00:00+00:00           NaN          NaN
2020-01-01 02:00:00+00:00           NaN          NaN
...                                 ...          ...
2021-12-31 21:00:00+00:00  47608.346563  3770.258246
2021-12-31 22:00:00+00:00  47608.346563  3770.258246
2021-12-31 23:00:00+00:00  47608.346563  3770.258246

[17513 rows x 2 columns]

>>> %%timeit
>>> SuperTrend.run(high, low, close)
2.35 ms ± 81.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

By the way, this is exactly how WorldQuant's Alphas are implemented in vectorbt. Never stop loving Python for the magic it enables

### Plotting¶

Remember how we previously plotted SuperTrend? We had to manually select the date range from each output array and add it to the plot by passing the figure around. Let's subclass SuperTrend and define a method plot that does all of this for us:

```python
SuperTrend
```

```python
plot
```

```python
>>> class SuperTrend(SuperTrend):
...     def plot(self, 
...              column=None,  # (1)!
...              close_kwargs=None,  # (2)!
...              superl_kwargs=None,
...              supers_kwargs=None,
...              fig=None,  # (3)!
...              **layout_kwargs):  # (4)!
...         close_kwargs = close_kwargs if close_kwargs else {}
...         superl_kwargs = superl_kwargs if superl_kwargs else {}
...         supers_kwargs = supers_kwargs if supers_kwargs else {}
...         
...         close = self.select_col_from_obj(self.close, column).rename('Close')
...         supers = self.select_col_from_obj(self.supers, column).rename('Short')
...         superl = self.select_col_from_obj(self.superl, column).rename('Long')
...         
...         fig = close.vbt.plot(fig=fig, **close_kwargs, **layout_kwargs)  # (5)!
...         supers.vbt.plot(fig=fig, **supers_kwargs)
...         superl.vbt.plot(fig=fig, **superl_kwargs)
...         
...         return fig
```

```python
>>> class SuperTrend(SuperTrend):
...     def plot(self, 
...              column=None,  # (1)!
...              close_kwargs=None,  # (2)!
...              superl_kwargs=None,
...              supers_kwargs=None,
...              fig=None,  # (3)!
...              **layout_kwargs):  # (4)!
...         close_kwargs = close_kwargs if close_kwargs else {}
...         superl_kwargs = superl_kwargs if superl_kwargs else {}
...         supers_kwargs = supers_kwargs if supers_kwargs else {}
...         
...         close = self.select_col_from_obj(self.close, column).rename('Close')
...         supers = self.select_col_from_obj(self.supers, column).rename('Short')
...         superl = self.select_col_from_obj(self.superl, column).rename('Long')
...         
...         fig = close.vbt.plot(fig=fig, **close_kwargs, **layout_kwargs)  # (5)!
...         supers.vbt.plot(fig=fig, **supers_kwargs)
...         superl.vbt.plot(fig=fig, **superl_kwargs)
...         
...         return fig
```

```python
kwargs
```

But how are we supposed to select the date range to plot? Pretty easy: the indicator factory made SuperTrend indexable just like any regular Pandas object! Let's plot the same date range and symbol but slightly change the color palette:

```python
SuperTrend
```

```python
>>> st = SuperTrend.run(high, low, close)
>>> st.loc[date_range, 'BTCUSDT'].plot(
...     superl_kwargs=dict(trace_kwargs=dict(line_color='limegreen')),
...     supers_kwargs=dict(trace_kwargs=dict(line_color='red'))
... ).show()
```

```python
>>> st = SuperTrend.run(high, low, close)
>>> st.loc[date_range, 'BTCUSDT'].plot(
...     superl_kwargs=dict(trace_kwargs=dict(line_color='limegreen')),
...     supers_kwargs=dict(trace_kwargs=dict(line_color='red'))
... ).show()
```

Beautiful!

## Backtesting¶

Backtesting is usually the simplest step in vectorbt: convert the indicator values into two signal arrays - entries and exits - and supply them to Portfolio.from_signals. To make the test better reflect the reality, let's do several adjustments. Since we're calculating the SuperTrend values based on the current close price and vectorbt executes orders right away, we'll shift the execution of the signals by one tick forward:

```python
entries
```

```python
exits
```

```python
>>> entries = (~st.superl.isnull()).vbt.signals.fshift()  # (1)!
>>> exits = (~st.supers.isnull()).vbt.signals.fshift()
```

```python
>>> entries = (~st.superl.isnull()).vbt.signals.fshift()  # (1)!
>>> exits = (~st.supers.isnull()).vbt.signals.fshift()
```

We'll also apply the commission of 0.1%:

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     entries=entries, 
...     exits=exits, 
...     fees=0.001, 
...     freq='1h'
... )
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     entries=entries, 
...     exits=exits, 
...     fees=0.001, 
...     freq='1h'
... )
```

We've got a portfolio with two columns that can be analyzed with numerous built-in tools. For example, let's calculate and display the statistics for the ETHUSDT symbol:

```python
ETHUSDT
```

```python
>>> pf['ETHUSDT'].stats()
Start                         2020-01-01 00:00:00+00:00
End                           2021-12-31 23:00:00+00:00
Period                                729 days 17:00:00
Start Value                                       100.0
Min Value                                     98.469385
Max Value                                   1805.987865
End Value                                   1135.272383
Total Return [%]                            1035.272383
Benchmark Return [%]                        2752.665477
Total Time Exposure [%]                       51.750128
Max Gross Exposure [%]                            100.0
Max Drawdown [%]                               37.39953
Max Drawdown Duration                  85 days 09:00:00
Total Orders                                        348
Total Fees Paid                              272.755758
Total Trades                                        174
Win Rate [%]                                  43.103448
Best Trade [%]                                33.286985
Worst Trade [%]                              -13.783496
Avg Winning Trade [%]                          7.815551
Avg Losing Trade [%]                          -3.021041
Avg Winning Trade Duration              3 days 06:43:12
Avg Losing Trade Duration     1 days 07:54:32.727272727
Profit Factor                                  1.390947
Expectancy                                     5.949841
Sharpe Ratio                                   2.258501
Calmar Ratio                                   6.320363
Omega Ratio                                    1.103525
Sortino Ratio                                   3.27869
Name: ETHUSDT, dtype: object
```

```python
>>> pf['ETHUSDT'].stats()
Start                         2020-01-01 00:00:00+00:00
End                           2021-12-31 23:00:00+00:00
Period                                729 days 17:00:00
Start Value                                       100.0
Min Value                                     98.469385
Max Value                                   1805.987865
End Value                                   1135.272383
Total Return [%]                            1035.272383
Benchmark Return [%]                        2752.665477
Total Time Exposure [%]                       51.750128
Max Gross Exposure [%]                            100.0
Max Drawdown [%]                               37.39953
Max Drawdown Duration                  85 days 09:00:00
Total Orders                                        348
Total Fees Paid                              272.755758
Total Trades                                        174
Win Rate [%]                                  43.103448
Best Trade [%]                                33.286985
Worst Trade [%]                              -13.783496
Avg Winning Trade [%]                          7.815551
Avg Losing Trade [%]                          -3.021041
Avg Winning Trade Duration              3 days 06:43:12
Avg Losing Trade Duration     1 days 07:54:32.727272727
Profit Factor                                  1.390947
Expectancy                                     5.949841
Sharpe Ratio                                   2.258501
Calmar Ratio                                   6.320363
Omega Ratio                                    1.103525
Sortino Ratio                                   3.27869
Name: ETHUSDT, dtype: object
```

### Optimization¶

Optimization in vectorbt can be performed in two ways: iteratively and column-wise.

The first approach involves a simple loop that goes through every combination of the strategy's parameters and runs the whole logic. This would require you to manually generate a proper parameter grid and concatenate the results for analysis. On the upside, you would be able to use Hyperopt and other tools that work on the per-iteration basis.

The second approach is natively supported by vectorbt and involves stacking columns. If you have 2 symbols and 5 parameters, vectorbt will generate 10 columns in total - one for each symbol and parameter, and backtest each column separately without leaving Numba (that's why most functions in vectorbt are specialized in processing two-dimensional data, by the way). Not only this has a huge performance benefit for small to medium-sized data, but this also enables parallelization with Numba and presentation of the results in a Pandas-friendly format.

Let's test the period values 4, 5, ..., 20, and the multiplier values 2, 2.1, 2.2, ..., 4, which would yield 336 parameter combinations in total. Since our indicator is now parametrized, we can pass those two parameter arrays directly to the SuperTrend.run method by also instructing it to do the Cartesian product using the param_product=True flag:

```python
4, 5, ..., 20
```

```python
2, 2.1, 2.2, ..., 4
```

```python
SuperTrend.run
```

```python
param_product=True
```

```python
>>> periods = np.arange(4, 20)
>>> multipliers = np.arange(20, 41) / 10  # (1)!

>>> st = SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True
... )
```

```python
>>> periods = np.arange(4, 20)
>>> multipliers = np.arange(20, 41) / 10  # (1)!

>>> st = SuperTrend.run(
...     high, low, close, 
...     period=periods, 
...     multiplier=multipliers,
...     param_product=True
... )
```

```python
np.arange(2, 4.1, 0.1)
```

```python
3.000000000000001
```

```python
3.0
```

Iteration 672/672

Iteration 672/672

The indicator did 672 iterations - 336 per symbol. Let's see the columns that have been stacked:

```python
>>> st.wrapper.columns
MultiIndex([( 4, 2.0, 'BTCUSDT'),
            ( 4, 2.0, 'ETHUSDT'),
            ( 4, 2.1, 'BTCUSDT'),
            ( 4, 2.1, 'ETHUSDT'),
            ( 4, 2.2, 'BTCUSDT'),
            ...
            (19, 3.8, 'ETHUSDT'),
            (19, 3.9, 'BTCUSDT'),
            (19, 3.9, 'ETHUSDT'),
            (19, 4.0, 'BTCUSDT'),
            (19, 4.0, 'ETHUSDT')],
           names=['st_period', 'st_multiplier', 'symbol'], length=672)
```

```python
>>> st.wrapper.columns
MultiIndex([( 4, 2.0, 'BTCUSDT'),
            ( 4, 2.0, 'ETHUSDT'),
            ( 4, 2.1, 'BTCUSDT'),
            ( 4, 2.1, 'ETHUSDT'),
            ( 4, 2.2, 'BTCUSDT'),
            ...
            (19, 3.8, 'ETHUSDT'),
            (19, 3.9, 'BTCUSDT'),
            (19, 3.9, 'ETHUSDT'),
            (19, 4.0, 'BTCUSDT'),
            (19, 4.0, 'ETHUSDT')],
           names=['st_period', 'st_multiplier', 'symbol'], length=672)
```

Each of the DataFrames has now 672 columns. Let's plot the latest combination by specifying the column as a regular tuple:

```python
>>> st.loc[date_range, (19, 4, 'ETHUSDT')].plot().show()
```

```python
>>> st.loc[date_range, (19, 4, 'ETHUSDT')].plot().show()
```

When stacking a huge number of columns, make sure that you are not running out of RAM. You can print the size of any pickleable object in vectorbt using the Pickleable.getsize method:

```python
>>> print(st.getsize())
377.6 MB
```

```python
>>> print(st.getsize())
377.6 MB
```

Which can be manually calculated as follows (without inputs and parameters):

```python
>>> output_size = st.wrapper.shape[0] * st.wrapper.shape[1]
>>> n_outputs = 4
>>> data_type_size = 8
>>> input_size * n_outputs * data_type_size / 1024 / 1024
359.173828125
```

```python
>>> output_size = st.wrapper.shape[0] * st.wrapper.shape[1]
>>> n_outputs = 4
>>> data_type_size = 8
>>> input_size * n_outputs * data_type_size / 1024 / 1024
359.173828125
```

Hint

To reduce the memory footprint, change the get_final_bands_nb function to produce the output arrays with a lesser floating point accuracy, such as np.float32 or even np.float16.

```python
get_final_bands_nb
```

```python
np.float32
```

```python
np.float16
```

The backtesting part remains the same, irrespective of the number of columns:

```python
>>> entries = (~st.superl.isnull()).vbt.signals.fshift()
>>> exits = (~st.supers.isnull()).vbt.signals.fshift()

>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     entries=entries, 
...     exits=exits, 
...     fees=0.001, 
...     freq='1h'
... )
```

```python
>>> entries = (~st.superl.isnull()).vbt.signals.fshift()
>>> exits = (~st.supers.isnull()).vbt.signals.fshift()

>>> pf = vbt.Portfolio.from_signals(
...     close=close, 
...     entries=entries, 
...     exits=exits, 
...     fees=0.001, 
...     freq='1h'
... )
```

Instead of computing all the statistics for each single combination, let's plot a heatmap of their Sharpe values with the periods laid out horizontally and the multipliers laid out vertically. Since we have an additional column level that contains symbols, we'll make it a slider:

```python
>>> pf.sharpe_ratio.vbt.heatmap(
...     x_level='st_period', 
...     y_level='st_multiplier',
...     slider_level='symbol'
... )
```

```python
>>> pf.sharpe_ratio.vbt.heatmap(
...     x_level='st_period', 
...     y_level='st_multiplier',
...     slider_level='symbol'
... )
```

We now have a nice overview of any parameter regions that performed well during the backtesting period, yay!

Hint

To see how those Sharpe values perform against holding:

```python
>>> vbt.Portfolio.from_holding(close, freq='1h').sharpe_ratio
symbol
BTCUSDT    1.561447
ETHUSDT    2.170813
Name: sharpe_ratio, dtype: float64
```

```python
>>> vbt.Portfolio.from_holding(close, freq='1h').sharpe_ratio
symbol
BTCUSDT    1.561447
ETHUSDT    2.170813
Name: sharpe_ratio, dtype: float64
```

Python code  Notebook

