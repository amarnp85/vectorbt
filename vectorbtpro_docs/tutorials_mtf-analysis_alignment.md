# Alignment¶

Comparing time series with different time frames is tricky to say the least. Consider an example where we want to calculate the ratio between the close price in H1 and H4.

```python
H1
```

```python
H4
```

## Pandas¶

Comparing both time series using Pandas yields the following results:

```python
>>> h1_close = h1_data.get("Close")
>>> h4_close = h4_data.get("Close")

>>> h1_close.iloc[:4]
Open time
2020-01-01 00:00:00+00:00    7177.02
2020-01-01 01:00:00+00:00    7216.27
2020-01-01 02:00:00+00:00    7242.85
2020-01-01 03:00:00+00:00    7225.01
Name: Close, dtype: float64

>>> h4_close.iloc[:1]
Open time
2020-01-01 00:00:00+00:00    7225.01
Freq: 4H, Name: Close, dtype: float64

>>> h1_h4_ratio = h1_close / h4_close
>>> h1_h4_ratio.iloc[:4]
Open time
2020-01-01 00:00:00+00:00    0.993358
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00         NaN
Name: Close, dtype: float64
```

```python
>>> h1_close = h1_data.get("Close")
>>> h4_close = h4_data.get("Close")

>>> h1_close.iloc[:4]
Open time
2020-01-01 00:00:00+00:00    7177.02
2020-01-01 01:00:00+00:00    7216.27
2020-01-01 02:00:00+00:00    7242.85
2020-01-01 03:00:00+00:00    7225.01
Name: Close, dtype: float64

>>> h4_close.iloc[:1]
Open time
2020-01-01 00:00:00+00:00    7225.01
Freq: 4H, Name: Close, dtype: float64

>>> h1_h4_ratio = h1_close / h4_close
>>> h1_h4_ratio.iloc[:4]
Open time
2020-01-01 00:00:00+00:00    0.993358
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00         NaN
Name: Close, dtype: float64
```

If you think the result is right, you're wrong

Here, only the timestamp 2020-01-01 00:00:00 in both time series is getting compared, while other timestamps become NaN. This is understandable since Pandas compares values by their index labels. So far so good. The actual problem lies in the fact that each timestamp is the open time and holds information all the way up to the next timestamp. In reality, the close price stored at 2020-01-01 00:00:00 happens right before 2020-01-01 01:00:00 in h1_close and right before 2020-01-01 04:00:00 in h4_close. This means that we are comparing information from the past with information from the future, effectively exposing ourselves to the look-ahead bias!

```python
2020-01-01 00:00:00
```

```python
2020-01-01 00:00:00
```

```python
2020-01-01 01:00:00
```

```python
h1_close
```

```python
2020-01-01 04:00:00
```

```python
h4_close
```

```python
flowchart TD;
    subgraph H4 [ ]
    id1["2020-01-01 00:00:00"]
    id2["2020-01-01 04:00:00"]

    id1 --o|"close_3"| id2;
    end
    subgraph H1 [ ]
    id3["2020-01-01 00:00:00"]
    id4["2020-01-01 01:00:00"]
    id5["2020-01-01 02:00:00"]
    id6["2020-01-01 03:00:00"]
    id7["2020-01-01 04:00:00"]

    id3 --o|"close_0"| id4;
    id4 --o|"close_1"| id5;
    id5 --o|"close_2"| id6;
    id6 --o|"close_3"| id7;
    end
```

```python
flowchart TD;
    subgraph H4 [ ]
    id1["2020-01-01 00:00:00"]
    id2["2020-01-01 04:00:00"]

    id1 --o|"close_3"| id2;
    end
    subgraph H1 [ ]
    id3["2020-01-01 00:00:00"]
    id4["2020-01-01 01:00:00"]
    id5["2020-01-01 02:00:00"]
    id6["2020-01-01 03:00:00"]
    id7["2020-01-01 04:00:00"]

    id3 --o|"close_0"| id4;
    id4 --o|"close_1"| id5;
    id5 --o|"close_2"| id6;
    id6 --o|"close_3"| id7;
    end
```

As we can take from the diagram above, we are only allowed to compare close_3 between both time frames. Comparing close_0 and close_3 won't cause any errors (!), but you will get burned hard in production without having any idea why the backtesting results are so much off.

```python
close_3
```

```python
close_0
```

```python
close_3
```

If we want a more fair comparison of the close price, we should compare each timestamp in h1_close with the previous timestamp in h4_close:

```python
h1_close
```

```python
h4_close
```

```python
>>> h4_close_shifted = h4_close.shift()
>>> h1_h4_ratio = h1_close / h4_close_shifted
>>> h1_h4_ratio.iloc[:8]
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00         NaN
2020-01-01 04:00:00+00:00    0.998929
2020-01-01 05:00:00+00:00         NaN
2020-01-01 06:00:00+00:00         NaN
2020-01-01 07:00:00+00:00         NaN
Name: Close, dtype: float64
```

```python
>>> h4_close_shifted = h4_close.shift()
>>> h1_h4_ratio = h1_close / h4_close_shifted
>>> h1_h4_ratio.iloc[:8]
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00         NaN
2020-01-01 04:00:00+00:00    0.998929
2020-01-01 05:00:00+00:00         NaN
2020-01-01 06:00:00+00:00         NaN
2020-01-01 07:00:00+00:00         NaN
Name: Close, dtype: float64
```

This comparison makes more sense, since any timestamp before 2020-01-01 04:00:00 doesn't know about the close price at the end of 2020-01-01 03:00:00 yet. But even this comparison can be further improved because the close price at 2020-01-01 03:00:00 in h1_close is the same close price as at 2020-01-01 00:00:00 in h4_close. Thus, we can safely shift the resulting series backward:

```python
2020-01-01 04:00:00
```

```python
2020-01-01 03:00:00
```

```python
2020-01-01 03:00:00
```

```python
h1_close
```

```python
2020-01-01 00:00:00
```

```python
h4_close
```

```python
>>> h1_h4_ratio.shift(-1).iloc[:8]
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00    0.998929
2020-01-01 04:00:00+00:00         NaN
2020-01-01 05:00:00+00:00         NaN
2020-01-01 06:00:00+00:00         NaN
2020-01-01 07:00:00+00:00    0.998725
Name: Close, dtype: float64
```

```python
>>> h1_h4_ratio.shift(-1).iloc[:8]
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00    0.998929
2020-01-01 04:00:00+00:00         NaN
2020-01-01 05:00:00+00:00         NaN
2020-01-01 06:00:00+00:00         NaN
2020-01-01 07:00:00+00:00    0.998725
Name: Close, dtype: float64
```

What about all those NaNs, why aren't they compared as well? For this, we need to upsample the h4_close to H1 and forward-fill NaN values:

```python
h4_close
```

```python
H1
```

```python
>>> h4_h1_close = h4_close.shift(1).resample("1h").last().shift(-1).ffill()
>>> h4_h1_close.iloc[:8]
Open time
2020-01-01 00:00:00+00:00        NaN
2020-01-01 01:00:00+00:00        NaN
2020-01-01 02:00:00+00:00        NaN
2020-01-01 03:00:00+00:00    7225.01  << first close
2020-01-01 04:00:00+00:00    7225.01
2020-01-01 05:00:00+00:00    7225.01
2020-01-01 06:00:00+00:00    7225.01
2020-01-01 07:00:00+00:00    7209.83  << second close
Freq: H, Name: Close, dtype: float64
```

```python
>>> h4_h1_close = h4_close.shift(1).resample("1h").last().shift(-1).ffill()
>>> h4_h1_close.iloc[:8]
Open time
2020-01-01 00:00:00+00:00        NaN
2020-01-01 01:00:00+00:00        NaN
2020-01-01 02:00:00+00:00        NaN
2020-01-01 03:00:00+00:00    7225.01  << first close
2020-01-01 04:00:00+00:00    7225.01
2020-01-01 05:00:00+00:00    7225.01
2020-01-01 06:00:00+00:00    7225.01
2020-01-01 07:00:00+00:00    7209.83  << second close
Freq: H, Name: Close, dtype: float64
```

Note

Don't forward and backward shift when downsampling, only when upsampling.

Let's plot the first 16 points of both time series for validation:

```python
>>> fig = h1_close.rename("H1").iloc[:16].vbt.plot()
>>> h4_h1_close.rename("H4_H1").iloc[:16].vbt.plot(fig=fig)
>>> fig.show()
```

```python
>>> fig = h1_close.rename("H1").iloc[:16].vbt.plot()
>>> h4_h1_close.rename("H4_H1").iloc[:16].vbt.plot(fig=fig)
>>> fig.show()
```

As we see, at each hour, H4_H1 contains the latest available information from the previous 4 hours. We can now compare both time frames safely:

```python
H4_H1
```

```python
>>> h1_h4_ratio = h1_close / h4_h1_close
>>> h1_h4_ratio
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00    1.000000
2020-01-01 04:00:00+00:00    0.998929
...                               ...
2020-12-31 19:00:00+00:00    1.000000
2020-12-31 20:00:00+00:00    1.007920
2020-12-31 21:00:00+00:00         NaN
2020-12-31 22:00:00+00:00         NaN
2020-12-31 23:00:00+00:00         NaN
Name: Close, Length: 8784, dtype: float64
```

```python
>>> h1_h4_ratio = h1_close / h4_h1_close
>>> h1_h4_ratio
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00    1.000000
2020-01-01 04:00:00+00:00    0.998929
...                               ...
2020-12-31 19:00:00+00:00    1.000000
2020-12-31 20:00:00+00:00    1.007920
2020-12-31 21:00:00+00:00         NaN
2020-12-31 22:00:00+00:00         NaN
2020-12-31 23:00:00+00:00         NaN
Name: Close, Length: 8784, dtype: float64
```

The same goes for high and low price since their information is also available only at the end of each bar. Open price, on the other hand, is already available at the beginning of each bar, so we don't need to shift H4 forward and backward:

```python
H4
```

```python
>>> h1_open = h1_data.get("Open")
>>> h4_open  = h4_data.get("Open")

>>> h1_open.iloc[:8]
Open time
2020-01-01 00:00:00+00:00    7195.24
2020-01-01 01:00:00+00:00    7176.47
2020-01-01 02:00:00+00:00    7215.52
2020-01-01 03:00:00+00:00    7242.66
2020-01-01 04:00:00+00:00    7225.00
2020-01-01 05:00:00+00:00    7217.26
2020-01-01 06:00:00+00:00    7224.24
2020-01-01 07:00:00+00:00    7225.88
Name: Open, dtype: float64

>>> h4_h1_open = h4_open.resample("1h").first().ffill()
>>> h4_h1_open.iloc[:8]
Open time
2020-01-01 00:00:00+00:00    7195.24  << first open
2020-01-01 01:00:00+00:00    7195.24
2020-01-01 02:00:00+00:00    7195.24
2020-01-01 03:00:00+00:00    7195.24
2020-01-01 04:00:00+00:00    7225.00  << second open
2020-01-01 05:00:00+00:00    7225.00
2020-01-01 06:00:00+00:00    7225.00
2020-01-01 07:00:00+00:00    7225.00
Freq: H, Name: Open, dtype: float64
```

```python
>>> h1_open = h1_data.get("Open")
>>> h4_open  = h4_data.get("Open")

>>> h1_open.iloc[:8]
Open time
2020-01-01 00:00:00+00:00    7195.24
2020-01-01 01:00:00+00:00    7176.47
2020-01-01 02:00:00+00:00    7215.52
2020-01-01 03:00:00+00:00    7242.66
2020-01-01 04:00:00+00:00    7225.00
2020-01-01 05:00:00+00:00    7217.26
2020-01-01 06:00:00+00:00    7224.24
2020-01-01 07:00:00+00:00    7225.88
Name: Open, dtype: float64

>>> h4_h1_open = h4_open.resample("1h").first().ffill()
>>> h4_h1_open.iloc[:8]
Open time
2020-01-01 00:00:00+00:00    7195.24  << first open
2020-01-01 01:00:00+00:00    7195.24
2020-01-01 02:00:00+00:00    7195.24
2020-01-01 03:00:00+00:00    7195.24
2020-01-01 04:00:00+00:00    7225.00  << second open
2020-01-01 05:00:00+00:00    7225.00
2020-01-01 06:00:00+00:00    7225.00
2020-01-01 07:00:00+00:00    7225.00
Freq: H, Name: Open, dtype: float64
```

## VBT¶

Seems like a lot of work is required to do everything right? But don't worry, vectorbt has got your back! To realign an array safely, we can use GenericAccessor.realign_opening for information available already at the beginning of each bar, and GenericAccessor.realign_closing for information available only at the end of each bar:

```python
>>> h4_close.vbt.realign_closing("1h")
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00     7225.01
2020-01-01 04:00:00+00:00     7225.01
...                               ...
2020-12-31 16:00:00+00:00    28770.00
2020-12-31 17:00:00+00:00    28770.00
2020-12-31 18:00:00+00:00    28770.00
2020-12-31 19:00:00+00:00    28897.83
2020-12-31 20:00:00+00:00    28897.83
Freq: H, Name: Close, Length: 8781, dtype: float64

>>> h4_open.vbt.realign_opening("1h")
Open time
2020-01-01 00:00:00+00:00     7195.24
2020-01-01 01:00:00+00:00     7195.24
2020-01-01 02:00:00+00:00     7195.24
2020-01-01 03:00:00+00:00     7195.24
2020-01-01 04:00:00+00:00     7225.00
...                               ...
2020-12-31 16:00:00+00:00    28782.01
2020-12-31 17:00:00+00:00    28782.01
2020-12-31 18:00:00+00:00    28782.01
2020-12-31 19:00:00+00:00    28782.01
2020-12-31 20:00:00+00:00    28897.84
Freq: H, Name: Open, Length: 8781, dtype: float64
```

```python
>>> h4_close.vbt.realign_closing("1h")
Open time
2020-01-01 00:00:00+00:00         NaN
2020-01-01 01:00:00+00:00         NaN
2020-01-01 02:00:00+00:00         NaN
2020-01-01 03:00:00+00:00     7225.01
2020-01-01 04:00:00+00:00     7225.01
...                               ...
2020-12-31 16:00:00+00:00    28770.00
2020-12-31 17:00:00+00:00    28770.00
2020-12-31 18:00:00+00:00    28770.00
2020-12-31 19:00:00+00:00    28897.83
2020-12-31 20:00:00+00:00    28897.83
Freq: H, Name: Close, Length: 8781, dtype: float64

>>> h4_open.vbt.realign_opening("1h")
Open time
2020-01-01 00:00:00+00:00     7195.24
2020-01-01 01:00:00+00:00     7195.24
2020-01-01 02:00:00+00:00     7195.24
2020-01-01 03:00:00+00:00     7195.24
2020-01-01 04:00:00+00:00     7225.00
...                               ...
2020-12-31 16:00:00+00:00    28782.01
2020-12-31 17:00:00+00:00    28782.01
2020-12-31 18:00:00+00:00    28782.01
2020-12-31 19:00:00+00:00    28782.01
2020-12-31 20:00:00+00:00    28897.84
Freq: H, Name: Open, Length: 8781, dtype: float64
```

Important

Use realign_opening only if information in the array happens exactly at the beginning of the bar (such as open price), and realign_closing if information happens after that (such as high, low, and close price).

```python
realign_opening
```

```python
realign_closing
```

That's it - the results above can now be safely combined with any H1 data

```python
H1
```

### Resampler¶

If you want to gain a deeper understanding of the inner workings of those two functions, let's first discuss what does "resampler" mean in vectorbt. Resampler is an instance of the Resampler class, which simply stores a source index and frequency, and a target index and frequency:

```python
>>> h4_h1_resampler = h4_close.vbt.wrapper.get_resampler("1h")  # (1)!
>>> h4_h1_resampler.source_index
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 04:00:00+00:00',
               '2020-01-01 08:00:00+00:00', '2020-01-01 12:00:00+00:00',
               '2020-01-01 16:00:00+00:00', '2020-01-01 20:00:00+00:00',
               ...
               '2020-12-31 00:00:00+00:00', '2020-12-31 04:00:00+00:00',
               '2020-12-31 08:00:00+00:00', '2020-12-31 12:00:00+00:00',
               '2020-12-31 16:00:00+00:00', '2020-12-31 20:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=2196, freq='4H')

>>> h4_h1_resampler.target_index
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00',
               '2020-01-01 02:00:00+00:00', '2020-01-01 03:00:00+00:00',
               '2020-01-01 04:00:00+00:00', '2020-01-01 05:00:00+00:00',
               ...
               '2020-12-31 15:00:00+00:00', '2020-12-31 16:00:00+00:00',
               '2020-12-31 17:00:00+00:00', '2020-12-31 18:00:00+00:00',
               '2020-12-31 19:00:00+00:00', '2020-12-31 20:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=8781, freq='H')

>>> h4_h1_resampler.source_freq
Timedelta('0 days 04:00:00')

>>> h4_h1_resampler.target_freq
Timedelta('0 days 01:00:00')
```

```python
>>> h4_h1_resampler = h4_close.vbt.wrapper.get_resampler("1h")  # (1)!
>>> h4_h1_resampler.source_index
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 04:00:00+00:00',
               '2020-01-01 08:00:00+00:00', '2020-01-01 12:00:00+00:00',
               '2020-01-01 16:00:00+00:00', '2020-01-01 20:00:00+00:00',
               ...
               '2020-12-31 00:00:00+00:00', '2020-12-31 04:00:00+00:00',
               '2020-12-31 08:00:00+00:00', '2020-12-31 12:00:00+00:00',
               '2020-12-31 16:00:00+00:00', '2020-12-31 20:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=2196, freq='4H')

>>> h4_h1_resampler.target_index
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00',
               '2020-01-01 02:00:00+00:00', '2020-01-01 03:00:00+00:00',
               '2020-01-01 04:00:00+00:00', '2020-01-01 05:00:00+00:00',
               ...
               '2020-12-31 15:00:00+00:00', '2020-12-31 16:00:00+00:00',
               '2020-12-31 17:00:00+00:00', '2020-12-31 18:00:00+00:00',
               '2020-12-31 19:00:00+00:00', '2020-12-31 20:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=8781, freq='H')

>>> h4_h1_resampler.source_freq
Timedelta('0 days 04:00:00')

>>> h4_h1_resampler.target_freq
Timedelta('0 days 01:00:00')
```

Having just those two pieces of information is enough to perform most resampling tasks, all we have to do is to iterate over both indexes and track which element in one index belongs to which element in the other one, which is done efficiently using Numba. This also means that, in contrast to Pandas, vectorbt accepts an arbitrary target index for resampling. In fact, Resampler has a bunch of convenient class methods to construct an instance out of various Pandas objects and functions. For example, to create a resampler out of a Pandas Resampler:

```python
>>> pd_resampler = h4_close.resample("1h")
>>> vbt.Resampler.from_pd_resampler(pd_resampler, source_freq="4h")
<vectorbtpro.base.resampling.base.Resampler at 0x7ff518c3f358>
```

```python
>>> pd_resampler = h4_close.resample("1h")
>>> vbt.Resampler.from_pd_resampler(pd_resampler, source_freq="4h")
<vectorbtpro.base.resampling.base.Resampler at 0x7ff518c3f358>
```

Or, we can use Resampler.from_date_range to build our custom hourly index starting from 10 AM and ending at 10 PM:

```python
>>> resampler = vbt.Resampler.from_date_range(
...     source_index=h4_close.index,
...     source_freq="4h",
...     start="2020-01-01 10:00:00",
...     end="2020-01-01 22:00:00",
...     freq="1h",
... )
```

```python
>>> resampler = vbt.Resampler.from_date_range(
...     source_index=h4_close.index,
...     source_freq="4h",
...     start="2020-01-01 10:00:00",
...     end="2020-01-01 22:00:00",
...     freq="1h",
... )
```

We can then pass the resampler directly to GenericAccessor.realign_closing to fill the latest information from the H4 time frame:

```python
H4
```

```python
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01 10:00:00    7209.83
2020-01-01 11:00:00    7197.20
2020-01-01 12:00:00    7197.20
2020-01-01 13:00:00    7197.20
2020-01-01 14:00:00    7197.20
2020-01-01 15:00:00    7234.19
2020-01-01 16:00:00    7234.19
2020-01-01 17:00:00    7234.19
2020-01-01 18:00:00    7234.19
2020-01-01 19:00:00    7229.48
2020-01-01 20:00:00    7229.48
2020-01-01 21:00:00    7229.48
2020-01-01 22:00:00    7229.48
Freq: H, Name: Close, dtype: float64
```

```python
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01 10:00:00    7209.83
2020-01-01 11:00:00    7197.20
2020-01-01 12:00:00    7197.20
2020-01-01 13:00:00    7197.20
2020-01-01 14:00:00    7197.20
2020-01-01 15:00:00    7234.19
2020-01-01 16:00:00    7234.19
2020-01-01 17:00:00    7234.19
2020-01-01 18:00:00    7234.19
2020-01-01 19:00:00    7229.48
2020-01-01 20:00:00    7229.48
2020-01-01 21:00:00    7229.48
2020-01-01 22:00:00    7229.48
Freq: H, Name: Close, dtype: float64
```

### Custom index¶

We can also specify our target index directly. For instance, let's get the latest information on H4 at the beginning of each month (= downsampling). Note that without providing the target frequency explicitly, vectorbt will infer it from the target index, which is MonthBegin of type DateOffset in our case. Date offsets like this cannot be converted into a timedelta required by the underlying Numba function - Numba accepts only numeric and np.timedelta64 for frequency (see this overview). To prevent inferring the frequency, we can set it to False. In this case, vectorbt will calculate the right bound for each index value using the next index value, as opposed to a fixed frequency.

```python
H4
```

```python
MonthBegin
```

```python
np.timedelta64
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
...     "2020-03-01",
...     "2020-04-01",
...     "2020-05-01",
...     "2020-06-01",
...     "2020-07-01",
...     "2020-08-01",
...     "2020-09-01",
...     "2020-10-01",
...     "2020-11-01",
...     "2020-12-01",
...     "2021-01-01"
... ])
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq=False)
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01     9352.89
2020-02-01     8523.61
2020-03-01     6410.44
2020-04-01     8620.00
2020-05-01     9448.27
2020-06-01     9138.55
2020-07-01    11335.46
2020-08-01    11649.51
2020-09-01    10776.59
2020-10-01    13791.00
2020-11-01    19695.87
2020-12-01    28923.63
2021-01-01    28923.63
Name: Close, dtype: float64
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
...     "2020-03-01",
...     "2020-04-01",
...     "2020-05-01",
...     "2020-06-01",
...     "2020-07-01",
...     "2020-08-01",
...     "2020-09-01",
...     "2020-10-01",
...     "2020-11-01",
...     "2020-12-01",
...     "2021-01-01"
... ])
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq=False)
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01     9352.89
2020-02-01     8523.61
2020-03-01     6410.44
2020-04-01     8620.00
2020-05-01     9448.27
2020-06-01     9138.55
2020-07-01    11335.46
2020-08-01    11649.51
2020-09-01    10776.59
2020-10-01    13791.00
2020-11-01    19695.87
2020-12-01    28923.63
2021-01-01    28923.63
Name: Close, dtype: float64
```

To make sure that the output is correct, let's validate the close value for 2020-08-01, which must be the latest value for that month:

```python
2020-08-01
```

```python
>>> h4_close[h4_close.index < "2020-09-01"].iloc[-1]
11649.51
```

```python
>>> h4_close[h4_close.index < "2020-09-01"].iloc[-1]
11649.51
```

Hint

Disabling the frequency is only required for offsets that cannot be translated to timedelta. Other offsets, such as for daily data, are converted automatically and without issues.

One major drawback of disabling or not being able to infer the frequency of a target index is that the bounds of each index value are not fixed anymore, but variable. Consider the following scenario where we want to downsample H4 to two dates, 2020-01-01 and 2020-02-01, knowing that they are months. If we do not specify the target frequency, vectorbt uses the latest close price after each of those dates:

```python
H4
```

```python
2020-01-01
```

```python
2020-02-01
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq=False)
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01     9352.89
2020-02-01    28923.63
Name: Close, dtype: float64
```

```python
>>> target_index = pd.Index([
...     "2020-01-01",
...     "2020-02-01",
... ])
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq=False)
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01     9352.89
2020-02-01    28923.63
Name: Close, dtype: float64
```

The value for 2020-02-01 is the latest value in H4, which is clearly not intended. To limit the bounds of all index values, set the target frequency:

```python
2020-02-01
```

```python
H4
```

```python
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq="30d")
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01    9513.21
2020-02-01    8531.88
Name: Close, dtype: float64
```

```python
>>> resampler = vbt.Resampler(h4_close.index, target_index, target_freq="30d")
>>> h4_close.vbt.realign_closing(resampler)
2020-01-01    9513.21
2020-02-01    8531.88
Name: Close, dtype: float64
```

Much better, but even this is wrong because a month may also be more or less than 30 days long. Since we cannot use date offsets, we need to specify the bounds for each index value. This is possible by using GenericAccessor.realign, which is the base method for both realign_closing and realign_opening we used above. This method is a true powerhouse that allows specifying every bit of information manually. The main idea is simple: return the latest available information from an array at each position in a target index.

```python
realign_closing
```

```python
realign_opening
```

For example, let's get the latest information on H4 on some specific time:

```python
H4
```

```python
>>> h4_open.vbt.realign("2020-06-07 12:15:00")  # (1)!
9576.57

>>> h4_close.vbt.realign(
...     "2020-06-07 12:15:00", 
...     source_rbound=True  # (2)!
... )
9575.59
```

```python
>>> h4_open.vbt.realign("2020-06-07 12:15:00")  # (1)!
9576.57

>>> h4_close.vbt.realign(
...     "2020-06-07 12:15:00", 
...     source_rbound=True  # (2)!
... )
9575.59
```

```python
source_rbound=False
```

```python
source_rbound=True
```

Note that the target datetime we provided is an exact point in time when information should become available. Let's get the latest highest value at the beginning of two months in target_index:

```python
target_index
```

```python
>>> h4_high = h4_data.get("High")
>>> h4_high.vbt.realign(
...     target_index, 
...     source_rbound=True
... )
2020-01-01       NaN
2020-02-01    9430.0
Name: High, dtype: float64
```

```python
>>> h4_high = h4_data.get("High")
>>> h4_high.vbt.realign(
...     target_index, 
...     source_rbound=True
... )
2020-01-01       NaN
2020-02-01    9430.0
Name: High, dtype: float64
```

Here, 2020-01-01 means exactly 2020-01-01 00:00:00 when there is no information yet, hence NaN. On 2020-02-01 though, we can use the information from 2020-01-31 20:00:00:

```python
2020-01-01
```

```python
2020-01-01 00:00:00
```

```python
2020-02-01
```

```python
2020-01-31 20:00:00
```

```python
>>> h4_high.index[h4_high.index < "2020-02-01"][-1]
Timestamp('2020-01-31 20:00:00+0000', tz='UTC', freq='4H')
```

```python
>>> h4_high.index[h4_high.index < "2020-02-01"][-1]
Timestamp('2020-01-31 20:00:00+0000', tz='UTC', freq='4H')
```

To make the target index behave like bars instead of exact points in time, we can turn on the right bound for it too:

```python
>>> h4_high.vbt.realign(
...     target_index, 
...     source_rbound=True,
...     target_rbound=True
... )
UserWarning: Using right bound of target index without frequency. Set target_freq.
2020-01-01     9430.00
2020-02-01    29169.55
Name: High, dtype: float64
```

```python
>>> h4_high.vbt.realign(
...     target_index, 
...     source_rbound=True,
...     target_rbound=True
... )
UserWarning: Using right bound of target index without frequency. Set target_freq.
2020-01-01     9430.00
2020-02-01    29169.55
Name: High, dtype: float64
```

We received a warning stating that vectorbt couldn't infer the frequency of target_index. But to not make the same mistake we did with a frequency of 30d (a month has a variable length after all), let's specify the right bound manually instead of enabling target_rbound. Thankfully, vectorbt has a nice method for doing that:

```python
target_index
```

```python
30d
```

```python
target_rbound
```

```python
>>> resampler = vbt.Resampler(h4_high.index, target_index)
>>> resampler.target_rbound_index  # (1)!
DatetimeIndex([
    '2020-01-31 23:59:59.999999999', 
    '2262-04-11 23:47:16.854775807'
], dtype='datetime64[ns]', freq=None)

>>> resampler = vbt.Resampler(
...     h4_high.index, 
...     target_index, 
...     target_freq=pd.offsets.MonthBegin(1))
>>> resampler.target_rbound_index  # (2)!
DatetimeIndex([
    '2020-01-31 23:59:59.999999999', 
    '2020-02-29 23:59:59.999999999'
], dtype='datetime64[ns]', freq=None)
```

```python
>>> resampler = vbt.Resampler(h4_high.index, target_index)
>>> resampler.target_rbound_index  # (1)!
DatetimeIndex([
    '2020-01-31 23:59:59.999999999', 
    '2262-04-11 23:47:16.854775807'
], dtype='datetime64[ns]', freq=None)

>>> resampler = vbt.Resampler(
...     h4_high.index, 
...     target_index, 
...     target_freq=pd.offsets.MonthBegin(1))
>>> resampler.target_rbound_index  # (2)!
DatetimeIndex([
    '2020-01-31 23:59:59.999999999', 
    '2020-02-29 23:59:59.999999999'
], dtype='datetime64[ns]', freq=None)
```

We can now use this right bound as the target index:

```python
>>> h4_high.vbt.realign(
...     resampler.replace(
...         target_index=resampler.target_rbound_index, 
...         target_freq=False
...     ),  # (1)!
...     wrap_kwargs=dict(index=target_index)  # (2)!
... )
2020-01-01    9430.00
2020-02-01    8671.61
Name: High, dtype: float64
```

```python
>>> h4_high.vbt.realign(
...     resampler.replace(
...         target_index=resampler.target_rbound_index, 
...         target_freq=False
...     ),  # (1)!
...     wrap_kwargs=dict(index=target_index)  # (2)!
... )
2020-01-01    9430.00
2020-02-01    8671.61
Name: High, dtype: float64
```

```python
target_index
```

Or conveniently using target_rbound="pandas" in GenericAccessor.realign:

```python
target_rbound="pandas"
```

```python
>>> h4_high.vbt.realign(
...     target_index, 
...     freq=pd.offsets.MonthBegin(1),
...     target_rbound="pandas"
... )
2020-01-01    9430.00
2020-02-01    8671.61
Name: High, dtype: float64
```

```python
>>> h4_high.vbt.realign(
...     target_index, 
...     freq=pd.offsets.MonthBegin(1),
...     target_rbound="pandas"
... )
2020-01-01    9430.00
2020-02-01    8671.61
Name: High, dtype: float64
```

Let's validate the output using Pandas:

```python
>>> h4_high[h4_high.index < "2020-03-01"].resample(vbt.offset("M")).last()  # (1)!
Open time
2020-01-01 00:00:00+00:00    9430.00
2020-02-01 00:00:00+00:00    8671.61
Freq: MS, Name: High, dtype: float64
```

```python
>>> h4_high[h4_high.index < "2020-03-01"].resample(vbt.offset("M")).last()  # (1)!
Open time
2020-01-01 00:00:00+00:00    9430.00
2020-02-01 00:00:00+00:00    8671.61
Freq: MS, Name: High, dtype: float64
```

A clear advantage of vectorbt over Pandas in this regard is:

Info

Why did we use vbt.offset("M") instead of just "M"? Pandas methods may have a different interpretation of offsets than VBT methods. For instance, Pandas interprets "M" as the month end while VBT interprets it as the month start (since we're working with bars most of the time). As a rule of thumb: explicitly translate any string offset if it must be passed to a Pandas method. If the method belongs to VBT, there's usually no need to perform this step.

```python
vbt.offset("M")
```

```python
"M"
```

### Numeric index¶

And we haven't even mentioned the ability to do resampling on numeric indexes. Below, we are getting the latest information at each 6th element from the H4 time frame, which is just another way of downsampling to the daily frequency (as long as there are no gaps):

```python
H4
```

```python
>>> resampler = vbt.Resampler(
...     source_index=np.arange(len(h4_high)),
...     target_index=np.arange(len(h4_high))[::6],
...     source_freq=1,
...     target_freq=6
... )
>>> h4_high.vbt.realign(
...     resampler, 
...     source_rbound=True,
...     target_rbound=True
... )
0        7242.98
6        6985.00
12       7361.00
...          ...
2178    27410.00
2184    28996.00
2190    29169.55
Name: High, Length: 366, dtype: float64
```

```python
>>> resampler = vbt.Resampler(
...     source_index=np.arange(len(h4_high)),
...     target_index=np.arange(len(h4_high))[::6],
...     source_freq=1,
...     target_freq=6
... )
>>> h4_high.vbt.realign(
...     resampler, 
...     source_rbound=True,
...     target_rbound=True
... )
0        7242.98
6        6985.00
12       7361.00
...          ...
2178    27410.00
2184    28996.00
2190    29169.55
Name: High, Length: 366, dtype: float64
```

Good luck doing the same with Pandas.

### Forward filling¶

By default, when upsampling or downsampling, vectorbt will forward fill missing values by propagating the latest known value. This is usually desired when the final task is to compare the resampled time series to another time series of the same timeframe. But this may not hold well for some more sensitive time series types, such as signals: repeating the same signal over and over again may give a distorted view of the original timeframe, especially when upsampling. To place each value only once, we can use the argument ffill. For example, let's upsample a 5min mask with 3 entries to a 1min mask with 15 entries, without and with forward filling. We'll assume that the 5min mask itself was generated using the close price:

```python
ffill
```

```python
>>> min5_index = vbt.date_range(start="2020", freq="5min", periods=3)
>>> min1_index = vbt.date_range(start="2020", freq="1min", periods=15)
>>> min5_mask = pd.Series(False, index=min5_index)
>>> min5_mask.iloc[0] = True  # (1)!
>>> min5_mask.iloc[2] = True

>>> resampler = vbt.Resampler(min5_index, min1_index)
>>> min1_mask = min5_mask.vbt.realign_closing(resampler)  # (2)!
>>> min1_mask
2020-01-01 00:00:00    NaN
2020-01-01 00:01:00    NaN
2020-01-01 00:02:00    NaN
2020-01-01 00:03:00    NaN
2020-01-01 00:04:00    1.0
2020-01-01 00:05:00    1.0
2020-01-01 00:06:00    1.0
2020-01-01 00:07:00    1.0
2020-01-01 00:08:00    1.0
2020-01-01 00:09:00    0.0
2020-01-01 00:10:00    0.0
2020-01-01 00:11:00    0.0
2020-01-01 00:12:00    0.0
2020-01-01 00:13:00    0.0
2020-01-01 00:14:00    1.0
Freq: T, dtype: float64

>>> min1_mask = min5_mask.vbt.realign_closing(resampler, ffill=False)  # (3)!
>>> min1_mask
2020-01-01 00:00:00    NaN
2020-01-01 00:01:00    NaN
2020-01-01 00:02:00    NaN
2020-01-01 00:03:00    NaN
2020-01-01 00:04:00    1.0
2020-01-01 00:05:00    NaN
2020-01-01 00:06:00    NaN
2020-01-01 00:07:00    NaN
2020-01-01 00:08:00    NaN
2020-01-01 00:09:00    0.0
2020-01-01 00:10:00    NaN
2020-01-01 00:11:00    NaN
2020-01-01 00:12:00    NaN
2020-01-01 00:13:00    NaN
2020-01-01 00:14:00    1.0
Freq: T, dtype: float64

>>> min1_mask = min1_mask.fillna(False).astype(bool)  # (4)!
>>> min1_mask
2020-01-01 00:00:00    False
2020-01-01 00:01:00    False
2020-01-01 00:02:00    False
2020-01-01 00:03:00    False
2020-01-01 00:04:00     True
2020-01-01 00:05:00    False
2020-01-01 00:06:00    False
2020-01-01 00:07:00    False
2020-01-01 00:08:00    False
2020-01-01 00:09:00    False
2020-01-01 00:10:00    False
2020-01-01 00:11:00    False
2020-01-01 00:12:00    False
2020-01-01 00:13:00    False
2020-01-01 00:14:00     True
Freq: T, dtype: bool
```

```python
>>> min5_index = vbt.date_range(start="2020", freq="5min", periods=3)
>>> min1_index = vbt.date_range(start="2020", freq="1min", periods=15)
>>> min5_mask = pd.Series(False, index=min5_index)
>>> min5_mask.iloc[0] = True  # (1)!
>>> min5_mask.iloc[2] = True

>>> resampler = vbt.Resampler(min5_index, min1_index)
>>> min1_mask = min5_mask.vbt.realign_closing(resampler)  # (2)!
>>> min1_mask
2020-01-01 00:00:00    NaN
2020-01-01 00:01:00    NaN
2020-01-01 00:02:00    NaN
2020-01-01 00:03:00    NaN
2020-01-01 00:04:00    1.0
2020-01-01 00:05:00    1.0
2020-01-01 00:06:00    1.0
2020-01-01 00:07:00    1.0
2020-01-01 00:08:00    1.0
2020-01-01 00:09:00    0.0
2020-01-01 00:10:00    0.0
2020-01-01 00:11:00    0.0
2020-01-01 00:12:00    0.0
2020-01-01 00:13:00    0.0
2020-01-01 00:14:00    1.0
Freq: T, dtype: float64

>>> min1_mask = min5_mask.vbt.realign_closing(resampler, ffill=False)  # (3)!
>>> min1_mask
2020-01-01 00:00:00    NaN
2020-01-01 00:01:00    NaN
2020-01-01 00:02:00    NaN
2020-01-01 00:03:00    NaN
2020-01-01 00:04:00    1.0
2020-01-01 00:05:00    NaN
2020-01-01 00:06:00    NaN
2020-01-01 00:07:00    NaN
2020-01-01 00:08:00    NaN
2020-01-01 00:09:00    0.0
2020-01-01 00:10:00    NaN
2020-01-01 00:11:00    NaN
2020-01-01 00:12:00    NaN
2020-01-01 00:13:00    NaN
2020-01-01 00:14:00    1.0
Freq: T, dtype: float64

>>> min1_mask = min1_mask.fillna(False).astype(bool)  # (4)!
>>> min1_mask
2020-01-01 00:00:00    False
2020-01-01 00:01:00    False
2020-01-01 00:02:00    False
2020-01-01 00:03:00    False
2020-01-01 00:04:00     True
2020-01-01 00:05:00    False
2020-01-01 00:06:00    False
2020-01-01 00:07:00    False
2020-01-01 00:08:00    False
2020-01-01 00:09:00    False
2020-01-01 00:10:00    False
2020-01-01 00:11:00    False
2020-01-01 00:12:00    False
2020-01-01 00:13:00    False
2020-01-01 00:14:00     True
Freq: T, dtype: bool
```

```python
True
```

```python
False
```

```python
True
```

## Indicators¶

So, how do we use the resampling logic from above in constructing indicators that combine multiple time frames? Quite easily:

Let's demonstrate this by calculating the crossover of two moving averages on the time frames H4 and D1. First, we will run the TA-Lib's SMA indicator on the close price of both time frames:

```python
H4
```

```python
D1
```

```python
>>> h4_sma = vbt.talib("SMA").run(
...     h4_data.get("Close"), 
...     skipna=True  # (1)!
... ).real
>>> d1_sma = vbt.talib("SMA").run(
...     d1_data.get("Close"), 
...     skipna=True
... ).real

>>> h4_sma = h4_sma.ffill()  # (2)!
>>> d1_sma = d1_sma.ffill()
```

```python
>>> h4_sma = vbt.talib("SMA").run(
...     h4_data.get("Close"), 
...     skipna=True  # (1)!
... ).real
>>> d1_sma = vbt.talib("SMA").run(
...     d1_data.get("Close"), 
...     skipna=True
... ).real

>>> h4_sma = h4_sma.ffill()  # (2)!
>>> d1_sma = d1_sma.ffill()
```

Then, upsample D1 to H4 such that both indicators have the same index:

```python
D1
```

```python
H4
```

```python
>>> resampler = vbt.Resampler(
...     d1_sma.index,  # (1)!
...     h4_sma.index,  # (2)!
...     source_freq="1d",
...     target_freq="4h"
... )
>>> d1_h4_sma = d1_sma.vbt.realign_closing(resampler)  # (3)!
```

```python
>>> resampler = vbt.Resampler(
...     d1_sma.index,  # (1)!
...     h4_sma.index,  # (2)!
...     source_freq="1d",
...     target_freq="4h"
... )
>>> d1_h4_sma = d1_sma.vbt.realign_closing(resampler)  # (3)!
```

```python
D1
```

```python
H1
```

Let's validate the result of resampling:

```python
>>> d1_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21746.412000
2020-12-31 00:00:00+00:00    22085.034333
Freq: D, Name: Close, dtype: float64

>>> d1_h4_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.423000
2020-12-30 04:00:00+00:00    21440.423000
2020-12-30 08:00:00+00:00    21440.423000
2020-12-30 12:00:00+00:00    21440.423000
2020-12-30 16:00:00+00:00    21440.423000
2020-12-30 20:00:00+00:00    21746.412000  << first value available
2020-12-31 00:00:00+00:00    21746.412000
2020-12-31 04:00:00+00:00    21746.412000
2020-12-31 08:00:00+00:00    21746.412000
2020-12-31 12:00:00+00:00    21746.412000
2020-12-31 16:00:00+00:00    21746.412000
2020-12-31 20:00:00+00:00    22085.034333  << second value available
Freq: 4H, Name: Close, dtype: float64
```

```python
>>> d1_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21746.412000
2020-12-31 00:00:00+00:00    22085.034333
Freq: D, Name: Close, dtype: float64

>>> d1_h4_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.423000
2020-12-30 04:00:00+00:00    21440.423000
2020-12-30 08:00:00+00:00    21440.423000
2020-12-30 12:00:00+00:00    21440.423000
2020-12-30 16:00:00+00:00    21440.423000
2020-12-30 20:00:00+00:00    21746.412000  << first value available
2020-12-31 00:00:00+00:00    21746.412000
2020-12-31 04:00:00+00:00    21746.412000
2020-12-31 08:00:00+00:00    21746.412000
2020-12-31 12:00:00+00:00    21746.412000
2020-12-31 16:00:00+00:00    21746.412000
2020-12-31 20:00:00+00:00    22085.034333  << second value available
Freq: 4H, Name: Close, dtype: float64
```

Finally, as usually, compare the new time series to produce entries and exits:

```python
>>> entries = h4_sma.vbt.crossed_above(d1_h4_sma)
>>> exits = h4_sma.vbt.crossed_below(d1_h4_sma)

>>> def plot_date_range(date_range):
...     fig = h4_sma[date_range].rename("H4").vbt.plot()
...     d1_h4_sma[date_range].rename("D1_H4").vbt.plot(fig=fig)
...     entries[date_range].rename("Entry").vbt.signals.plot_as_entries(
...         y=h4_sma[date_range], fig=fig)
...     exits[date_range].rename("Exit").vbt.signals.plot_as_exits(
...         y=h4_sma[date_range], fig=fig)
...     return fig

>>> plot_date_range(slice("2020-02-01", "2020-03-01")).show()
```

```python
>>> entries = h4_sma.vbt.crossed_above(d1_h4_sma)
>>> exits = h4_sma.vbt.crossed_below(d1_h4_sma)

>>> def plot_date_range(date_range):
...     fig = h4_sma[date_range].rename("H4").vbt.plot()
...     d1_h4_sma[date_range].rename("D1_H4").vbt.plot(fig=fig)
...     entries[date_range].rename("Entry").vbt.signals.plot_as_entries(
...         y=h4_sma[date_range], fig=fig)
...     exits[date_range].rename("Exit").vbt.signals.plot_as_exits(
...         y=h4_sma[date_range], fig=fig)
...     return fig

>>> plot_date_range(slice("2020-02-01", "2020-03-01")).show()
```

In case any calculation was performed on the open price, we can account for that by directly using GenericAccessor.realign and disabling the right bound of the affected index:

```python
>>> d1_open_sma = vbt.talib("SMA").run(
...     d1_data.get("Open"),  # (1)!
...     skipna=True
... ).real
>>> d1_open_sma = d1_open_sma.ffill()

>>> d1_h4_open_sma = d1_open_sma.vbt.realign(
...     resampler, 
...     source_rbound=False,  # (2)!
...     target_rbound=True,  # (3)!
... )
```

```python
>>> d1_open_sma = vbt.talib("SMA").run(
...     d1_data.get("Open"),  # (1)!
...     skipna=True
... ).real
>>> d1_open_sma = d1_open_sma.ffill()

>>> d1_h4_open_sma = d1_open_sma.vbt.realign(
...     resampler, 
...     source_rbound=False,  # (2)!
...     target_rbound=True,  # (3)!
... )
```

```python
D1
```

```python
H4
```

Let's validate the result of resampling:

```python
>>> d1_open_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.420333
2020-12-31 00:00:00+00:00    21746.409667
Freq: D, Name: Open, dtype: float64

>>> d1_h4_open_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.420333  << first value available
2020-12-30 04:00:00+00:00    21440.420333
2020-12-30 08:00:00+00:00    21440.420333
2020-12-30 12:00:00+00:00    21440.420333
2020-12-30 16:00:00+00:00    21440.420333
2020-12-30 20:00:00+00:00    21440.420333
2020-12-31 00:00:00+00:00    21746.409667  << second value available
2020-12-31 04:00:00+00:00    21746.409667
2020-12-31 08:00:00+00:00    21746.409667
2020-12-31 12:00:00+00:00    21746.409667
2020-12-31 16:00:00+00:00    21746.409667
2020-12-31 20:00:00+00:00    21746.409667
Freq: 4H, Name: Open, dtype: float64
```

```python
>>> d1_open_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.420333
2020-12-31 00:00:00+00:00    21746.409667
Freq: D, Name: Open, dtype: float64

>>> d1_h4_open_sma["2020-12-30":]
Open time
2020-12-30 00:00:00+00:00    21440.420333  << first value available
2020-12-30 04:00:00+00:00    21440.420333
2020-12-30 08:00:00+00:00    21440.420333
2020-12-30 12:00:00+00:00    21440.420333
2020-12-30 16:00:00+00:00    21440.420333
2020-12-30 20:00:00+00:00    21440.420333
2020-12-31 00:00:00+00:00    21746.409667  << second value available
2020-12-31 04:00:00+00:00    21746.409667
2020-12-31 08:00:00+00:00    21746.409667
2020-12-31 12:00:00+00:00    21746.409667
2020-12-31 16:00:00+00:00    21746.409667
2020-12-31 20:00:00+00:00    21746.409667
Freq: 4H, Name: Open, dtype: float64
```

Let's do something more fun: calculate the bandwidth of the Bollinger Bands indicator on a set of arbitrary frequencies and pack everything into a single DataFrame:

```python
>>> def generate_bandwidths(freqs):
...     bandwidths = []
...     for freq in freqs:
...         close = h1_data.resample(freq).get("Close")  # (1)!
...         bbands = vbt.talib("BBANDS").run(close, skipna=True)
...         upperband = bbands.upperband.ffill()
...         middleband = bbands.middleband.ffill()
...         lowerband = bbands.lowerband.ffill()
...         bandwidth = (upperband - lowerband) / middleband
...         bandwidths.append(bandwidth.vbt.realign_closing("1h"))  # (2)!
...     df = pd.concat(bandwidths, axis=1, keys=pd.Index(freqs, name="timeframe"))  # (3)!
...     return df.ffill()  # (4)!

>>> bandwidths = generate_bandwidths(["1h", "4h", "1d", "7d"])
>>> bandwidths
timeframe                        1h        4h        1d        7d
Open time                                                        
2020-01-01 00:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 01:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 04:00:00+00:00  0.011948       NaN       NaN       NaN
...                             ...       ...       ...       ...
2020-12-31 19:00:00+00:00  0.027320  0.017939  0.134607  0.652958
2020-12-31 20:00:00+00:00  0.036515  0.017939  0.134607  0.652958
2020-12-31 21:00:00+00:00  0.025027  0.017939  0.134607  0.652958
2020-12-31 22:00:00+00:00  0.014318  0.017939  0.134607  0.652958
2020-12-31 23:00:00+00:00  0.012875  0.017939  0.134607  0.652958

[8784 rows x 4 columns]
```

```python
>>> def generate_bandwidths(freqs):
...     bandwidths = []
...     for freq in freqs:
...         close = h1_data.resample(freq).get("Close")  # (1)!
...         bbands = vbt.talib("BBANDS").run(close, skipna=True)
...         upperband = bbands.upperband.ffill()
...         middleband = bbands.middleband.ffill()
...         lowerband = bbands.lowerband.ffill()
...         bandwidth = (upperband - lowerband) / middleband
...         bandwidths.append(bandwidth.vbt.realign_closing("1h"))  # (2)!
...     df = pd.concat(bandwidths, axis=1, keys=pd.Index(freqs, name="timeframe"))  # (3)!
...     return df.ffill()  # (4)!

>>> bandwidths = generate_bandwidths(["1h", "4h", "1d", "7d"])
>>> bandwidths
timeframe                        1h        4h        1d        7d
Open time                                                        
2020-01-01 00:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 01:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 04:00:00+00:00  0.011948       NaN       NaN       NaN
...                             ...       ...       ...       ...
2020-12-31 19:00:00+00:00  0.027320  0.017939  0.134607  0.652958
2020-12-31 20:00:00+00:00  0.036515  0.017939  0.134607  0.652958
2020-12-31 21:00:00+00:00  0.025027  0.017939  0.134607  0.652958
2020-12-31 22:00:00+00:00  0.014318  0.017939  0.134607  0.652958
2020-12-31 23:00:00+00:00  0.012875  0.017939  0.134607  0.652958

[8784 rows x 4 columns]
```

```python
H1
```

```python
H1
```

```python
H1
```

We can then plot the entire thing as a heatmap:

```python
>>> bandwidths.loc[:, ::-1].vbt.ts_heatmap().show()  # (1)!
```

```python
>>> bandwidths.loc[:, ::-1].vbt.ts_heatmap().show()  # (1)!
```

We just created such a badass visualization in 10 lines of code!

But we can make the code even shorter: each TA-Lib indicator has a timeframe parameter

```python
timeframe
```

```python
>>> bbands = vbt.talib("BBANDS").run(
...     h1_data.get("Close"), 
...     skipna=True, 
...     timeframe=["1h", "4h", "1d", "7d"],
...     broadcast_kwargs=dict(wrapper_kwargs=dict(freq="1h"))  # (1)!
... )
>>> bandwidths = (bbands.upperband - bbands.lowerband) / bbands.middleband
>>> bandwidths
timeframe                        1h        4h        1d        7d
Open time                                                        
2020-01-01 00:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 01:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 04:00:00+00:00  0.011948       NaN       NaN       NaN
...                             ...       ...       ...       ...
2020-12-31 19:00:00+00:00  0.027320  0.017939  0.134607  0.652958
2020-12-31 20:00:00+00:00  0.036515  0.017939  0.134607  0.652958
2020-12-31 21:00:00+00:00  0.025027  0.017939  0.134607  0.652958
2020-12-31 22:00:00+00:00  0.014318  0.017939  0.134607  0.652958
2020-12-31 23:00:00+00:00  0.012875  0.017939  0.134607  0.652958

[8784 rows x 4 columns]
```

```python
>>> bbands = vbt.talib("BBANDS").run(
...     h1_data.get("Close"), 
...     skipna=True, 
...     timeframe=["1h", "4h", "1d", "7d"],
...     broadcast_kwargs=dict(wrapper_kwargs=dict(freq="1h"))  # (1)!
... )
>>> bandwidths = (bbands.upperband - bbands.lowerband) / bbands.middleband
>>> bandwidths
timeframe                        1h        4h        1d        7d
Open time                                                        
2020-01-01 00:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 01:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 02:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 03:00:00+00:00       NaN       NaN       NaN       NaN
2020-01-01 04:00:00+00:00  0.011948       NaN       NaN       NaN
...                             ...       ...       ...       ...
2020-12-31 19:00:00+00:00  0.027320  0.017939  0.134607  0.652958
2020-12-31 20:00:00+00:00  0.036515  0.017939  0.134607  0.652958
2020-12-31 21:00:00+00:00  0.025027  0.017939  0.134607  0.652958
2020-12-31 22:00:00+00:00  0.014318  0.017939  0.134607  0.652958
2020-12-31 23:00:00+00:00  0.012875  0.017939  0.134607  0.652958

[8784 rows x 4 columns]
```

## Testing¶

As with everything in vectorbt, time frames are just another dimension that can be tested by iteration (loop over frequencies and simulate each one independently) or by stacking columns. If you don't want to inflate the data by storing multiple time frames in a single array, use the first approach. If you want to make decisions based on multiple time frames, or you want to test them from the same angle and using the same conditions (which is a prerequisite for a fair experiment), and you don't have much data to actually hit memory hard, use the second approach.

Let's demonstrate the second approach. Below, for each frequency, we are computing the SMA crossover on the open price of H1. We then align and concatenate all time frames, and simulate them as a single entity using the close price of H1and some stop loss. This way, we can test multiple time frames by keeping order execution as granular as possible.

```python
H1
```

```python
H1
```

```python
>>> def generate_signals(data, freq, fast_window, slow_window):
...     open_price = data.resample(freq).get("Open")  # (1)!
...     fast_sma = vbt.talib("SMA")\
...         .run(
...             open_price, 
...             fast_window, 
...             skipna=True, 
...             short_name="fast_sma"
...         )\
...         .real.ffill()\  # (2)!
...         .vbt.realign(data.wrapper.index)  # (3)!
...     slow_sma = vbt.talib("SMA")\
...         .run(
...             open_price, 
...             slow_window, 
...             skipna=True, 
...             short_name="slow_sma"
...         )\
...         .real.ffill()\
...         .vbt.realign(data.wrapper.index)
...     entries = fast_sma.vbt.crossed_above(slow_sma)  # (4)!
...     exits = fast_sma.vbt.crossed_below(slow_sma)
...     return entries, exits

>>> fast_window = [10, 20]  # (5)!
>>> slow_window = [20, 30]
>>> h1_entries, h1_exits = generate_signals(h1_data, "1h", fast_window, slow_window)
>>> h4_entries, h4_exits = generate_signals(h1_data, "4h", fast_window, slow_window)
>>> d1_entries, d1_exits = generate_signals(h1_data, "1d", fast_window, slow_window)

>>> entries = pd.concat(  # (6)!
...     (h1_entries, h4_entries, d1_entries), 
...     axis=1, 
...     keys=pd.Index(["1h", "4h", "1d"], name="timeframe")
... )
>>> exits = pd.concat(
...     (h1_exits, h4_exits, d1_exits), 
...     axis=1, 
...     keys=pd.Index(["1h", "4h", "1d"], name="timeframe")
... )

>>> (entries.astype(int) - exits.astype(int))\
...     .resample("1d").sum()\
...     .vbt.ts_heatmap(
...         trace_kwargs=dict(
...             colorscale=["#ef553b", "rgba(0, 0, 0, 0)", "#17becf"],
...             colorbar=dict(
...                 tickvals=[-1, 0, 1], 
...                 ticktext=["Exit", "", "Entry"]
...             )
...         )
...     ).show()  # (7)!
```

```python
>>> def generate_signals(data, freq, fast_window, slow_window):
...     open_price = data.resample(freq).get("Open")  # (1)!
...     fast_sma = vbt.talib("SMA")\
...         .run(
...             open_price, 
...             fast_window, 
...             skipna=True, 
...             short_name="fast_sma"
...         )\
...         .real.ffill()\  # (2)!
...         .vbt.realign(data.wrapper.index)  # (3)!
...     slow_sma = vbt.talib("SMA")\
...         .run(
...             open_price, 
...             slow_window, 
...             skipna=True, 
...             short_name="slow_sma"
...         )\
...         .real.ffill()\
...         .vbt.realign(data.wrapper.index)
...     entries = fast_sma.vbt.crossed_above(slow_sma)  # (4)!
...     exits = fast_sma.vbt.crossed_below(slow_sma)
...     return entries, exits

>>> fast_window = [10, 20]  # (5)!
>>> slow_window = [20, 30]
>>> h1_entries, h1_exits = generate_signals(h1_data, "1h", fast_window, slow_window)
>>> h4_entries, h4_exits = generate_signals(h1_data, "4h", fast_window, slow_window)
>>> d1_entries, d1_exits = generate_signals(h1_data, "1d", fast_window, slow_window)

>>> entries = pd.concat(  # (6)!
...     (h1_entries, h4_entries, d1_entries), 
...     axis=1, 
...     keys=pd.Index(["1h", "4h", "1d"], name="timeframe")
... )
>>> exits = pd.concat(
...     (h1_exits, h4_exits, d1_exits), 
...     axis=1, 
...     keys=pd.Index(["1h", "4h", "1d"], name="timeframe")
... )

>>> (entries.astype(int) - exits.astype(int))\
...     .resample("1d").sum()\
...     .vbt.ts_heatmap(
...         trace_kwargs=dict(
...             colorscale=["#ef553b", "rgba(0, 0, 0, 0)", "#17becf"],
...             colorbar=dict(
...                 tickvals=[-1, 0, 1], 
...                 ticktext=["Exit", "", "Entry"]
...             )
...         )
...     ).show()  # (7)!
```

```python
H1
```

```python
h1_data
```

```python
H1
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     h1_data,
...     entries,
...     exits,
...     sl_stop=0.1,
...     freq="1h"
... )

>>> pf.orders.count()
timeframe  fast_sma_timeperiod  slow_sma_timeperiod
1h         10                   20                     504
           20                   30                     379
4h         10                   20                     111
           20                   30                      85
1d         10                   20                      13
           20                   30                       7
Name: count, dtype: int64

>>> pf.sharpe_ratio
timeframe  fast_sma_timeperiod  slow_sma_timeperiod
1h         10                   20                     3.400095
           20                   30                     2.051091
4h         10                   20                     2.751626
           20                   30                     1.559501
1d         10                   20                     3.239846
           20                   30                     2.755367
Name: sharpe_ratio, dtype: float64
```

```python
>>> pf = vbt.Portfolio.from_signals(
...     h1_data,
...     entries,
...     exits,
...     sl_stop=0.1,
...     freq="1h"
... )

>>> pf.orders.count()
timeframe  fast_sma_timeperiod  slow_sma_timeperiod
1h         10                   20                     504
           20                   30                     379
4h         10                   20                     111
           20                   30                      85
1d         10                   20                      13
           20                   30                       7
Name: count, dtype: int64

>>> pf.sharpe_ratio
timeframe  fast_sma_timeperiod  slow_sma_timeperiod
1h         10                   20                     3.400095
           20                   30                     2.051091
4h         10                   20                     2.751626
           20                   30                     1.559501
1d         10                   20                     3.239846
           20                   30                     2.755367
Name: sharpe_ratio, dtype: float64
```

Python code  Notebook

