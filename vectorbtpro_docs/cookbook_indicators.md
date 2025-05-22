# Indicators¶

Question

Learn more in Indicators documentation.

## Listing¶

To list the currently supported indicators, use IndicatorFactory.list_indicators. The returned indicator names can be filtered by location, which can be listed with IndicatorFactory.list_locations, or by evaluating a glob or regex pattern.

```python
indicator_names = vbt.IF.list_indicators()  # (1)!
indicator_names = vbt.IF.list_indicators("vbt")  # (2)!
indicator_names = vbt.IF.list_indicators("talib")  # (3)!
indicator_names = vbt.IF.list_indicators("RSI*")  # (4)!
indicator_names = vbt.IF.list_indicators("*ma")  # (5)!
indicator_names = vbt.IF.list_indicators("[a-z]+ma$", use_regex=True)  # (6)!
indicator_names = vbt.IF.list_indicators("*ma", location="pandas_ta")  # (7)!

location_names = vbt.IF.list_locations()  # (8)!
```

```python
indicator_names = vbt.IF.list_indicators()  # (1)!
indicator_names = vbt.IF.list_indicators("vbt")  # (2)!
indicator_names = vbt.IF.list_indicators("talib")  # (3)!
indicator_names = vbt.IF.list_indicators("RSI*")  # (4)!
indicator_names = vbt.IF.list_indicators("*ma")  # (5)!
indicator_names = vbt.IF.list_indicators("[a-z]+ma$", use_regex=True)  # (6)!
indicator_names = vbt.IF.list_indicators("*ma", location="pandas_ta")  # (7)!

location_names = vbt.IF.list_locations()  # (8)!
```

Note

Without specifying a location, indicators across all the locations will be parsed, which may take some time. Thus, make sure to not repeatedly call this function; instead, save the results to a variable.

To get the class of an indicator, use IndicatorFactory.get_indicator.

```python
vbt.BBANDS  # (1)!

BBANDS = vbt.IF.get_indicator("pandas_ta:BBANDS")  # (2)!
BBANDS = vbt.indicator("pandas_ta:BBANDS")  # (3)!
BBANDS = vbt.IF.from_pandas_ta("BBANDS")  # (4)!
BBANDS = vbt.pandas_ta("BBANDS")  # (5)!

RSI = vbt.indicator("RSI")  # (6)!
```

```python
vbt.BBANDS  # (1)!

BBANDS = vbt.IF.get_indicator("pandas_ta:BBANDS")  # (2)!
BBANDS = vbt.indicator("pandas_ta:BBANDS")  # (3)!
BBANDS = vbt.IF.from_pandas_ta("BBANDS")  # (4)!
BBANDS = vbt.pandas_ta("BBANDS")  # (5)!

RSI = vbt.indicator("RSI")  # (6)!
```

```python
list_indicators
```

To get familiar with an indicator class, call phelp on the run class method, which is used to run the indicator. Alternatively, the specification such as input names is also available via various properties to be accessed in a programmable fashion.

```python
run
```

```python
vbt.phelp(vbt.OLS.run)  # (1)!

print(vbt.OLS.input_names)  # (2)!
print(vbt.OLS.param_names)  # (3)!
print(vbt.OLS.param_defaults)  # (4)!
print(vbt.OLS.in_output_names)  # (5)!
print(vbt.OLS.output_names)  # (6)!
print(vbt.OLS.lazy_output_names)  # (7)!
```

```python
vbt.phelp(vbt.OLS.run)  # (1)!

print(vbt.OLS.input_names)  # (2)!
print(vbt.OLS.param_names)  # (3)!
print(vbt.OLS.param_defaults)  # (4)!
print(vbt.OLS.in_output_names)  # (5)!
print(vbt.OLS.output_names)  # (6)!
print(vbt.OLS.lazy_output_names)  # (7)!
```

```python
run
```

```python
talib:SMA
```

```python
talib:SMA
```

```python
vbt:STX
```

```python
vbt:RANDNX
```

```python
vbt:BBANDS
```

## Running¶

To run an indicator, call the IndicatorBase.run class method of its class by manually passing the input arrays (which can be any array-like objects such as Pandas DataFrames and NumPy arrays), parameters (which can be single values and lists for testing multiple parameter combinations), and other arguments expected by the indicator. The result of running the indicator is an indicator instance (not the actual arrays!).

```python
bbands = vbt.BBANDS.run(close)  # (1)!
bbands = vbt.BBANDS.run(open)  # (2)!
bbands = vbt.BBANDS.run(close, window=20)  # (3)!
bbands = vbt.BBANDS.run(close, window=vbt.Default(20))  # (4)!
bbands = vbt.BBANDS.run(close, window=20, hide_params=["window"])  # (5)!
bbands = vbt.BBANDS.run(close, window=20, hide_params=True)  # (6)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30])  # (7)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30], alpha=[2, 3, 4])  # (8)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30], alpha=[2, 3, 4], param_product=True)  # (9)!
```

```python
bbands = vbt.BBANDS.run(close)  # (1)!
bbands = vbt.BBANDS.run(open)  # (2)!
bbands = vbt.BBANDS.run(close, window=20)  # (3)!
bbands = vbt.BBANDS.run(close, window=vbt.Default(20))  # (4)!
bbands = vbt.BBANDS.run(close, window=20, hide_params=["window"])  # (5)!
bbands = vbt.BBANDS.run(close, window=20, hide_params=True)  # (6)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30])  # (7)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30], alpha=[2, 3, 4])  # (8)!
bbands = vbt.BBANDS.run(close, window=[10, 20, 30], alpha=[2, 3, 4], param_product=True)  # (9)!
```

```python
Default
```

Warning

Testing a wide grid of parameter combinations will produce wide arrays. For example, testing 10000 parameter combinations on one year of daily data would produce an array that takes 30MB of RAM. If the indicator returns three arrays, the RAM consumption would be at least 120MB. One year of minute data would result in staggering 40GB. Thus, for testing wide parameter grids it's recommended to test only a subset of combinations at a time, such as with the use of parameterization or chunking.

Often, there's a need to make an indicator skip missing values. For this, use skipna=True. This argument not only works for TA-Lib indicators but for any indicators, the only requirement: the jitted loop must be disabled. Also, when a two-dimensional input array is passed, you need to additionally pass split_columns=True to split its columns and process one column at once.

```python
skipna=True
```

```python
split_columns=True
```

```python
bbands = vbt.BBANDS.run(close_1d, skipna=True)
bbands = vbt.BBANDS.run(close_2d, split_columns=True, skipna=True)
```

```python
bbands = vbt.BBANDS.run(close_1d, skipna=True)
bbands = vbt.BBANDS.run(close_2d, split_columns=True, skipna=True)
```

Another way is to remove missing values altogether.

```python
bbands = bbands.dropna()  # (1)!
bbands = bbands.dropna(how="all")  # (2)!
```

```python
bbands = bbands.dropna()  # (1)!
bbands = bbands.dropna(how="all")  # (2)!
```

To retrieve the output arrays from an indicator instance, either access each as an attribute, or use various unpacking options such as IndicatorBase.unpack.

```python
bbands = vbt.talib("BBANDS").run(close)
upperband_df = bbands.upperband  # (1)!
middleband_df = bbands.middleband
lowerband_df = bbands.lowerband
upperband_df, middleband_df, lowerband_df = bbands.unpack()  # (2)!
output_dict = bbands.to_dict()  # (3)!
output_df = bbands.to_frame()  # (4)!

sma = vbt.talib("SMA").run(close)
sma_df = sma.real  # (5)!
sma_df = sma.sma  # (6)!
sma_df = sma.output  # (7)!
sma_df = sma.unpack()
```

```python
bbands = vbt.talib("BBANDS").run(close)
upperband_df = bbands.upperband  # (1)!
middleband_df = bbands.middleband
lowerband_df = bbands.lowerband
upperband_df, middleband_df, lowerband_df = bbands.unpack()  # (2)!
output_dict = bbands.to_dict()  # (3)!
output_df = bbands.to_frame()  # (4)!

sma = vbt.talib("SMA").run(close)
sma_df = sma.real  # (5)!
sma_df = sma.sma  # (6)!
sma_df = sma.output  # (7)!
sma_df = sma.unpack()
```

```python
bbands.output_names
```

To keep outputs in the NumPy format and/or omit any shape checks, use return_raw="outputs".

```python
return_raw="outputs"
```

```python
upperband, middleband, lowerband = vbt.talib("BBANDS").run(close, return_raw="outputs")
```

```python
upperband, middleband, lowerband = vbt.talib("BBANDS").run(close, return_raw="outputs")
```

An even simpler way to run indicators is by using Data.run, which takes an indicator name or class, identifies what input names the indicator expects, and then runs the indicator while passing all the inputs found in the data instance automatically. This method also allows unpacking and running multiple indicators, which is very useful for feature engineering.

```python
bbands = data.run("vbt:BBANDS")  # (1)!
bbands = data.run("vbt:BBANDS", window=20)  # (2)!
upper, middle, lower = data.run("vbt:BBANDS", unpack=True)  # (3)!

features_df = data.run(["talib:BBANDS", "talib:RSI"])  # (4)!
bbands, rsi = data.run(["talib:BBANDS", "talib:RSI"], concat=False)  # (5)!
features_df = data.run(  # (6)!
    ["talib:BBANDS", "talib:RSI"], 
    timeperiod=vbt.run_func_dict(talib_bbands=20, talib_rsi=30),
    hide_params=True
)
features_df = data.run(  # (7)!
    ["talib:BBANDS", "vbt:RSI"], 
    talib_bbands=vbt.run_arg_dict(timeperiod=20),
    vbt_rsi=vbt.run_arg_dict(window=30),
    hide_params=True
)
features_df = data.run("talib_all")  # (8)!
```

```python
bbands = data.run("vbt:BBANDS")  # (1)!
bbands = data.run("vbt:BBANDS", window=20)  # (2)!
upper, middle, lower = data.run("vbt:BBANDS", unpack=True)  # (3)!

features_df = data.run(["talib:BBANDS", "talib:RSI"])  # (4)!
bbands, rsi = data.run(["talib:BBANDS", "talib:RSI"], concat=False)  # (5)!
features_df = data.run(  # (6)!
    ["talib:BBANDS", "talib:RSI"], 
    timeperiod=vbt.run_func_dict(talib_bbands=20, talib_rsi=30),
    hide_params=True
)
features_df = data.run(  # (7)!
    ["talib:BBANDS", "vbt:RSI"], 
    talib_bbands=vbt.run_arg_dict(timeperiod=20),
    vbt_rsi=vbt.run_arg_dict(window=30),
    hide_params=True
)
features_df = data.run("talib_all")  # (8)!
```

```python
vbt.BBANDS
```

```python
data.close
```

```python
run
```

```python
unpack
```

```python
run_func_dict
```

```python
run_arg_dict
```

To quickly run and plot a TA-Lib indicator on a single parameter combination without using the indicator factory, use talib_func and talib_plot_func respectively. In contrast to the official TA-Lib implementation, it can properly handle DataFrames, NaNs, broadcasting, and timeframes. The indicator factory's TA-Lib version is based on these two functions.

```python
run_bbands = vbt.talib_func("BBANDS")
upperband, middleband, lowerband = run_bbands(close, timeperiod=2)
upperband, middleband, lowerband = data.run("talib_func:BBANDS", timeperiod=2)  # (1)!

plot_bbands = vbt.talib_plot_func("BBANDS")
fig = plot_bbands(upperband, middleband, lowerband)
```

```python
run_bbands = vbt.talib_func("BBANDS")
upperband, middleband, lowerband = run_bbands(close, timeperiod=2)
upperband, middleband, lowerband = data.run("talib_func:BBANDS", timeperiod=2)  # (1)!

plot_bbands = vbt.talib_plot_func("BBANDS")
fig = plot_bbands(upperband, middleband, lowerband)
```

### Parallelization¶

Parameter combinations are processed using execute such that it's fairly easy to parallelize their execution.

```python
any_indicator.run(...)  # (1)!

# ______________________________________________________________

numba_indicator.run(  # (2)!
    ...,
    jitted_loop=True,  # (3)!
    jitted_warmup=True,    # (4)!
    execute_kwargs=dict(n_chunks="auto", engine="threadpool")
)

# ______________________________________________________________

python_indicator.run(  # (5)!
    ...,
    execute_kwargs=dict(n_chunks="auto", distribute="chunks", engine="pathos")
)
```

```python
any_indicator.run(...)  # (1)!

# ______________________________________________________________

numba_indicator.run(  # (2)!
    ...,
    jitted_loop=True,  # (3)!
    jitted_warmup=True,    # (4)!
    execute_kwargs=dict(n_chunks="auto", engine="threadpool")
)

# ______________________________________________________________

python_indicator.run(  # (5)!
    ...,
    execute_kwargs=dict(n_chunks="auto", distribute="chunks", engine="pathos")
)
```

```python
nogil
```

## Registration¶

Custom indicators can be registered by the indicator factory to appear in the list of all indicators. This is convenient to be able to refer to the indicator by its name when running a data instance. Upon registration, you can assign the indicator to a custom location (the default location is "custom"), which acts as a tag or group; this can be used to build arbitrary indicator groups. One indicator can be assigned to multiple locations. Custom indicators have priority over built-in indicators.

```python
vbt.IF.register_custom_indicator(sma_indicator)  # (1)!
vbt.IF.register_custom_indicator(sma_indicator, "SMA")  # (2)!
vbt.IF.register_custom_indicator(sma_indicator, "SMA", location="rolling")  # (3)!
vbt.IF.register_custom_indicator(sma_indicator, "rolling:SMA")
vbt.IF.register_custom_indicator("talib:sma", location="rolling")  # (4)!

vbt.IF.deregister_custom_indicator("SMA", location="rolling")  # (5)!
vbt.IF.deregister_custom_indicator("rolling:SMA")
vbt.IF.deregister_custom_indicator("SMA")  # (6)!
vbt.IF.deregister_custom_indicator(location="rolling")  # (7)!
vbt.IF.deregister_custom_indicator()  # (8)!
```

```python
vbt.IF.register_custom_indicator(sma_indicator)  # (1)!
vbt.IF.register_custom_indicator(sma_indicator, "SMA")  # (2)!
vbt.IF.register_custom_indicator(sma_indicator, "SMA", location="rolling")  # (3)!
vbt.IF.register_custom_indicator(sma_indicator, "rolling:SMA")
vbt.IF.register_custom_indicator("talib:sma", location="rolling")  # (4)!

vbt.IF.deregister_custom_indicator("SMA", location="rolling")  # (5)!
vbt.IF.deregister_custom_indicator("rolling:SMA")
vbt.IF.deregister_custom_indicator("SMA")  # (6)!
vbt.IF.deregister_custom_indicator(location="rolling")  # (7)!
vbt.IF.deregister_custom_indicator()  # (8)!
```

