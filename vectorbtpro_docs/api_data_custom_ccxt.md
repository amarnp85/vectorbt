# ccxt module¶

Module with CCXTData.

## CCXTData class¶

```python
CCXTData(
    wrapper,
    data,
    single_key=True,
    classes=None,
    level_name=None,
    fetch_kwargs=None,
    returned_kwargs=None,
    last_index=None,
    delisted=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    **kwargs
)
```

```python
CCXTData(
    wrapper,
    data,
    single_key=True,
    classes=None,
    level_name=None,
    fetch_kwargs=None,
    returned_kwargs=None,
    last_index=None,
    delisted=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    **kwargs
)
```

Data class for fetching using CCXT.

See https://github.com/ccxt/ccxt for API.

See CCXTData.fetch_symbol() for arguments.

Usage

```python
>>> from vectorbtpro import *

>>> vbt.CCXTData.set_exchange_settings(
...     exchange_name="binance",
...     populate_=True,
...     exchange_config=dict(
...         apiKey="YOUR_KEY",
...         secret="YOUR_SECRET"
...     )
... )
```

```python
>>> from vectorbtpro import *

>>> vbt.CCXTData.set_exchange_settings(
...     exchange_name="binance",
...     populate_=True,
...     exchange_config=dict(
...         apiKey="YOUR_KEY",
...         secret="YOUR_SECRET"
...     )
... )
```

```python
>>> data = vbt.CCXTData.pull(
...     "BTCUSDT",
...     exchange="binance",
...     start="2020-01-01",
...     end="2021-01-01",
...     timeframe="1 day"
... )
```

```python
>>> data = vbt.CCXTData.pull(
...     "BTCUSDT",
...     exchange="binance",
...     start="2020-01-01",
...     end="2021-01-01",
...     timeframe="1 day"
... )
```

Superclasses

Inherited members

### fetch_symbol class method¶

```python
CCXTData.fetch_symbol(
    symbol,
    exchange=None,
    exchange_config=None,
    start=None,
    end=None,
    timeframe=None,
    tz=None,
    find_earliest_date=None,
    limit=None,
    delay=None,
    retries=None,
    fetch_params=None,
    show_progress=None,
    pbar_kwargs=None,
    silence_warnings=None,
    return_fetch_method=False
)
```

```python
CCXTData.fetch_symbol(
    symbol,
    exchange=None,
    exchange_config=None,
    start=None,
    end=None,
    timeframe=None,
    tz=None,
    find_earliest_date=None,
    limit=None,
    delay=None,
    retries=None,
    fetch_params=None,
    show_progress=None,
    pbar_kwargs=None,
    silence_warnings=None,
    return_fetch_method=False
)
```

Override Data.fetch_symbol() to fetch a symbol from CCXT.

Args

```python
symbol
```

```python
str
```

Symbol.

Symbol can be in the EXCHANGE:SYMBOL format, in this case exchange argument will be ignored.

```python
EXCHANGE:SYMBOL
```

```python
exchange
```

```python
exchange
```

```python
str
```

```python
object
```

Exchange identifier or an exchange object.

See CCXTData.resolve_exchange().

```python
exchange_config
```

```python
dict
```

Exchange config.

See CCXTData.resolve_exchange().

```python
start
```

```python
any
```

Start datetime.

See to_tzaware_datetime().

```python
end
```

```python
any
```

End datetime.

See to_tzaware_datetime().

```python
timeframe
```

```python
str
```

Timeframe.

Allows human-readable strings such as "15 minutes".

```python
tz
```

```python
any
```

Timezone.

See to_timezone().

```python
find_earliest_date
```

```python
bool
```

```python
limit
```

```python
int
```

```python
delay
```

```python
float
```

Time to sleep after each request (in seconds).

Note

Use only if enableRateLimit is not set.

```python
enableRateLimit
```

```python
retries
```

```python
int
```

```python
fetch_params
```

```python
dict
```

```python
fetch_ohlcv
```

```python
show_progress
```

```python
bool
```

```python
pbar_kwargs
```

```python
dict
```

```python
silence_warnings
```

```python
bool
```

```python
return_fetch_method
```

```python
bool
```

For defaults, see custom.ccxt in data. Global settings can be provided per exchange id using the exchanges dictionary.

```python
custom.ccxt
```

```python
exchanges
```

### find_earliest_date class method¶

```python
CCXTData.find_earliest_date(
    symbol,
    for_internal_use=False,
    **kwargs
)
```

```python
CCXTData.find_earliest_date(
    symbol,
    for_internal_use=False,
    **kwargs
)
```

Find the earliest date using binary search.

See CCXTData.fetch_symbol() for arguments.

### get_exchange_setting class method¶

```python
CCXTData.get_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.get_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.get_custom_setting() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

### get_exchange_settings class method¶

```python
CCXTData.get_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.get_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.get_custom_settings() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

### has_exchange_setting class method¶

```python
CCXTData.has_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.has_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.has_custom_setting() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

### has_exchange_settings class method¶

```python
CCXTData.has_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.has_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.has_custom_settings() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

### list_symbols class method¶

```python
CCXTData.list_symbols(
    pattern=None,
    use_regex=False,
    sort=True,
    exchange=None,
    exchange_config=None
)
```

```python
CCXTData.list_symbols(
    pattern=None,
    use_regex=False,
    sort=True,
    exchange=None,
    exchange_config=None
)
```

List all symbols.

Uses CustomData.key_match() to check each symbol against pattern.

```python
pattern
```

### resolve_exchange class method¶

```python
CCXTData.resolve_exchange(
    exchange=None,
    **exchange_config
)
```

```python
CCXTData.resolve_exchange(
    exchange=None,
    **exchange_config
)
```

Resolve the exchange.

If provided, must be of the type ccxt.base.exchange.Exchange. Otherwise, will be created using exchange_config.

```python
ccxt.base.exchange.Exchange
```

```python
exchange_config
```

### resolve_exchange_setting class method¶

```python
CCXTData.resolve_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.resolve_exchange_setting(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.resolve_custom_setting() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

### set_exchange_settings class method¶

```python
CCXTData.set_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

```python
CCXTData.set_exchange_settings(
    *args,
    exchange_name=None,
    **kwargs
)
```

CustomData.set_custom_settings() with sub_path=exchange_name.

```python
sub_path=exchange_name
```

