# Data¶

VectorBT® PRO works on Pandas and NumPy arrays, but where those arrays are coming from? Getting the financial data manually is a challenging task, especially when an exchange can return only one bunch of data at a time such that iteration over time ranges, concatenation of results, and alignment of index and columns are effectively outsourced to the user. The task gets only trickier when multiple symbols are involved.

To simplify and automate data retrieval and management, vectorbt implements the Data class, which allows seamless handling of features (such as OHLC) and symbols (such as "BTC-USD"). It's a semi-abstract class, meaning you have to subclass it and define your own logic at various places to be able to use its rich functionality to the full extent. Gladly, there is a collection of custom data classes already implemented for us, but it's always good to know how to create such a data class on our own.

The steps discussed below can be visualized using the following graph:

```python
flowchart TD;
    dataclass["Data class"]
    fetching["Fetching"]
    pdobjs["Pandas objects"]
    wrapping["Wrapping"]
    dataobjs["Data objects"]
    merging["Merging"]
    alignment["Alignment"]
    dataobj["Data object"]
    updating["Updating"]
    transforming["Transforming"]
    resampling["Resampling"]
    indexing["Indexing"]
    getting["Getting"]
    pdobj["Pandas object"]
    running["Running"]
    output["Output"]

    dataclass --> fetching;
    fetching --> alignment;
    pdobjs --> wrapping;
    wrapping --> alignment;
    dataobjs --> merging;
    merging --> alignment;
    alignment -->|"creates new"| dataobj;
    dataobj --> updating;
    updating --> alignment;
    dataobj --> transforming;
    transforming --> alignment;
    dataobj --> getting;
    getting --> pdobj;
    dataobj --> running;
    running --> output;
    dataobj --> resampling;
    resampling -->|"creates new"| dataobj;
    dataobj --> indexing;
    indexing -->|"creates new"| dataobj;
```

```python
flowchart TD;
    dataclass["Data class"]
    fetching["Fetching"]
    pdobjs["Pandas objects"]
    wrapping["Wrapping"]
    dataobjs["Data objects"]
    merging["Merging"]
    alignment["Alignment"]
    dataobj["Data object"]
    updating["Updating"]
    transforming["Transforming"]
    resampling["Resampling"]
    indexing["Indexing"]
    getting["Getting"]
    pdobj["Pandas object"]
    running["Running"]
    output["Output"]

    dataclass --> fetching;
    fetching --> alignment;
    pdobjs --> wrapping;
    wrapping --> alignment;
    dataobjs --> merging;
    merging --> alignment;
    alignment -->|"creates new"| dataobj;
    dataobj --> updating;
    updating --> alignment;
    dataobj --> transforming;
    transforming --> alignment;
    dataobj --> getting;
    getting --> pdobj;
    dataobj --> running;
    running --> output;
    dataobj --> resampling;
    resampling -->|"creates new"| dataobj;
    dataobj --> indexing;
    indexing -->|"creates new"| dataobj;
```

(Reload the page if the diagram doesn't show up)

## Fetching¶

Class Data implements an abstract class method Data.fetch_symbol for generating, loading, or fetching one symbol of data from any data source. It has to be overridden and implemented by the user, and return a single (Pandas or NumPy) array given some set of parameters, such as the starting date, the ending date, and the frequency.

Let's write a function that returns any symbol of data from Yahoo Finance using yfinance:

```python
>>> from vectorbtpro import *

>>> def get_yf_symbol(symbol, period="max", start=None, end=None, **kwargs):
...     import yfinance as yf
...     if start is not None:
...         start = vbt.local_datetime(start)  # (1)!
...     if end is not None:
...         end = vbt.local_datetime(end)
...     return yf.Ticker(symbol).history(
...         period=period, 
...         start=start, 
...         end=end, 
...         **kwargs
...     )

>>> get_yf_symbol("BTC-USD", start="2020-01-01", end="2020-01-05")
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

```python
>>> from vectorbtpro import *

>>> def get_yf_symbol(symbol, period="max", start=None, end=None, **kwargs):
...     import yfinance as yf
...     if start is not None:
...         start = vbt.local_datetime(start)  # (1)!
...     if end is not None:
...         end = vbt.local_datetime(end)
...     return yf.Ticker(symbol).history(
...         period=period, 
...         start=start, 
...         end=end, 
...         **kwargs
...     )

>>> get_yf_symbol("BTC-USD", start="2020-01-01", end="2020-01-05")
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

Info

Why the returned data starts from 2019-12-31 and not from 2020-01-01? The provided start and end dates are defined in the local timezone and then converted into UTC. In the Europe/Berlin timezone, depending upon the time of the year, 2020-01-01 gets translated into 2019-12-31 22:00:00, which is the date Yahoo Finance actually receives. To provide any date directly as a UTC date, append "UTC": 2020-01-01 UTC or construct a proper Timestamp instance.

```python
2019-12-31
```

```python
2020-01-01
```

```python
2020-01-01
```

```python
2019-12-31 22:00:00
```

```python
2020-01-01 UTC
```

Managing data in a Pandas format is acceptable when we are dealing with one symbol, but what about multiple symbols? Remember how vectorbt wants us to provide each of the open price, high price, and other features as separate variables? Each of those variables must have symbols laid out as columns, which means that we would have to manually fetch all symbols and properly reorganize their data layout. Having some symbols with different index or columns would just add to our headache.

Luckily, there is a class method Data.pull that solves most of the issues related to iterating over, fetching, and merging symbols. It takes from one to multiple symbols, fetches each one with Data.fetch_symbol, puts it into a dictionary, and passes this dictionary to Data.from_data for post-processing and class instantiation.

Building upon our example, let's subclass Data and override the Data.fetch_symbol method to call our get_yf_symbol function:

```python
get_yf_symbol
```

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         return get_yf_symbol(symbol, **kwargs)
```

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         return get_yf_symbol(symbol, **kwargs)
```

Hint

You can replace get_yf_symbol with any other function that returns any array-like data!

```python
get_yf_symbol
```

That's it, YFData is now a full-blown data class capable of pulling data from Yahoo Finance and storing it:

```python
YFData
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-01-05"
... )
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-01-05"
... )
```

Symbol 2/2

Symbol 2/2

The pulled data is stored inside the Data.data dictionary with symbols being keys and values being Pandas objects returned by Data.fetch_symbol:

```python
>>> yf_data.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00  132.612274  133.732681  128.798157  129.610855   
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00   8936866397        0.0           0.0  
2020-01-01 00:00:00+00:00   7935230330        0.0           0.0  
2020-01-02 00:00:00+00:00   8032709256        0.0           0.0  
2020-01-03 00:00:00+00:00  10476845358        0.0           0.0  
2020-01-04 00:00:00+00:00   7430904515        0.0           0.0
```

```python
>>> yf_data.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00  132.612274  133.732681  128.798157  129.610855   
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00   8936866397        0.0           0.0  
2020-01-01 00:00:00+00:00   7935230330        0.0           0.0  
2020-01-02 00:00:00+00:00   8032709256        0.0           0.0  
2020-01-03 00:00:00+00:00  10476845358        0.0           0.0  
2020-01-04 00:00:00+00:00   7430904515        0.0           0.0
```

### Exception handling¶

If Data.fetch_symbol returned None or an empty Pandas object or NumPy array, the symbol will be skipped entirely. Data.pull will also catch any exception raised in Data.fetch_symbol and skip the symbol if the argument skip_on_error is True (it's False by default!), otherwise, it will abort the procedure.

```python
None
```

```python
skip_on_error
```

Generally, it's the task of Data.fetch_symbol to handle issues. Whenever there is a lot of data points to fetch and the fetcher relies upon a loop to concatenate different data bunches together, the best approach is to show the user a warning whenever an exception is thrown and return the data fetched up to the most recent point in time, similarly to how this was implemented in BinanceData and CCXTData. In such a case, vectorbt will replace the missing data points with NaN or drop them altogether, and keep track of the last index. You can then wait until your connection is stable and re-fetch the missing data using Data.update.

### Custom context¶

Along with the data, Data.fetch_symbol can also return a dictionary with custom keyword arguments acting as a context of the fetching operation. This context can later be accessed in the symbol dictionary Data.returned_kwargs. For instance, this context may include any information on why the fetching process failed, the length of the remaining data left to fetch, or which rows the fetched data represents when reading a local file (as implemented by CSVData for data updates).

Just for the sake of example, let's save the current timestamp:

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         returned_kwargs = dict(timestamp=vbt.timestamp())
...         return get_yf_symbol(symbol, **kwargs), returned_kwargs

>>> yf_data = YFData.pull("BTC-USD", start="2020-01-01", end="2020-01-05")
>>> yf_data.returned_kwargs
symbol_dict({'BTC-USD': {'timestamp': Timestamp('2023-08-28 20:08:50.893763')}})
```

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         returned_kwargs = dict(timestamp=vbt.timestamp())
...         return get_yf_symbol(symbol, **kwargs), returned_kwargs

>>> yf_data = YFData.pull("BTC-USD", start="2020-01-01", end="2020-01-05")
>>> yf_data.returned_kwargs
symbol_dict({'BTC-USD': {'timestamp': Timestamp('2023-08-28 20:08:50.893763')}})
```

Info

symbol_dict is a regular dictionary where information is grouped by symbol.

## Alignment¶

Like most classes that hold data, the class Data subclasses Analyzable, so we can perform Pandas indexing on the class instance itself to select rows and columns in all Pandas objects stored inside that instance. Doing a single Pandas indexing operation on multiple Pandas objects with different labels is impossible, so what happens if we fetched symbol data from different date ranges or with different columns? Whenever Data.pull passes the (unaligned) data dictionary to Data.from_data, it calls Data.align_data, which does the following:

Let's illustrate this workflow in practice:

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({  # (1)!
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({  # (1)!
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

Symbol 2/2

Symbol 2/2

```python
>>> yf_data.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00          NaN          NaN          NaN          NaN   
2020-01-04 00:00:00+00:00          NaN          NaN          NaN          NaN   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0  
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00           NaN        NaN           NaN  
2020-01-04 00:00:00+00:00           NaN        NaN           NaN  

>>> yf_data.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-01 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00           NaN        NaN           NaN  
2020-01-01 00:00:00+00:00           NaN        NaN           NaN  
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0
```

```python
>>> yf_data.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00          NaN          NaN          NaN          NaN   
2020-01-04 00:00:00+00:00          NaN          NaN          NaN          NaN   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0  
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00           NaN        NaN           NaN  
2020-01-04 00:00:00+00:00           NaN        NaN           NaN  

>>> yf_data.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-01 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00           NaN        NaN           NaN  
2020-01-01 00:00:00+00:00           NaN        NaN           NaN  
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0
```

Notice how we ended up with the same index and columns across all Pandas objects. We can now use this data in any vectorbt function without fearing any indexing errors.

### NaNs¶

If some rows are present in one symbol and are missing in another, vectorbt will raise a warning with the text "Symbols have mismatching index". By default, the missing rows will be replaced by NaN. To drop them or raise an error instead, use the missing_index argument:

```python
missing_index
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({  # (1)!
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     }),
...     missing_index="drop"
... )
UserWarning: Symbols have mismatching index. Dropping missing data points.
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({  # (1)!
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     }),
...     missing_index="drop"
... )
UserWarning: Symbols have mismatching index. Dropping missing data points.
```

Symbol 2/2

Symbol 2/2

```python
>>> yf_data.data["BTC-USD"]
                                 Open         High         Low        Close  \
Date                                                                          
2020-01-02 00:00:00+00:00  7202.55127  7212.155273  6935.27002  6985.470215   

                                Volume  Dividends  Stock Splits  
Date                                                             
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  

>>> yf_data.data["ETH-USD"]
                                 Open        High        Low       Close  \
Date                                                                       
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.95491  127.410179   

                               Volume  Dividends  Stock Splits  
Date                                                            
2020-01-02 00:00:00+00:00  8032709256        0.0           0.0
```

```python
>>> yf_data.data["BTC-USD"]
                                 Open         High         Low        Close  \
Date                                                                          
2020-01-02 00:00:00+00:00  7202.55127  7212.155273  6935.27002  6985.470215   

                                Volume  Dividends  Stock Splits  
Date                                                             
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  

>>> yf_data.data["ETH-USD"]
                                 Open        High        Low       Close  \
Date                                                                       
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.95491  127.410179   

                               Volume  Dividends  Stock Splits  
Date                                                            
2020-01-02 00:00:00+00:00  8032709256        0.0           0.0
```

## Updating¶

Updating is a regular fetching operation that can be used both to update the existing data points and to add new ones. It requires specifying the first timestamp or row index of the update, and assumes that the data points prior to this timestamp or row index remain unchanged.

Similarly to Data.fetch_symbol, updating must be manually implemented by overriding a method Data.update_symbol. In contrast to the fetcher, the updater is an instance method and can access the data fetched earlier. For instance, it can access the keyword arguments initially passed to the fetcher, accessible in the symbol dictionary Data.fetch_kwargs. Those arguments can be used as default arguments or be overridden by any argument passed directly to the updater. Every data instance has also a symbol dictionary Data.last_index, which holds the last fetched index per symbol. We can use this index as the starting point of the next update.

Let's build a new YFData class that can also perform updates to the stored data:

```python
YFData
```

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         return get_yf_symbol(symbol, **kwargs)
...
...     def update_symbol(self, symbol, **kwargs):
...         defaults = self.select_fetch_kwargs(symbol)  # (1)!
...         defaults["start"] = self.select_last_index(symbol)  # (2)!
...         kwargs = vbt.merge_dicts(defaults, kwargs)  # (3)!
...         return self.fetch_symbol(symbol, **kwargs)  # (4)!
```

```python
>>> class YFData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         return get_yf_symbol(symbol, **kwargs)
...
...     def update_symbol(self, symbol, **kwargs):
...         defaults = self.select_fetch_kwargs(symbol)  # (1)!
...         defaults["start"] = self.select_last_index(symbol)  # (2)!
...         kwargs = vbt.merge_dicts(defaults, kwargs)  # (3)!
...         return self.fetch_symbol(symbol, **kwargs)  # (4)!
```

```python
kwargs
```

Once the Data.update_symbol method is implemented, we can call the method Data.update to iterate over each symbol and update its data. Under the hood, this method also aligns the index and column labels of all the returned Pandas objects, appends the new data to the old data through concatenation along rows, and updates the last index of each symbol for the use in the next data update. Finally, it produces a new instance of Data by using Configured.replace.

Important

Updating data never overwrites the existing data instance but always returns a new instance. Remember that most classes in vectorbt are read-only to enable caching and avoid side effects.

First, we'll fetch the same data as previously:

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

Symbol 2/2

Symbol 2/2

Even though both DataFrames end with the same date, our YFData instance knows that the BTC-USD symbol is 2 rows behind the ETH-USD symbol:

```python
YFData
```

```python
BTC-USD
```

```python
ETH-USD
```

```python
>>> yf_data.last_index
symbol_dict({
    'BTC-USD': Timestamp('2020-01-02 00:00:00+0000', tz='UTC'), 
    'ETH-USD': Timestamp('2020-01-04 00:00:00+0000', tz='UTC')
})
```

```python
>>> yf_data.last_index
symbol_dict({
    'BTC-USD': Timestamp('2020-01-02 00:00:00+0000', tz='UTC'), 
    'ETH-USD': Timestamp('2020-01-04 00:00:00+0000', tz='UTC')
})
```

We can also access the keyword arguments passed to the initial fetching operation:

```python
>>> yf_data.fetch_kwargs
symbol_dict({
    'BTC-USD': {'start': '2020-01-01', 'end': '2020-01-03'}, 
    'ETH-USD': {'start': '2020-01-03', 'end': '2020-01-05'}
})
```

```python
>>> yf_data.fetch_kwargs
symbol_dict({
    'BTC-USD': {'start': '2020-01-01', 'end': '2020-01-03'}, 
    'ETH-USD': {'start': '2020-01-03', 'end': '2020-01-05'}
})
```

The start argument of each symbol will be replaced by its respective entry in Data.last_index, while the end argument can be overridden by any date that we specify during the update.

```python
start
```

```python
end
```

Note

Without specifying the end date, vectorbt will update only the latest data point of each symbol.

Let's update both symbols up to the same date:

```python
>>> yf_data_updated = yf_data.update(end="2020-01-06")  # (1)!

>>> yf_data_updated.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0  
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00  2.811148e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  1.844427e+10        0.0           0.0  
2020-01-05 00:00:00+00:00  1.972507e+10        0.0           0.0  

>>> yf_data_updated.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-01 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   
2020-01-05 00:00:00+00:00  135.072098  139.410202  135.045624  136.276779   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00           NaN        NaN           NaN  
2020-01-01 00:00:00+00:00           NaN        NaN           NaN  
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0  
2020-01-05 00:00:00+00:00  7.526675e+09        0.0           0.0
```

```python
>>> yf_data_updated = yf_data.update(end="2020-01-06")  # (1)!

>>> yf_data_updated.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0  
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00  2.811148e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  1.844427e+10        0.0           0.0  
2020-01-05 00:00:00+00:00  1.972507e+10        0.0           0.0  

>>> yf_data_updated.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-01 00:00:00+00:00         NaN         NaN         NaN         NaN   
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   
2020-01-05 00:00:00+00:00  135.072098  139.410202  135.045624  136.276779   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00           NaN        NaN           NaN  
2020-01-01 00:00:00+00:00           NaN        NaN           NaN  
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0  
2020-01-05 00:00:00+00:00  7.526675e+09        0.0           0.0
```

Each symbol has been updated separately based on their last_index value: the symbol BTC-USD has received new rows ranging from 2020-01-02 to 2020-01-05, while the symbol ETH-USD has only received new rows between 2020-01-04 to 2020-01-05. We can now see that both symbols have been successfully synced up to the same ending date:

```python
last_index
```

```python
BTC-USD
```

```python
2020-01-02
```

```python
2020-01-05
```

```python
ETH-USD
```

```python
2020-01-04
```

```python
2020-01-05
```

```python
>>> yf_data_updated.last_index
symbol_dict({
    'BTC-USD': Timestamp('2020-01-05 00:00:00+0000', tz='UTC'), 
    'ETH-USD': Timestamp('2020-01-05 00:00:00+0000', tz='UTC')
})
```

```python
>>> yf_data_updated.last_index
symbol_dict({
    'BTC-USD': Timestamp('2020-01-05 00:00:00+0000', tz='UTC'), 
    'ETH-USD': Timestamp('2020-01-05 00:00:00+0000', tz='UTC')
})
```

If the last index of the data update lies before the current last_index (that is, we want to update any data in the middle), all the data after the new last index will be disregarded:

```python
last_index
```

```python
>>> yf_data_updated = yf_data_updated.update(start="2020-01-01", end="2020-01-02")

>>> yf_data_updated.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0 

>>> yf_data_updated.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00  132.612274  133.732681  128.798157  129.610855   
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  8.936866e+09        0.0           0.0  
2020-01-01 00:00:00+00:00  7.935230e+09        0.0           0.0
```

```python
>>> yf_data_updated = yf_data_updated.update(start="2020-01-01", end="2020-01-02")

>>> yf_data_updated.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  2.116795e+10        0.0           0.0  
2020-01-01 00:00:00+00:00  1.856566e+10        0.0           0.0 

>>> yf_data_updated.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2019-12-31 00:00:00+00:00  132.612274  133.732681  128.798157  129.610855   
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2019-12-31 00:00:00+00:00  8.936866e+09        0.0           0.0  
2020-01-01 00:00:00+00:00  7.935230e+09        0.0           0.0
```

Note

The last data point of an update is considered to be the most up-to-date point, thus no data stored previously can come after it.

### Concatenation¶

By default, the returned data instance contains the whole data - the old data with the new data concatenated together. To return only the updated data, disable concat:

```python
concat
```

```python
>>> yf_data_new = yf_data.update(end="2020-01-06", concat=False)

>>> yf_data_new.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00  2.811148e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  1.844427e+10        0.0           0.0  
2020-01-05 00:00:00+00:00  1.972507e+10        0.0           0.0  

>>> yf_data_new.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   
2020-01-05 00:00:00+00:00  135.072098  139.410202  135.045624  136.276779   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0  
2020-01-05 00:00:00+00:00  7.526675e+09        0.0           0.0
```

```python
>>> yf_data_new = yf_data.update(end="2020-01-06", concat=False)

>>> yf_data_new.data["BTC-USD"]
                                  Open         High          Low        Close  \
Date                                                                            
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   
2020-01-05 00:00:00+00:00  7410.451660  7544.497070  7400.535645  7411.317383   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2020-01-02 00:00:00+00:00  2.080208e+10        0.0           0.0  
2020-01-03 00:00:00+00:00  2.811148e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  1.844427e+10        0.0           0.0  
2020-01-05 00:00:00+00:00  1.972507e+10        0.0           0.0  

>>> yf_data_new.data["ETH-USD"]
                                 Open        High         Low       Close  \
Date                                                                        
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179   
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707   
2020-01-04 00:00:00+00:00  134.168518  136.052719  133.040558  135.069366   
2020-01-05 00:00:00+00:00  135.072098  139.410202  135.045624  136.276779   

                                 Volume  Dividends  Stock Splits  
Date                                                              
2020-01-02 00:00:00+00:00  8.032709e+09        0.0           0.0  
2020-01-03 00:00:00+00:00  1.047685e+10        0.0           0.0  
2020-01-04 00:00:00+00:00  7.430905e+09        0.0           0.0  
2020-01-05 00:00:00+00:00  7.526675e+09        0.0           0.0
```

The returned data instance skips two timestamps: 2019-12-31 and 2020-01-01, which weren't changed during that update. But even though the symbol ETH-USD only received new rows between 2020-01-04 to 2020-01-05, it contains the old data for 2020-01-02 and 2020-01-03 as well, why so? Those timestamps were updated in the BTC-USD dataset, and because the index across all symbols must be aligned, we need to include some old data to avoid setting NaNs.

```python
2019-12-31
```

```python
2020-01-01
```

```python
ETH-USD
```

```python
2020-01-04
```

```python
2020-01-05
```

```python
2020-01-02
```

```python
2020-01-03
```

```python
BTC-USD
```

## Getting¶

After the data has been fetched and a new Data instance has been created, getting the data is straight-forward using the Data.data dictionary or the method Data.get.

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-01-05"
... )
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-01-05"
... )
```

Symbol 2/2

Symbol 2/2

Get all features of one symbol of data:

```python
>>> yf_data.get(symbols="BTC-USD")
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

```python
>>> yf_data.get(symbols="BTC-USD")
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

Get specific features of one symbol of data:

```python
>>> yf_data.get(features=["High", "Low"], symbols="BTC-USD")
                                  High          Low
Date                                               
2019-12-31 00:00:00+00:00  7335.290039  7169.777832
2020-01-01 00:00:00+00:00  7254.330566  7174.944336
2020-01-02 00:00:00+00:00  7212.155273  6935.270020
2020-01-03 00:00:00+00:00  7413.715332  6914.996094
2020-01-04 00:00:00+00:00  7427.385742  7309.514160
```

```python
>>> yf_data.get(features=["High", "Low"], symbols="BTC-USD")
                                  High          Low
Date                                               
2019-12-31 00:00:00+00:00  7335.290039  7169.777832
2020-01-01 00:00:00+00:00  7254.330566  7174.944336
2020-01-02 00:00:00+00:00  7212.155273  6935.270020
2020-01-03 00:00:00+00:00  7413.715332  6914.996094
2020-01-04 00:00:00+00:00  7427.385742  7309.514160
```

Get one feature of all symbols of data:

```python
>>> yf_data.get(features="Close")
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

```python
>>> yf_data.get(features="Close")
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

Notice how symbols have become columns in the returned DataFrame? This is the format so much loved by vectorbt.

Get multiple features of multiple symbols of data:

```python
>>> open_price, close_price = yf_data.get(features=["Open", "Close"])  # (1)!
```

```python
>>> open_price, close_price = yf_data.get(features=["Open", "Close"])  # (1)!
```

Hint

As you might have noticed, vectorbt returns different formats depending upon when there is one or multiple features/symbols captured by the data instance. To produce a consisting format irrespective of the number of features/symbols, pass features/symbols as a list or any other collection.

```python
features
```

```python
symbols
```

For example, running yf_data.get(features="Close") when there is only one symbol will produce a Series instead of a DataFrame. To force vectorbt to always return a DataFrame, pass features=["Close"].

```python
yf_data.get(features="Close")
```

```python
features=["Close"]
```

### Magnet features¶

Magnet features are features with case-insensitive names that the Data class knows how to detect and query. They include static features such as OHLCV, but also those that can be computed dynamically such as VWAP, HLC/3, OHLC/4, and returns. Each feature is also associated with an instance property that returns that feature for all symbols in a data instance. For example, to get the close price and returns:

```python
>>> yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366

>>> yf_data.returns
symbol                      BTC-USD   ETH-USD
Date                                         
2019-12-31 00:00:00+00:00  0.000000  0.000000
2020-01-01 00:00:00+00:00  0.000914  0.009190
2020-01-02 00:00:00+00:00 -0.029819 -0.025931
2020-01-03 00:00:00+00:00  0.051452  0.053069
2020-01-04 00:00:00+00:00  0.008955  0.006690
```

```python
>>> yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366

>>> yf_data.returns
symbol                      BTC-USD   ETH-USD
Date                                         
2019-12-31 00:00:00+00:00  0.000000  0.000000
2020-01-01 00:00:00+00:00  0.000914  0.009190
2020-01-02 00:00:00+00:00 -0.029819 -0.025931
2020-01-03 00:00:00+00:00  0.051452  0.053069
2020-01-04 00:00:00+00:00  0.008955  0.006690
```

## Running¶

Thanks to the unambiguous nature of magnet features, we can use them in feeding many functions across vectorbt, and since most functions don't accept data directly but expect features such as close to be provided separately, there is an urgent need for a method that can recognize what a function wants and pass the data to it accordingly. Such a method is Data.run: it accepts a function, parses its arguments, and upon recognition of a magnet feature, simply forwards it. This is especially useful for quickly running indicators, which are recognized automatically by their names:

```python
data
```

```python
close
```

```python
>>> yf_data.run("sma", 3)
<vectorbtpro.indicators.factory.talib.SMA at 0x7ff75218b5b0>
```

```python
>>> yf_data.run("sma", 3)
<vectorbtpro.indicators.factory.talib.SMA at 0x7ff75218b5b0>
```

If there are multiple third-party libraries that have the same indicator name, it's advisable to also provide a prefix with the name of the library to avoid any confusion:

```python
>>> yf_data.run("talib_sma", 3)
<vectorbtpro.indicators.factory.talib.SMA at 0x7ff752111070>
```

```python
>>> yf_data.run("talib_sma", 3)
<vectorbtpro.indicators.factory.talib.SMA at 0x7ff752111070>
```

This method also accepts names of all the simulation methods available in Portfolio, such as Portfolio.from_holding:

```python
>>> yf_data.run("from_holding")
<vectorbtpro.portfolio.base.Portfolio at 0x7ff767e18970>
```

```python
>>> yf_data.run("from_holding")
<vectorbtpro.portfolio.base.Portfolio at 0x7ff767e18970>
```

## Features and symbols¶

Class Data implements various dictionaries that hold data per symbol, but also methods that let us manipulate that data.

We can view the list of features and symbols using the Data.features and Data.symbols property respectively:

```python
>>> yf_data.features
['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

>>> yf_data.symbols
['BTC-USD', 'ETH-USD']
```

```python
>>> yf_data.features
['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']

>>> yf_data.symbols
['BTC-USD', 'ETH-USD']
```

Additionally, there is a flag Data.single_key that is True if this instance holds only one symbol of data (or feature in case the instance is feature-oriented!). This has implications on Getting as we discussed in the hints above.

```python
>>> yf_data.single_key
False
```

```python
>>> yf_data.single_key
False
```

### Dicts¶

Each data instance holds at least 5 dictionaries:

Each dictionary is a regular dictionary of either the type symbol_dict (mostly when the instance is symbol-oriented) or feature_dict (mostly when the instance is feature-oriented).

```python
>>> yf_data.last_index["BTC-USD"]
Timestamp('2020-01-04 00:00:00+0000', tz='UTC')
```

```python
>>> yf_data.last_index["BTC-USD"]
Timestamp('2020-01-04 00:00:00+0000', tz='UTC')
```

Important

Do not change the values of the above dictionaries in-place. Whenever working with keyword arguments, make sure to build a new dict after selecting a symbol: dict(data.fetch_kwargs[symbol]) - this won't change the parent dict in case you want to modify the keyword arguments for some task.

```python
dict(data.fetch_kwargs[symbol])
```

### Selecting¶

One or more symbols can be selected using Data.select:

```python
>>> yf_data.select("BTC-USD")
<__main__.YFData at 0x7ff6a97f4b38>

>>> yf_data.select("BTC-USD").get()
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

```python
>>> yf_data.select("BTC-USD")
<__main__.YFData at 0x7ff6a97f4b38>

>>> yf_data.select("BTC-USD").get()
                                  Open         High          Low        Close  \
Date                                                                            
2019-12-31 00:00:00+00:00  7294.438965  7335.290039  7169.777832  7193.599121   
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316   
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215   
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277   
2020-01-04 00:00:00+00:00  7345.375488  7427.385742  7309.514160  7410.656738   

                                Volume  Dividends  Stock Splits  
Date                                                             
2019-12-31 00:00:00+00:00  21167946112        0.0           0.0  
2020-01-01 00:00:00+00:00  18565664997        0.0           0.0  
2020-01-02 00:00:00+00:00  20802083465        0.0           0.0  
2020-01-03 00:00:00+00:00  28111481032        0.0           0.0  
2020-01-04 00:00:00+00:00  18444271275        0.0           0.0
```

The operation above produced a new YFData instance with only one symbol - BTC-USD.

```python
YFData
```

```python
BTC-USD
```

Note

Updating the data in a child instance won't affect the parent instance we copied from because updating creates a new Pandas object. But changing the data in-place will also propagate the change to the parent instance. To make both instances fully independent, pass copy_mode_="deep" (see Configured.replace).

```python
copy_mode_="deep"
```

Info

If the instance is feature-oriented, this method will apply to features rather than symbols.

### Renaming¶

Symbols can be renamed using Data.rename:

```python
>>> yf_data.rename({
...     "BTC-USD": "BTC/USD",
...     "ETH-USD": "ETH/USD"
... }).get("Close")
symbol                         BTC/USD     ETH/USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

```python
>>> yf_data.rename({
...     "BTC-USD": "BTC/USD",
...     "ETH-USD": "ETH/USD"
... }).get("Close")
symbol                         BTC/USD     ETH/USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-04 00:00:00+00:00  7410.656738  135.069366
```

Warning

Renaming symbols may (and mostly will) break their updating. Use this only for getting.

Info

If the instance is feature-oriented, this method will apply to features rather than symbols.

### Classes¶

Classes come in handy when we want to introduce another level of abstraction over symbols, such as to further divide symbols into industries and sectors; this would allow us to analyze symbols within their classes, and entire classes themselves. Classes can be provided to the fetcher via the argument classes; they must be specified per symbol, unless there is only one class that should be applied to all symbols. In the end, they will be converted into a (multi-)index and stacked on top of symbol columns when getting the symbol wrapper using Data.get_symbol_wrapper. Each class can be either provided as a string (which will be stored under the class name symbol_class), or as a dictionary where keys are class names and values are class values:

```python
classes
```

```python
symbol_class
```

```python
>>> cls_yfdata = vbt.YFData.pull(
...     ["META", "GOOGL", "NFLX", "BAC", "WFC", "TLT", "SHV"],
...     classes=[
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Financial"),
...         dict(class1="Equity", class2="Financial"),
...         dict(class1="Fixed Income", class2="Treasury"),
...         dict(class1="Fixed Income", class2="Treasury"),
...     ],
...     start="2010-01-01",
...     missing_columns="nan"
... )
>>> cls_yfdata.close
class1                         Equity                                     \
class2                     Technology                          Financial   
symbol                           META       GOOGL        NFLX        BAC   
Date                                                                       
2010-01-04 00:00:00-05:00         NaN   15.684434    7.640000  12.977036   
2010-01-05 00:00:00-05:00         NaN   15.615365    7.358571  13.398854   
2010-01-06 00:00:00-05:00         NaN   15.221722    7.617143  13.555999   
...                               ...         ...         ...        ...   
2023-08-24 00:00:00-04:00  286.750000  129.779999  406.929993  28.620001   
2023-08-25 00:00:00-04:00  285.500000  129.880005  416.029999  28.500000   
2023-08-28 00:00:00-04:00  290.260010  131.009995  418.059998  28.764999   

class1                               Fixed Income              
class2                                   Treasury              
symbol                           WFC          TLT         SHV  
Date                                                           
2010-01-04 00:00:00-05:00  19.073046    62.717960   99.920975  
2010-01-05 00:00:00-05:00  19.596645    63.123013   99.884712  
2010-01-06 00:00:00-05:00  19.624567    62.278038   99.893806  
...                              ...          ...         ...  
2023-08-24 00:00:00-04:00  41.430000    94.910004  110.389999  
2023-08-25 00:00:00-04:00  41.230000    95.220001  110.389999  
2023-08-28 00:00:00-04:00  41.880001    95.320000  110.400002  

[3436 rows x 7 columns]
```

```python
>>> cls_yfdata = vbt.YFData.pull(
...     ["META", "GOOGL", "NFLX", "BAC", "WFC", "TLT", "SHV"],
...     classes=[
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Technology"),
...         dict(class1="Equity", class2="Financial"),
...         dict(class1="Equity", class2="Financial"),
...         dict(class1="Fixed Income", class2="Treasury"),
...         dict(class1="Fixed Income", class2="Treasury"),
...     ],
...     start="2010-01-01",
...     missing_columns="nan"
... )
>>> cls_yfdata.close
class1                         Equity                                     \
class2                     Technology                          Financial   
symbol                           META       GOOGL        NFLX        BAC   
Date                                                                       
2010-01-04 00:00:00-05:00         NaN   15.684434    7.640000  12.977036   
2010-01-05 00:00:00-05:00         NaN   15.615365    7.358571  13.398854   
2010-01-06 00:00:00-05:00         NaN   15.221722    7.617143  13.555999   
...                               ...         ...         ...        ...   
2023-08-24 00:00:00-04:00  286.750000  129.779999  406.929993  28.620001   
2023-08-25 00:00:00-04:00  285.500000  129.880005  416.029999  28.500000   
2023-08-28 00:00:00-04:00  290.260010  131.009995  418.059998  28.764999   

class1                               Fixed Income              
class2                                   Treasury              
symbol                           WFC          TLT         SHV  
Date                                                           
2010-01-04 00:00:00-05:00  19.073046    62.717960   99.920975  
2010-01-05 00:00:00-05:00  19.596645    63.123013   99.884712  
2010-01-06 00:00:00-05:00  19.624567    62.278038   99.893806  
...                              ...          ...         ...  
2023-08-24 00:00:00-04:00  41.430000    94.910004  110.389999  
2023-08-25 00:00:00-04:00  41.230000    95.220001  110.389999  
2023-08-28 00:00:00-04:00  41.880001    95.320000  110.400002  

[3436 rows x 7 columns]
```

Apart from feeding classes to the fetcher, we can also replace them in any existing data instance, which will return a new data instance:

```python
>>> new_cls_yfdata = cls_yfdata.replace(
...     classes=vbt.symbol_dict({
...         "META": dict(class1="Equity", class2="Technology"),
...         "GOOGL": dict(class1="Equity", class2="Technology"),
...         "NFLX": dict(class1="Equity", class2="Technology"),
...         "BAC": dict(class1="Equity", class2="Financial"),
...         "WFC": dict(class1="Equity", class2="Financial"),
...         "TLT": dict(class1="Fixed Income", class2="Treasury"),
...         "SHV": dict(class1="Fixed Income", class2="Treasury"),
...     })
... )
```

```python
>>> new_cls_yfdata = cls_yfdata.replace(
...     classes=vbt.symbol_dict({
...         "META": dict(class1="Equity", class2="Technology"),
...         "GOOGL": dict(class1="Equity", class2="Technology"),
...         "NFLX": dict(class1="Equity", class2="Technology"),
...         "BAC": dict(class1="Equity", class2="Financial"),
...         "WFC": dict(class1="Equity", class2="Financial"),
...         "TLT": dict(class1="Fixed Income", class2="Treasury"),
...         "SHV": dict(class1="Fixed Income", class2="Treasury"),
...     })
... )
```

Or by using Data.update_classes:

```python
>>> new_cls_yfdata = cls_yfdata.update_classes(
...     class1=vbt.symbol_dict({
...         "META": "Equity",
...         "GOOGL": "Equity",
...         "NFLX": "Equity",
...         "BAC": "Equity",
...         "WFC": "Equity",
...         "TLT": "Fixed Income",
...         "SHV": "Fixed Income",
...     }),
...     class2=vbt.symbol_dict({
...         "META": "Technology",
...         "GOOGL": "Technology",
...         "NFLX": "Technology",
...         "BAC": "Financial",
...         "WFC": "Financial",
...         "TLT": "Treasury",
...         "SHV": "Treasury",
...     })
... )
```

```python
>>> new_cls_yfdata = cls_yfdata.update_classes(
...     class1=vbt.symbol_dict({
...         "META": "Equity",
...         "GOOGL": "Equity",
...         "NFLX": "Equity",
...         "BAC": "Equity",
...         "WFC": "Equity",
...         "TLT": "Fixed Income",
...         "SHV": "Fixed Income",
...     }),
...     class2=vbt.symbol_dict({
...         "META": "Technology",
...         "GOOGL": "Technology",
...         "NFLX": "Technology",
...         "BAC": "Financial",
...         "WFC": "Financial",
...         "TLT": "Treasury",
...         "SHV": "Treasury",
...     })
... )
```

Info

If the instance is feature-oriented and the dictionary with classes is of the type feature_dict, the classes will be applied to features rather than symbols.

## Wrapping¶

We don't need data instances to work with vectorbt since the main objects of vectorbt's operation are Pandas and NumPy arrays, but sometimes it's much more convenient having all the data located under the same Data container because it can be managed (aligned, resampled, transformed, etc.) in a standardized way. To wrap any custom Pandas object with a Data class, we can use the class method Data.from_data, which can take either a single Pandas object (will be stored under the symbol symbol), a symbol dictionary consisting of multiple Pandas objects - one per symbol, or a feature dictionary consisting of multiple Pandas objects - one per feature.

```python
symbol
```

The Series/DataFrame to be wrapped normally has columns associated with features such as OHLC as opposed to symbols such as BTCUSDT, for example:

```python
BTCUSDT
```

```python
>>> btc_df = pd.DataFrame({
...     "Open": [7194.89, 7202.55, 6984.42],
...     "High": [7254.33, 7212.15, 7413.71],
...     "Low": [7174.94, 6935.27, 6985.47],
...     "Close": [7200.17, 6985.47, 7344.88]
... }, index=vbt.date_range("2020-01-01", periods=3))
>>> btc_df
               Open     High      Low    Close
2020-01-01  7194.89  7254.33  7174.94  7200.17
2020-01-02  7202.55  7212.15  6935.27  6985.47
2020-01-03  6984.42  7413.71  6985.47  7344.88

>>> my_data = vbt.Data.from_data(btc_df)
>>> my_data.hlc3
2020-01-01 00:00:00+00:00    7209.813333
2020-01-02 00:00:00+00:00    7044.296667
2020-01-03 00:00:00+00:00    7248.020000
Freq: D, dtype: float64
```

```python
>>> btc_df = pd.DataFrame({
...     "Open": [7194.89, 7202.55, 6984.42],
...     "High": [7254.33, 7212.15, 7413.71],
...     "Low": [7174.94, 6935.27, 6985.47],
...     "Close": [7200.17, 6985.47, 7344.88]
... }, index=vbt.date_range("2020-01-01", periods=3))
>>> btc_df
               Open     High      Low    Close
2020-01-01  7194.89  7254.33  7174.94  7200.17
2020-01-02  7202.55  7212.15  6935.27  6985.47
2020-01-03  6984.42  7413.71  6985.47  7344.88

>>> my_data = vbt.Data.from_data(btc_df)
>>> my_data.hlc3
2020-01-01 00:00:00+00:00    7209.813333
2020-01-02 00:00:00+00:00    7044.296667
2020-01-03 00:00:00+00:00    7248.020000
Freq: D, dtype: float64
```

We can also wrap multiple Pandas objects keyed by symbol:

```python
>>> eth_df = pd.DataFrame({
...     "Open": [127.41, 134.16, 135.07],
...     "High": [134.55, 136.05, 139.41],
...     "Low": [126.49, 133.04, 135.04],
...     "Close": [134.17, 135.06, 136.27]
... }, index=vbt.date_range("2020-01-03", periods=3))  # (1)!
>>> eth_df
              Open    High     Low   Close
2020-01-03  127.41  134.55  126.49  134.17
2020-01-04  134.16  136.05  133.04  135.06
2020-01-05  135.07  139.41  135.04  136.27

>>> my_data = vbt.Data.from_data({"BTCUSDT": btc_df, "ETHUSDT": eth_df})
>>> my_data.hlc3
symbol                         BTCUSDT     ETHUSDT
2020-01-01 00:00:00+00:00  7209.813333         NaN
2020-01-02 00:00:00+00:00  7044.296667         NaN
2020-01-03 00:00:00+00:00  7248.020000  131.736667
2020-01-04 00:00:00+00:00          NaN  134.716667
2020-01-05 00:00:00+00:00          NaN  136.906667
```

```python
>>> eth_df = pd.DataFrame({
...     "Open": [127.41, 134.16, 135.07],
...     "High": [134.55, 136.05, 139.41],
...     "Low": [126.49, 133.04, 135.04],
...     "Close": [134.17, 135.06, 136.27]
... }, index=vbt.date_range("2020-01-03", periods=3))  # (1)!
>>> eth_df
              Open    High     Low   Close
2020-01-03  127.41  134.55  126.49  134.17
2020-01-04  134.16  136.05  133.04  135.06
2020-01-05  135.07  139.41  135.04  136.27

>>> my_data = vbt.Data.from_data({"BTCUSDT": btc_df, "ETHUSDT": eth_df})
>>> my_data.hlc3
symbol                         BTCUSDT     ETHUSDT
2020-01-01 00:00:00+00:00  7209.813333         NaN
2020-01-02 00:00:00+00:00  7044.296667         NaN
2020-01-03 00:00:00+00:00  7248.020000  131.736667
2020-01-04 00:00:00+00:00          NaN  134.716667
2020-01-05 00:00:00+00:00          NaN  136.906667
```

If our data happen to have symbols as columns, enable columns_are_symbols:

```python
columns_are_symbols
```

```python
>>> hlc3_data = vbt.Data.from_data(my_data.hlc3, columns_are_symbols=True)
>>> hlc3_data.get()
symbol                         BTCUSDT     ETHUSDT
2020-01-01 00:00:00+00:00  7209.813333         NaN
2020-01-02 00:00:00+00:00  7044.296667         NaN
2020-01-03 00:00:00+00:00  7248.020000  131.736667
2020-01-04 00:00:00+00:00          NaN  134.716667
2020-01-05 00:00:00+00:00          NaN  136.906667
```

```python
>>> hlc3_data = vbt.Data.from_data(my_data.hlc3, columns_are_symbols=True)
>>> hlc3_data.get()
symbol                         BTCUSDT     ETHUSDT
2020-01-01 00:00:00+00:00  7209.813333         NaN
2020-01-02 00:00:00+00:00  7044.296667         NaN
2020-01-03 00:00:00+00:00  7248.020000  131.736667
2020-01-04 00:00:00+00:00          NaN  134.716667
2020-01-05 00:00:00+00:00          NaN  136.906667
```

In this case, the instance will become feature-oriented, that is, the DataFrame above will be stored in a feature_dict and the behavior of symbols and features will be swapped across many methods. To make the instance symbol-oriented as in most of our examples, additionally pass invert_data=True.

```python
invert_data=True
```

## Merging¶

As you might have already noticed, the process of aligning data is logically separated from the process of fetching data, enabling us to merge and align any data retrospectively.

Instead of storing and managing all symbols as a single monolithic entity, we can manage them separately and merge into one data instance whenever this is actually needed. Such an approach may be particularly useful when symbols are distributed over multiple data classes, such as a mixture of remote and local data sources. For this, we can use the class method Data.merge, which takes two or more data instances, merges their information, and forwards the merged information to Data.from_data:

```python
>>> yf_data_btc = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-01", 
...     end="2020-01-03"
... )
>>> yf_data_eth = YFData.pull(
...     "ETH-USD", 
...     start="2020-01-03", 
...     end="2020-01-05"
... )
>>> merged_yf_data = YFData.merge(yf_data_btc, yf_data_eth)
>>> merged_yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00          NaN  134.171707
2020-01-04 00:00:00+00:00          NaN  135.069366
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

```python
>>> yf_data_btc = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-01", 
...     end="2020-01-03"
... )
>>> yf_data_eth = YFData.pull(
...     "ETH-USD", 
...     start="2020-01-03", 
...     end="2020-01-05"
... )
>>> merged_yf_data = YFData.merge(yf_data_btc, yf_data_eth)
>>> merged_yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00          NaN  134.171707
2020-01-04 00:00:00+00:00          NaN  135.069366
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
```

The benefit of this method is that it not only merges different symbols across different data instances, but it can also merge Pandas objects corresponding to the same symbol:

```python
>>> yf_data_btc1 = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-01", 
...     end="2020-01-03"
... )
>>> yf_data_btc2 = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-05", 
...     end="2020-01-07"
... )
>>> yf_data_eth = YFData.pull(
...     "ETH-USD", 
...     start="2020-01-06", 
...     end="2020-01-08"
... )
>>> merged_yf_data = YFData.merge(yf_data_btc1, yf_data_btc2, yf_data_eth)
>>> merged_yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215         NaN
2020-01-04 00:00:00+00:00  7410.656738         NaN
2020-01-05 00:00:00+00:00  7411.317383  136.276779
2020-01-06 00:00:00+00:00  7769.219238  144.304153
2020-01-07 00:00:00+00:00          NaN  143.543991
```

```python
>>> yf_data_btc1 = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-01", 
...     end="2020-01-03"
... )
>>> yf_data_btc2 = YFData.pull(
...     "BTC-USD", 
...     start="2020-01-05", 
...     end="2020-01-07"
... )
>>> yf_data_eth = YFData.pull(
...     "ETH-USD", 
...     start="2020-01-06", 
...     end="2020-01-08"
... )
>>> merged_yf_data = YFData.merge(yf_data_btc1, yf_data_btc2, yf_data_eth)
>>> merged_yf_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2019-12-31 00:00:00+00:00  7193.599121         NaN
2020-01-01 00:00:00+00:00  7200.174316         NaN
2020-01-02 00:00:00+00:00  6985.470215         NaN
2020-01-04 00:00:00+00:00  7410.656738         NaN
2020-01-05 00:00:00+00:00  7411.317383  136.276779
2020-01-06 00:00:00+00:00  7769.219238  144.304153
2020-01-07 00:00:00+00:00          NaN  143.543991
```

### Subclassing¶

We called Data on the class YFData, which automatically creates an instance of that class. Having an instance of YFData, we can update the data the same way as we did before.But what if the data instances to be merged originate from different data classes? If we used YFData for merging CCXTData and BinanceData instances, we wouldn't be able to update the data objects anymore since the method YFData.update_symbol was implemented specifically for the symbols supported by Yahoo Finance.

```python
YFData
```

```python
YFData
```

```python
YFData
```

```python
YFData.update_symbol
```

In such case, either use Data, which will raise an error when attempting to update, or create a subclass of it to handle updates using different data providers (which is fairly easy if you know which symbol belongs to which data class - just call the respective fetch_symbol or update_symbol method):

```python
fetch_symbol
```

```python
update_symbol
```

```python
>>> bn_data_btc = vbt.BinanceData.pull(
...     "BTCUSDT", 
...     start="2020-01-01", 
...     end="2020-01-04")
>>> bn_data_btc.close
Open time
2020-01-01 00:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    6965.71
2020-01-03 00:00:00+00:00    7344.96
Freq: D, Name: Close, dtype: float64

>>> ccxt_data_eth = vbt.CCXTData.pull(
...     "ETH/USDT", 
...     start="2020-01-03", 
...     end="2020-01-06")
>>> ccxt_data_eth.close
Open time
2020-01-03 00:00:00+00:00    134.35
2020-01-04 00:00:00+00:00    134.20
2020-01-05 00:00:00+00:00    135.37
Freq: D, Name: Close, dtype: float64

>>> class MergedData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         if symbol.startswith("BN_"):
...             return vbt.BinanceData.fetch_symbol(symbol[3:], **kwargs)
...         if symbol.startswith("CCXT_"):
...             return vbt.CCXTData.fetch_symbol(symbol[5:], **kwargs)
...         raise ValueError(f"Unknown symbol '{symbol}'")
...
...     def update_symbol(self, symbol, **kwargs):
...         fetch_kwargs = self.select_fetch_kwargs(symbol)
...         fetch_kwargs["start"] = self.select_last_index(symbol)
...         kwargs = vbt.merge_dicts(fetch_kwargs, kwargs)
...         return self.fetch_symbol(symbol, **kwargs)

>>> merged_data = MergedData.merge(
...     bn_data_btc, 
...     ccxt_data_eth,
...     rename={
...         "BTCUSDT": "BN_BTCUSDT", 
...         "ETH/USDT": "CCXT_ETH/USDT"
...     },
...     missing_columns="drop"
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
UserWarning: Symbols have mismatching columns. Dropping missing data points.

>>> merged_data = merged_data.update(end="2020-01-07")
>>> merged_data.close
symbol                     BN_BTCUSDT  CCXT_ETH/USDT
Open time                                           
2020-01-01 00:00:00+00:00     7200.85            NaN
2020-01-02 00:00:00+00:00     6965.71            NaN
2020-01-03 00:00:00+00:00     7344.96         134.35
2020-01-04 00:00:00+00:00     7354.11         134.20
2020-01-05 00:00:00+00:00     7358.75         135.37
2020-01-06 00:00:00+00:00     7758.00         144.15
```

```python
>>> bn_data_btc = vbt.BinanceData.pull(
...     "BTCUSDT", 
...     start="2020-01-01", 
...     end="2020-01-04")
>>> bn_data_btc.close
Open time
2020-01-01 00:00:00+00:00    7200.85
2020-01-02 00:00:00+00:00    6965.71
2020-01-03 00:00:00+00:00    7344.96
Freq: D, Name: Close, dtype: float64

>>> ccxt_data_eth = vbt.CCXTData.pull(
...     "ETH/USDT", 
...     start="2020-01-03", 
...     end="2020-01-06")
>>> ccxt_data_eth.close
Open time
2020-01-03 00:00:00+00:00    134.35
2020-01-04 00:00:00+00:00    134.20
2020-01-05 00:00:00+00:00    135.37
Freq: D, Name: Close, dtype: float64

>>> class MergedData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, **kwargs):
...         if symbol.startswith("BN_"):
...             return vbt.BinanceData.fetch_symbol(symbol[3:], **kwargs)
...         if symbol.startswith("CCXT_"):
...             return vbt.CCXTData.fetch_symbol(symbol[5:], **kwargs)
...         raise ValueError(f"Unknown symbol '{symbol}'")
...
...     def update_symbol(self, symbol, **kwargs):
...         fetch_kwargs = self.select_fetch_kwargs(symbol)
...         fetch_kwargs["start"] = self.select_last_index(symbol)
...         kwargs = vbt.merge_dicts(fetch_kwargs, kwargs)
...         return self.fetch_symbol(symbol, **kwargs)

>>> merged_data = MergedData.merge(
...     bn_data_btc, 
...     ccxt_data_eth,
...     rename={
...         "BTCUSDT": "BN_BTCUSDT", 
...         "ETH/USDT": "CCXT_ETH/USDT"
...     },
...     missing_columns="drop"
... )
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.
UserWarning: Symbols have mismatching columns. Dropping missing data points.

>>> merged_data = merged_data.update(end="2020-01-07")
>>> merged_data.close
symbol                     BN_BTCUSDT  CCXT_ETH/USDT
Open time                                           
2020-01-01 00:00:00+00:00     7200.85            NaN
2020-01-02 00:00:00+00:00     6965.71            NaN
2020-01-03 00:00:00+00:00     7344.96         134.35
2020-01-04 00:00:00+00:00     7354.11         134.20
2020-01-05 00:00:00+00:00     7358.75         135.37
2020-01-06 00:00:00+00:00     7758.00         144.15
```

We just created a flexible data class that can fetch, update, and manage symbols from multiple data providers. Great!

## Resampling¶

As a subclass of Wrapping, each data instance stores the normalized metadata of all Pandas objects stored in that instance. This metadata can be used for resampling (i.e., changing the time frame) of all Pandas objects at once. Since many data classes, such as CCXTData, have a fixed feature layout, we can define the resampling function for each of their features in a special config called "feature config" (stored under Data.feature_config) and bind that config to the class itself for the use by all instances. Similar to field configs in Records, this config also can be attached to an entire data class or on any of its instances. Whenever a new instance is created, the config of the class is copied over such that rewriting it wouldn't affect the class config.

Here's, for example, how the feature config of BinanceData looks like:

```python
>>> vbt.pprint(vbt.BinanceData.feature_config)
HybridConfig({
    'Quote volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d7280>
    ),
    'Taker base volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d7310>
    ),
    'Taker quote volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d73a0>
    )
})
```

```python
>>> vbt.pprint(vbt.BinanceData.feature_config)
HybridConfig({
    'Quote volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d7280>
    ),
    'Taker base volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d7310>
    ),
    'Taker quote volume': dict(
        resample_func=<function BinanceData.<lambda> at 0x7ff7648d73a0>
    )
})
```

Wondering where are the resampling functions for all the OHLCV features? Those features are universal, and recognized and resampled automatically.

Let's resample the entire daily BTC/USD data from Yahoo Finance to the monthly frequency:

```python
>>> full_yf_data = vbt.YFData.pull("BTC-USD")  # (1)!
>>> ms_yf_data = full_yf_data.resample("M")
>>> ms_yf_data.close
Date
2014-09-01 00:00:00+00:00      386.944000
2014-10-01 00:00:00+00:00      338.321014
2014-11-01 00:00:00+00:00      378.046997
                                      ...     
2023-06-01 00:00:00+00:00    30477.251953
2023-07-01 00:00:00+00:00    29230.111328
2023-08-01 00:00:00+00:00    25995.177734
Freq: MS, Name: Close, Length: 108, dtype: float64
```

```python
>>> full_yf_data = vbt.YFData.pull("BTC-USD")  # (1)!
>>> ms_yf_data = full_yf_data.resample("M")
>>> ms_yf_data.close
Date
2014-09-01 00:00:00+00:00      386.944000
2014-10-01 00:00:00+00:00      338.321014
2014-11-01 00:00:00+00:00      378.046997
                                      ...     
2023-06-01 00:00:00+00:00    30477.251953
2023-07-01 00:00:00+00:00    29230.111328
2023-08-01 00:00:00+00:00    25995.177734
Freq: MS, Name: Close, Length: 108, dtype: float64
```

Since vectorbt works with custom target indexes just as well as with frequencies, we can provide a custom index to resample to:

```python
>>> resampler = vbt.Resampler.from_date_range(
...     full_yf_data.wrapper.index,
...     start=full_yf_data.wrapper.index[0],
...     end=full_yf_data.wrapper.index[-1],
...     freq="Y",
...     silence_warnings=True
... )
>>> y_yf_data = full_yf_data.resample(resampler)
>>> y_yf_data.close
2014-12-31 00:00:00+00:00      426.619995
2015-12-31 00:00:00+00:00      961.237976
2016-12-31 00:00:00+00:00    12952.200195
2017-12-31 00:00:00+00:00     3865.952637
2018-12-31 00:00:00+00:00     7292.995117
2019-12-31 00:00:00+00:00    28840.953125
2020-12-31 00:00:00+00:00    47178.125000
2021-12-31 00:00:00+00:00    39194.972656
Freq: A-DEC, Name: Close, dtype: float64
```

```python
>>> resampler = vbt.Resampler.from_date_range(
...     full_yf_data.wrapper.index,
...     start=full_yf_data.wrapper.index[0],
...     end=full_yf_data.wrapper.index[-1],
...     freq="Y",
...     silence_warnings=True
... )
>>> y_yf_data = full_yf_data.resample(resampler)
>>> y_yf_data.close
2014-12-31 00:00:00+00:00      426.619995
2015-12-31 00:00:00+00:00      961.237976
2016-12-31 00:00:00+00:00    12952.200195
2017-12-31 00:00:00+00:00     3865.952637
2018-12-31 00:00:00+00:00     7292.995117
2019-12-31 00:00:00+00:00    28840.953125
2020-12-31 00:00:00+00:00    47178.125000
2021-12-31 00:00:00+00:00    39194.972656
Freq: A-DEC, Name: Close, dtype: float64
```

Note

Whenever providing a custom index, vectorbt will aggregate all the values after each index entry. The last entry aggregates all the values up to infinity. See GenericAccessor.resample_to_index.

If a data class doesn't have a fixed feature layout, such as HDFData, we need to adapt the feature config to each data instance instead of setting it to the entire data class. For example, if we convert bn_data_btc to a generic Data instance:

```python
bn_data_btc
```

```python
>>> data_btc = vbt.Data.from_data(bn_data_btc.data, single_key=True)
>>> data_btc.resample("M")
ValueError: Cannot resample feature 'Quote volume'. Specify resample_func in feature_config.

>>> for k, v in vbt.BinanceData.feature_config.items():
...     data_btc.feature_config[k] = v
>>> data_btc.resample("M")
<vectorbtpro.data.base.Data at 0x7fc0dfce1630>
```

```python
>>> data_btc = vbt.Data.from_data(bn_data_btc.data, single_key=True)
>>> data_btc.resample("M")
ValueError: Cannot resample feature 'Quote volume'. Specify resample_func in feature_config.

>>> for k, v in vbt.BinanceData.feature_config.items():
...     data_btc.feature_config[k] = v
>>> data_btc.resample("M")
<vectorbtpro.data.base.Data at 0x7fc0dfce1630>
```

The same can be done with a single copy operation using Data.use_feature_config_of:

```python
>>> data_btc = vbt.Data.from_data(bn_data_btc.data, single_key=True)
>>> data_btc.use_feature_config_of(vbt.BinanceData)
>>> data_btc.resample("M").close
Open time
2020-01-01 00:00:00+00:00    7344.96
Freq: MS, Name: Close, dtype: float64
```

```python
>>> data_btc = vbt.Data.from_data(bn_data_btc.data, single_key=True)
>>> data_btc.use_feature_config_of(vbt.BinanceData)
>>> data_btc.resample("M").close
Open time
2020-01-01 00:00:00+00:00    7344.96
Freq: MS, Name: Close, dtype: float64
```

### Realignment¶

Similarly to resampling, realignment also changes the frequency of data, but in contrast to resampling, it doesn't aggregate the data but includes only the latest data point available at each step in the target index. It uses GenericAccessor.realign_opening for "open" and GenericAccessor.realign_closing for any other feature. This has two major use cases: aligning multiple symbols from different timezones to a single index, and upsampling data. Let's align symbols with different timings:

```python
>>> data = vbt.YFData.pull(
...     ["BTC-USD", "AAPL"], 
...     start="2020-01-01", 
...     end="2020-01-04"
... )
>>> data.close
symbol                         BTC-USD       AAPL
Date                                             
2020-01-01 00:00:00+00:00  7200.174316        NaN
2020-01-02 00:00:00+00:00  6985.470215        NaN
2020-01-02 05:00:00+00:00          NaN  73.249016
2020-01-03 00:00:00+00:00  7344.884277        NaN
2020-01-03 05:00:00+00:00          NaN  72.536888

>>> new_index = data.index.ceil("D").drop_duplicates()
>>> new_data = data.realign(new_index, ffill=False)
>>> new_data.close
symbol                         BTC-USD       AAPL
Date                                             
2020-01-01 00:00:00+00:00  7200.174316        NaN
2020-01-02 00:00:00+00:00  6985.470215        NaN
2020-01-03 00:00:00+00:00  7344.884277  73.249016
2020-01-04 00:00:00+00:00          NaN  72.536888
```

```python
>>> data = vbt.YFData.pull(
...     ["BTC-USD", "AAPL"], 
...     start="2020-01-01", 
...     end="2020-01-04"
... )
>>> data.close
symbol                         BTC-USD       AAPL
Date                                             
2020-01-01 00:00:00+00:00  7200.174316        NaN
2020-01-02 00:00:00+00:00  6985.470215        NaN
2020-01-02 05:00:00+00:00          NaN  73.249016
2020-01-03 00:00:00+00:00  7344.884277        NaN
2020-01-03 05:00:00+00:00          NaN  72.536888

>>> new_index = data.index.ceil("D").drop_duplicates()
>>> new_data = data.realign(new_index, ffill=False)
>>> new_data.close
symbol                         BTC-USD       AAPL
Date                                             
2020-01-01 00:00:00+00:00  7200.174316        NaN
2020-01-02 00:00:00+00:00  6985.470215        NaN
2020-01-03 00:00:00+00:00  7344.884277  73.249016
2020-01-04 00:00:00+00:00          NaN  72.536888
```

## Transforming¶

The main challenge in transforming any data is that each symbol must have the same index and columns because we need to concatenate them into one Pandas object later, thus any transformation operation must ensure that it's applied on each symbol in the same way. To enforce that, the method Data.transform concatenates the data across all symbols and features into one big DataFrame, and passes it to an UDF for transformation. Once transformed, the method splits the result back into multiple smaller Pandas objects - one per symbol, aligns them, creates a new data wrapper based on the aligned index and columns, and finally, initializes a new data instance.

Let's drop any row that contains at least one NaN:

```python
>>> full_yf_data = YFData.pull(["BTC-USD", "ETH-USD"])
>>> full_yf_data.close.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 3268 entries, 2014-09-17 00:00:00+00:00 to 2023-08-28 00:00:00+00:00
Freq: D
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   BTC-USD  3268 non-null   float64
 1   ETH-USD  2119 non-null   float64
dtypes: float64(2)
memory usage: 76.6 KB

>>> new_full_yf_data = full_yf_data.transform(lambda df: df.dropna())
>>> new_full_yf_data.close.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2119 entries, 2017-11-09 00:00:00+00:00 to 2023-08-28 00:00:00+00:00
Freq: D
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   BTC-USD  2119 non-null   float64
 1   ETH-USD  2119 non-null   float64
dtypes: float64(2)
memory usage: 49.7 KB
```

```python
>>> full_yf_data = YFData.pull(["BTC-USD", "ETH-USD"])
>>> full_yf_data.close.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 3268 entries, 2014-09-17 00:00:00+00:00 to 2023-08-28 00:00:00+00:00
Freq: D
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   BTC-USD  3268 non-null   float64
 1   ETH-USD  2119 non-null   float64
dtypes: float64(2)
memory usage: 76.6 KB

>>> new_full_yf_data = full_yf_data.transform(lambda df: df.dropna())
>>> new_full_yf_data.close.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2119 entries, 2017-11-09 00:00:00+00:00 to 2023-08-28 00:00:00+00:00
Freq: D
Data columns (total 2 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   BTC-USD  2119 non-null   float64
 1   ETH-USD  2119 non-null   float64
dtypes: float64(2)
memory usage: 49.7 KB
```

We can also decide to pass only one feature or symbol at a time by setting per_feature=True and per_symbol=True respectively. By enabling both arguments simultaneously, we can instruct vectorbt to pass only one feature and symbol combination as a Pandas Series at a time.

```python
per_feature=True
```

```python
per_symbol=True
```

## Analysis¶

Each data class subclasses Analyzable, which makes it analyzable and indexable.

### Indexing¶

We can perform Pandas indexing on the data instance to select rows and columns in all fetched Pandas objects. Supported operations are iloc, loc, xs, and []:

```python
iloc
```

```python
loc
```

```python
xs
```

```python
[]
```

```python
>>> sub_yf_data = yf_data.loc["2020-01-01":"2020-01-03"]  # (1)!
>>> sub_yf_data
<__main__.YFData at 0x7fa9a0012396>

>>> sub_yf_data = sub_yf_data[["Open", "High", "Low", "Close"]]  # (2)!
>>> sub_yf_data
<__main__.YFData at 0x7fa9a0032358>

>>> sub_yf_data.data["BTC-USD"]
                                  Open         High          Low        Close
Date                                                                         
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277

>>> sub_yf_data.data["ETH-USD"]
                                 Open        High         Low       Close
Date                                                                     
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707
```

```python
>>> sub_yf_data = yf_data.loc["2020-01-01":"2020-01-03"]  # (1)!
>>> sub_yf_data
<__main__.YFData at 0x7fa9a0012396>

>>> sub_yf_data = sub_yf_data[["Open", "High", "Low", "Close"]]  # (2)!
>>> sub_yf_data
<__main__.YFData at 0x7fa9a0032358>

>>> sub_yf_data.data["BTC-USD"]
                                  Open         High          Low        Close
Date                                                                         
2020-01-01 00:00:00+00:00  7194.892090  7254.330566  7174.944336  7200.174316
2020-01-02 00:00:00+00:00  7202.551270  7212.155273  6935.270020  6985.470215
2020-01-03 00:00:00+00:00  6984.428711  7413.715332  6914.996094  7344.884277

>>> sub_yf_data.data["ETH-USD"]
                                 Open        High         Low       Close
Date                                                                     
2020-01-01 00:00:00+00:00  129.630661  132.835358  129.198288  130.802002
2020-01-02 00:00:00+00:00  130.820038  130.820038  126.954910  127.410179
2020-01-03 00:00:00+00:00  127.411263  134.554016  126.490021  134.171707
```

```python
loc
```

Note

Don't attempt to select symbols in this way - this notation is reserved for rows and columns only. Use Data.select instead.

Info

If the instance is feature-oriented, this method will apply to features rather than symbols.

### Stats and plots¶

As with every Analyzable instance, we can compute and plot various properties of the data stored in the instance.

Very often, a simple call of DataFrame.info and DataFrame.describe on any of the stored Series or DataFrames is enough to print a concise summary:

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start=vbt.symbol_dict({
...         "BTC-USD": "2020-01-01", 
...         "ETH-USD": "2020-01-03"
...     }),
...     end=vbt.symbol_dict({
...         "BTC-USD": "2020-01-03", 
...         "ETH-USD": "2020-01-05"
...     })
... )
```

Symbol 2/2

Symbol 2/2

```python
>>> yf_data.data["BTC-USD"].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 5 entries, 2019-12-31 00:00:00+00:00 to 2020-01-04 00:00:00+00:00
Freq: D
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Open          3 non-null      float64
 1   High          3 non-null      float64
 2   Low           3 non-null      float64
 3   Close         3 non-null      float64
 4   Volume        3 non-null      float64
 5   Dividends     3 non-null      float64
 6   Stock Splits  3 non-null      float64
dtypes: float64(7)
memory usage: 320.0 bytes

>>> yf_data.data["BTC-USD"].describe()
              Open         High          Low        Close        Volume  \
count     3.000000     3.000000     3.000000     3.000000  3.000000e+00   
mean   7230.627441  7267.258626  7093.330729  7126.414551  2.017856e+10   
std      55.394933    62.577102   136.908963   122.105641  1.408740e+09   
min    7194.892090  7212.155273  6935.270020  6985.470215  1.856566e+10   
25%    7198.721680  7233.242920  7052.523926  7089.534668  1.968387e+10   
50%    7202.551270  7254.330566  7169.777832  7193.599121  2.080208e+10   
75%    7248.495117  7294.810303  7172.361084  7196.886719  2.098501e+10   
max    7294.438965  7335.290039  7174.944336  7200.174316  2.116795e+10   

       Dividends  Stock Splits  
count        3.0           3.0  
mean         0.0           0.0  
std          0.0           0.0  
min          0.0           0.0  
25%          0.0           0.0  
50%          0.0           0.0  
75%          0.0           0.0  
max          0.0           0.0
```

```python
>>> yf_data.data["BTC-USD"].info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 5 entries, 2019-12-31 00:00:00+00:00 to 2020-01-04 00:00:00+00:00
Freq: D
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Open          3 non-null      float64
 1   High          3 non-null      float64
 2   Low           3 non-null      float64
 3   Close         3 non-null      float64
 4   Volume        3 non-null      float64
 5   Dividends     3 non-null      float64
 6   Stock Splits  3 non-null      float64
dtypes: float64(7)
memory usage: 320.0 bytes

>>> yf_data.data["BTC-USD"].describe()
              Open         High          Low        Close        Volume  \
count     3.000000     3.000000     3.000000     3.000000  3.000000e+00   
mean   7230.627441  7267.258626  7093.330729  7126.414551  2.017856e+10   
std      55.394933    62.577102   136.908963   122.105641  1.408740e+09   
min    7194.892090  7212.155273  6935.270020  6985.470215  1.856566e+10   
25%    7198.721680  7233.242920  7052.523926  7089.534668  1.968387e+10   
50%    7202.551270  7254.330566  7169.777832  7193.599121  2.080208e+10   
75%    7248.495117  7294.810303  7172.361084  7196.886719  2.098501e+10   
max    7294.438965  7335.290039  7174.944336  7200.174316  2.116795e+10   

       Dividends  Stock Splits  
count        3.0           3.0  
mean         0.0           0.0  
std          0.0           0.0  
min          0.0           0.0  
25%          0.0           0.0  
50%          0.0           0.0  
75%          0.0           0.0  
max          0.0           0.0
```

But since any data instance can capture multiple symbols, using StatsBuilderMixin.stats can provide us with information on symbols as well:

```python
>>> yf_data.stats()
Start                   2019-12-31 00:00:00+00:00
End                     2020-01-04 00:00:00+00:00
Period                            5 days 00:00:00
Total Symbols                                   2
Null Counts: BTC-USD                           14
Null Counts: ETH-USD                           14
Name: agg_stats, dtype: object

>>> yf_data.stats(column="Volume")  # (1)!
Start                   2019-12-31 00:00:00+00:00
End                     2020-01-04 00:00:00+00:00
Period                            5 days 00:00:00
Total Symbols                                   2
Null Counts: BTC-USD                            2
Null Counts: ETH-USD                            2
Name: Volume, dtype: object
```

```python
>>> yf_data.stats()
Start                   2019-12-31 00:00:00+00:00
End                     2020-01-04 00:00:00+00:00
Period                            5 days 00:00:00
Total Symbols                                   2
Null Counts: BTC-USD                           14
Null Counts: ETH-USD                           14
Name: agg_stats, dtype: object

>>> yf_data.stats(column="Volume")  # (1)!
Start                   2019-12-31 00:00:00+00:00
End                     2020-01-04 00:00:00+00:00
Period                            5 days 00:00:00
Total Symbols                                   2
Null Counts: BTC-USD                            2
Null Counts: ETH-USD                            2
Name: Volume, dtype: object
```

```python
Volume
```

To plot the data, we can use the method Data.plot, which produces an OHLC(V) chart whenever the Pandas object is a DataFrame with regular price features, and a line chart otherwise. The former can plot only one symbol of data, while the latter can plot only one feature of data; both can be specified with the symbol and feature argument respectively.

```python
symbol
```

```python
feature
```

Since different symbols mostly have different starting values, we can provide an argument base, which will rebase the time series to start from the same point on chart:

```python
base
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-06-01"
... )
```

```python
>>> yf_data = YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     start="2020-01-01", 
...     end="2020-06-01"
... )
```

Symbol 2/2

Symbol 2/2

```python
>>> yf_data.plot(column="Close", base=100).show()  # (1)!
```

```python
>>> yf_data.plot(column="Close", base=100).show()  # (1)!
```

```python
column
```

```python
feature
```

Info

This only works for line traces since we cannot plot multiple OHLC(V) traces on the same chart.

Like most things, the same can be replicated using a chain of simple commands:

```python
>>> yf_data["Close"].get().vbt.rebase(100).vbt.plot()  # (1)!
```

```python
>>> yf_data["Close"].get().vbt.rebase(100).vbt.plot()  # (1)!
```

In addition, Data can display a subplot per symbol using PlotsBuilderMixin.plots, which utilizes Data.plot under the hood:

```python
>>> yf_data.plots().show()
```

```python
>>> yf_data.plots().show()
```

By also specifying a column, we can plot one feature per symbol of data:

```python
>>> yf_data.plots(column="Close").show()
```

```python
>>> yf_data.plots(column="Close").show()
```

We can select one or more symbols by passing them via the template_context dictionary:

```python
template_context
```

```python
>>> yf_data.plots(
...     column="Close", 
...     template_context=dict(symbols=["BTC-USD"])
... ).show()
```

```python
>>> yf_data.plots(
...     column="Close", 
...     template_context=dict(symbols=["BTC-USD"])
... ).show()
```

If you look into the Data.subplots config, you'll notice only one subplot defined as a template. During the resolution phase, the template will be evaluated and the subplot will be expanded into multiple subplots - one per symbol - with the same name plot but prefixed with the index of that subplot in the expansion. For illustration, let's change the colors of both lines and plot their moving averages:

```python
plot
```

```python
>>> from vectorbtpro.utils.colors import adjust_opacity

>>> fig = yf_data.plots(
...     column="Close",
...     subplot_settings=dict(
...         plot_0=dict(trace_kwargs=dict(line_color="mediumslateblue")),
...         plot_1=dict(trace_kwargs=dict(line_color="limegreen"))
...     )
... )
>>> sma = vbt.talib("SMA").run(yf_data.close, vbt.Default(10))  # (1)!
>>> sma["BTC-USD"].real.rename("SMA(10)").vbt.plot(
...     trace_kwargs=dict(line_color=adjust_opacity("mediumslateblue", 0.7)),
...     add_trace_kwargs=dict(row=1, col=1),  # (2)!
...     fig=fig
... )
>>> sma["ETH-USD"].real.rename("SMA(10)").vbt.plot(
...     trace_kwargs=dict(line_color=adjust_opacity("limegreen", 0.7)),
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> fig.show()
```

```python
>>> from vectorbtpro.utils.colors import adjust_opacity

>>> fig = yf_data.plots(
...     column="Close",
...     subplot_settings=dict(
...         plot_0=dict(trace_kwargs=dict(line_color="mediumslateblue")),
...         plot_1=dict(trace_kwargs=dict(line_color="limegreen"))
...     )
... )
>>> sma = vbt.talib("SMA").run(yf_data.close, vbt.Default(10))  # (1)!
>>> sma["BTC-USD"].real.rename("SMA(10)").vbt.plot(
...     trace_kwargs=dict(line_color=adjust_opacity("mediumslateblue", 0.7)),
...     add_trace_kwargs=dict(row=1, col=1),  # (2)!
...     fig=fig
... )
>>> sma["ETH-USD"].real.rename("SMA(10)").vbt.plot(
...     trace_kwargs=dict(line_color=adjust_opacity("limegreen", 0.7)),
...     add_trace_kwargs=dict(row=2, col=1),
...     fig=fig
... )
>>> fig.show()
```

```python
timeperiod
```

If you're hungry for a challenge, subclass the YFData class and override the Data.plot method such that it also runs and plots the SMA over the time series. This would make the plotting procedure ultra-flexible because now you can display the SMA for every feature and symbol without caring about the subplot's position and other things.

```python
YFData
```

Python code

