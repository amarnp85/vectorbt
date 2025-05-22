# Data¶

Question

Learn more in Data documentation.

There are plenty of supported data sources for OHLC and indicator data. For the full list, see the custom module.

## Listing¶

Many data classes have a class method to list all symbols that can be fetched. Usually, such as method starts with list_, for example, TVData.list_symbols, SQLData.list_tables, or CSVData.list_paths. In addition, most methods allow client-side filtering of symbols by a glob-style or regex-style pattern.

```python
list_
```

```python
all_symbols = vbt.BinanceData.list_symbols()  # (1)!
usdt_symbols = vbt.BinanceData.list_symbols("*USDT")  # (2)!
usdt_symbols = vbt.BinanceData.list_symbols(r"^.+USDT$", use_regex=True)

all_symbols = vbt.TVData.list_symbols()  # (3)!
nasdaq_symbols = vbt.TVData.list_symbols(exchange_pattern="NASDAQ")  # (4)!
btc_symbols = vbt.TVData.list_symbols(symbol_pattern="BTC*")  # (5)!
pl_symbols = vbt.TVData.list_symbols(market="poland")  # (6)!
usdt_symbols = vbt.TVData.list_symbols(fields=["currency"], filter_by=["USDT"])  # (7)!

def filter_by(market_cap_basic):
    if market_cap_basic is None:
        return False
    return market_cap_basic >= 1_000_000_000_000

trillion_symbols = vbt.TVData.list_symbols(  # (8)!
    fields=["market_cap_basic"], 
    filter_by=vbt.RepFunc(filter_by)
)

all_paths = vbt.FileData.list_paths()  # (9)!
csv_paths = vbt.CSVData.list_paths()  # (10)!
all_csv_paths = vbt.CSVData.list_paths("**/*.csv")  # (11)!
all_data_paths = vbt.HDFData.list_paths("data.h5")  # (12)!
all_paths = vbt.HDFData.list_paths()  # (13)!

all_schemas = vbt.SQLData.list_schemas(engine=engine)  # (14)!
all_tables = vbt.SQLData.list_tables(engine=engine)  # (15)!
```

```python
all_symbols = vbt.BinanceData.list_symbols()  # (1)!
usdt_symbols = vbt.BinanceData.list_symbols("*USDT")  # (2)!
usdt_symbols = vbt.BinanceData.list_symbols(r"^.+USDT$", use_regex=True)

all_symbols = vbt.TVData.list_symbols()  # (3)!
nasdaq_symbols = vbt.TVData.list_symbols(exchange_pattern="NASDAQ")  # (4)!
btc_symbols = vbt.TVData.list_symbols(symbol_pattern="BTC*")  # (5)!
pl_symbols = vbt.TVData.list_symbols(market="poland")  # (6)!
usdt_symbols = vbt.TVData.list_symbols(fields=["currency"], filter_by=["USDT"])  # (7)!

def filter_by(market_cap_basic):
    if market_cap_basic is None:
        return False
    return market_cap_basic >= 1_000_000_000_000

trillion_symbols = vbt.TVData.list_symbols(  # (8)!
    fields=["market_cap_basic"], 
    filter_by=vbt.RepFunc(filter_by)
)

all_paths = vbt.FileData.list_paths()  # (9)!
csv_paths = vbt.CSVData.list_paths()  # (10)!
all_csv_paths = vbt.CSVData.list_paths("**/*.csv")  # (11)!
all_data_paths = vbt.HDFData.list_paths("data.h5")  # (12)!
all_paths = vbt.HDFData.list_paths()  # (13)!

all_schemas = vbt.SQLData.list_schemas(engine=engine)  # (14)!
all_tables = vbt.SQLData.list_tables(engine=engine)  # (15)!
```

## Pulling¶

Each data class has the method fetch_symbol() for fetching a single symbol and returning raw data, usually in a form of a DataFrame. To return a data instance, the method pull() should be used, which takes one or multiple symbols, calls fetch_symbol() on each one, and aligns all DataFrames. For testing, use YFData, which is easy to use but poor in terms of quality. For production, use more reliable data sources, such as CCXTData for crypto and AlpacaData for stocks. For technical analysis based on the most recent data, use TVData (TradingView).

```python
fetch_symbol()
```

```python
pull()
```

```python
fetch_symbol()
```

Hint

To see what arguments a data class like YFData accepts, use vbt.phelp(vbt.YFData.fetch_symbol).

```python
YFData
```

```python
vbt.phelp(vbt.YFData.fetch_symbol)
```

```python
data = vbt.YFData.pull("AAPL")  # (1)!
data = vbt.YFData.pull(["AAPL", "MSFT"])  # (2)!
data = vbt.YFData.pull("AAPL", start="2020")  # (3)!
data = vbt.YFData.pull("AAPL", start="2020", end="2021")  # (4)!
data = vbt.YFData.pull("AAPL", start="1 month ago")  # (5)!
data = vbt.YFData.pull("AAPL", start="1 month ago", timeframe="hourly")  # (6)!
data = vbt.YFData.pull("AAPL", tz="UTC")  # (7)!
data = vbt.YFData.pull(symbols, execute_kwargs=dict(engine="threadpool"))  # (8)!

data = vbt.YFData.pull("AAPL", auto_adjust=False)  # (9)!
data = vbt.BinanceData.pull("BTCUSDT", klines_type="futures")  # (10)!
data = vbt.CCXTData.pull("BTCUSDT", exchange="binanceusdm")  # (11)!
data = vbt.BinanceData.pull("BTCUSDT", tld="us")  # (12)!
data = vbt.TVData.pull("CRYPTOCAP:TOTAL")  # (13)!
```

```python
data = vbt.YFData.pull("AAPL")  # (1)!
data = vbt.YFData.pull(["AAPL", "MSFT"])  # (2)!
data = vbt.YFData.pull("AAPL", start="2020")  # (3)!
data = vbt.YFData.pull("AAPL", start="2020", end="2021")  # (4)!
data = vbt.YFData.pull("AAPL", start="1 month ago")  # (5)!
data = vbt.YFData.pull("AAPL", start="1 month ago", timeframe="hourly")  # (6)!
data = vbt.YFData.pull("AAPL", tz="UTC")  # (7)!
data = vbt.YFData.pull(symbols, execute_kwargs=dict(engine="threadpool"))  # (8)!

data = vbt.YFData.pull("AAPL", auto_adjust=False)  # (9)!
data = vbt.BinanceData.pull("BTCUSDT", klines_type="futures")  # (10)!
data = vbt.CCXTData.pull("BTCUSDT", exchange="binanceusdm")  # (11)!
data = vbt.BinanceData.pull("BTCUSDT", tld="us")  # (12)!
data = vbt.TVData.pull("CRYPTOCAP:TOTAL")  # (13)!
```

```python
datetime
```

```python
pd.Timestamp
```

```python
start
```

```python
end
```

```python
tld
```

To provide different keyword arguments for different symbols, either pass an argument as symbol_dict or pass a dictionary with keyword arguments keyed by symbol as the first argument.

```python
data = vbt.TVData.pull(
    ["SPX", "NDX", "VIX"],
    exchange=vbt.symbol_dict({"SPX": "SP", "NDX": "NASDAQ", "VIX": "CBOE"})
)
data = vbt.TVData.pull({  # (1)!
    "SPX": dict(exchange="SP"),
    "NDX": dict(exchange="NASDAQ"),
    "VIX": dict(exchange="CBOE")
})
data = vbt.TVData.pull(["SP:SPX", "NASDAQ:NDX", "CBOE:VIX"])  # (2)!
```

```python
data = vbt.TVData.pull(
    ["SPX", "NDX", "VIX"],
    exchange=vbt.symbol_dict({"SPX": "SP", "NDX": "NASDAQ", "VIX": "CBOE"})
)
data = vbt.TVData.pull({  # (1)!
    "SPX": dict(exchange="SP"),
    "NDX": dict(exchange="NASDAQ"),
    "VIX": dict(exchange="CBOE")
})
data = vbt.TVData.pull(["SP:SPX", "NASDAQ:NDX", "CBOE:VIX"])  # (2)!
```

If your data provider of choice takes credentials and you want to fetch multiple symbols, the client will be created for each symbol leading to multiple authentications and a slower execution. To avoid that, create the client in advance and then pass to the fetch() method.

```python
fetch()
```

```python
client = vbt.TVData.resolve_client(username="YOUR_USERNAME", password="YOUR_PASSWORD")
```

```python
client = vbt.TVData.resolve_client(username="YOUR_USERNAME", password="YOUR_PASSWORD")
```

```python
data = vbt.TVData.pull(["NASDAQ:AAPL", "NASDAQ:MSFT"], client=client)

# ______________________________________________________________

vbt.TVData.set_custom_settings(client=client)
data = vbt.TVData.pull(["NASDAQ:AAPL", "NASDAQ:MSFT"])
```

```python
data = vbt.TVData.pull(["NASDAQ:AAPL", "NASDAQ:MSFT"], client=client)

# ______________________________________________________________

vbt.TVData.set_custom_settings(client=client)
data = vbt.TVData.pull(["NASDAQ:AAPL", "NASDAQ:MSFT"])
```

## Persisting¶

Once fetched, the data can be saved in a variety of ways. The most common and recommended way is by pickling the data, which will save the entire object including the arguments used during fetching. Another ways include CSV files (Data.to_csv), HDF files (Data.to_hdf) and more, which will save only the data but not the accompanied metadata such as the timeframe.

```python
data.save()  # (1)!
data.save(compression="blosc")  # (2)!

data.to_csv("data", mkdir_kwargs=dict(mkdir=True))  # (3)!
data.to_csv("AAPL.csv")  # (4)!
data.to_csv("AAPL.tsv", sep="\t")  # (5)!
data.to_csv(vbt.symbol_dict(AAPL="AAPL.csv", MSFT="MSFT.csv"))  # (6)!
data.to_csv(vbt.RepEval("symbol + '.csv'"))  # (7)!

data.to_hdf("data")  # (8)!
data.to_hdf("data.h5")  # (9)!
data.to_hdf("data.h5", key=vbt.RepFunc(lambda symbol: symbol.replace(" ", "_")))  # (10)!
data.to_hdf("data.h5", key=vbt.RepFunc(lambda symbol: "stocks/" + symbol))  # (11)!
data.to_hdf(vbt.RepEval("symbol + '.h5'"), key="df")  # (12)!

data.to_parquet("data")  # (13)!
data.to_parquet(vbt.symbol_dict(
    AAPL="data/AAPL.parquet", 
    MSFT="data/MSFT.parquet"
))  # (14)!
data.to_parquet("data", partition_by="Y")  # (15)!
data.to_parquet(vbt.symbol_dict(
    AAPL="data/AAPL", 
    MSFT="data/MSFT"
), partition_by="Y")  # (16)!

data.to_sql(engine="sqlite:///data.db")  # (17)!
data.to_sql(engine="postgresql+psycopg2://postgres:admin@localhost:5432/data")  # (18)!
data.to_sql(engine=engine, schema="yahoo")  # (19)!
data.to_sql(engine=engine, table=vbt.symbol_dict(AAPL="AAPL", MSFT="MSFT"))  # (20)!
data.to_sql(engine=engine, if_exists="replace")  # (21)!
data.to_sql(engine=engine, attach_row_number=True)  # (22)!
data.to_sql(
    engine=engine, 
    attach_row_number=True, 
    row_number_column="RN",
    from_row_number=vbt.symbol_dict(AAPL=100, MSFT=200), 
    if_exists="append"
)  # (23)!
```

```python
data.save()  # (1)!
data.save(compression="blosc")  # (2)!

data.to_csv("data", mkdir_kwargs=dict(mkdir=True))  # (3)!
data.to_csv("AAPL.csv")  # (4)!
data.to_csv("AAPL.tsv", sep="\t")  # (5)!
data.to_csv(vbt.symbol_dict(AAPL="AAPL.csv", MSFT="MSFT.csv"))  # (6)!
data.to_csv(vbt.RepEval("symbol + '.csv'"))  # (7)!

data.to_hdf("data")  # (8)!
data.to_hdf("data.h5")  # (9)!
data.to_hdf("data.h5", key=vbt.RepFunc(lambda symbol: symbol.replace(" ", "_")))  # (10)!
data.to_hdf("data.h5", key=vbt.RepFunc(lambda symbol: "stocks/" + symbol))  # (11)!
data.to_hdf(vbt.RepEval("symbol + '.h5'"), key="df")  # (12)!

data.to_parquet("data")  # (13)!
data.to_parquet(vbt.symbol_dict(
    AAPL="data/AAPL.parquet", 
    MSFT="data/MSFT.parquet"
))  # (14)!
data.to_parquet("data", partition_by="Y")  # (15)!
data.to_parquet(vbt.symbol_dict(
    AAPL="data/AAPL", 
    MSFT="data/MSFT"
), partition_by="Y")  # (16)!

data.to_sql(engine="sqlite:///data.db")  # (17)!
data.to_sql(engine="postgresql+psycopg2://postgres:admin@localhost:5432/data")  # (18)!
data.to_sql(engine=engine, schema="yahoo")  # (19)!
data.to_sql(engine=engine, table=vbt.symbol_dict(AAPL="AAPL", MSFT="MSFT"))  # (20)!
data.to_sql(engine=engine, if_exists="replace")  # (21)!
data.to_sql(engine=engine, attach_row_number=True)  # (22)!
data.to_sql(
    engine=engine, 
    attach_row_number=True, 
    row_number_column="RN",
    from_row_number=vbt.symbol_dict(AAPL=100, MSFT=200), 
    if_exists="append"
)  # (23)!
```

```python
{class_name}.pickle
```

```python
{symbol}.csv
```

```python
{class_name}.h5
```

```python
{symbol}.parquet
```

Once saved, the data can be loaded with the corresponding class method.

```python
data = vbt.YFData.load()  # (1)!

data = vbt.Data.from_csv("data")  # (2)!
data = vbt.Data.from_csv("data/*.csv")  # (3)!
data = vbt.Data.from_csv("data/*/**.csv")  # (4)!
data = vbt.Data.from_csv(symbols=["BTC-USD.csv", "ETH-USD.csv"])  # (5)!
data = vbt.Data.from_csv(features=["High.csv", "Low.csv"])  # (6)!
data = vbt.Data.from_csv("BTC-USD", paths="polygon_btc_1hour.csv")  # (7)!
data = vbt.Data.from_csv("AAPL.tsv", sep="\t")  # (8)!
data = vbt.Data.from_csv(["MSFT.csv", "AAPL.tsv"], sep=vbt.symbol_dict(MSFT=",", AAPL="\t"))  # (9)!
data = vbt.Data.from_csv("https://datahub.io/core/s-and-p-500/r/data.csv", match_paths=False)  # (10)!

data = vbt.Data.from_hdf("data")  # (11)!
data = vbt.Data.from_hdf("data.h5")  # (12)!
data = vbt.Data.from_hdf("data.h5/AAPL")  # (13)!
data = vbt.Data.from_hdf(["data.h5/AAPL", "data.h5/MSFT"])  # (14)!
data = vbt.Data.from_hdf(["AAPL", "MSFT"], paths="data.h5", match_paths=False)
data = vbt.Data.from_hdf("data.h5/stocks/*")  # (15)!

data = vbt.Data.from_parquet("data")  # (16)!
data = vbt.Data.from_parquet("AAPL.parquet")  # (17)!
data = vbt.Data.from_parquet("AAPL")  # (18)!

data = vbt.Data.from_sql(engine="sqlite:///data.db")  # (19)!
data = vbt.Data.from_sql("AAPL", engine=engine)  # (20)!
data = vbt.Data.from_sql("yahoo:AAPL", engine=engine)  # (21)!
data = vbt.Data.from_sql("AAPL", schema="yahoo", engine=engine)  # (22)!
data = vbt.Data.from_sql("AAPL", query="SELECT * FROM AAPL", engine=engine)  # (23)!

data = vbt.BinanceData.from_csv("BTCUSDT.csv", fetch_kwargs=dict(timeframe="hourly"))  # (24)!
```

```python
data = vbt.YFData.load()  # (1)!

data = vbt.Data.from_csv("data")  # (2)!
data = vbt.Data.from_csv("data/*.csv")  # (3)!
data = vbt.Data.from_csv("data/*/**.csv")  # (4)!
data = vbt.Data.from_csv(symbols=["BTC-USD.csv", "ETH-USD.csv"])  # (5)!
data = vbt.Data.from_csv(features=["High.csv", "Low.csv"])  # (6)!
data = vbt.Data.from_csv("BTC-USD", paths="polygon_btc_1hour.csv")  # (7)!
data = vbt.Data.from_csv("AAPL.tsv", sep="\t")  # (8)!
data = vbt.Data.from_csv(["MSFT.csv", "AAPL.tsv"], sep=vbt.symbol_dict(MSFT=",", AAPL="\t"))  # (9)!
data = vbt.Data.from_csv("https://datahub.io/core/s-and-p-500/r/data.csv", match_paths=False)  # (10)!

data = vbt.Data.from_hdf("data")  # (11)!
data = vbt.Data.from_hdf("data.h5")  # (12)!
data = vbt.Data.from_hdf("data.h5/AAPL")  # (13)!
data = vbt.Data.from_hdf(["data.h5/AAPL", "data.h5/MSFT"])  # (14)!
data = vbt.Data.from_hdf(["AAPL", "MSFT"], paths="data.h5", match_paths=False)
data = vbt.Data.from_hdf("data.h5/stocks/*")  # (15)!

data = vbt.Data.from_parquet("data")  # (16)!
data = vbt.Data.from_parquet("AAPL.parquet")  # (17)!
data = vbt.Data.from_parquet("AAPL")  # (18)!

data = vbt.Data.from_sql(engine="sqlite:///data.db")  # (19)!
data = vbt.Data.from_sql("AAPL", engine=engine)  # (20)!
data = vbt.Data.from_sql("yahoo:AAPL", engine=engine)  # (21)!
data = vbt.Data.from_sql("AAPL", schema="yahoo", engine=engine)  # (22)!
data = vbt.Data.from_sql("AAPL", query="SELECT * FROM AAPL", engine=engine)  # (23)!

data = vbt.BinanceData.from_csv("BTCUSDT.csv", fetch_kwargs=dict(timeframe="hourly"))  # (24)!
```

```python
BinanceData
```

```python
fetch_kwargs
```

## Updating¶

Some data classes support fetching and appending new data to previously saved data by overriding the method Data.update_symbol, which scans the data for the latest timestamp and uses it as the start timestamp for fetching new data with Data.fetch_symbol. The method Data.update then does it for each symbol in the data instance. There's no need to provide the client, timeframe, or other arguments since they were captured during fetching and are reused automatically (unless they were lost by converting the data instance to Pandas, CSV, or HDF!).

```python
data = vbt.YFData.pull("AAPL", timeframe="1 minute")

# (1)!

data = data.update()  # (2)!
```

```python
data = vbt.YFData.pull("AAPL", timeframe="1 minute")

# (1)!

data = data.update()  # (2)!
```

```python
start = 2010
end = 2020
data = None
while start < end:
    if data is None:
        data = vbt.YFData.pull("AAPL", start=str(start), end=str(start + 1))
    else:
        data = data.update(end=str(start + 1))
    start += 1
```

```python
start = 2010
end = 2020
data = None
while start < end:
    if data is None:
        data = vbt.YFData.pull("AAPL", start=str(start), end=str(start + 1))
    else:
        data = data.update(end=str(start + 1))
    start += 1
```

## Wrapping¶

Custom DataFrame can be wrapped into a data instance by using Data.from_data, which takes either a single DataFrame for one symbol, or a dict with more DataFrames keyed by their symbols.

```python
data = ohlc_df.vbt.ohlcv.to_data()  # (1)!
data = vbt.Data.from_data(ohlc_df)

data = close_df.vbt.to_data()  # (2)!
data = vbt.Data.from_data(close_df, columns_are_symbols=True)

data = close_df.vbt.to_data(invert_data=True)  # (3)!
data = vbt.Data.from_data(close_df, columns_are_symbols=True, invert_data=True)

data = vbt.Data.from_data(vbt.symbol_dict({"AAPL": aapl_ohlc_df, "MSFT": msft_ohlc_df}))  # (4)!
data = vbt.Data.from_data(vbt.feature_dict({"High": high_df, "Low": low_df}))  # (5)!
```

```python
data = ohlc_df.vbt.ohlcv.to_data()  # (1)!
data = vbt.Data.from_data(ohlc_df)

data = close_df.vbt.to_data()  # (2)!
data = vbt.Data.from_data(close_df, columns_are_symbols=True)

data = close_df.vbt.to_data(invert_data=True)  # (3)!
data = vbt.Data.from_data(close_df, columns_are_symbols=True, invert_data=True)

data = vbt.Data.from_data(vbt.symbol_dict({"AAPL": aapl_ohlc_df, "MSFT": msft_ohlc_df}))  # (4)!
data = vbt.Data.from_data(vbt.feature_dict({"High": high_df, "Low": low_df}))  # (5)!
```

Tip

You aren't required to use data instances, you can proceed with Pandas and even NumPy arrays as well since VBT converts every array-like object to a NumPy array anyway. But beware that the Pandas format is more suitable than the NumPy format because the former also contains datetime index and backtest configuration metadata such as symbols and parameter combinations in form of columns. Where data instances are essential are symbol alignment, stacking, resampling, and updating.

## Extracting¶

Depending on the use case, there are multiple ways to extract the actual Pandas Series/DataFrame from an instance. To retrieve the original data with one DataFrame per symbol, query the data attribute. Such data contains OHLC and other features (of various data types too) concatenated together, which may be helpful in plotting. But note that VBT doesn't support this format: instead, you're encouraged to represent each feature as a separate DataFrame where columns are symbols. Such a feature can be queried as an attribute (data.close for close price, for example), or by using Data.get.

```python
data
```

```python
data.close
```

```python
data_per_symbol = data.data  # (1)!
aapl_data = data_per_symbol["AAPL"]  # (2)!

sr_or_df = data.get("Close")  # (3)!
sr_or_df = data["Close"].get()
sr_or_df = data.close

sr_or_df = data.get(["Close"])  # (4)!
sr_or_df = data[["Close"]].get()

sr = data.get("Close", "AAPL")  # (5)!
sr = data["Close"].get(symbols="AAPL")
sr = data.select("AAPL").close

df = data.get("Close", ["AAPL"])  # (6)!
df = data["Close"].get(symbols=["AAPL"])
df = data.select(["AAPL"]).close

aapl_df = data.get(["Open", "Close"], "AAPL")  # (7)!
close_df = data.get("Close", ["AAPL", "MSFT"])  # (8)!
open_df, close_df = data.get(["Open", "Close"], ["AAPL", "MSFT"])  # (9)!
```

```python
data_per_symbol = data.data  # (1)!
aapl_data = data_per_symbol["AAPL"]  # (2)!

sr_or_df = data.get("Close")  # (3)!
sr_or_df = data["Close"].get()
sr_or_df = data.close

sr_or_df = data.get(["Close"])  # (4)!
sr_or_df = data[["Close"]].get()

sr = data.get("Close", "AAPL")  # (5)!
sr = data["Close"].get(symbols="AAPL")
sr = data.select("AAPL").close

df = data.get("Close", ["AAPL"])  # (6)!
df = data["Close"].get(symbols=["AAPL"])
df = data.select(["AAPL"]).close

aapl_df = data.get(["Open", "Close"], "AAPL")  # (7)!
close_df = data.get("Close", ["AAPL", "MSFT"])  # (8)!
open_df, close_df = data.get(["Open", "Close"], ["AAPL", "MSFT"])  # (9)!
```

If a data instance is feature-oriented, the behavior of features and symbols is reversed.

```python
data_per_feature = feat_data.data  # (1)!
close_data = data_per_feature["Close"]

sr_or_df = data.get("Close") 
sr_or_df = data.select("Close").get()
sr_or_df = data.close

sr = feat_data.get("Close", "AAPL")
sr = feat_data["AAPL"].get(features="Close")  # (2)!
sr = feat_data.select("Close").get(symbols="AAPL")  # (3)!

aapl_df = data.get(["Open", "Close"], "AAPL")
close_df = data.get("Close", ["AAPL", "MSFT"])
aapl_df, msft_df = data.get(["Open", "Close"], ["AAPL", "MSFT"])  # (4)!
```

```python
data_per_feature = feat_data.data  # (1)!
close_data = data_per_feature["Close"]

sr_or_df = data.get("Close") 
sr_or_df = data.select("Close").get()
sr_or_df = data.close

sr = feat_data.get("Close", "AAPL")
sr = feat_data["AAPL"].get(features="Close")  # (2)!
sr = feat_data.select("Close").get(symbols="AAPL")  # (3)!

aapl_df = data.get(["Open", "Close"], "AAPL")
close_df = data.get("Close", ["AAPL", "MSFT"])
aapl_df, msft_df = data.get(["Open", "Close"], ["AAPL", "MSFT"])  # (4)!
```

```python
[]
```

```python
select
```

Tip

To get the same behavior between symbol-oriented and feature-oriented instances, always use Data.get to extract the data.

## Changing¶

There are four main operations to change features and symbols: adding, selecting, renaming, and removing. The first operation can be done on one feature or symbol at a time, while other operations can be done on a multiple of such. Usually, you won't need to specify whether you want to perform the operation on symbols or features as this will be determined automatically. Features and symbols are also case-insensitive. Also note that each operation doesn't change the original data instance but returns a new one.

```python
new_data = data.add_symbol("BTC-USD")  # (1)!
new_data = data.add_symbol("BTC-USD", fetch_kwargs=dict(start="2020"))  # (2)!
btc_df = vbt.YFData.pull("ETH-USD", start="2020").get()
new_data = data.add_symbol("BTC-USD", btc_df)  # (3)!

new_data = data.add_feature("SMA")  # (4)!
new_data = data.add_feature("SMA", run_kwargs=dict(timeperiod=20, hide_params=True))  # (5)!
sma_df = data.run("SMA", timeperiod=20, hide_params=True, unpack=True)
new_data = data.add_feature("SMA", sma_df)  # (6)!

new_data = data.add("BTC-USD", btc_df)  # (7)!
new_data = data.add("SMA", sma_df)  # (8)!
```

```python
new_data = data.add_symbol("BTC-USD")  # (1)!
new_data = data.add_symbol("BTC-USD", fetch_kwargs=dict(start="2020"))  # (2)!
btc_df = vbt.YFData.pull("ETH-USD", start="2020").get()
new_data = data.add_symbol("BTC-USD", btc_df)  # (3)!

new_data = data.add_feature("SMA")  # (4)!
new_data = data.add_feature("SMA", run_kwargs=dict(timeperiod=20, hide_params=True))  # (5)!
sma_df = data.run("SMA", timeperiod=20, hide_params=True, unpack=True)
new_data = data.add_feature("SMA", sma_df)  # (6)!

new_data = data.add("BTC-USD", btc_df)  # (7)!
new_data = data.add("SMA", sma_df)  # (8)!
```

Note

Only one feature or symbol can be added at a time. To add another data instance, use merge instead.

```python
merge
```

```python
new_data = data.select_symbols("BTC-USD")  # (1)!
new_data = data.select_symbols(["BTC-USD", "ETH-USD"])  # (2)!

new_data = data.select_features("SMA")  # (3)!
new_data = data.select_features(["SMA", "EMA"])  # (4)!

new_data = data.select("BTC-USD")  # (5)!
new_data = data.select("SMA")  # (6)!
new_data = data.select("sma")  # (7)!
```

```python
new_data = data.select_symbols("BTC-USD")  # (1)!
new_data = data.select_symbols(["BTC-USD", "ETH-USD"])  # (2)!

new_data = data.select_features("SMA")  # (3)!
new_data = data.select_features(["SMA", "EMA"])  # (4)!

new_data = data.select("BTC-USD")  # (5)!
new_data = data.select("SMA")  # (6)!
new_data = data.select("sma")  # (7)!
```

```python
new_data = data.rename_symbols("BTC-USD", "BTCUSDT")  # (1)!
new_data = data.rename_symbols(["BTC-USD", "ETH-USD"], ["BTCUSDT", "ETHUSDT"])  # (2)!
new_data = data.rename_symbols({"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"})

new_data = data.rename_features("Price", "Close")  # (3)!
new_data = data.rename_features(["Price", "MovAvg"], ["Close", "SMA"])  # (4)!
new_data = data.rename_features({"Price": "Close", "MovAvg": "SMA"})

new_data = data.rename("BTC-USD", "BTCUSDT")  # (5)!
new_data = data.rename("Price", "Close")  # (6)!
new_data = data.rename("price", "Close")  # (7)!
```

```python
new_data = data.rename_symbols("BTC-USD", "BTCUSDT")  # (1)!
new_data = data.rename_symbols(["BTC-USD", "ETH-USD"], ["BTCUSDT", "ETHUSDT"])  # (2)!
new_data = data.rename_symbols({"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"})

new_data = data.rename_features("Price", "Close")  # (3)!
new_data = data.rename_features(["Price", "MovAvg"], ["Close", "SMA"])  # (4)!
new_data = data.rename_features({"Price": "Close", "MovAvg": "SMA"})

new_data = data.rename("BTC-USD", "BTCUSDT")  # (5)!
new_data = data.rename("Price", "Close")  # (6)!
new_data = data.rename("price", "Close")  # (7)!
```

```python
new_data = data.remove_symbols("BTC-USD")  # (1)!
new_data = data.remove_symbols(["BTC-USD", "ETH-USD"])  # (2)!

new_data = data.remove_features("SMA")  # (3)!
new_data = data.remove_features(["SMA", "EMA"])  # (4)!

new_data = data.remove("BTC-USD")  # (5)!
new_data = data.remove("SMA")  # (6)!
new_data = data.remove("sma")  # (7)!
```

```python
new_data = data.remove_symbols("BTC-USD")  # (1)!
new_data = data.remove_symbols(["BTC-USD", "ETH-USD"])  # (2)!

new_data = data.remove_features("SMA")  # (3)!
new_data = data.remove_features(["SMA", "EMA"])  # (4)!

new_data = data.remove("BTC-USD")  # (5)!
new_data = data.remove("SMA")  # (6)!
new_data = data.remove("sma")  # (7)!
```

Instances can be merged together along symbols, rows, and columns by using Data.merge.

```python
data1 = vbt.YFData.pull("BTC-USD")
data2 = vbt.BinanceData.pull("BTCUSDT")
data3 = vbt.CCXTData.pull("BTC-USDT", exchange="kucoin")
data = vbt.Data.merge(data1, data2, data3, missing_columns="drop")
```

```python
data1 = vbt.YFData.pull("BTC-USD")
data2 = vbt.BinanceData.pull("BTCUSDT")
data3 = vbt.CCXTData.pull("BTC-USDT", exchange="kucoin")
data = vbt.Data.merge(data1, data2, data3, missing_columns="drop")
```

To apply a function to each DataFrame and return a new instance, the method Data.transform can be used. By default, it passes one single DataFrame where all individual DataFrames are concatenated along columns. This is useful for dropping missing values across all symbols. To transform the DataFrames individually, use per_symbol=True and/or per_feature=True. The only requirement is that the returned column names are identical across all features and symbols.

```python
per_symbol=True
```

```python
per_feature=True
```

```python
new_data = data.transform(lambda df: df.dropna(how="any"))  # (1)!
new_data = data.dropna()  # (2)!
new_data = data.dropna(how="all")  # (3)!

new_data = data.transform(your_func, per_feature=True)
new_data = data.transform(your_func, per_symbol=True)
new_data = data.transform(your_func, per_feature=True, per_symbol=True)  # (4)!
new_data = data.transform(your_func, per_feature=True, per_symbol=True, pass_frame=True)  # (5)!
```

```python
new_data = data.transform(lambda df: df.dropna(how="any"))  # (1)!
new_data = data.dropna()  # (2)!
new_data = data.dropna(how="all")  # (3)!

new_data = data.transform(your_func, per_feature=True)
new_data = data.transform(your_func, per_symbol=True)
new_data = data.transform(your_func, per_feature=True, per_symbol=True)  # (4)!
new_data = data.transform(your_func, per_feature=True, per_symbol=True, pass_frame=True)  # (5)!
```

If symbols have different timezones, the final timezone will become "UTC". This will make some symbols shifted in time; for example, one symbol with UTC+0200 and another with UTC+0400 will effectively double the common index and produce missing values half of the time. To align their indexes into a single one, use Data.realign, which is a special form of resampling that produces a single index where data is correctly ordered by time.

```python
new_data = data.realign()  # (1)!
```

```python
new_data = data.realign()  # (1)!
```

```python
ffill=False
```

Operations that return a new data instance can be easily chained using the dot notation or the method pipe.

```python
pipe
```

```python
data = (
    vbt.YFData.pull("BTC-USD")
    .add_symbol("ETH-USD")
    .rename({"btc-usd": "BTCUSDT", "eth-usd": "ETHUSDT"})
    .remove(["dividends", "stock splits"])
    .add_feature("SMA")
    .add_feature("EMA")
)

# ______________________________________________________________

data = (
    vbt.YFData
    .pipe("pull", "BTC-USD")  # (1)!
    .pipe("add_symbol", "ETH-USD")
    .pipe("rename", {"btc-usd": "BTCUSDT", "eth-usd": "ETHUSDT"})
    .pipe("remove", ["dividends", "stock splits"])
    .pipe("add_feature", "SMA")
    .pipe("add_feature", "EMA")
)
```

```python
data = (
    vbt.YFData.pull("BTC-USD")
    .add_symbol("ETH-USD")
    .rename({"btc-usd": "BTCUSDT", "eth-usd": "ETHUSDT"})
    .remove(["dividends", "stock splits"])
    .add_feature("SMA")
    .add_feature("EMA")
)

# ______________________________________________________________

data = (
    vbt.YFData
    .pipe("pull", "BTC-USD")  # (1)!
    .pipe("add_symbol", "ETH-USD")
    .pipe("rename", {"btc-usd": "BTCUSDT", "eth-usd": "ETHUSDT"})
    .pipe("remove", ["dividends", "stock splits"])
    .pipe("add_feature", "SMA")
    .pipe("add_feature", "EMA")
)
```

