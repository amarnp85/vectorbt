# Data¶

## FinDataPy¶

```python
>>> vbt.FinPyData.list_symbols(data_source="dukascopy")
['fx.dukascopy.tick.NYC.AUDCAD',
 'fx.dukascopy.tick.NYC.AUDCHF',
 'fx.dukascopy.tick.NYC.AUDJPY',
 ...
 'fx.dukascopy.tick.NYC.USDTRY',
 'fx.dukascopy.tick.NYC.USDZAR',
 'fx.dukascopy.tick.NYC.ZARJPY']

>>> data = vbt.FinPyData.pull(  # (1)!
...     "fx.dukascopy.tick.NYC.EURUSD",
...     start="14 Jun 2016", 
...     end="15 Jun 2016"
... )
>>> data.get()
                                     close
Date                                      
2016-06-14 00:00:00.844000+00:00  1.128795
2016-06-14 00:00:01.591000+00:00  1.128790
2016-06-14 00:00:01.743000+00:00  1.128775
2016-06-14 00:00:02.464000+00:00  1.128770
2016-06-14 00:00:02.971000+00:00  1.128760
...                                    ...
2016-06-14 23:59:57.733000+00:00  1.121020
2016-06-14 23:59:58.239000+00:00  1.121030
2016-06-14 23:59:58.953000+00:00  1.121035
2016-06-14 23:59:59.004000+00:00  1.121050
2016-06-14 23:59:59.934000+00:00  1.121055

[82484 rows x 1 columns]

>>> data = vbt.FinPyData.pull(  # (2)!
...     "EURUSD",
...     start="14 Jun 2016", 
...     end="15 Jun 2016", 
...     timeframe="tick", 
...     category="fx", 
...     data_source="dukascopy",
...     fields=["bid", "ask", "bidv", "askv"]
... )
>>> data.get()
                                      bid      ask  bidv   askv
Date                                                           
2016-06-14 00:00:00.844000+00:00  1.12877  1.12882  1.00  10.12
2016-06-14 00:00:01.591000+00:00  1.12877  1.12881  1.00   1.00
2016-06-14 00:00:01.743000+00:00  1.12875  1.12880  3.11   3.00
2016-06-14 00:00:02.464000+00:00  1.12875  1.12879  2.21   1.00
2016-06-14 00:00:02.971000+00:00  1.12875  1.12877  2.21   1.00
...                                   ...      ...   ...    ...
2016-06-14 23:59:57.733000+00:00  1.12100  1.12104  1.24   1.50
2016-06-14 23:59:58.239000+00:00  1.12101  1.12105  9.82   1.12
2016-06-14 23:59:58.953000+00:00  1.12102  1.12105  1.50   1.12
2016-06-14 23:59:59.004000+00:00  1.12103  1.12107  1.50   1.12
2016-06-14 23:59:59.934000+00:00  1.12103  1.12108  1.87   2.25

[82484 rows x 4 columns]
```

```python
>>> vbt.FinPyData.list_symbols(data_source="dukascopy")
['fx.dukascopy.tick.NYC.AUDCAD',
 'fx.dukascopy.tick.NYC.AUDCHF',
 'fx.dukascopy.tick.NYC.AUDJPY',
 ...
 'fx.dukascopy.tick.NYC.USDTRY',
 'fx.dukascopy.tick.NYC.USDZAR',
 'fx.dukascopy.tick.NYC.ZARJPY']

>>> data = vbt.FinPyData.pull(  # (1)!
...     "fx.dukascopy.tick.NYC.EURUSD",
...     start="14 Jun 2016", 
...     end="15 Jun 2016"
... )
>>> data.get()
                                     close
Date                                      
2016-06-14 00:00:00.844000+00:00  1.128795
2016-06-14 00:00:01.591000+00:00  1.128790
2016-06-14 00:00:01.743000+00:00  1.128775
2016-06-14 00:00:02.464000+00:00  1.128770
2016-06-14 00:00:02.971000+00:00  1.128760
...                                    ...
2016-06-14 23:59:57.733000+00:00  1.121020
2016-06-14 23:59:58.239000+00:00  1.121030
2016-06-14 23:59:58.953000+00:00  1.121035
2016-06-14 23:59:59.004000+00:00  1.121050
2016-06-14 23:59:59.934000+00:00  1.121055

[82484 rows x 1 columns]

>>> data = vbt.FinPyData.pull(  # (2)!
...     "EURUSD",
...     start="14 Jun 2016", 
...     end="15 Jun 2016", 
...     timeframe="tick", 
...     category="fx", 
...     data_source="dukascopy",
...     fields=["bid", "ask", "bidv", "askv"]
... )
>>> data.get()
                                      bid      ask  bidv   askv
Date                                                           
2016-06-14 00:00:00.844000+00:00  1.12877  1.12882  1.00  10.12
2016-06-14 00:00:01.591000+00:00  1.12877  1.12881  1.00   1.00
2016-06-14 00:00:01.743000+00:00  1.12875  1.12880  3.11   3.00
2016-06-14 00:00:02.464000+00:00  1.12875  1.12879  2.21   1.00
2016-06-14 00:00:02.971000+00:00  1.12875  1.12877  2.21   1.00
...                                   ...      ...   ...    ...
2016-06-14 23:59:57.733000+00:00  1.12100  1.12104  1.24   1.50
2016-06-14 23:59:58.239000+00:00  1.12101  1.12105  9.82   1.12
2016-06-14 23:59:58.953000+00:00  1.12102  1.12105  1.50   1.12
2016-06-14 23:59:59.004000+00:00  1.12103  1.12107  1.50   1.12
2016-06-14 23:59:59.934000+00:00  1.12103  1.12108  1.87   2.25

[82484 rows x 4 columns]
```

## Databento¶

```python
>>> vbt.BentoData.set_custom_settings(
...     client_config=dict(
...         key="YOUR_KEY"
...     )
... )
>>> params = dict(
...     symbols="ESH3",
...     dataset="GLBX.MDP3",
...     start=vbt.timestamp("2022-10-28 20:30:00"),
...     end=vbt.timestamp("2022-10-28 21:00:00"),
...     schema="tbbo"
... )
>>> vbt.BentoData.get_cost(**params)
1.2002885341644287e-05

>>> data = vbt.BentoData.pull(**params)
>>> data.get()
                                                               ts_event  \
ts_recv                                                                   
2022-10-28 20:30:59.047138053+00:00 2022-10-28 20:30:59.046914657+00:00   
2022-10-28 20:37:53.112494436+00:00 2022-10-28 20:37:53.112246421+00:00   
...
2022-10-28 20:59:15.075191111+00:00 2022-10-28 20:59:15.074953895+00:00   
2022-10-28 20:59:34.607239899+00:00 2022-10-28 20:59:34.606984277+00:00   

                                     rtype  publisher_id  instrument_id  \
ts_recv                                                                   
2022-10-28 20:30:59.047138053+00:00      1             1         206299   
2022-10-28 20:37:53.112494436+00:00      1             1         206299   
...
2022-10-28 20:59:15.075191111+00:00      1             1         206299   
2022-10-28 20:59:34.607239899+00:00      1             1         206299   

                                    action side  depth    price  size  flags  \
ts_recv                                                                        
2022-10-28 20:30:59.047138053+00:00      T    B      0  3955.25     1      0   
2022-10-28 20:37:53.112494436+00:00      T    A      0  3955.00     1      0   
...
2022-10-28 20:59:15.075191111+00:00      T    A      0  3953.75     1      0   
2022-10-28 20:59:34.607239899+00:00      T    A      0  3954.50     2      0   

                                     ts_in_delta  sequence  bid_px_00  \
ts_recv                                                                 
2022-10-28 20:30:59.047138053+00:00        18553  73918214    3954.75   
2022-10-28 20:37:53.112494436+00:00        18334  73926240    3955.00   
...
2022-10-28 20:59:15.075191111+00:00        19294  73945515    3953.75   
2022-10-28 20:59:34.607239899+00:00        18701  73945932    3954.50   

                                     ask_px_00  bid_sz_00  ask_sz_00  \
ts_recv                                                                
2022-10-28 20:30:59.047138053+00:00    3955.25          1          1   
2022-10-28 20:37:53.112494436+00:00    3955.75          1          1   
...
2022-10-28 20:59:15.075191111+00:00    3956.00          1          1   
2022-10-28 20:59:34.607239899+00:00    3956.00          4          1   

                                     bid_ct_00  ask_ct_00 symbol  
ts_recv                                                           
2022-10-28 20:30:59.047138053+00:00          1          1   ESH3  
2022-10-28 20:37:53.112494436+00:00          1          1   ESH3  
...
2022-10-28 20:59:15.075191111+00:00          1          1   ESH3  
2022-10-28 20:59:34.607239899+00:00          3          1   ESH3
```

```python
>>> vbt.BentoData.set_custom_settings(
...     client_config=dict(
...         key="YOUR_KEY"
...     )
... )
>>> params = dict(
...     symbols="ESH3",
...     dataset="GLBX.MDP3",
...     start=vbt.timestamp("2022-10-28 20:30:00"),
...     end=vbt.timestamp("2022-10-28 21:00:00"),
...     schema="tbbo"
... )
>>> vbt.BentoData.get_cost(**params)
1.2002885341644287e-05

>>> data = vbt.BentoData.pull(**params)
>>> data.get()
                                                               ts_event  \
ts_recv                                                                   
2022-10-28 20:30:59.047138053+00:00 2022-10-28 20:30:59.046914657+00:00   
2022-10-28 20:37:53.112494436+00:00 2022-10-28 20:37:53.112246421+00:00   
...
2022-10-28 20:59:15.075191111+00:00 2022-10-28 20:59:15.074953895+00:00   
2022-10-28 20:59:34.607239899+00:00 2022-10-28 20:59:34.606984277+00:00   

                                     rtype  publisher_id  instrument_id  \
ts_recv                                                                   
2022-10-28 20:30:59.047138053+00:00      1             1         206299   
2022-10-28 20:37:53.112494436+00:00      1             1         206299   
...
2022-10-28 20:59:15.075191111+00:00      1             1         206299   
2022-10-28 20:59:34.607239899+00:00      1             1         206299   

                                    action side  depth    price  size  flags  \
ts_recv                                                                        
2022-10-28 20:30:59.047138053+00:00      T    B      0  3955.25     1      0   
2022-10-28 20:37:53.112494436+00:00      T    A      0  3955.00     1      0   
...
2022-10-28 20:59:15.075191111+00:00      T    A      0  3953.75     1      0   
2022-10-28 20:59:34.607239899+00:00      T    A      0  3954.50     2      0   

                                     ts_in_delta  sequence  bid_px_00  \
ts_recv                                                                 
2022-10-28 20:30:59.047138053+00:00        18553  73918214    3954.75   
2022-10-28 20:37:53.112494436+00:00        18334  73926240    3955.00   
...
2022-10-28 20:59:15.075191111+00:00        19294  73945515    3953.75   
2022-10-28 20:59:34.607239899+00:00        18701  73945932    3954.50   

                                     ask_px_00  bid_sz_00  ask_sz_00  \
ts_recv                                                                
2022-10-28 20:30:59.047138053+00:00    3955.25          1          1   
2022-10-28 20:37:53.112494436+00:00    3955.75          1          1   
...
2022-10-28 20:59:15.075191111+00:00    3956.00          1          1   
2022-10-28 20:59:34.607239899+00:00    3956.00          4          1   

                                     bid_ct_00  ask_ct_00 symbol  
ts_recv                                                           
2022-10-28 20:30:59.047138053+00:00          1          1   ESH3  
2022-10-28 20:37:53.112494436+00:00          1          1   ESH3  
...
2022-10-28 20:59:15.075191111+00:00          1          1   ESH3  
2022-10-28 20:59:34.607239899+00:00          3          1   ESH3
```

## SQL queries¶

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> data.sql("""
...     SELECT datetime, AVG(Close) OVER(
...         ORDER BY "datetime" ASC
...         RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND CURRENT ROW
...     ) AS "Moving Average"
...     FROM "AAPL";
... """)
datetime
2023-09-11 09:30:00-04:00    180.080000
2023-09-11 09:31:00-04:00    179.965000
2023-09-11 09:32:00-04:00    180.000000
2023-09-11 09:33:00-04:00    180.022500
2023-09-11 09:34:00-04:00    179.984000
                                    ...
2023-10-20 15:55:00-04:00    177.786669
2023-10-20 15:56:00-04:00    177.785492
2023-10-20 15:57:00-04:00    177.784322
2023-10-20 15:58:00-04:00    177.783166
2023-10-20 15:59:00-04:00    177.781986
Name: Moving Average, Length: 11700, dtype: float64
```

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> data.sql("""
...     SELECT datetime, AVG(Close) OVER(
...         ORDER BY "datetime" ASC
...         RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND CURRENT ROW
...     ) AS "Moving Average"
...     FROM "AAPL";
... """)
datetime
2023-09-11 09:30:00-04:00    180.080000
2023-09-11 09:31:00-04:00    179.965000
2023-09-11 09:32:00-04:00    180.000000
2023-09-11 09:33:00-04:00    180.022500
2023-09-11 09:34:00-04:00    179.984000
                                    ...
2023-10-20 15:55:00-04:00    177.786669
2023-10-20 15:56:00-04:00    177.785492
2023-10-20 15:57:00-04:00    177.784322
2023-10-20 15:58:00-04:00    177.783166
2023-10-20 15:59:00-04:00    177.781986
Name: Moving Average, Length: 11700, dtype: float64
```

## DuckDB¶

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> URL = "database.duckdb"
>>> data.to_duckdb(connection=URL)

>>> day_data = vbt.DuckDBData.pull(
...     "AAPL", 
...     start="2023-10-02 09:30:00", 
...     end="2023-10-02 16:00:00", 
...     tz="America/New_York",
...     connection=URL
... )
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> URL = "database.duckdb"
>>> data.to_duckdb(connection=URL)

>>> day_data = vbt.DuckDBData.pull(
...     "AAPL", 
...     start="2023-10-02 09:30:00", 
...     end="2023-10-02 16:00:00", 
...     tz="America/New_York",
...     connection=URL
... )
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

## SQLAlchemy¶

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> URL = "postgresql://postgres:postgres@localhost:5432"
>>> data.to_sql(engine=URL)

>>> day_data = vbt.SQLData.pull(
...     "AAPL", 
...     start="2023-10-02 09:30:00", 
...     end="2023-10-02 16:00:00", 
...     tz="America/New_York",
...     engine=URL
... )
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> URL = "postgresql://postgres:postgres@localhost:5432"
>>> data.to_sql(engine=URL)

>>> day_data = vbt.SQLData.pull(
...     "AAPL", 
...     start="2023-10-02 09:30:00", 
...     end="2023-10-02 16:00:00", 
...     tz="America/New_York",
...     engine=URL
... )
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

## PyArrow & FastParquet¶

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> data.to_parquet(partition_by="day")  # (1)!

>>> day_data = vbt.ParquetData.pull("AAPL", filters=[("group", "==", "2023-10-02")])
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

```python
>>> data = vbt.TVData.pull(
...     "AAPL", 
...     exchange="NASDAQ", 
...     timeframe="1 minute",
...     tz="America/New_York"
... )

>>> data.to_parquet(partition_by="day")  # (1)!

>>> day_data = vbt.ParquetData.pull("AAPL", filters=[("group", "==", "2023-10-02")])
>>> day_data.get()
                              Open    High     Low   Close    Volume
datetime                                                            
2023-10-02 09:30:00-04:00  171.260  171.34  170.93  171.10   61654.0
2023-10-02 09:31:00-04:00  171.130  172.37  171.13  172.30   53481.0
2023-10-02 09:32:00-04:00  172.310  172.64  172.16  172.64   44750.0
2023-10-02 09:33:00-04:00  172.640  172.97  172.54  172.78   53195.0
2023-10-02 09:34:00-04:00  172.780  173.07  172.75  173.00   47416.0
...                            ...     ...     ...     ...       ...
2023-10-02 15:55:00-04:00  173.300  173.51  173.26  173.51   61619.0
2023-10-02 15:56:00-04:00  173.525  173.59  173.42  173.43   45066.0
2023-10-02 15:57:00-04:00  173.430  173.55  173.42  173.50   45220.0
2023-10-02 15:58:00-04:00  173.510  173.60  173.46  173.56   47371.0
2023-10-02 15:59:00-04:00  173.560  173.78  173.56  173.75  161253.0

[390 rows x 5 columns]
```

```python
partition_by
```

## Feature-oriented data¶

```python
data.close
```

```python
>>> data = vbt.YFData.pull(["AAPL", "MSFT", "GOOG"])
>>> pf = data.run("from_random_signals", n=[10, 20, 30])

>>> pf_data = vbt.Data.from_data(
...     vbt.feature_dict({
...         "cash": pf.cash,
...         "assets": pf.assets,
...         "asset_value": pf.asset_value,
...         "value": pf.value
...     })
... )
>>> pf_data.get(feature="cash", symbol=(10, "AAPL"))
Date
1980-12-12 05:00:00+00:00      100.000000
1980-12-15 05:00:00+00:00      100.000000
1980-12-16 05:00:00+00:00      100.000000
1980-12-17 05:00:00+00:00      100.000000
1980-12-18 05:00:00+00:00      100.000000
                                      ...
2023-08-25 04:00:00+00:00    81193.079771
2023-08-28 04:00:00+00:00    81193.079771
2023-08-29 04:00:00+00:00    81193.079771
2023-08-30 04:00:00+00:00    81193.079771
2023-08-31 04:00:00+00:00    81193.079771
Name: (10, AAPL), Length: 10770, dtype: float64
```

```python
>>> data = vbt.YFData.pull(["AAPL", "MSFT", "GOOG"])
>>> pf = data.run("from_random_signals", n=[10, 20, 30])

>>> pf_data = vbt.Data.from_data(
...     vbt.feature_dict({
...         "cash": pf.cash,
...         "assets": pf.assets,
...         "asset_value": pf.asset_value,
...         "value": pf.value
...     })
... )
>>> pf_data.get(feature="cash", symbol=(10, "AAPL"))
Date
1980-12-12 05:00:00+00:00      100.000000
1980-12-15 05:00:00+00:00      100.000000
1980-12-16 05:00:00+00:00      100.000000
1980-12-17 05:00:00+00:00      100.000000
1980-12-18 05:00:00+00:00      100.000000
                                      ...
2023-08-25 04:00:00+00:00    81193.079771
2023-08-28 04:00:00+00:00    81193.079771
2023-08-29 04:00:00+00:00    81193.079771
2023-08-30 04:00:00+00:00    81193.079771
2023-08-31 04:00:00+00:00    81193.079771
Name: (10, AAPL), Length: 10770, dtype: float64
```

## Parallel data¶

```python
>>> symbols = ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"]

>>> with vbt.Timer() as timer:
...     data = vbt.YFData.pull(symbols)
>>> print(timer.elapsed())
4.52 seconds

>>> with vbt.Timer() as timer:
...     data = vbt.YFData.pull(symbols, execute_kwargs=dict(engine="threadpool"))
>>> print(timer.elapsed())
918.54 milliseconds
```

```python
>>> symbols = ["SPY", "TLT", "XLF", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV"]

>>> with vbt.Timer() as timer:
...     data = vbt.YFData.pull(symbols)
>>> print(timer.elapsed())
4.52 seconds

>>> with vbt.Timer() as timer:
...     data = vbt.YFData.pull(symbols, execute_kwargs=dict(engine="threadpool"))
>>> print(timer.elapsed())
918.54 milliseconds
```

## Trading View¶

```python
>>> data = vbt.TVData.pull(
...     "NASDAQ:AAPL",
...     timeframe="1 minute",
...     tz="US/Eastern"
... )
>>> data.get()
                             Open    High     Low   Close   Volume
datetime                                                          
2022-12-05 09:30:00-05:00  147.75  148.31  147.50  148.28  37769.0
2022-12-05 09:31:00-05:00  148.28  148.67  148.28  148.49  10525.0
2022-12-05 09:32:00-05:00  148.50  148.73  148.30  148.30   4860.0
2022-12-05 09:33:00-05:00  148.25  148.73  148.25  148.64   5306.0
2022-12-05 09:34:00-05:00  148.62  148.97  148.52  148.97   5808.0
...                           ...     ...     ...     ...      ...
2023-01-17 15:55:00-05:00  135.80  135.91  135.80  135.86  37573.0
2023-01-17 15:56:00-05:00  135.85  135.88  135.80  135.88  18796.0
2023-01-17 15:57:00-05:00  135.88  135.93  135.85  135.91  21019.0
2023-01-17 15:58:00-05:00  135.90  135.97  135.89  135.95  20934.0
2023-01-17 15:59:00-05:00  135.94  136.00  135.84  135.94  86696.0

[11310 rows x 5 columns]
```

```python
>>> data = vbt.TVData.pull(
...     "NASDAQ:AAPL",
...     timeframe="1 minute",
...     tz="US/Eastern"
... )
>>> data.get()
                             Open    High     Low   Close   Volume
datetime                                                          
2022-12-05 09:30:00-05:00  147.75  148.31  147.50  148.28  37769.0
2022-12-05 09:31:00-05:00  148.28  148.67  148.28  148.49  10525.0
2022-12-05 09:32:00-05:00  148.50  148.73  148.30  148.30   4860.0
2022-12-05 09:33:00-05:00  148.25  148.73  148.25  148.64   5306.0
2022-12-05 09:34:00-05:00  148.62  148.97  148.52  148.97   5808.0
...                           ...     ...     ...     ...      ...
2023-01-17 15:55:00-05:00  135.80  135.91  135.80  135.86  37573.0
2023-01-17 15:56:00-05:00  135.85  135.88  135.80  135.88  18796.0
2023-01-17 15:57:00-05:00  135.88  135.93  135.85  135.91  21019.0
2023-01-17 15:58:00-05:00  135.90  135.97  135.89  135.95  20934.0
2023-01-17 15:59:00-05:00  135.94  136.00  135.84  135.94  86696.0

[11310 rows x 5 columns]
```

## Symbol search¶

```python
>>> vbt.BinanceData.list_symbols("XRP*")
{'XRPAUD',
 'XRPBEARBUSD',
 'XRPBEARUSDT',
 'XRPBIDR',
 'XRPBKRW',
 'XRPBNB',
 'XRPBRL',
 'XRPBTC',
 'XRPBULLBUSD',
 'XRPBULLUSDT',
 'XRPBUSD',
 'XRPDOWNUSDT',
 'XRPETH',
 'XRPEUR',
 'XRPGBP',
 'XRPNGN',
 'XRPPAX',
 'XRPRUB',
 'XRPTRY',
 'XRPTUSD',
 'XRPUPUSDT',
 'XRPUSDC',
 'XRPUSDT'}
```

```python
>>> vbt.BinanceData.list_symbols("XRP*")
{'XRPAUD',
 'XRPBEARBUSD',
 'XRPBEARUSDT',
 'XRPBIDR',
 'XRPBKRW',
 'XRPBNB',
 'XRPBRL',
 'XRPBTC',
 'XRPBULLBUSD',
 'XRPBULLUSDT',
 'XRPBUSD',
 'XRPDOWNUSDT',
 'XRPETH',
 'XRPEUR',
 'XRPGBP',
 'XRPNGN',
 'XRPPAX',
 'XRPRUB',
 'XRPTRY',
 'XRPTUSD',
 'XRPUPUSDT',
 'XRPUSDC',
 'XRPUSDT'}
```

## Symbol classes¶

```python
>>> classes = vbt.symbol_dict({
...     "MSFT": dict(sector="Technology"),
...     "GOOGL": dict(sector="Technology"),
...     "META": dict(sector="Technology"),
...     "JPM": dict(sector="Finance"),
...     "BAC": dict(sector="Finance"),
...     "WFC": dict(sector="Finance"),
...     "AMZN": dict(sector="Retail"),
...     "WMT": dict(sector="Retail"),
...     "BABA": dict(sector="Retail"),
... })
>>> data = vbt.YFData.pull(
...     list(classes.keys()), 
...     classes=classes,
...     missing_index="drop"
... )
>>> pf = vbt.PF.from_orders(
...     data, 
...     size=vbt.index_dict({0: 1 / 3}),  # (1)!
...     size_type="targetpercent",
...     group_by="sector", 
...     cash_sharing=True
... )
>>> pf.value.vbt.plot().show()
```

```python
>>> classes = vbt.symbol_dict({
...     "MSFT": dict(sector="Technology"),
...     "GOOGL": dict(sector="Technology"),
...     "META": dict(sector="Technology"),
...     "JPM": dict(sector="Finance"),
...     "BAC": dict(sector="Finance"),
...     "WFC": dict(sector="Finance"),
...     "AMZN": dict(sector="Retail"),
...     "WMT": dict(sector="Retail"),
...     "BABA": dict(sector="Retail"),
... })
>>> data = vbt.YFData.pull(
...     list(classes.keys()), 
...     classes=classes,
...     missing_index="drop"
... )
>>> pf = vbt.PF.from_orders(
...     data, 
...     size=vbt.index_dict({0: 1 / 3}),  # (1)!
...     size_type="targetpercent",
...     group_by="sector", 
...     cash_sharing=True
... )
>>> pf.value.vbt.plot().show()
```

## Runnable data¶

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> stochrsi = data.run("stochrsi")
>>> stochrsi.fastd
Date
2014-09-17 00:00:00+00:00          NaN
2014-09-18 00:00:00+00:00          NaN
2014-09-19 00:00:00+00:00          NaN
2014-09-20 00:00:00+00:00          NaN
2014-09-21 00:00:00+00:00          NaN
                                   ...
2023-01-15 00:00:00+00:00    96.168788
2023-01-16 00:00:00+00:00    91.733393
2023-01-17 00:00:00+00:00    78.295255
2023-01-18 00:00:00+00:00    48.793133
2023-01-20 00:00:00+00:00    26.242474
Name: Close, Length: 3047, dtype: float64
```

```python
>>> data = vbt.YFData.pull("BTC-USD")
>>> stochrsi = data.run("stochrsi")
>>> stochrsi.fastd
Date
2014-09-17 00:00:00+00:00          NaN
2014-09-18 00:00:00+00:00          NaN
2014-09-19 00:00:00+00:00          NaN
2014-09-20 00:00:00+00:00          NaN
2014-09-21 00:00:00+00:00          NaN
                                   ...
2023-01-15 00:00:00+00:00    96.168788
2023-01-16 00:00:00+00:00    91.733393
2023-01-17 00:00:00+00:00    78.295255
2023-01-18 00:00:00+00:00    48.793133
2023-01-20 00:00:00+00:00    26.242474
Name: Close, Length: 3047, dtype: float64
```

## Data transformation¶

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], start="2020-01-01", end="2020-01-14")
>>> new_data = data.transform(lambda df: df[~df.index.weekday.isin([5, 6])])
>>> new_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-06 00:00:00+00:00  7769.219238  144.304153
2020-01-07 00:00:00+00:00  8163.692383  143.543991
2020-01-08 00:00:00+00:00  8079.862793  141.258133
2020-01-09 00:00:00+00:00  7879.071289  138.979202
2020-01-10 00:00:00+00:00  8166.554199  143.963776
2020-01-13 00:00:00+00:00  8144.194336  144.226593
```

```python
>>> data = vbt.YFData.pull(["BTC-USD", "ETH-USD"], start="2020-01-01", end="2020-01-14")
>>> new_data = data.transform(lambda df: df[~df.index.weekday.isin([5, 6])])
>>> new_data.close
symbol                         BTC-USD     ETH-USD
Date                                              
2020-01-01 00:00:00+00:00  7200.174316  130.802002
2020-01-02 00:00:00+00:00  6985.470215  127.410179
2020-01-03 00:00:00+00:00  7344.884277  134.171707
2020-01-06 00:00:00+00:00  7769.219238  144.304153
2020-01-07 00:00:00+00:00  8163.692383  143.543991
2020-01-08 00:00:00+00:00  8079.862793  141.258133
2020-01-09 00:00:00+00:00  7879.071289  138.979202
2020-01-10 00:00:00+00:00  8166.554199  143.963776
2020-01-13 00:00:00+00:00  8144.194336  144.226593
```

## Synthetic OHLC¶

```python
>>> data = vbt.GBMOHLCData.pull("R", start="2022-01", end="2022-04")
>>> data.plot().show()
```

```python
>>> data = vbt.GBMOHLCData.pull("R", start="2022-01", end="2022-04")
>>> data.plot().show()
```

## Data saver¶

```python
from vectorbtpro import *

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if vbt.CSVDataSaver.file_exists():
        csv_saver = vbt.CSVDataSaver.load()
        csv_saver.update()
        init_save = False
    else:
        data = vbt.BinanceData.pull(
            "BTCUSDT", 
            start="10 minutes ago UTC",
            timeframe="1 minute"
        )
        csv_saver = vbt.CSVDataSaver(data)
        init_save = True
    csv_saver.update_every(1, "minute", init_save=init_save)
    csv_saver.save()  # (1)!
```

```python
from vectorbtpro import *

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if vbt.CSVDataSaver.file_exists():
        csv_saver = vbt.CSVDataSaver.load()
        csv_saver.update()
        init_save = False
    else:
        data = vbt.BinanceData.pull(
            "BTCUSDT", 
            start="10 minutes ago UTC",
            timeframe="1 minute"
        )
        csv_saver = vbt.CSVDataSaver(data)
        init_save = True
    csv_saver.update_every(1, "minute", init_save=init_save)
    csv_saver.save()  # (1)!
```

```python
$ python BTCUSDT_1m_saver.py
2023-02-01 12:26:36.744000+00:00 - 2023-02-01 12:36:00+00:00: : 1it [00:01,  1.22s/it]
INFO:vectorbtpro.data.saver:Saved initial 10 rows from 2023-02-01 12:27:00+00:00 to 2023-02-01 12:36:00+00:00
INFO:vectorbtpro.utils.schedule_:Starting schedule manager with jobs [Every 1 minute do update(save_kwargs=None) (last run: [never], next run: 2023-02-01 13:37:38)]
INFO:vectorbtpro.data.saver:Saved 2 rows from 2023-02-01 12:36:00+00:00 to 2023-02-01 12:37:00+00:00
INFO:vectorbtpro.data.saver:Saved 2 rows from 2023-02-01 12:37:00+00:00 to 2023-02-01 12:38:00+00:00
INFO:vectorbtpro.utils.schedule_:Stopping schedule manager
```

```python
$ python BTCUSDT_1m_saver.py
2023-02-01 12:26:36.744000+00:00 - 2023-02-01 12:36:00+00:00: : 1it [00:01,  1.22s/it]
INFO:vectorbtpro.data.saver:Saved initial 10 rows from 2023-02-01 12:27:00+00:00 to 2023-02-01 12:36:00+00:00
INFO:vectorbtpro.utils.schedule_:Starting schedule manager with jobs [Every 1 minute do update(save_kwargs=None) (last run: [never], next run: 2023-02-01 13:37:38)]
INFO:vectorbtpro.data.saver:Saved 2 rows from 2023-02-01 12:36:00+00:00 to 2023-02-01 12:37:00+00:00
INFO:vectorbtpro.data.saver:Saved 2 rows from 2023-02-01 12:37:00+00:00 to 2023-02-01 12:38:00+00:00
INFO:vectorbtpro.utils.schedule_:Stopping schedule manager
```

```python
$ python BTCUSDT_1m_saver.py
INFO:vectorbtpro.utils.schedule_:Starting schedule manager with jobs [Every 1 minute do update(save_kwargs=None) (last run: [never], next run: 2023-02-01 13:42:08)]
INFO:vectorbtpro.data.saver:Saved 5 rows from 2023-02-01 12:38:00+00:00 to 2023-02-01 12:42:00+00:00
INFO:vectorbtpro.utils.schedule_:Stopping schedule manager
```

```python
$ python BTCUSDT_1m_saver.py
INFO:vectorbtpro.utils.schedule_:Starting schedule manager with jobs [Every 1 minute do update(save_kwargs=None) (last run: [never], next run: 2023-02-01 13:42:08)]
INFO:vectorbtpro.data.saver:Saved 5 rows from 2023-02-01 12:38:00+00:00 to 2023-02-01 12:42:00+00:00
INFO:vectorbtpro.utils.schedule_:Stopping schedule manager
```

## Polygon.io¶

```python
>>> vbt.PolygonData.set_custom_settings(
...     client_config=dict(
...         api_key="YOUR_API_KEY"
...     )
... )
>>> data = vbt.PolygonData.pull(
...     "AAPL",
...     start="2022-12-01",  # (1)!
...     end="2023-01-01",
...     timeframe="30 minutes",
...     tz="US/Eastern"
... )
>>> data.get()
                             Open    High     Low     Close   Volume  \
Open time                                                              
2022-12-01 04:00:00-05:00  148.08  148.08  147.04  147.3700  50886.0   
2022-12-01 04:30:00-05:00  147.37  147.37  147.12  147.2600  16575.0   
2022-12-01 05:00:00-05:00  147.31  147.51  147.20  147.3800  20753.0   
2022-12-01 05:30:00-05:00  147.43  147.56  147.38  147.3800   7388.0   
2022-12-01 06:00:00-05:00  147.30  147.38  147.24  147.2400   7416.0   
...                           ...     ...     ...       ...      ...   
2022-12-30 17:30:00-05:00  129.94  130.05  129.91  129.9487  35694.0   
2022-12-30 18:00:00-05:00  129.95  130.00  129.94  129.9500  15595.0   
2022-12-30 18:30:00-05:00  129.94  130.05  129.94  130.0100  20287.0   
2022-12-30 19:00:00-05:00  129.99  130.04  129.99  130.0000  12490.0   
2022-12-30 19:30:00-05:00  130.00  130.04  129.97  129.9700  28271.0   

                           Trade count      VWAP  
Open time                                         
2022-12-01 04:00:00-05:00         1024  147.2632  
2022-12-01 04:30:00-05:00          412  147.2304  
2022-12-01 05:00:00-05:00          306  147.3466  
2022-12-01 05:30:00-05:00          201  147.4818  
2022-12-01 06:00:00-05:00          221  147.2938  
...                                ...       ...  
2022-12-30 17:30:00-05:00          350  129.9672  
2022-12-30 18:00:00-05:00          277  129.9572  
2022-12-30 18:30:00-05:00          312  130.0034  
2022-12-30 19:00:00-05:00          176  130.0140  
2022-12-30 19:30:00-05:00          366  129.9941  

[672 rows x 7 columns]
```

```python
>>> vbt.PolygonData.set_custom_settings(
...     client_config=dict(
...         api_key="YOUR_API_KEY"
...     )
... )
>>> data = vbt.PolygonData.pull(
...     "AAPL",
...     start="2022-12-01",  # (1)!
...     end="2023-01-01",
...     timeframe="30 minutes",
...     tz="US/Eastern"
... )
>>> data.get()
                             Open    High     Low     Close   Volume  \
Open time                                                              
2022-12-01 04:00:00-05:00  148.08  148.08  147.04  147.3700  50886.0   
2022-12-01 04:30:00-05:00  147.37  147.37  147.12  147.2600  16575.0   
2022-12-01 05:00:00-05:00  147.31  147.51  147.20  147.3800  20753.0   
2022-12-01 05:30:00-05:00  147.43  147.56  147.38  147.3800   7388.0   
2022-12-01 06:00:00-05:00  147.30  147.38  147.24  147.2400   7416.0   
...                           ...     ...     ...       ...      ...   
2022-12-30 17:30:00-05:00  129.94  130.05  129.91  129.9487  35694.0   
2022-12-30 18:00:00-05:00  129.95  130.00  129.94  129.9500  15595.0   
2022-12-30 18:30:00-05:00  129.94  130.05  129.94  130.0100  20287.0   
2022-12-30 19:00:00-05:00  129.99  130.04  129.99  130.0000  12490.0   
2022-12-30 19:30:00-05:00  130.00  130.04  129.97  129.9700  28271.0   

                           Trade count      VWAP  
Open time                                         
2022-12-01 04:00:00-05:00         1024  147.2632  
2022-12-01 04:30:00-05:00          412  147.2304  
2022-12-01 05:00:00-05:00          306  147.3466  
2022-12-01 05:30:00-05:00          201  147.4818  
2022-12-01 06:00:00-05:00          221  147.2938  
...                                ...       ...  
2022-12-30 17:30:00-05:00          350  129.9672  
2022-12-30 18:00:00-05:00          277  129.9572  
2022-12-30 18:30:00-05:00          312  130.0034  
2022-12-30 19:00:00-05:00          176  130.0140  
2022-12-30 19:30:00-05:00          366  129.9941  

[672 rows x 7 columns]
```

```python
tz
```

## Alpha Vantage¶

```python
>>> data = vbt.AVData.pull(
...     "IBM",
...     category="technical-indicators",
...     function="STOCHRSI",
...     params=dict(fastkperiod=14)
... )
>>> data.get()
                              FastD     FastK
1999-12-07 00:00:00+00:00  100.0000  100.0000
1999-12-08 00:00:00+00:00  100.0000  100.0000
1999-12-09 00:00:00+00:00   77.0255   31.0765
1999-12-10 00:00:00+00:00   43.6922    0.0000
1999-12-13 00:00:00+00:00   12.0197    4.9826
...                             ...       ...
2023-01-26 00:00:00+00:00   11.7960    0.0000
2023-01-27 00:00:00+00:00    3.7773    0.0000
2023-01-30 00:00:00+00:00    4.4824   13.4471
2023-01-31 00:00:00+00:00    7.8258   10.0302
2023-02-01 16:00:01+00:00   13.0966   15.8126

[5826 rows x 2 columns]
```

```python
>>> data = vbt.AVData.pull(
...     "IBM",
...     category="technical-indicators",
...     function="STOCHRSI",
...     params=dict(fastkperiod=14)
... )
>>> data.get()
                              FastD     FastK
1999-12-07 00:00:00+00:00  100.0000  100.0000
1999-12-08 00:00:00+00:00  100.0000  100.0000
1999-12-09 00:00:00+00:00   77.0255   31.0765
1999-12-10 00:00:00+00:00   43.6922    0.0000
1999-12-13 00:00:00+00:00   12.0197    4.9826
...                             ...       ...
2023-01-26 00:00:00+00:00   11.7960    0.0000
2023-01-27 00:00:00+00:00    3.7773    0.0000
2023-01-30 00:00:00+00:00    4.4824   13.4471
2023-01-31 00:00:00+00:00    7.8258   10.0302
2023-02-01 16:00:01+00:00   13.0966   15.8126

[5826 rows x 2 columns]
```

## Nasdaq Data Link¶

```python
>>> data = vbt.NDLData.pull("UMICH/SOC1")
>>> data.get()
                           Index
Date                            
1952-11-30 00:00:00+00:00   86.2
1953-02-28 00:00:00+00:00   90.7
1953-08-31 00:00:00+00:00   80.8
1953-11-30 00:00:00+00:00   80.7
1954-02-28 00:00:00+00:00   82.0
...                          ...
2022-08-31 00:00:00+00:00   58.2
2022-09-30 00:00:00+00:00   58.6
2022-10-31 00:00:00+00:00   59.9
2022-11-30 00:00:00+00:00   56.8
2022-12-31 00:00:00+00:00   59.7

[632 rows x 1 columns]
```

```python
>>> data = vbt.NDLData.pull("UMICH/SOC1")
>>> data.get()
                           Index
Date                            
1952-11-30 00:00:00+00:00   86.2
1953-02-28 00:00:00+00:00   90.7
1953-08-31 00:00:00+00:00   80.8
1953-11-30 00:00:00+00:00   80.7
1954-02-28 00:00:00+00:00   82.0
...                          ...
2022-08-31 00:00:00+00:00   58.2
2022-09-30 00:00:00+00:00   58.6
2022-10-31 00:00:00+00:00   59.9
2022-11-30 00:00:00+00:00   56.8
2022-12-31 00:00:00+00:00   59.7

[632 rows x 1 columns]
```

## Data merging¶

```python
>>> binance_data = vbt.CCXTData.pull("BTCUSDT", exchange="binance")
>>> bybit_data = vbt.CCXTData.pull("BTCUSDT", exchange="bybit")
>>> bitfinex_data = vbt.CCXTData.pull("BTC/USDT", exchange="bitfinex")
>>> kucoin_data = vbt.CCXTData.pull("BTC-USDT", exchange="kucoin")

>>> data = vbt.Data.merge([
...     binance_data.rename({"BTCUSDT": "Binance"}),
...     bybit_data.rename({"BTCUSDT": "Bybit"}),
...     bitfinex_data.rename({"BTC/USDT": "Bitfinex"}),
...     kucoin_data.rename({"BTC-USDT": "KuCoin"}),
... ], missing_index="drop", silence_warnings=True)

>>> @njit
... def rescale_nb(x):
...     return (x - x.mean()) / x.mean()

>>> rescaled_close = data.close.vbt.row_apply(rescale_nb)
>>> rescaled_close = rescaled_close.vbt.rolling_mean(30)
>>> rescaled_close.loc["2023":"2023"].vbt.plot().show()
```

```python
>>> binance_data = vbt.CCXTData.pull("BTCUSDT", exchange="binance")
>>> bybit_data = vbt.CCXTData.pull("BTCUSDT", exchange="bybit")
>>> bitfinex_data = vbt.CCXTData.pull("BTC/USDT", exchange="bitfinex")
>>> kucoin_data = vbt.CCXTData.pull("BTC-USDT", exchange="kucoin")

>>> data = vbt.Data.merge([
...     binance_data.rename({"BTCUSDT": "Binance"}),
...     bybit_data.rename({"BTCUSDT": "Bybit"}),
...     bitfinex_data.rename({"BTC/USDT": "Bitfinex"}),
...     kucoin_data.rename({"BTC-USDT": "KuCoin"}),
... ], missing_index="drop", silence_warnings=True)

>>> @njit
... def rescale_nb(x):
...     return (x - x.mean()) / x.mean()

>>> rescaled_close = data.close.vbt.row_apply(rescale_nb)
>>> rescaled_close = rescaled_close.vbt.rolling_mean(30)
>>> rescaled_close.loc["2023":"2023"].vbt.plot().show()
```

## Alpaca¶

```python
>>> vbt.AlpacaData.set_custom_settings(
...     client_config=dict(
...         api_key="YOUR_API_KEY",
...         secret_key="YOUR_API_SECRET"
...     )
... )
>>> data = vbt.AlpacaData.pull(
...     "AAPL",
...     start="one week ago 00:00",  # (1)!
...     end="15 minutes ago",  # (2)!
...     timeframe="1 minute",
...     adjustment="all",
...     tz="US/Eastern"
... )
>>> data.get()
                               Open      High       Low     Close  Volume  \
Open time                                                                   
2023-01-30 04:00:00-05:00  145.5400  145.5400  144.0100  144.0200  5452.0   
2023-01-30 04:01:00-05:00  144.0800  144.0800  144.0000  144.0500  3616.0   
2023-01-30 04:02:00-05:00  144.0300  144.0400  144.0100  144.0100  1671.0   
2023-01-30 04:03:00-05:00  144.0100  144.0300  144.0000  144.0300  4721.0   
2023-01-30 04:04:00-05:00  144.0200  144.0200  144.0200  144.0200  1343.0   
...                             ...       ...       ...       ...     ...   
2023-02-03 19:54:00-05:00  154.3301  154.3301  154.3301  154.3301   347.0   
2023-02-03 19:55:00-05:00  154.3300  154.3400  154.3200  154.3400  1438.0   
2023-02-03 19:56:00-05:00  154.3400  154.3400  154.3300  154.3300   588.0   
2023-02-03 19:58:00-05:00  154.3500  154.3500  154.3500  154.3500   555.0   
2023-02-03 19:59:00-05:00  154.3400  154.3900  154.3300  154.3900  3835.0   

                           Trade count        VWAP  
Open time                                           
2023-01-30 04:00:00-05:00          165  144.376126  
2023-01-30 04:01:00-05:00           81  144.036336  
2023-01-30 04:02:00-05:00           52  144.035314  
2023-01-30 04:03:00-05:00           56  144.012680  
2023-01-30 04:04:00-05:00           40  144.021854  
...                                ...         ...  
2023-02-03 19:54:00-05:00           21  154.331340  
2023-02-03 19:55:00-05:00           38  154.331756  
2023-02-03 19:56:00-05:00           17  154.338971  
2023-02-03 19:58:00-05:00           27  154.343090  
2023-02-03 19:59:00-05:00           58  154.357219  

[4224 rows x 7 columns]
```

```python
>>> vbt.AlpacaData.set_custom_settings(
...     client_config=dict(
...         api_key="YOUR_API_KEY",
...         secret_key="YOUR_API_SECRET"
...     )
... )
>>> data = vbt.AlpacaData.pull(
...     "AAPL",
...     start="one week ago 00:00",  # (1)!
...     end="15 minutes ago",  # (2)!
...     timeframe="1 minute",
...     adjustment="all",
...     tz="US/Eastern"
... )
>>> data.get()
                               Open      High       Low     Close  Volume  \
Open time                                                                   
2023-01-30 04:00:00-05:00  145.5400  145.5400  144.0100  144.0200  5452.0   
2023-01-30 04:01:00-05:00  144.0800  144.0800  144.0000  144.0500  3616.0   
2023-01-30 04:02:00-05:00  144.0300  144.0400  144.0100  144.0100  1671.0   
2023-01-30 04:03:00-05:00  144.0100  144.0300  144.0000  144.0300  4721.0   
2023-01-30 04:04:00-05:00  144.0200  144.0200  144.0200  144.0200  1343.0   
...                             ...       ...       ...       ...     ...   
2023-02-03 19:54:00-05:00  154.3301  154.3301  154.3301  154.3301   347.0   
2023-02-03 19:55:00-05:00  154.3300  154.3400  154.3200  154.3400  1438.0   
2023-02-03 19:56:00-05:00  154.3400  154.3400  154.3300  154.3300   588.0   
2023-02-03 19:58:00-05:00  154.3500  154.3500  154.3500  154.3500   555.0   
2023-02-03 19:59:00-05:00  154.3400  154.3900  154.3300  154.3900  3835.0   

                           Trade count        VWAP  
Open time                                           
2023-01-30 04:00:00-05:00          165  144.376126  
2023-01-30 04:01:00-05:00           81  144.036336  
2023-01-30 04:02:00-05:00           52  144.035314  
2023-01-30 04:03:00-05:00           56  144.012680  
2023-01-30 04:04:00-05:00           40  144.021854  
...                                ...         ...  
2023-02-03 19:54:00-05:00           21  154.331340  
2023-02-03 19:55:00-05:00           38  154.331756  
2023-02-03 19:56:00-05:00           17  154.338971  
2023-02-03 19:58:00-05:00           27  154.343090  
2023-02-03 19:59:00-05:00           58  154.357219  

[4224 rows x 7 columns]
```

```python
tz
```

## Local data¶

Once remote data has been fetched, you most likely want to persist it on disk. There are two new options for this: either serialize the entire data class, or save the actual data to CSV or HDF5. Each dataset can be stored in a single flat file, which makes it easier to work with than a database. Upon saving, the data can be effortlessly loaded back either by deserializing, or by using data classes that specialize in loading data from CSV and HDF5 files. These classes support a variety of features, including filtering by row and datetime ranges, updating, chunking, and even a smart dataset search that can traverse sub-directories recursively and return datasets that match a specific glob pattern or regular expression

```python
>>> btc_data = vbt.BinanceData.pull("BTCUSDT")
>>> eth_data = vbt.BinanceData.pull("ETHUSDT")

>>> btc_data.to_hdf()
>>> eth_data.to_hdf()

>>> data = vbt.BinanceData.from_hdf(start="2020", end="2021")
```

```python
>>> btc_data = vbt.BinanceData.pull("BTCUSDT")
>>> eth_data = vbt.BinanceData.pull("ETHUSDT")

>>> btc_data.to_hdf()
>>> eth_data.to_hdf()

>>> data = vbt.BinanceData.from_hdf(start="2020", end="2021")
```

Key 2/2

Key 2/2

```python
>>> data.close
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2020-01-01 00:00:00+00:00   7200.85   130.77
2020-01-02 00:00:00+00:00   6965.71   127.19
2020-01-03 00:00:00+00:00   7344.96   134.35
2020-01-04 00:00:00+00:00   7354.11   134.20
2020-01-05 00:00:00+00:00   7358.75   135.37
...                             ...      ...
2020-12-27 00:00:00+00:00  26281.66   685.11
2020-12-28 00:00:00+00:00  27079.41   730.41
2020-12-29 00:00:00+00:00  27385.00   732.00
2020-12-30 00:00:00+00:00  28875.54   752.17
2020-12-31 00:00:00+00:00  28923.63   736.42

[366 rows x 2 columns]
```

```python
>>> data.close
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2020-01-01 00:00:00+00:00   7200.85   130.77
2020-01-02 00:00:00+00:00   6965.71   127.19
2020-01-03 00:00:00+00:00   7344.96   134.35
2020-01-04 00:00:00+00:00   7354.11   134.20
2020-01-05 00:00:00+00:00   7358.75   135.37
...                             ...      ...
2020-12-27 00:00:00+00:00  26281.66   685.11
2020-12-28 00:00:00+00:00  27079.41   730.41
2020-12-29 00:00:00+00:00  27385.00   732.00
2020-12-30 00:00:00+00:00  28875.54   752.17
2020-12-31 00:00:00+00:00  28923.63   736.42

[366 rows x 2 columns]
```

Python code

