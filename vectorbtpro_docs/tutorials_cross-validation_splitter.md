# Splitter¶

The manual approach that we did previously can be decomposed into three distinctive steps: splitting the whole period into sub-periods, applying a UDF on each sub-period and merging its outputs, and analyzing the merged outputs using available data science tools. The first and the second step can be well automated; for example, scikit-learn has a range of classes for cross-validation, each taking an array and performing some job on chunks of that array. The issue with that (otherwise excellent) Python package is that it lacks robust CV schemes for time series data, charting and analysis tools for split distributions, as well as an easy-to-extend interface for custom use cases. It also focuses on machine learning (ML) models that are trained on one data and validated on another by doing predictions (that's why it's called scikit-learn, after all), while rule-based algorithms that aren't predicting but producing a range of scores (one per test rather than data point) aren't receiving enough love.

That's why vectorbt needs to go its own way and implement a functionality cut to the needs of quantitative analysts rather than ML enthusiasts. At the heart of this functionality is the class Splitter, whose main responsibility is to produce arbitrary splits and perform operations on those splits. The workings of this class are dead-simple: the user calls one of the class methods with the prefix from_ (sounds familiar, right?) to generate splits; in return, a splitter instance is returned with splits and their labels being saved in a memory-efficient array format. This instance can be used to analyze the split distribution, to chunk array-like objects, and to run UDFs. Get yourself ready, because this class alone has twice as many lines of code as the entire backtesting.py library

```python
from_
```

Let's create a splitter for the schema of our first example:

```python
>>> splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length=360, 
...     split=0.5,
...     set_labels=["IS", "OOS"]
... )
>>> splitter.plot().show()
```

```python
>>> splitter = vbt.Splitter.from_rolling(
...     data.index, 
...     length=360, 
...     split=0.5,
...     set_labels=["IS", "OOS"]
... )
>>> splitter.plot().show()
```

That's it! We've got a splitter that can manipulate the periods (blue and orange boxes on the plot) and the data under those periods to our liking - no more while-loops. But before we dive into the dozens of implemented generation and analysis techniques, let's take a look under the hood and make ourselves comfortable with some basic concepts first.

## Schema¶

The smallest unit of a splitter is a range, which is a period of time that can be mapped onto data. On the plot above, we can count a total of 18 ranges - 9 blue ones and 9 orange ones. Multiple ranges next to each other and representing a single test are called a split; there are 9 splits present in the chart, such that we expect one pipeline to be tested on 9 different data ranges. Different range types within each split are called sets. Usually, there is either one set, two sets - "training" and "test" (commonly used in backtesting), or three sets - "training", "validation", and "test" (commonly used in ML). The number of sets is fixed throughout all splits.

This schema fits perfectly into the philosophy of vectorbt because we can represent things in an array format where rows are splits, columns are sets, and elements are ranges:

```python
>>> splitter.splits
set                         IS                      OOS
split                                                  
0          slice(0, 180, None)    slice(180, 360, None)
1        slice(180, 360, None)    slice(360, 540, None)
2        slice(360, 540, None)    slice(540, 720, None)
3        slice(540, 720, None)    slice(720, 900, None)
4        slice(720, 900, None)   slice(900, 1080, None)
5       slice(900, 1080, None)  slice(1080, 1260, None)
6      slice(1080, 1260, None)  slice(1260, 1440, None)
7      slice(1260, 1440, None)  slice(1440, 1620, None)
8      slice(1440, 1620, None)  slice(1620, 1800, None)
```

```python
>>> splitter.splits
set                         IS                      OOS
split                                                  
0          slice(0, 180, None)    slice(180, 360, None)
1        slice(180, 360, None)    slice(360, 540, None)
2        slice(360, 540, None)    slice(540, 720, None)
3        slice(540, 720, None)    slice(720, 900, None)
4        slice(720, 900, None)   slice(900, 1080, None)
5       slice(900, 1080, None)  slice(1080, 1260, None)
6      slice(1080, 1260, None)  slice(1260, 1440, None)
7      slice(1260, 1440, None)  slice(1440, 1620, None)
8      slice(1440, 1620, None)  slice(1620, 1800, None)
```

Notice how the index are split labels and columns are set labels: in contrast to other classes in vectorbt, the wrapper of this class doesn't represent time and assets, but splits and sets. Time is being tracked separately as Splitter.index while assets aren't being tracked at all since they have no implications on splitting.

```python
>>> splitter.index
DatetimeIndex(['2017-08-17 00:00:00+00:00', '2017-08-18 00:00:00+00:00',
               '2017-08-19 00:00:00+00:00', '2017-08-20 00:00:00+00:00',
               ...
               '2022-10-28 00:00:00+00:00', '2022-10-29 00:00:00+00:00',
               '2022-10-30 00:00:00+00:00', '2022-10-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=1902, freq='D')

>>> splitter.wrapper.index
RangeIndex(start=0, stop=9, step=1, name='split')

>>> splitter.wrapper.columns
Index(['IS', 'OOS'], dtype='object', name='set')
```

```python
>>> splitter.index
DatetimeIndex(['2017-08-17 00:00:00+00:00', '2017-08-18 00:00:00+00:00',
               '2017-08-19 00:00:00+00:00', '2017-08-20 00:00:00+00:00',
               ...
               '2022-10-28 00:00:00+00:00', '2022-10-29 00:00:00+00:00',
               '2022-10-30 00:00:00+00:00', '2022-10-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=1902, freq='D')

>>> splitter.wrapper.index
RangeIndex(start=0, stop=9, step=1, name='split')

>>> splitter.wrapper.columns
Index(['IS', 'OOS'], dtype='object', name='set')
```

Such a design has one nice property: we can apply indexing directly on a splitter instance to select specific splits and sets. Let's select the OOS set:

```python
>>> oos_splitter = splitter["OOS"]
>>> oos_splitter.splits
split
0      slice(180, 360, None)
1      slice(360, 540, None)
2      slice(540, 720, None)
3      slice(720, 900, None)
4     slice(900, 1080, None)
5    slice(1080, 1260, None)
6    slice(1260, 1440, None)
7    slice(1440, 1620, None)
8    slice(1620, 1800, None)
Name: OOS, dtype: object
```

```python
>>> oos_splitter = splitter["OOS"]
>>> oos_splitter.splits
split
0      slice(180, 360, None)
1      slice(360, 540, None)
2      slice(540, 720, None)
3      slice(720, 900, None)
4     slice(900, 1080, None)
5    slice(1080, 1260, None)
6    slice(1260, 1440, None)
7    slice(1440, 1620, None)
8    slice(1620, 1800, None)
Name: OOS, dtype: object
```

This operation has created a completely new splitter for OOS ranges

Why bother? Because we can select one set and apply a UDF on it, and then select the next set and apply a completely different UDF on it. Sounds like a prerequisite for CV, right?

### Range format¶

So, how do ranges look like? In the first example of this tutorial, we used a start and end date to slice the data using loc. But as we also learned that the end date in a loc operation should be inclusive, which makes it annoyingly difficult to make sure that neighboring ranges do not overlap. Also, dates cannot be used to slice NumPy arrays, unless they are translated into positions beforehand. That's why the splitter does integer-location based indexing and accepts the following range formats that can be used to slice both Pandas (using pandas.DataFrame.iloc) and NumPy arrays:

```python
loc
```

```python
loc
```

```python
[4, 3, 0]
```

```python
slice(1, 7)
```

For example, the slice slice(1, 7) covers the indices [0, 1, 2, 3, 4, 5, 6]:

```python
slice(1, 7)
```

```python
[0, 1, 2, 3, 4, 5, 6]
```

```python
>>> index = vbt.date_range("2020", periods=14)
>>> index[slice(1, 7)]  # (1)!
DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
               '2020-01-06', '2020-01-07'],
              dtype='datetime64[ns]', freq='D')

>>> index[1], index[6]
(Timestamp('2020-01-02 00:00:00', freq='D'),
 Timestamp('2020-01-07 00:00:00', freq='D'))
```

```python
>>> index = vbt.date_range("2020", periods=14)
>>> index[slice(1, 7)]  # (1)!
DatetimeIndex(['2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
               '2020-01-06', '2020-01-07'],
              dtype='datetime64[ns]', freq='D')

>>> index[1], index[6]
(Timestamp('2020-01-02 00:00:00', freq='D'),
 Timestamp('2020-01-07 00:00:00', freq='D'))
```

```python
index[1:7]
```

Having the index and integer-location based ranges kept separately makes designing non-overlapping, bug-free ranges super-easy.

#### Relative¶

The range format introduced above is called "fixed" because ranges do not depend on each other. But there's another range format called "relative" that makes one range depend on the previous range. For example, instead of defining a range between fixed index positions 110 and 150, we can issue an instruction to create a range that starts 10 points after the end point of the previous range and has the length of 40. Such an instruction can be created using RelRange:

```python
110
```

```python
150
```

```python
10
```

```python
40
```

```python
>>> rel_range = vbt.RelRange(offset=10, length=40)
>>> rel_range
RelRange(
    offset=10, 
    offset_anchor='prev_end', 
    offset_space='free', 
    length=40, 
    length_space='free', 
    out_of_bounds='warn',
    is_gap=False
)
```

```python
>>> rel_range = vbt.RelRange(offset=10, length=40)
>>> rel_range
RelRange(
    offset=10, 
    offset_anchor='prev_end', 
    offset_space='free', 
    length=40, 
    length_space='free', 
    out_of_bounds='warn',
    is_gap=False
)
```

This instruction will be evaluated later in time by calling RelRange.to_slice:

```python
>>> rel_range.to_slice(total_len=len(splitter.index), prev_end=100)  # (1)!
slice(110, 150, None)
```

```python
>>> rel_range.to_slice(total_len=len(splitter.index), prev_end=100)  # (1)!
slice(110, 150, None)
```

```python
prev_end
```

Relative ranges are usually converted into fixed ones prior to constructing a splitter instance, but splitter instances can also hold relative ranges in case this is wanted by the user.

### Array format¶

But how do we store such range formats efficiently? The most flexible range formats are apparently indices and masks because they allow gaps and can enable a more traditional k-fold cross-validation, but they may also produce a huge memory footprint even for simple use cases. For example, consider a range stretching over 1 year of 1-minute data; it would take roughly 4MB of RAM as an integer array and 0.5MB as a mask:

```python
>>> index = vbt.date_range("2020", "2021", freq="1min")
>>> range_ = np.arange(len(index))
>>> range_.nbytes / 1024 / 1024
4.02099609375

>>> range_ = np.full(len(index), True)
>>> range_.nbytes / 1024 / 1024
0.50262451171875
```

```python
>>> index = vbt.date_range("2020", "2021", freq="1min")
>>> range_ = np.arange(len(index))
>>> range_.nbytes / 1024 / 1024
4.02099609375

>>> range_ = np.full(len(index), True)
>>> range_.nbytes / 1024 / 1024
0.50262451171875
```

This means that only 100 splits and 2 sets would consume 800MB and 100MB of RAM respectively, and this is only to keep the splitter metadata in memory! Moreover, most "ranges" don't need to be that complex: they have predefined start and end points (that should occupy at most 18 bytes of memory), while being capable of pulling the same exact period of data as their integer and boolean array counterparts. That's why the Splitter class tries to convert any array into a slice, by the way.

To make sure that the user can make use of lightweight ranges, complex arrays, and relative ranges using the same API, the array that holds ranges has an object data type:

```python
>>> splitter.splits_arr.dtype  # (1)!
dtype('O')
```

```python
>>> splitter.splits_arr.dtype  # (1)!
dtype('O')
```

Info

If an element of an array is a complex object, it doesn't mean that the array holds this entire object - it only holds the reference to that object:

```python
>>> id(slice(0, 180, None))
140627839366784
```

```python
>>> id(slice(0, 180, None))
140627839366784
```

The object data type is completely legit; it only becomes a burden once attempted to be passed to Numba, but the splitting functionality is entirely written in Python because the number of ranges (splits x sets) is usually kept relatively low such that the main bottleneck lies in running UDFs and not in iterating over ranges (and no worries, UDFs can still be run in Numba ) The other drawback is that the array cannot be numerically processed with NumPy or Pandas anymore, but that's why we have Splitter that can extract the meaning out of such an array!

This array format has even more benefits: we can use different range formats across different splits, we can store index arrays of different lengths, and since the splits array only stores references, we don't need to duplicate an array if two range values are pointing to the same array object. For example, let's construct a splitter where (differently-sized) ranges are stored as integer-location based arrays:

```python
splits
```

```python
>>> range_00 = np.arange(0, 5)
>>> range_01 = np.arange(5, 15)
>>> range_10 = np.arange(15, 30)
>>> range_11 = np.arange(30, 50)

>>> ind_splitter = vbt.Splitter.from_splits(
...     data.index,
...     [[range_00, range_01], [range_10, range_11]],
...     fix_ranges=False
... )
>>> ind_splitter.splits
set                                                set_0  \
split                                                      
0                FixRange(range_=array([0, 1, 2, 3, 4]))   
1      FixRange(range_=array([15, 16, 17, 18, 19, 20,...   

set                                                set_1  
split                                                     
0      FixRange(range_=array([ 5,  6,  7,  8,  9, 10,...  
1      FixRange(range_=array([30, 31, 32, 33, 34, 35,... 

>>> ind_splitter.splits.loc[0, "set_1"]  # (1)!
FixRange(range_=array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]))

>>> ind_splitter.splits.loc[0, "set_1"].range_  # (2)!
array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
```

```python
>>> range_00 = np.arange(0, 5)
>>> range_01 = np.arange(5, 15)
>>> range_10 = np.arange(15, 30)
>>> range_11 = np.arange(30, 50)

>>> ind_splitter = vbt.Splitter.from_splits(
...     data.index,
...     [[range_00, range_01], [range_10, range_11]],
...     fix_ranges=False
... )
>>> ind_splitter.splits
set                                                set_0  \
split                                                      
0                FixRange(range_=array([0, 1, 2, 3, 4]))   
1      FixRange(range_=array([15, 16, 17, 18, 19, 20,...   

set                                                set_1  
split                                                     
0      FixRange(range_=array([ 5,  6,  7,  8,  9, 10,...  
1      FixRange(range_=array([30, 31, 32, 33, 34, 35,... 

>>> ind_splitter.splits.loc[0, "set_1"]  # (1)!
FixRange(range_=array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]))

>>> ind_splitter.splits.loc[0, "set_1"].range_  # (2)!
array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
```

```python
range_01
```

Hint

Why is the value of FixRange called range_ and not range? Because range is a reserved keyword in Python.

```python
range_
```

```python
range
```

```python
range
```

As we see, each element of the splits array is... a NumPy array wrapped with FixRange, which is perfectly fine. Why are arrays wrapped and slices not? Because sub-arrays would expand the entire array to three dimensions, which are disliked by Pandas.

## Preparation¶

In a nutshell, a splitter instance keeps track of three objects:

```python
wrapper
```

```python
index
```

```python
splits
```

Although we can prepare these objects manually, there are convenient methods that automate this workflow for us. The base class method that most other class methods are based upon is Splitter.from_splits, which takes a sequence of splits, optionally does some pre-processing on each split, and converts that sequence into a suitable array format. It also prepares the labels and the wrapper. But let's focus on preparing the splits first.

### Splits¶

Splitting is a process of dividing a bigger range into chunks of smaller ranges, which is implemented by the method Splitter.split_range. What it takes is a fixed range range_ and a split specification new_split, and returns a tuple of new fixed ranges. Keep in mind that the returned ranges are always fixed, hence this method is also used to convert relative ranges to fixed. But the main use case of this method is to check whether the provided specification makes any sense and doesn't violate any bounds. Let's generate two ranges: one that takes 75% of space and another that takes the remaining 25%.

```python
range_
```

```python
new_split
```

```python
>>> vbt.Splitter.split_range(
...     slice(None),  # (1)!
...     (vbt.RelRange(length=0.75), vbt.RelRange()),  # (2)!
...     index=data.index
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None),  # (1)!
...     (vbt.RelRange(length=0.75), vbt.RelRange()),  # (2)!
...     index=data.index
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

```python
index[None:None]
```

```python
index[:]
```

Hint

This method is a hybrid method: it can be called both as a class method and instance method. If the latter, we don't need to provide the index since it's already stored:

```python
>>> splitter.split_range(
...     slice(None),
...     (vbt.RelRange(length=0.75), vbt.RelRange())
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

```python
>>> splitter.split_range(
...     slice(None),
...     (vbt.RelRange(length=0.75), vbt.RelRange())
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

These two slices can then be used to slice the data:

```python
>>> data[slice(0, 1426, None)]  # (1)!
<vectorbtpro.data.custom.binance.BinanceData at 0x7fe12a7df310>
```

```python
>>> data[slice(0, 1426, None)]  # (1)!
<vectorbtpro.data.custom.binance.BinanceData at 0x7fe12a7df310>
```

```python
data.iloc[:1426]
```

The two relative ranges can be substituted by just one number, which translates into the length reserved for the first range, while the second range gets the remaining space:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     0.75, 
...     index=data.index
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     0.75, 
...     index=data.index
... )
(slice(0, 1426, None), slice(1426, 1902, None))
```

Very often in CV, we want to fix the length of the OOS period and put everything else to the IS period. This can be specified by a negative number, which effectively reverses the processing order. For example, let's set the length of the OOS period to 25%:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     -0.25,
...     index=data.index
... )
(slice(0, 1427, None), slice(1427, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     -0.25,
...     index=data.index
... )
(slice(0, 1427, None), slice(1427, 1902, None))
```

Hint

Why are both results not identical? Because of rounding:

```python
>>> int(0.75 * len(data.index))
1426

>>> len(data.index) - int(0.25 * len(data.index))
1427
```

```python
>>> int(0.75 * len(data.index))
1426

>>> len(data.index) - int(0.25 * len(data.index))
1427
```

Or manually:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (vbt.RelRange(), vbt.RelRange(length=0.25)),
...     backwards=True,
...     index=data.index
... )
(slice(0, 1427, None), slice(1427, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (vbt.RelRange(), vbt.RelRange(length=0.25)),
...     backwards=True,
...     index=data.index
... )
(slice(0, 1427, None), slice(1427, 1902, None))
```

Relative ranges with just the length defined can be substituted by numbers for more convenience. For example, make the OOS period 30 data points long:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (1.0, 30), 
...     backwards=True,
...     index=data.index
... )
(slice(0, 1872, None), slice(1872, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (1.0, 30), 
...     backwards=True,
...     index=data.index
... )
(slice(0, 1872, None), slice(1872, 1902, None))
```

Hint

How does the method decide whether the length is relative or absolute? If the number is between 0 and 1, the length is relative to the surrounding space, otherwise it reflects the number of data points.

When using relative lengths, we can also specify the space the length should be relative to using RelRange.length_space. By default, a length is relative to the remaining space (i.e., from the right-most bound of the previous range to the right-most bound of the whole period), but we can force it to be relative to the entire space instead. For example, let's define three ranges with 40%, 40%, and 20% respectively:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length=0.4, length_space="all"), 
...         vbt.RelRange(length=0.4, length_space="all"),
...         vbt.RelRange()
...     ),
...     index=data.index
... )
(slice(0, 760, None), slice(760, 1520, None), slice(1520, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length=0.4, length_space="all"), 
...         vbt.RelRange(length=0.4, length_space="all"),
...         vbt.RelRange()
...     ),
...     index=data.index
... )
(slice(0, 760, None), slice(760, 1520, None), slice(1520, 1902, None))
```

To introduce a gap between two ranges, we can use an offset. Similarly to lengths, offsets can be relative or absolute too. Also, offsets have an anchor, which defaults to the right-most bound of the previous range (if any, otherwise 0). Let's require both ranges to have a gap of 1 point:

```python
>>> vbt.Splitter.split_range(
...     slice(None),
...     (vbt.RelRange(length=0.75), vbt.RelRange(offset=1)),
...     index=data.index
... )
(slice(0, 1426, None), slice(1427, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None),
...     (vbt.RelRange(length=0.75), vbt.RelRange(offset=1)),
...     index=data.index
... )
(slice(0, 1426, None), slice(1427, 1902, None))
```

The exact same result can be achieved by placing a relative range of 1 data point between both ranges and enabling RelRange.is_gap:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length=0.75), 
...         vbt.RelRange(length=1, is_gap=True),
...         vbt.RelRange()
...     ),
...     index=data.index
... )
(slice(0, 1426, None), slice(1427, 1902, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length=0.75), 
...         vbt.RelRange(length=1, is_gap=True),
...         vbt.RelRange()
...     ),
...     index=data.index
... )
(slice(0, 1426, None), slice(1427, 1902, None))
```

This method's power is not only in converting relative ranges into fixed ones, but also in trying to optimize the target ranges for best memory efficiency. Let's make the first range an array without gaps and the second range an array with gaps, and see what happens:

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (np.array([3, 4, 5]), np.array([6, 8, 10])),
...     index=data.index
... )
(slice(3, 6, None), array([ 6,  8, 10]))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (np.array([3, 4, 5]), np.array([6, 8, 10])),
...     index=data.index
... )
(slice(3, 6, None), array([ 6,  8, 10]))
```

We can see that the method was successful in optimizing the first array into a slice, but not the second. If such a conversion is not desired, we can disable it using the argument range_format:

```python
range_format
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (np.array([3, 4, 5]), np.array([6, 8, 10])),
...     range_format="indices",
...     index=data.index
... )
(array([3, 4, 5]), array([ 6,  8, 10]))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (np.array([3, 4, 5]), np.array([6, 8, 10])),
...     range_format="indices",
...     index=data.index
... )
(array([3, 4, 5]), array([ 6,  8, 10]))
```

Since Splitter uses integer-location and mask based indexing under the hood, we cannot use dates and times to slice arrays. Gladly, Splitter.split_range can take any slices and arrays as pd.Timestamp, np.datetime64, datetime.datetime, and even datetime-looking strings, and convert them into integers for us! It even takes care of the timezone.

```python
pd.Timestamp
```

```python
np.datetime64
```

```python
datetime.datetime
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (slice("2020", "2021"), slice("2021", "2022")),  # (1)!
...     index=data.index
... )
(slice(867, 1233, None), slice(1233, 1598, None))

>>> data.index[867:1233]
DatetimeIndex(['2020-01-01 00:00:00+00:00', ..., '2020-12-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=366, freq='D')

>>> data.index[1233:1598]
DatetimeIndex(['2021-01-01 00:00:00+00:00', ..., '2021-12-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=365, freq='D')
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (slice("2020", "2021"), slice("2021", "2022")),  # (1)!
...     index=data.index
... )
(slice(867, 1233, None), slice(1233, 1598, None))

>>> data.index[867:1233]
DatetimeIndex(['2020-01-01 00:00:00+00:00', ..., '2020-12-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=366, freq='D')

>>> data.index[1233:1598]
DatetimeIndex(['2021-01-01 00:00:00+00:00', ..., '2021-12-31 00:00:00+00:00'],
    dtype='datetime64[ns, UTC]', name='Open time', length=365, freq='D')
```

The same goes for relative ranges where the arguments offset and length can be provided as pd.Timedelta, np.timedelta64, datetime.timedelta, and timedelta-looking strings:

```python
offset
```

```python
length
```

```python
pd.Timedelta
```

```python
np.timedelta64
```

```python
datetime.timedelta
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length="180 days"), 
...         vbt.RelRange(offset="1 day", length="90 days")
...     ),
...     index=data.index
... )
(slice(0, 180, None), slice(181, 271, None))
```

```python
>>> vbt.Splitter.split_range(
...     slice(None), 
...     (
...         vbt.RelRange(length="180 days"), 
...         vbt.RelRange(offset="1 day", length="90 days")
...     ),
...     index=data.index
... )
(slice(0, 180, None), slice(181, 271, None))
```

### Method¶

Back to Splitter.from_splits. We've learned that each split is being prepared by converting a split specification into a sequence of ranges, one per set. By passing multiple of such specifications, we thus get a two-dimensional array available under Splitter.splits. Let's manually generate expanding splits with a OOS set having a fixed length of 25%:

```python
>>> manual_splitter = vbt.Splitter.from_splits(
...     data.index,
...     [
...         (vbt.RelRange(), vbt.RelRange(offset=0.5, length=0.25, length_space="all")),  # (1)!
...         (vbt.RelRange(), vbt.RelRange(offset=0.25, length=0.25, length_space="all")),
...         (vbt.RelRange(), vbt.RelRange(offset=0, length=0.25, length_space="all")),
...     ],
...     split_range_kwargs=dict(backwards=True),  # (2)!
...     set_labels=["IS", "OOS"]
... )
>>> manual_splitter.splits
set                      IS                      OOS
split                                               
0       slice(0, 476, None)    slice(476, 951, None)
1       slice(0, 952, None)   slice(952, 1427, None)
2      slice(0, 1427, None)  slice(1427, 1902, None)

>>> manual_splitter.plot().show()
```

```python
>>> manual_splitter = vbt.Splitter.from_splits(
...     data.index,
...     [
...         (vbt.RelRange(), vbt.RelRange(offset=0.5, length=0.25, length_space="all")),  # (1)!
...         (vbt.RelRange(), vbt.RelRange(offset=0.25, length=0.25, length_space="all")),
...         (vbt.RelRange(), vbt.RelRange(offset=0, length=0.25, length_space="all")),
...     ],
...     split_range_kwargs=dict(backwards=True),  # (2)!
...     set_labels=["IS", "OOS"]
... )
>>> manual_splitter.splits
set                      IS                      OOS
split                                               
0       slice(0, 476, None)    slice(476, 951, None)
1       slice(0, 952, None)   slice(952, 1427, None)
2      slice(0, 1427, None)  slice(1427, 1902, None)

>>> manual_splitter.plot().show()
```

## Generation¶

We know how to build a splitter manually, but most CV schemes involve generation through iteration, just like we did with the while-loop in our first example. Moreover, the start point of a split usually depends on the preceding split, which would require us to explicitly call Splitter.split_range on each split to get its boundaries. To reduce the amount of a boilerplate code required to enable this workflow, Splitter implements a collection of class methods, such as Splitter.from_rolling, that can produce a logically-coherent schema from a simple user query.

Most of these methods first divide the entire period into windows (either in advance or iteratively), and then split each sub-period using the argument split, which is simply being passed to Splitter.split_range as new_split. This way, the split specification becomes relative the sub-period and not the entire period as we did above.

```python
split
```

```python
new_split
```

Info

Internally, slice(None) (that we used every time previously) is being replaced by the window slice such that 0.5 would split only the window in half, not the entire period.

```python
slice(None)
```

```python
0.5
```

### Rolling¶

The most important method for CV is Splitter.from_rolling, which deploys a simple while-loop that appends splits until any split exceeds the right bound of the index. If the last split has the length less than the one requested, it gets discarded, such that there is usually some unused space at the end of the backtesting period.

But the most interesting question is: where do we place the next split? By default, if there's only one set, the next split is placed right after the previous one. If there are multiple sets though, the next split is placed right after the first (IS) range in the previous split, such that IS ranges never overlap across splits. And of course, we can control the offset behavior using offset_anchor_set (which range in the previous split acts as an anchor?), offset_anchor (left or right bound of that range acts as an anchor?), offset (positive/negative distance from anchor), and offset_space (see RelRange).

```python
offset_anchor_set
```

```python
offset_anchor
```

```python
offset
```

```python
offset_space
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index,
...     length=360,  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index,
...     length=360,  # (1)!
... ).plot().show()
```

```python
0.1
```

```python
"360 days"
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     offset=90  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     offset=90  # (1)!
... ).plot().show()
```

```python
offset_space
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     offset=-0.5  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     offset=-0.5  # (1)!
... ).plot().show()
```

```python
"-180 days"
```

```python
offset_space
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     split=0.5  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index, 
...     length=360,
...     split=0.5  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index,
...     length=360,
...     split=0.5,
...     offset_anchor_set=None  # (1)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_rolling(
...     data.index,
...     length=360,
...     split=0.5,
...     offset_anchor_set=None  # (1)!
... ).plot().show()
```

```python
set_0 + set_1
```

```python
set_0
```

Another popular approach is by dividing the entire period into n equally-spaced, potentially-overlapping windows, which is implemented by Splitter.from_n_rolling. If the length of the window is None (i.e., not provided), it simply calls Splitter.from_rolling with the length set to len(index) // n. Note that in contrast to the previous method, this one doesn't allow us to control the offset.

```python
n
```

```python
None
```

```python
len(index) // n
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=5,
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=5,
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=3,  # (1)!
...     length=360,
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=3,  # (1)!
...     length=360,
...     split=0.5
... ).plot().show()
```

```python
n
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=7,  # (1)!
...     length=360,
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_rolling(
...     data.index,
...     n=7,  # (1)!
...     length=360,
...     split=0.5
... ).plot().show()
```

```python
n
```

The windows that we've generated above have all the same length, which makes it easier to conduct fair experiments in backtesting. But sometimes, especially when training ML models, we need every training period to incorporate all the previous history. Such windows are called expanding and can be generated automatically using Splitter.from_expanding, which works similarly to its rolling counterpart except that the offset controls the number of windows, the offset anchor is always the end of the previous split (window), and there's an argument min_length for a minimum window length. There's also a method Splitter.from_n_expanding that allows us to generate a predefined number of expanding windows.

```python
min_length
```

```python
>>> vbt.Splitter.from_expanding(
...     data.index, 
...     min_length=360,  # (1)!
...     offset=180,  # (2)!
...     split=-180  # (3)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_expanding(
...     data.index, 
...     min_length=360,  # (1)!
...     offset=180,  # (2)!
...     split=-180  # (3)!
... ).plot().show()
```

```python
length
```

```python
offset
```

```python
>>> vbt.Splitter.from_n_expanding(
...     data.index, 
...     n=5,
...     min_length=360,
...     split=-180
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_expanding(
...     data.index, 
...     n=5,
...     min_length=360,
...     split=-180
... ).plot().show()
```

### Anchored¶

Let's consider a scenario where we want to generate a set of one-year long splits. Using any of the approaches above, we would get splits that last for one year but most likely start in somewhere in the middle of the year. But what if our requirement was for each split to start exactly at the beginning of the year? Such time anchors are only possible by grouping or resampling. There are two class methods in Splitter that enable this behavior: Splitter.from_ranges and Splitter.from_grouper.

The first method uses get_index_ranges to translate a user query into a set of start and end indices. It allows us to provide custom start and end dates, to resample using a lookback period, to select a time range within each day, and more, - just like resampling but on steroids

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Y",  # (1)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Y",  # (1)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Q",  # (1)!
...     lookback_period="Y",  # (2)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Q",  # (1)!
...     lookback_period="Y",  # (2)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Q",
...     lookback_period="Y",
...     split=(  # (1)!
...         vbt.RepEval("index.month != index.month[-1]"),  # (2)!
...         vbt.RepEval("index.month == index.month[-1]")
...     )
... ).plot().show()
```

```python
>>> vbt.Splitter.from_ranges(
...     data.index,
...     every="Q",
...     lookback_period="Y",
...     split=(  # (1)!
...         vbt.RepEval("index.month != index.month[-1]"),  # (2)!
...         vbt.RepEval("index.month == index.month[-1]")
...     )
... ).plot().show()
```

```python
index
```

```python
>>> def qyear(index):  # (1)!
...     return index.to_period("Q")

>>> vbt.Splitter.from_ranges(
...     data.index,
...     start=0,
...     fixed_start=True,  # (2)!
...     every="Q",
...     split=(
...         lambda index: qyear(index) != qyear(index)[-1],  # (3)!
...         lambda index: qyear(index) == qyear(index)[-1]  # (4)!
...     )
... ).plot().show()
```

```python
>>> def qyear(index):  # (1)!
...     return index.to_period("Q")

>>> vbt.Splitter.from_ranges(
...     data.index,
...     start=0,
...     fixed_start=True,  # (2)!
...     every="Q",
...     split=(
...         lambda index: qyear(index) != qyear(index)[-1],  # (3)!
...         lambda index: qyear(index) == qyear(index)[-1]  # (4)!
...     )
... ).plot().show()
```

The second method takes a grouping or resampling instruction and converts each group into a split. It's based on the method BaseIDXAccessor.get_grouper, and accepts a variety of formats from both vectorbt and Pandas, even pandas.Grouper and pandas.Resampler. The only issue that we may encounter are incomplete splits, which can be filtered out using a template provided as split_check_template and forwarded down to Splitter.from_splits.

```python
split_check_template
```

```python
>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",  # (1)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",  # (1)!
...     split=0.5
... ).plot().show()
```

```python
>>> def is_split_complete(index, split):  # (1)!
...     first_range = split[0]
...     first_index = index[first_range][0]
...     last_range = split[-1]
...     last_index = index[last_range][-1]
...     return first_index.is_year_start and last_index.is_year_end

>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete)  # (2)!
... ).plot().show()
```

```python
>>> def is_split_complete(index, split):  # (1)!
...     first_range = split[0]
...     first_index = index[first_range][0]
...     last_range = split[-1]
...     last_index = index[last_range][-1]
...     return first_index.is_year_start and last_index.is_year_end

>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete)  # (2)!
... ).plot().show()
```

```python
index
```

```python
split
```

```python
is_split_complete
```

```python
>>> def format_split_labels(index, splits_arr):  # (1)!
...     years = map(lambda x: index[x[0]][0].year, splits_arr)  # (2)!
...     return pd.Index(years, name="split_year")

>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete),
...     split_labels=vbt.RepFunc(format_split_labels)  # (3)!
... ).plot().show()
```

```python
>>> def format_split_labels(index, splits_arr):  # (1)!
...     years = map(lambda x: index[x[0]][0].year, splits_arr)  # (2)!
...     return pd.Index(years, name="split_year")

>>> vbt.Splitter.from_grouper(
...     data.index,
...     by="Y",
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete),
...     split_labels=vbt.RepFunc(format_split_labels)  # (3)!
... ).plot().show()
```

```python
index
```

```python
splits_arr
```

```python
>>> vbt.Splitter.from_grouper(
...     data.index,
...     by=data.index.year,  # (1)!
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete)  # (2)!
... ).plot().show()
```

```python
>>> vbt.Splitter.from_grouper(
...     data.index,
...     by=data.index.year,  # (1)!
...     split=0.5,
...     split_check_template=vbt.RepFunc(is_split_complete)  # (2)!
... ).plot().show()
```

### Random¶

So far, we generated windows based on some pre-defined schema. But there is also a special place for randomness in CV, especially when it comes to bootstrapping and block bootstrap in particular. The method Splitter.from_n_random draws a predefined number of windows of a (optionally) variable length. At the heart of this method are two callbacks: length_choice_func and start_choice_func, selecting the next window's length and start point respectively. By default, they are set to numpy.random.Generator.choice, which generates a random sample with replacement (i.e., the same window can occur more than once). Another two callbacks, length_p_func and start_p_func, can control the probabilities of picking each entry (e.g., to select more windows to the right of the period).

```python
length_choice_func
```

```python
start_choice_func
```

```python
length_p_func
```

```python
start_p_func
```

```python
>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=360,  # (1)!
...     seed=42,  # (2)!
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=360,  # (1)!
...     seed=42,  # (2)!
...     split=0.5
... ).plot().show()
```

```python
max_length
```

```python
>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=30,
...     max_length=300,
...     seed=42,
...     split=0.5
... ).plot().show()
```

```python
>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=30,
...     max_length=300,
...     seed=42,
...     split=0.5
... ).plot().show()
```

```python
>>> def start_p_func(i, indices):  # (1)!
...     return indices / indices.sum()

>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=30,
...     max_length=300,
...     seed=42,
...     start_p_func=start_p_func,
...     split=0.5
... ).plot().show()
```

```python
>>> def start_p_func(i, indices):  # (1)!
...     return indices / indices.sum()

>>> vbt.Splitter.from_n_random(
...     data.index,
...     n=50,
...     min_length=30,
...     max_length=300,
...     seed=42,
...     start_p_func=start_p_func,
...     split=0.5
... ).plot().show()
```

### Scikit-learn¶

For k-fold and many other standard CV schemes where scikit-learn has an upper hand, there is a method Splitter.from_sklearn that can parse just about every cross-validator that subclasses the scikit-learn's BaseCrossValidator class.

```python
BaseCrossValidator
```

```python
>>> from sklearn.model_selection import KFold

>>> vbt.Splitter.from_sklearn(
...     data.index, 
...     KFold(n_splits=5)
... ).plot().show()
```

```python
>>> from sklearn.model_selection import KFold

>>> vbt.Splitter.from_sklearn(
...     data.index, 
...     KFold(n_splits=5)
... ).plot().show()
```

Warning

There is a temporal dependency between observations: it makes no sense to use the values from the future to forecast values in the past, thus make sure that the test period always succeeds the training period.

### Dynamic¶

The final and the most flexible generation method involves calling a UDF that takes a context including all the splits generated previously, and returns a new split to be appended. This all happens in an infinite while-loop; to break out of the loop, the UDF must return None. Similarly to many other methods in vectorbt that take functions as arguments, this one also makes use of templates to substitute them for information from the context. The context itself includes the appended and resolved splits (splits) and the bounds of each range in each split (but only when fix_ranges=True), which makes the generation process a child's play.

```python
None
```

```python
splits
```

```python
fix_ranges=True
```

```python
>>> def split_func(index, prev_start):
...     if prev_start is None:  # (1)!
...         prev_start = index[0]
...     new_start = prev_start + pd.offsets.MonthBegin(1)  # (2)!
...     new_end = new_start + pd.DateOffset(years=1)  # (3)!
...     if new_end > index[-1] + index.freq:  # (4)!
...         return None
...     return [
...         slice(new_start, new_start + pd.offsets.MonthBegin(9)),  # (5)!
...         slice(new_start + pd.offsets.MonthBegin(9), new_end)
...     ]

>>> vbt.Splitter.from_split_func(
...     data.index,
...     split_func=split_func,
...     split_args=(vbt.Rep("index"), vbt.Rep("prev_start")),  # (6)!
...     range_bounds_kwargs=dict(index_bounds=True)  # (7)!
... ).plot().show()
```

```python
>>> def split_func(index, prev_start):
...     if prev_start is None:  # (1)!
...         prev_start = index[0]
...     new_start = prev_start + pd.offsets.MonthBegin(1)  # (2)!
...     new_end = new_start + pd.DateOffset(years=1)  # (3)!
...     if new_end > index[-1] + index.freq:  # (4)!
...         return None
...     return [
...         slice(new_start, new_start + pd.offsets.MonthBegin(9)),  # (5)!
...         slice(new_start + pd.offsets.MonthBegin(9), new_end)
...     ]

>>> vbt.Splitter.from_split_func(
...     data.index,
...     split_func=split_func,
...     split_args=(vbt.Rep("index"), vbt.Rep("prev_start")),  # (6)!
...     range_bounds_kwargs=dict(index_bounds=True)  # (7)!
... ).plot().show()
```

```python
prev_start
```

```python
None
```

```python
prev_start
```

```python
>>> def get_next_monday(from_date):
...     if from_date.weekday == 0 and from_date.ceil("H").hour <= 9:  # (1)!
...         return from_date.floor("D")
...     return from_date.floor("D") + pd.offsets.Week(n=0, weekday=0)  # (2)!

>>> def get_next_business_range(from_date):
...     monday_0000 = get_next_monday(from_date)
...     monday_0900 = monday_0000 + pd.DateOffset(hours=9)  # (3)!
...     friday_1700 = monday_0900 + pd.DateOffset(days=4, hours=8)  # (4)!
...     return slice(monday_0900, friday_1700)

>>> def split_func(index, bounds):
...     if len(bounds) == 0:
...         from_date = index[0]  # (5)!
...     else:
...         from_date = bounds[-1][1][0]  # (6)!
...     train_range = get_next_business_range(from_date)
...     test_range = get_next_business_range(train_range.stop)
...     if test_range.stop > index[-1] + index.freq:
...         return None
...     return train_range, test_range

>>> vbt.Splitter.from_split_func(
...     vbt.date_range("2020-01", "2020-03", freq="15min"),  # (7)!
...     split_func=split_func,
...     split_args=(vbt.Rep("index"), vbt.Rep("bounds")),
...     range_bounds_kwargs=dict(index_bounds=True)
... ).plot().show()
```

```python
>>> def get_next_monday(from_date):
...     if from_date.weekday == 0 and from_date.ceil("H").hour <= 9:  # (1)!
...         return from_date.floor("D")
...     return from_date.floor("D") + pd.offsets.Week(n=0, weekday=0)  # (2)!

>>> def get_next_business_range(from_date):
...     monday_0000 = get_next_monday(from_date)
...     monday_0900 = monday_0000 + pd.DateOffset(hours=9)  # (3)!
...     friday_1700 = monday_0900 + pd.DateOffset(days=4, hours=8)  # (4)!
...     return slice(monday_0900, friday_1700)

>>> def split_func(index, bounds):
...     if len(bounds) == 0:
...         from_date = index[0]  # (5)!
...     else:
...         from_date = bounds[-1][1][0]  # (6)!
...     train_range = get_next_business_range(from_date)
...     test_range = get_next_business_range(train_range.stop)
...     if test_range.stop > index[-1] + index.freq:
...         return None
...     return train_range, test_range

>>> vbt.Splitter.from_split_func(
...     vbt.date_range("2020-01", "2020-03", freq="15min"),  # (7)!
...     split_func=split_func,
...     split_args=(vbt.Rep("index"), vbt.Rep("bounds")),
...     range_bounds_kwargs=dict(index_bounds=True)
... ).plot().show()
```

```python
from_date
```

## Validation¶

Consider the following splitter that splits the entire period into years, one per split:

```python
>>> splitter = vbt.Splitter.from_ranges(
...     data.index,
...     every="Y",
...     split=0.5,
...     set_labels=["IS", "OOS"]
... )
>>> splitter.plot().show()
```

```python
>>> splitter = vbt.Splitter.from_ranges(
...     data.index,
...     every="Y",
...     split=0.5,
...     set_labels=["IS", "OOS"]
... )
>>> splitter.plot().show()
```

By default, closed_end is False such that any neighboring ranges do not overlap, but we deliberately made a mistake here by setting closed_end to True, hence we've produced splits that overlap by exactly one bar. How do we detect such a mistake post-factum? The Splitter class has various tools exactly for that.

```python
closed_end
```

```python
False
```

```python
closed_end
```

```python
True
```

### Bounds¶

The first tool involves computing the bounds of each range. The bounds consist of two numbers (index_bounds=False) or dates (index_bounds=True): the start (always inclusive) and the end (exclusive, but can be inclusive using right_inclusive=True). Depending on what the analysis goal is, they can be returned in two different formats. The first format is a three-dimensional NumPy array returned by Splitter.get_bounds_arr where the first axis are splits, the second axis are sets, and the third axis are bounds:

```python
index_bounds=False
```

```python
index_bounds=True
```

```python
right_inclusive=True
```

```python
>>> bounds_arr = splitter.get_bounds_arr()  # (1)!
>>> bounds_arr.shape
(4, 2, 2)

>>> bounds_arr
[[[ 137  320]
  [ 320  503]]
 [[ 502  685]
  [ 685  868]]
 [[ 867 1050]
  [1050 1234]]
 [[1233 1416]
  [1416 1599]]]
```

```python
>>> bounds_arr = splitter.get_bounds_arr()  # (1)!
>>> bounds_arr.shape
(4, 2, 2)

>>> bounds_arr
[[[ 137  320]
  [ 320  503]]
 [[ 502  685]
  [ 685  868]]
 [[ 867 1050]
  [1050 1234]]
 [[1233 1416]
  [1416 1599]]]
```

Another, probably a user-friendlier, format is a DataFrame returned by Splitter.get_bounds where rows represent ranges and columns represent bounds:

```python
>>> bounds = splitter.get_bounds(index_bounds=True)  # (1)!
>>> bounds.shape
(8, 2)

>>> bounds
bound                         start                       end
split set                                                    
0     IS  2018-01-01 00:00:00+00:00 2018-07-03 00:00:00+00:00
      OOS 2018-07-03 00:00:00+00:00 2019-01-02 00:00:00+00:00
1     IS  2019-01-01 00:00:00+00:00 2019-07-03 00:00:00+00:00
      OOS 2019-07-03 00:00:00+00:00 2020-01-02 00:00:00+00:00
2     IS  2020-01-01 00:00:00+00:00 2020-07-02 00:00:00+00:00
      OOS 2020-07-02 00:00:00+00:00 2021-01-02 00:00:00+00:00
3     IS  2021-01-01 00:00:00+00:00 2021-07-03 00:00:00+00:00
      OOS 2021-07-03 00:00:00+00:00 2022-01-02 00:00:00+00:00
```

```python
>>> bounds = splitter.get_bounds(index_bounds=True)  # (1)!
>>> bounds.shape
(8, 2)

>>> bounds
bound                         start                       end
split set                                                    
0     IS  2018-01-01 00:00:00+00:00 2018-07-03 00:00:00+00:00
      OOS 2018-07-03 00:00:00+00:00 2019-01-02 00:00:00+00:00
1     IS  2019-01-01 00:00:00+00:00 2019-07-03 00:00:00+00:00
      OOS 2019-07-03 00:00:00+00:00 2020-01-02 00:00:00+00:00
2     IS  2020-01-01 00:00:00+00:00 2020-07-02 00:00:00+00:00
      OOS 2020-07-02 00:00:00+00:00 2021-01-02 00:00:00+00:00
3     IS  2021-01-01 00:00:00+00:00 2021-07-03 00:00:00+00:00
      OOS 2021-07-03 00:00:00+00:00 2022-01-02 00:00:00+00:00
```

We can then detect some IS ranges starting before the preceding OOS range ends:

```python
>>> bounds.loc[(0, "OOS"), "end"]
Timestamp('2019-01-02 00:00:00+0000', tz='UTC')

>>> bounds.loc[(1, "IS"), "start"]
Timestamp('2019-01-01 00:00:00+0000', tz='UTC')
```

```python
>>> bounds.loc[(0, "OOS"), "end"]
Timestamp('2019-01-02 00:00:00+0000', tz='UTC')

>>> bounds.loc[(1, "IS"), "start"]
Timestamp('2019-01-01 00:00:00+0000', tz='UTC')
```

### Masks¶

Another tool revolves around range masks. Since the index is the same for each single range, we can translate a range into a mask of the same length as the index and then stack all masks into a single array. Another advantage of masks is that we can combine them, apply logical operators on them, reduce them, and test them for True values. The only downside is clearly their memory consumption: as we calculated previously, 100 splits and 2 sets of 1 year of 1-minute data would consume whooping 100MB of RAM. Similarly to bounds, we have two methods that return a three-dimensional NumPy array and a DataFrame respectively: Splitter.get_mask_arr and Splitter.get_mask. Let's get the mask in the DataFrame format:

```python
True
```

```python
>>> mask = splitter.get_mask()  # (1)!
>>> mask.shape
(1902, 8)

>>> mask
split                          0             1             2             3  \
set                           IS    OOS     IS    OOS     IS    OOS     IS   
Open time                                                                    
2017-08-17 00:00:00+00:00  False  False  False  False  False  False  False   
2017-08-18 00:00:00+00:00  False  False  False  False  False  False  False   
2017-08-19 00:00:00+00:00  False  False  False  False  False  False  False   
...                          ...    ...    ...    ...    ...    ...    ...   
2022-10-29 00:00:00+00:00  False  False  False  False  False  False  False   
2022-10-30 00:00:00+00:00  False  False  False  False  False  False  False   
2022-10-31 00:00:00+00:00  False  False  False  False  False  False  False   

split                             
set                          OOS  
Open time                         
2017-08-17 00:00:00+00:00  False  
2017-08-18 00:00:00+00:00  False  
2017-08-19 00:00:00+00:00  False  
...                          ...  
2022-10-29 00:00:00+00:00  False  
2022-10-30 00:00:00+00:00  False  
2022-10-31 00:00:00+00:00  False  

[1902 rows x 8 columns]
```

```python
>>> mask = splitter.get_mask()  # (1)!
>>> mask.shape
(1902, 8)

>>> mask
split                          0             1             2             3  \
set                           IS    OOS     IS    OOS     IS    OOS     IS   
Open time                                                                    
2017-08-17 00:00:00+00:00  False  False  False  False  False  False  False   
2017-08-18 00:00:00+00:00  False  False  False  False  False  False  False   
2017-08-19 00:00:00+00:00  False  False  False  False  False  False  False   
...                          ...    ...    ...    ...    ...    ...    ...   
2022-10-29 00:00:00+00:00  False  False  False  False  False  False  False   
2022-10-30 00:00:00+00:00  False  False  False  False  False  False  False   
2022-10-31 00:00:00+00:00  False  False  False  False  False  False  False   

split                             
set                          OOS  
Open time                         
2017-08-17 00:00:00+00:00  False  
2017-08-18 00:00:00+00:00  False  
2017-08-19 00:00:00+00:00  False  
...                          ...  
2022-10-29 00:00:00+00:00  False  
2022-10-30 00:00:00+00:00  False  
2022-10-31 00:00:00+00:00  False  

[1902 rows x 8 columns]
```

As opposed to bounds, the DataFrame lists split and set labels (i.e., range labels) in columns rather than rows. To demonstrate the power of masks, let's answer the following question: what ranges cover the year 2021?

```python
>>> mask["2021":"2021"].any()
split  set
0      IS     False
       OOS    False
1      IS     False
       OOS    False
2      IS     False
       OOS     True
3      IS      True
       OOS     True
dtype: bool
```

```python
>>> mask["2021":"2021"].any()
split  set
0      IS     False
       OOS    False
1      IS     False
       OOS    False
2      IS     False
       OOS     True
3      IS      True
       OOS     True
dtype: bool
```

We can spot the mistake once again: there's an OOS range that clearly overflows into the next year. Here's another question: get the number of dates covered by each set in each year.

```python
>>> mask.resample(vbt.offset("Y")).sum()  # (1)!
split                        0         1         2         3     
set                         IS  OOS   IS  OOS   IS  OOS   IS  OOS
Open time                                                        
2017-01-01 00:00:00+00:00    0    0    0    0    0    0    0    0
2018-01-01 00:00:00+00:00  183  182    0    0    0    0    0    0
2019-01-01 00:00:00+00:00    0    1  183  182    0    0    0    0
2020-01-01 00:00:00+00:00    0    0    0    1  183  183    0    0
2021-01-01 00:00:00+00:00    0    0    0    0    0    1  183  182
2022-01-01 00:00:00+00:00    0    0    0    0    0    0    0    1
```

```python
>>> mask.resample(vbt.offset("Y")).sum()  # (1)!
split                        0         1         2         3     
set                         IS  OOS   IS  OOS   IS  OOS   IS  OOS
Open time                                                        
2017-01-01 00:00:00+00:00    0    0    0    0    0    0    0    0
2018-01-01 00:00:00+00:00  183  182    0    0    0    0    0    0
2019-01-01 00:00:00+00:00    0    1  183  182    0    0    0    0
2020-01-01 00:00:00+00:00    0    0    0    1  183  183    0    0
2021-01-01 00:00:00+00:00    0    0    0    0    0    1  183  182
2022-01-01 00:00:00+00:00    0    0    0    0    0    0    0    1
```

To mitigate potential memory issues, there are special approaches that translate only a subset of ranges into a mask at a time. Those approaches are based on two iteration schemas: by split and by set, implemented by Splitter.get_iter_split_masks and Splitter.get_iter_set_masks respectively. Each method returns a Python generator that can be iterated over in a loop. Let's answer the question above in a memory-friendly manner (if really needed):

```python
>>> results = []
>>> for mask in splitter.get_iter_split_masks():
...     results.append(mask.resample(vbt.offset("Y")).sum())
>>> pd.concat(results, axis=1, keys=splitter.split_labels)
split                        0         1         2         3     
set                         IS  OOS   IS  OOS   IS  OOS   IS  OOS
Open time                                                        
2017-01-01 00:00:00+00:00    0    0    0    0    0    0    0    0
2018-01-01 00:00:00+00:00  183  182    0    0    0    0    0    0
2019-01-01 00:00:00+00:00    0    1  183  182    0    0    0    0
2020-01-01 00:00:00+00:00    0    0    0    1  183  183    0    0
2021-01-01 00:00:00+00:00    0    0    0    0    0    1  183  182
2022-01-01 00:00:00+00:00    0    0    0    0    0    0    0    1
```

```python
>>> results = []
>>> for mask in splitter.get_iter_split_masks():
...     results.append(mask.resample(vbt.offset("Y")).sum())
>>> pd.concat(results, axis=1, keys=splitter.split_labels)
split                        0         1         2         3     
set                         IS  OOS   IS  OOS   IS  OOS   IS  OOS
Open time                                                        
2017-01-01 00:00:00+00:00    0    0    0    0    0    0    0    0
2018-01-01 00:00:00+00:00  183  182    0    0    0    0    0    0
2019-01-01 00:00:00+00:00    0    1  183  182    0    0    0    0
2020-01-01 00:00:00+00:00    0    0    0    1  183  183    0    0
2021-01-01 00:00:00+00:00    0    0    0    0    0    1  183  182
2022-01-01 00:00:00+00:00    0    0    0    0    0    0    0    1
```

### Coverage¶

Bounds and masks are convenient range formats that enable us to analyze ranges from various perspectives. To not delegate too much work to the user, there are additional methods that automate this analysis. Since we're mostly interested in knowing whether and how much do splits, sets, and ranges overlap, there are 4 methods that provide us with quick insights into that matter: Splitter.get_split_coverage for the coverage by split, Splitter.get_set_coverage for the coverage by set, Splitter.get_range_coverage for the coverage by range, and Splitter.get_coverage for the coverage by any of the above. For example, the split-relative coverage will return the % of bars in the index covered by each split, while the total coverage will return the % of bars in the index covered by any range.

Warning

Most of the methods below, apart from the plotting method, require the entire mask array to be in memory.

```python
>>> splitter.get_split_coverage()  # (1)!
split
0    0.192429
1    0.192429
2    0.192955
3    0.192429
Name: split_coverage, dtype: float64

>>> splitter.get_set_coverage()  # (2)!
set
IS     0.384858
OOS    0.385384
Name: set_coverage, dtype: float64

>>> splitter.get_range_coverage()  # (3)!
split  set
0      IS     0.096215
       OOS    0.096215
1      IS     0.096215
       OOS    0.096215
2      IS     0.096215
       OOS    0.096740
3      IS     0.096215
       OOS    0.096215
Name: range_coverage, dtype: float64

>>> splitter.get_coverage()  # (4)!
0.768664563617245
```

```python
>>> splitter.get_split_coverage()  # (1)!
split
0    0.192429
1    0.192429
2    0.192955
3    0.192429
Name: split_coverage, dtype: float64

>>> splitter.get_set_coverage()  # (2)!
set
IS     0.384858
OOS    0.385384
Name: set_coverage, dtype: float64

>>> splitter.get_range_coverage()  # (3)!
split  set
0      IS     0.096215
       OOS    0.096215
1      IS     0.096215
       OOS    0.096215
2      IS     0.096215
       OOS    0.096740
3      IS     0.096215
       OOS    0.096215
Name: range_coverage, dtype: float64

>>> splitter.get_coverage()  # (4)!
0.768664563617245
```

Note

The default arguments will always return a metric relative to the length of the index.

As we can see above, the first split covers 19.24% of the entire period, while both ranges in that split take 9.62%, or exactly 50% of the split. Both sets cover roughly the same period of time - 38.53% of the entire period. The last metric tells us that 23.14% of the entire period aren't covered by the splitter, which makes sense because the years 2017 and 2022 are incomplete such that no split was produced for either of them. Finally, why do all ranges cover the same period of time except the OOS set in the split 2? It's because that year was a leap year such that the last months had one day more than the same months in other years:

```python
2
```

```python
>>> splitter.index_bounds.loc[(2, "OOS"), "start"].is_leap_year
True
```

```python
>>> splitter.index_bounds.loc[(2, "OOS"), "start"].is_leap_year
True
```

So far, we analyzed coverage in relation to the full index. But is a special argument relative that allows us to analyze splits and sets relatively to the total coverage, as well as ranges relatively to the split coverage. For example, let's get the fraction of IS and OOS sets in their respective splits:

```python
relative
```

```python
>>> splitter.get_range_coverage(relative=True)
split  set
0      IS     0.500000
       OOS    0.500000
1      IS     0.500000
       OOS    0.500000
2      IS     0.498638
       OOS    0.501362
3      IS     0.500000
       OOS    0.500000
Name: range_coverage, dtype: float64
```

```python
>>> splitter.get_range_coverage(relative=True)
split  set
0      IS     0.500000
       OOS    0.500000
1      IS     0.500000
       OOS    0.500000
2      IS     0.498638
       OOS    0.501362
3      IS     0.500000
       OOS    0.500000
Name: range_coverage, dtype: float64
```

Most periods except the leap year have a perfect 50/50 split, as we wanted. We can expand our analysis to answer whether sets in general follow the same 50/50 split:

```python
>>> splitter.get_set_coverage(relative=True)
set
IS     0.500684
OOS    0.501368
Name: set_coverage, dtype: float64
```

```python
>>> splitter.get_set_coverage(relative=True)
set
IS     0.500684
OOS    0.501368
Name: set_coverage, dtype: float64
```

Both numbers do not sum to one, which (again) hints at overlapping ranges. By using the argument overlapping, we can assess any overlaps of sets within each split, any overlaps of splits within each set, and any overlaps of ranges globally:

```python
overlapping
```

```python
>>> splitter.get_split_coverage(overlapping=True)
split
0    0.0
1    0.0
2    0.0
3    0.0
Name: split_coverage, dtype: float64

>>> splitter.get_set_coverage(overlapping=True)
set
IS     0.0
OOS    0.0
Name: set_coverage, dtype: float64

>>> splitter.get_coverage(overlapping=True)
0.002051983584131327
```

```python
>>> splitter.get_split_coverage(overlapping=True)
split
0    0.0
1    0.0
2    0.0
3    0.0
Name: split_coverage, dtype: float64

>>> splitter.get_set_coverage(overlapping=True)
set
IS     0.0
OOS    0.0
Name: set_coverage, dtype: float64

>>> splitter.get_coverage(overlapping=True)
0.002051983584131327
```

Ranges neither overlap within each split, nor within each set. But there are still some overlaps of ranges recorded globally, which means they belong to opposite splits and sets. To have a better look, let's visualize the coverage using Splitter.plot_coverage:

```python
>>> splitter.plot_coverage().show()
```

```python
>>> splitter.plot_coverage().show()
```

The Y-axis represents the total number of ranges that cover one particular date. We can see that there are three dates covered by two ranges simultaneously.

The last and the most powerful tool in overlap detection are overlap matrices, which compute overlaps either between splits (by="split"), sets (by="set"), or ranges (by="range"). If normalize is True, which is the default value, the intersection of two range masks will be normalized by their union. Even though the operation is Numba-compiled, it's still very expensive: 100 splits and 2 sets would require going through all 200 * 200 = 40000 range pairs to build the range overlap matrix. Let's finally shed some light on the overlapping ranges:

```python
by="split"
```

```python
by="set"
```

```python
by="range"
```

```python
normalize
```

```python
200 * 200 = 40000
```

```python
>>> splitter.get_overlap_matrix(by="range", normalize=False)  # (1)!
split        0         1         2         3     
set         IS  OOS   IS  OOS   IS  OOS   IS  OOS
split set                                        
0     IS   183    0    0    0    0    0    0    0
      OOS    0  183    1    0    0    0    0    0
1     IS     0    1  183    0    0    0    0    0
      OOS    0    0    0  183    1    0    0    0
2     IS     0    0    0    1  183    0    0    0
      OOS    0    0    0    0    0  184    1    0
3     IS     0    0    0    0    0    1  183    0
      OOS    0    0    0    0    0    0    0  183
```

```python
>>> splitter.get_overlap_matrix(by="range", normalize=False)  # (1)!
split        0         1         2         3     
set         IS  OOS   IS  OOS   IS  OOS   IS  OOS
split set                                        
0     IS   183    0    0    0    0    0    0    0
      OOS    0  183    1    0    0    0    0    0
1     IS     0    1  183    0    0    0    0    0
      OOS    0    0    0  183    1    0    0    0
2     IS     0    0    0    1  183    0    0    0
      OOS    0    0    0    0    0  184    1    0
3     IS     0    0    0    0    0    1  183    0
      OOS    0    0    0    0    0    0    0  183
```

### Grouping¶

Each of the methods above (and many others including Splitter.plot) accept the arguments split_group_by and set_group_by, which allow grouping splits and sets respectively. Their format is identical to the format of the argument group_by, which appears just about everywhere in the vectorbt's codebase. For example, by passing True, we can put all ranges into the same bucket and merge them. We can also pass a list of the same length as the number of splits/sets such that the splits/sets under the same unique value will get merged. The actual merging part is being done by the method Splitter.merge_split.

```python
split_group_by
```

```python
set_group_by
```

```python
group_by
```

```python
True
```

For example, let's get the bounds of each entire split:

```python
>>> splitter.get_bounds(index_bounds=True, set_group_by=True)
bound                               start                       end
split set_group                                                    
0     group     2018-01-01 00:00:00+00:00 2019-01-02 00:00:00+00:00
1     group     2019-01-01 00:00:00+00:00 2020-01-02 00:00:00+00:00
2     group     2020-01-01 00:00:00+00:00 2021-01-02 00:00:00+00:00
3     group     2021-01-01 00:00:00+00:00 2022-01-02 00:00:00+00:00
```

```python
>>> splitter.get_bounds(index_bounds=True, set_group_by=True)
bound                               start                       end
split set_group                                                    
0     group     2018-01-01 00:00:00+00:00 2019-01-02 00:00:00+00:00
1     group     2019-01-01 00:00:00+00:00 2020-01-02 00:00:00+00:00
2     group     2020-01-01 00:00:00+00:00 2021-01-02 00:00:00+00:00
3     group     2021-01-01 00:00:00+00:00 2022-01-02 00:00:00+00:00
```

This makes certain kinds of analysis much easier

## Manipulation¶

We'll end this page with an overview of the methods that can be used to change a splitter. Let's build a splitter that has just one set representing the current year:

```python
>>> splitter = vbt.Splitter.from_grouper(
...     data.index, 
...     by=data.index.year.rename("split_year")  # (1)!
... )
```

```python
>>> splitter = vbt.Splitter.from_grouper(
...     data.index, 
...     by=data.index.year.rename("split_year")  # (1)!
... )
```

Since the class Splitter subclasses the class Analyzable, we can get a quick and nice overview of the most important metrics and plots:

```python
>>> splitter.stats()
Index Start             2017-08-17 00:00:00+00:00
Index End               2022-10-31 00:00:00+00:00
Index Length                                 1902
Splits                                          6
Sets                                            1
Coverage [%]                                100.0
Overlap Coverage [%]                          0.0
Name: agg_stats, dtype: object

>>> splitter.plots().show()
```

```python
>>> splitter.stats()
Index Start             2017-08-17 00:00:00+00:00
Index End               2022-10-31 00:00:00+00:00
Index Length                                 1902
Splits                                          6
Sets                                            1
Coverage [%]                                100.0
Overlap Coverage [%]                          0.0
Name: agg_stats, dtype: object

>>> splitter.plots().show()
```

Info

Since we have only one split, most metrics were hidden from the statistics.

As we already know, we can select specific splits and sets using regular Pandas indexing (which is another great property of Analyzable). Since we're not interested in incomplete years, let's remove the first and the last split:

```python
>>> splitter = splitter.iloc[1:-1]
>>> splitter.stats()
Index Start             2017-08-17 00:00:00+00:00
Index End               2022-10-31 00:00:00+00:00
Index Length                                 1902
Splits                                          4  << changed
Sets                                            1
Coverage [%]                             76.81388  << changed
Overlap Coverage [%]                          0.0
Name: agg_stats, dtype: object
```

```python
>>> splitter = splitter.iloc[1:-1]
>>> splitter.stats()
Index Start             2017-08-17 00:00:00+00:00
Index End               2022-10-31 00:00:00+00:00
Index Length                                 1902
Splits                                          4  << changed
Sets                                            1
Coverage [%]                             76.81388  << changed
Overlap Coverage [%]                          0.0
Name: agg_stats, dtype: object
```

Now, let's split the only set into three: a train set covering the first two quarters, a validation set covering the third one, and a test set covering the last one. This is possible thanks to the method Splitter.split_set, which takes the split specification and the labels of the new split as new_split and new_set_labels respectively. We'll use a function template to divide the set:

```python
new_split
```

```python
new_set_labels
```

```python
>>> def new_split(index):  # (1)!
...     return [
...         np.isin(index.quarter, [1, 2]),  # (2)!
...         index.quarter == 3, 
...         index.quarter == 4
...     ]

>>> splitter = splitter.split_set(
...     vbt.RepFunc(new_split),  # (3)!
...     new_set_labels=["train", "valid", "test"]  # (4)!
... )
```

```python
>>> def new_split(index):  # (1)!
...     return [
...         np.isin(index.quarter, [1, 2]),  # (2)!
...         index.quarter == 3, 
...         index.quarter == 4
...     ]

>>> splitter = splitter.split_set(
...     vbt.RepFunc(new_split),  # (3)!
...     new_set_labels=["train", "valid", "test"]  # (4)!
... )
```

```python
index
```

```python
index.quarter
```

Info

Each operation on a splitter returns a new splitter: no information is changed in place to not mess up with caching and to keep the splitter (as any other vectorbt object) side effect free.

Let's take a look at the new splitter:

```python
>>> splitter.stats()
Index Start                     2017-08-17 00:00:00+00:00
Index End                       2022-10-31 00:00:00+00:00
Index Length                                         1902
Splits                                                  4
Sets                                                    3
Coverage [%]                                     76.81388
Coverage [%]: train                             38.117771
Coverage [%]: valid                             19.348055
Coverage [%]: test                              19.348055
Mean Rel Coverage [%]: train                    49.623475
Mean Rel Coverage [%]: valid                    25.188263
Mean Rel Coverage [%]: test                     25.188263
Overlap Coverage [%]                                  0.0
Overlap Coverage [%]: train                           0.0
Overlap Coverage [%]: valid                           0.0
Overlap Coverage [%]: test                            0.0
Name: agg_stats, dtype: object

>>> splitter.plots().show()
```

```python
>>> splitter.stats()
Index Start                     2017-08-17 00:00:00+00:00
Index End                       2022-10-31 00:00:00+00:00
Index Length                                         1902
Splits                                                  4
Sets                                                    3
Coverage [%]                                     76.81388
Coverage [%]: train                             38.117771
Coverage [%]: valid                             19.348055
Coverage [%]: test                              19.348055
Mean Rel Coverage [%]: train                    49.623475
Mean Rel Coverage [%]: valid                    25.188263
Mean Rel Coverage [%]: test                     25.188263
Overlap Coverage [%]                                  0.0
Overlap Coverage [%]: train                           0.0
Overlap Coverage [%]: valid                           0.0
Overlap Coverage [%]: test                            0.0
Name: agg_stats, dtype: object

>>> splitter.plots().show()
```

As you might have guessed, there's also a method that can merge multiple sets - Splitter.merge_sets. Your homework is to merge the "valid" and "test" sets into "test"

We did our job perfectly, let's move on to applications!

Python code  Notebook

