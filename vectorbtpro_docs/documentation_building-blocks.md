# Building blocks¶

In what follows, we will look at sub-packages, modules, and especially classes that act as building blocks for more advanced functionalities distributed across vectorbt, such as Portfolio. For illustration, we will gradually build our custom class CorrStats that will let us analyze the correlation between two arrays in the most performant and flexible way

```python
CorrStats
```

## Utilities¶

```python
flowchart TD;
    Config -->|inherits| Pickleable;
    Config -->|inherits| Prettified;
    Configured -->|inherits| Pickleable;
    Configured -->|inherits| Prettified;
    Configured -.->|references| Config;
    AttrResolverMixin;
```

```python
flowchart TD;
    Config -->|inherits| Pickleable;
    Config -->|inherits| Prettified;
    Configured -->|inherits| Pickleable;
    Configured -->|inherits| Prettified;
    Configured -.->|references| Config;
    AttrResolverMixin;
```

(Reload the page if the diagram doesn't show up)

VectorBT® PRO deploys a modular project structure that is composed of a range of subpackages. Each subpackage is applicable to a certain area of analysis.

Subpackage utils contains a set of utilities powering every part of vectorbt  They are loosely connected and provide small but powerful re-usable code snippets that can be used independently of other functionality.

Info

The main reason why we don't import third-party packages but implement many utilities from scratch is because we want to retain full control over execution and code quality.

### Formatting¶

VectorBT® PRO implements its own formatting engine that can pretty-print any Python object. It's far more superior to formatting with JSON because it respects native Python data types and injects some smart formatting logic when it comes to more structured data types, such as np.dtype and namedtuple. Often, we can even convert the string back into Python using eval.

```python
np.dtype
```

```python
namedtuple
```

```python
eval
```

Let's beautify a nested dictionary using prettify and convert the string back into an object:

```python
>>> from vectorbtpro import *

>>> dct = {'planet' : {'has': {'plants': 'yes', 'animals': 'yes', 'cryptonite': 'no'}, 'name': 'Earth'}}
>>> print(vbt.prettify(dct))
{
    'planet': {
        'has': {
            'plants': 'yes',
            'animals': 'yes',
            'cryptonite': 'no'
        },
        'name': 'Earth'
    }
}

>>> eval(vbt.prettify(dct)) == dct
True
```

```python
>>> from vectorbtpro import *

>>> dct = {'planet' : {'has': {'plants': 'yes', 'animals': 'yes', 'cryptonite': 'no'}, 'name': 'Earth'}}
>>> print(vbt.prettify(dct))
{
    'planet': {
        'has': {
            'plants': 'yes',
            'animals': 'yes',
            'cryptonite': 'no'
        },
        'name': 'Earth'
    }
}

>>> eval(vbt.prettify(dct)) == dct
True
```

Hint

Wondering why we used vbt.prettify instead of vbt.utils.formatting.prettify? Any utility that may prove useful to the end user can be accessed directly from vbt.

```python
vbt.prettify
```

```python
vbt.utils.formatting.prettify
```

```python
vbt
```

To see which utilities are accessible from the root of the package, visit vectorbtpro/utils/__init__.py or any other subpackage, and look for the objects that are listed in __all__.

```python
__all__
```

Class Prettified implements the abstract method Prettified.prettify, which a subclass can override to pretty-print an instance using prettify. Read below to learn how instances of various classes can be introspected using this method.

### Pickling¶

Pickling is the process of converting a Python object into a byte stream to store it in a file/database. Class Pickleable enables pickling of objects of any complexity using Dill (or pickle if Dill is not installed). Each of its subclasses inherits ready-to-use methods for serializing, de-serializing, saving to a file, and loading from a file. This is truly amazing because it allows us to persist objects holding any type of data, including instances of Data and Portfolio.

### Configuring¶

VectorBT® PRO heavily relies upon automation based on some sort of specification. The specification for most repetitive tasks is usually stored inside so-called "configs", which act as settings for a certain task, a data structure, or even a class. This makes most places in vectorbt transparent and easily traversable and changeable programmatically.

Class Config is a dictionary on steroids: it extends the Python's dict with various configuration features, such as frozen keys, read-only values, accessing keys via the dot notation, and nested updates. The most notable feature is the ability to reset a config to its initial state and even make checkpoints, which is particularly useful for settings. In addition, since Config inherits Pickleable, we can save any configuration to disk, while subclassing Prettified allows us to beautify it (by the way, this approach is being used to generate the API reference):

```python
dict
```

```python
>>> print(vbt.Records.field_config)
Config(
    dtype=None,
    settings={
        'id': {
            'name': 'id',
            'title': 'Id'
        },
        'col': {
            'name': 'col',
            'title': 'Column',
            'mapping': 'columns'
        },
        'idx': {
            'name': 'idx',
            'title': 'Timestamp',
            'mapping': 'index'
        }
    }
)
```

```python
>>> print(vbt.Records.field_config)
Config(
    dtype=None,
    settings={
        'id': {
            'name': 'id',
            'title': 'Id'
        },
        'col': {
            'name': 'col',
            'title': 'Column',
            'mapping': 'columns'
        },
        'idx': {
            'name': 'idx',
            'title': 'Timestamp',
            'mapping': 'index'
        }
    }
)
```

Configs are very common structures in vectorbt. There are three main types of configs (that either subclass or partially call Config) used throughout vectorbt:

```python
dict
```

```python
vbt.settings.portfolio.log
```

```python
vbt.settings['portfolio']['log']
```

A config can be created like a regular dict. To provide options for this config, use the options_ argument (notice the trailing underscore):

```python
dict
```

```python
options_
```

```python
>>> cfg = vbt.Config(
...     hello="world",
...     options_=dict(readonly=True)
... )
>>> print(cfg)
Config(
    hello='world'
)

>>> cfg["change"] = "something"
TypeError: Config is read-only
```

```python
>>> cfg = vbt.Config(
...     hello="world",
...     options_=dict(readonly=True)
... )
>>> print(cfg)
Config(
    hello='world'
)

>>> cfg["change"] = "something"
TypeError: Config is read-only
```

Apart from the use cases listed above, Config is also being used by the class Configured, which is a base class to most core classes in vectorbt. It's a read-only class that holds a config of type Config with all arguments passed during the initialization. Whenever we initialize any subclass of Configured, any named arguments we passed to the initializer (__init__) are stored inside Configured.config. This way, the created instance is described and managed solely by its config:

```python
__init__
```

```python
new_instance = ConfiguredClass(**old_instance.config)
```

The main requirement for all of this to work properly is immutability. And here we have arrived at the first very important design decision: most classes in vectorbt are meant to be immutable (read-only) and it's discouraged to change any attribute unless it's listed in a special variable called _writeable_attrs. There are multiple reasons why we require immutability:

```python
_writeable_attrs
```

Let's create our custom class that returns some correlation statistics of two arrays. In particular, it will compute the Pearson correlation coefficient and its rolling version using Pandas:

```python
>>> class CorrStats(vbt.Configured):
...     def __init__(self, obj1, obj2):
...         vbt.Configured.__init__(self, obj1=obj1, obj2=obj2)
...
...         self._obj1 = obj1
...         self._obj2 = obj2
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         if isinstance(self.obj1, pd.Series):
...             return self.obj1.corr(self.obj2)
...         return self.obj1.corrwith(self.obj2)
...
...     def rolling_corr(self, window):
...         return self.obj1.rolling(window).corr(self.obj2)
```

```python
>>> class CorrStats(vbt.Configured):
...     def __init__(self, obj1, obj2):
...         vbt.Configured.__init__(self, obj1=obj1, obj2=obj2)
...
...         self._obj1 = obj1
...         self._obj2 = obj2
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         if isinstance(self.obj1, pd.Series):
...             return self.obj1.corr(self.obj2)
...         return self.obj1.corrwith(self.obj2)
...
...     def rolling_corr(self, window):
...         return self.obj1.rolling(window).corr(self.obj2)
```

This is how most configured classes in vectorbt, such as Portfolio, are designed. Any argument that is being passed to CorrStats is forwarded down to Configured to initialize a new config:

```python
CorrStats
```

```python
>>> index = vbt.date_range("2020", periods=5)
>>> df1 = pd.DataFrame({
...     'a': [1, 5, 2, 4, 3],
...     'b': [3, 2, 4, 1, 5]
... }, index=index)
>>> df2 = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=index)

>>> corrstats = CorrStats(df1, df2)
>>> print(corrstats.config)
Config(
    obj1=<DataFrame object at 0x7fd490461400 of shape (5, 2)>,
    obj2=<DataFrame object at 0x7fd490461240 of shape (5, 2)>
)
```

```python
>>> index = vbt.date_range("2020", periods=5)
>>> df1 = pd.DataFrame({
...     'a': [1, 5, 2, 4, 3],
...     'b': [3, 2, 4, 1, 5]
... }, index=index)
>>> df2 = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=index)

>>> corrstats = CorrStats(df1, df2)
>>> print(corrstats.config)
Config(
    obj1=<DataFrame object at 0x7fd490461400 of shape (5, 2)>,
    obj2=<DataFrame object at 0x7fd490461240 of shape (5, 2)>
)
```

Access to any attribute is read-only: whenever we try to set a read-only property or modify the config, an (expected) error will be thrown:

```python
>>> df3 = pd.DataFrame({
...     'a': [3, 2, 1, 5, 4],
...     'b': [4, 5, 1, 2, 3]
... }, index=index)
>>> corrstats.obj1 = df3
AttributeError: can't set attribute

>>> corrstats.config['obj1'] = df3
TypeError: Config is read-only
```

```python
>>> df3 = pd.DataFrame({
...     'a': [3, 2, 1, 5, 4],
...     'b': [4, 5, 1, 2, 3]
... }, index=index)
>>> corrstats.obj1 = df3
AttributeError: can't set attribute

>>> corrstats.config['obj1'] = df3
TypeError: Config is read-only
```

However, it won't (and can't) throw an error when setting a private attribute (with a leading underscore) or if any of the attributes were modified in place, which is a common pitfall you should avoid.

```python
>>> corrstats._obj1 = df3  # (1)!
>>> corrstats.obj1.iloc[:] = df3  # (2)!
```

```python
>>> corrstats._obj1 = df3  # (1)!
>>> corrstats.obj1.iloc[:] = df3  # (2)!
```

Warning

vectorbt assumes that the data in a configured instance always stays the same. Whenever there is a change to data, vectorbt won't register it and likely deliver erroneous results at some point in time.

The only approved way to change any data in an instance is to create another instance!

To change anything, pass the new data to Configured.replace, which takes the same arguments as the class but in the keyword-only format, merges them over the old config, and passes as keyword arguments to the class for instantiation.

```python
>>> new_corrstats = corrstats.replace(obj1=df3)
>>> new_corrstats.obj1  # (1)!
            a  b
2020-01-01  3  4
2020-01-02  2  5
2020-01-03  1  1
2020-01-04  5  2
2020-01-05  4  3

>>> new_corrstats.obj2  # (2)!
            a  b
2020-01-01  1  5
2020-01-02  2  4
2020-01-03  3  3
2020-01-04  4  2
2020-01-05  5  1
```

```python
>>> new_corrstats = corrstats.replace(obj1=df3)
>>> new_corrstats.obj1  # (1)!
            a  b
2020-01-01  3  4
2020-01-02  2  5
2020-01-03  1  1
2020-01-04  5  2
2020-01-05  4  3

>>> new_corrstats.obj2  # (2)!
            a  b
2020-01-01  1  5
2020-01-02  2  4
2020-01-03  3  3
2020-01-04  4  2
2020-01-05  5  1
```

```python
df1
```

```python
df3
```

```python
df2
```

Since all of our data is now stored inside a config, we can perform many actions on the instance as if we performed them on the config itself, such as saving to disk (thanks to Pickling):

```python
>>> corrstats.save('corrstats')  # (1)!

>>> corrstats = CorrStats.load('corrstats')
```

```python
>>> corrstats.save('corrstats')  # (1)!

>>> corrstats = CorrStats.load('corrstats')
```

### Attribute resolution¶

Attribute resolution is handy when it comes to accessing attributes based on strings or some other logic, which is realized by the mixin AttrResolverMixin. You can imagine it implementing an arbitrary logic for a custom getattr operation. It's widely used in StatsBuilderMixin and PlotsBuilderMixin to execute metrics and subplots respectively as a chain of commands. In other classes, such as Portfolio, it's also being used for accessing shortcut properties, caching attribute access, and more. It works in conjunction with deep_getattr, which accesses a chain of attributes provided as a specification.

```python
getattr
```

Let's compute the min of the rolling mean solely using Pandas and deep attribute resolution:

```python
>>> sr = pd.Series([1, 2, 3, 4, 5])
>>> attr_chain = [('rolling', (3,)), 'mean', 'min']
>>> vbt.deep_getattr(sr, attr_chain)
2.0
```

```python
>>> sr = pd.Series([1, 2, 3, 4, 5])
>>> attr_chain = [('rolling', (3,)), 'mean', 'min']
>>> vbt.deep_getattr(sr, attr_chain)
2.0
```

If any of the above operations were done on a subclass AttrResolverMixin, they could have been preprocessed and postprocessed easily.

### Templating¶

Templates play an important role in the vectorbt's ecosystem. They allow postponing data resolution to a later point in time when there is more information available. There are many different templating classes, such as Rep for replacing an entire string and Sub for substituting only parts of it (those beginning with $).

```python
$
```

You can imagine templates being callbacks that are executed at some point during the execution, mostly after broadcasting or merging keyword arguments. Also, there exist functions that offer multiple potential substitution points; in such case, they either attempt to substitute the template multiple times until they succeed, or they match the template with a specific evaluation id (eval_id), if provided. The actual evaluation operation is performed by substitute_templates.

```python
eval_id
```

```python
>>> def some_function(*args, **kwargs):
...     context = {}
...     args = vbt.substitute_templates(args, context=context, strict=False)
...     kwargs = vbt.substitute_templates(kwargs, context=context, strict=False)
...     print(args)
...     print(kwargs)
...     
...     context['result'] = 100  # (1)!
...     args = vbt.substitute_templates(args, context=context)
...     kwargs = vbt.substitute_templates(kwargs, context=context)
...     print(args)
...     print(kwargs)

>>> some_function(vbt.Rep('result'), double_result=vbt.RepEval('result * 2'))
(Rep(template='result', context=None, strict=None, eval_id=None),)
{'double_result': RepEval(template='result * 2', context=None, strict=None, eval_id=None)}
(100,)
{'double_result': 200}
```

```python
>>> def some_function(*args, **kwargs):
...     context = {}
...     args = vbt.substitute_templates(args, context=context, strict=False)
...     kwargs = vbt.substitute_templates(kwargs, context=context, strict=False)
...     print(args)
...     print(kwargs)
...     
...     context['result'] = 100  # (1)!
...     args = vbt.substitute_templates(args, context=context)
...     kwargs = vbt.substitute_templates(kwargs, context=context)
...     print(args)
...     print(kwargs)

>>> some_function(vbt.Rep('result'), double_result=vbt.RepEval('result * 2'))
(Rep(template='result', context=None, strict=None, eval_id=None),)
{'double_result': RepEval(template='result * 2', context=None, strict=None, eval_id=None)}
(100,)
{'double_result': 200}
```

## Base¶

```python
flowchart TD;
    Grouper -->|inherits| Configured;
    ArrayWrapper -->|inherits| Configured;
    ArrayWrapper -->|inherits| PandasIndexer;
    ArrayWrapper -.->|references| Grouper;
    Wrapping -->|inherits| Configured;
    Wrapping -->|inherits| PandasIndexer;
    Wrapping -->|inherits| AttrResolverMixin;
    Wrapping -.->|references| ArrayWrapper;
    BaseAccessor -->|inherits| Wrapping;
```

```python
flowchart TD;
    Grouper -->|inherits| Configured;
    ArrayWrapper -->|inherits| Configured;
    ArrayWrapper -->|inherits| PandasIndexer;
    ArrayWrapper -.->|references| Grouper;
    Wrapping -->|inherits| Configured;
    Wrapping -->|inherits| PandasIndexer;
    Wrapping -->|inherits| AttrResolverMixin;
    Wrapping -.->|references| ArrayWrapper;
    BaseAccessor -->|inherits| Wrapping;
```

Subpackage base is the non-computational core of vectorbt. It offers a range of modules for working with and converting between Pandas and NumPy objects. In particular, it provides functions and classes for broadcasting, combining and wrapping NumPy arrays, grouping columns, managing MultiIndex, and more. These operations are essential for extending Pandas and replicating some of its functionality in custom classes.

### Grouping¶

Since vectorbt usually associates with processing multi-column data, where each column (or "line") represents a separate backtesting instance, the ability to group those columns into some sort of groups is a must-have feature.

Class Grouper implements functionality to validate and build groups of any Pandas Index, especially columns. It's capable of translating various metadata such as GroupBy objects and column levels into special NumPy arrays that can be used by Numba-compiled functions to aggregate multiple columns of data. This is especially useful for multi-asset portfolios where each group is composed of one or more assets.

```python
>>> columns = pd.MultiIndex.from_tuples([
...     ('BTC-USD', 'group1'), 
...     ('ETH-USD', 'group1'), 
...     ('ADA-USD', 'group2'),
...     ('SOL-USD', 'group2')
... ], names=['symbol', 'group'])
>>> vbt.Grouper(columns, 'group').get_groups()
array([0, 0, 1, 1])
```

```python
>>> columns = pd.MultiIndex.from_tuples([
...     ('BTC-USD', 'group1'), 
...     ('ETH-USD', 'group1'), 
...     ('ADA-USD', 'group2'),
...     ('SOL-USD', 'group2')
... ], names=['symbol', 'group'])
>>> vbt.Grouper(columns, 'group').get_groups()
array([0, 0, 1, 1])
```

### Indexing¶

The main purpose of indexing in vectorbt is to provide Pandas indexing to any custom class holding Pandas-like objects, in particular, to select rows, columns, and groups in each. This is done by forwarding a Pandas indexing operation to each Pandas-like object and instantiating the class using them, which is fairly easy using Configured. This way, one can index complex classes with dozens of Pandas-like objects using a single command.

The main indexer class PandasIndexer mimics a regular Pandas object by exposing properties PandasIndexer.iloc, PandasIndexer.loc, and PandasIndexer.xs. All we have to do is to subclass this class and override IndexingBase.indexing_func, which should take pd_indexing_func, apply it on each Pandas-like object, and initialize a new instance.

```python
pd_indexing_func
```

Let's extend our newly created CorrStats class with Pandas indexing:

```python
CorrStats
```

```python
>>> class CorrStats(vbt.Configured, vbt.PandasIndexer):
...     def __init__(self, obj1, obj2):
...         vbt.Configured.__init__(self, obj1=obj1, obj2=obj2)
...         vbt.PandasIndexer.__init__(self)
...
...         self._obj1 = obj1
...         self._obj2 = obj2
...
...     def indexing_func(self, pd_indexing_func):  # (1)!
...         return self.replace(
...             obj1=pd_indexing_func(self.obj1),
...             obj2=pd_indexing_func(self.obj2)
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         if isinstance(self.obj1, pd.Series):
...             return self.obj1.corr(self.obj2)
...         return self.obj1.corrwith(self.obj2)
...
...     def rolling_corr(self, window):
...         return self.obj1.rolling(window).corr(self.obj2)

>>> corrstats = CorrStats(df1, df2)
>>> corrstats.corr()
a    0.3
b   -0.3
dtype: float64

>>> corrstats.loc['2020-01-01':'2020-01-03', 'a'].corr()  # (2)!
0.24019223070763066
```

```python
>>> class CorrStats(vbt.Configured, vbt.PandasIndexer):
...     def __init__(self, obj1, obj2):
...         vbt.Configured.__init__(self, obj1=obj1, obj2=obj2)
...         vbt.PandasIndexer.__init__(self)
...
...         self._obj1 = obj1
...         self._obj2 = obj2
...
...     def indexing_func(self, pd_indexing_func):  # (1)!
...         return self.replace(
...             obj1=pd_indexing_func(self.obj1),
...             obj2=pd_indexing_func(self.obj2)
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         if isinstance(self.obj1, pd.Series):
...             return self.obj1.corr(self.obj2)
...         return self.obj1.corrwith(self.obj2)
...
...     def rolling_corr(self, window):
...         return self.obj1.rolling(window).corr(self.obj2)

>>> corrstats = CorrStats(df1, df2)
>>> corrstats.corr()
a    0.3
b   -0.3
dtype: float64

>>> corrstats.loc['2020-01-01':'2020-01-03', 'a'].corr()  # (2)!
0.24019223070763066
```

```python
pd_indexing_func
```

```python
lambda x: x.loc[key]
```

We just indexed two Pandas objects as a single entity. Yay!

### Wrapping¶

Remember how vectorbt specializes at taking a Pandas object, extracting its NumPy array, processing the array, and converting the results back into a Pandas format? The last part is done by the class ArrayWrapper, which captures all the necessary metadata, such as the index, columns, and number of dimensions, and exposes methods such as ArrayWrapper.wrap to convert a NumPy object back into a Pandas format.

Class ArrayWrapper combines many concepts we introduced earlier to behave just like a (supercharged) Pandas object. In particular, it uses Grouping to build and manage groups of columns, and Indexing to select rows, columns, and groups using Pandas indexing. Probably the most powerful features of an array wrapper is 1) the ability to behave like a grouped object, which isn't possible with Pandas alone, and 2) the ability to translate a Pandas indexing operation to a range of integer arrays that can be used to index NumPy arrays. The latter allows indexing without the need to hold Pandas objects, only the wrapper.

We can construct a wrapper in multiple ways, the easiest being using a Pandas object:

```python
>>> df = pd.DataFrame({
...     'a': range(0, 5),
...     'b': range(5, 10),
...     'c': range(10, 15),
...     'd': range(15, 20)
... }, index=index)
>>> wrapper = vbt.ArrayWrapper.from_obj(df)
>>> print(wrapper)
ArrayWrapper(
    index=<DatetimeIndex object at 0x7ff6d8528d68 of shape (5,)>,
    columns=<Index object at 0x7ff6d857fcc0 of shape (4,)>,
    ndim=2,
    freq=None,
    column_only_select=None,
    group_select=None,
    grouped_ndim=None,
    grouper=Grouper(
        index=<Index object at 0x7ff6d857fcc0 of shape (4,)>,
        group_by=None,
        allow_enable=True,
        allow_disable=True,
        allow_modify=True
    )
)
```

```python
>>> df = pd.DataFrame({
...     'a': range(0, 5),
...     'b': range(5, 10),
...     'c': range(10, 15),
...     'd': range(15, 20)
... }, index=index)
>>> wrapper = vbt.ArrayWrapper.from_obj(df)
>>> print(wrapper)
ArrayWrapper(
    index=<DatetimeIndex object at 0x7ff6d8528d68 of shape (5,)>,
    columns=<Index object at 0x7ff6d857fcc0 of shape (4,)>,
    ndim=2,
    freq=None,
    column_only_select=None,
    group_select=None,
    grouped_ndim=None,
    grouper=Grouper(
        index=<Index object at 0x7ff6d857fcc0 of shape (4,)>,
        group_by=None,
        allow_enable=True,
        allow_disable=True,
        allow_modify=True
    )
)
```

Let's create a function that sums all elements over each column using NumPy and returns a regular Pandas object:

```python
>>> def sum_per_column(df):
...     wrapper = vbt.ArrayWrapper.from_obj(df)
...     result = np.sum(df.values, axis=0)
...     return wrapper.wrap_reduced(result)

>>> sum_per_column(df)
a    10
b    35
c    60
d    85
dtype: int64
```

```python
>>> def sum_per_column(df):
...     wrapper = vbt.ArrayWrapper.from_obj(df)
...     result = np.sum(df.values, axis=0)
...     return wrapper.wrap_reduced(result)

>>> sum_per_column(df)
a    10
b    35
c    60
d    85
dtype: int64
```

The function above is already 20x faster than Pandas

```python
>>> big_df = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_df.sum()
2.52 ms ± 3.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit sum_per_column(big_df)
428 µs ± 1.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

```python
>>> big_df = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_df.sum()
2.52 ms ± 3.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit sum_per_column(big_df)
428 µs ± 1.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Since ArrayWrapper can manage groups of columns, let's adapt our function to sum all elements over each group of columns:

```python
>>> def sum_per_group(df, group_by):
...     wrapper = vbt.ArrayWrapper.from_obj(df, group_by=group_by)
...     results = []
...     for group_idxs in wrapper.grouper.iter_group_idxs():
...         group_result = np.sum(df.values[:, group_idxs])
...         results.append(group_result)
...     return wrapper.wrap_reduced(results)

>>> sum_per_group(df, False)  # (1)!
a    10
b    35
c    60
d    85
dtype: int64

>>> sum_per_group(df, True)  # (2)!
190

>>> group_by = pd.Index(['group1', 'group1', 'group2', 'group2'])
>>> sum_per_group(df, group_by)  # (3)!
group1     45
group2    145
dtype: int64
```

```python
>>> def sum_per_group(df, group_by):
...     wrapper = vbt.ArrayWrapper.from_obj(df, group_by=group_by)
...     results = []
...     for group_idxs in wrapper.grouper.iter_group_idxs():
...         group_result = np.sum(df.values[:, group_idxs])
...         results.append(group_result)
...     return wrapper.wrap_reduced(results)

>>> sum_per_group(df, False)  # (1)!
a    10
b    35
c    60
d    85
dtype: int64

>>> sum_per_group(df, True)  # (2)!
190

>>> group_by = pd.Index(['group1', 'group1', 'group2', 'group2'])
>>> sum_per_group(df, group_by)  # (3)!
group1     45
group2    145
dtype: int64
```

To avoid creating multiple array wrappers with the same metadata, there is the class Wrapping, which binds a single instance of ArrayWrapper to manage an arbitrary number of shape-compatible array-like objects. Instead of accepting multiple Pandas objects, it takes an array wrapper, and all other objects and arrays in any format (preferably NumPy), and wraps them using this wrapper. Additionally, any Wrapping subclass can utilize its wrapper to perform Pandas indexing on any kind of objects, including NumPy arrays, because ArrayWrapper can translate a Pandas indexing operation into universal row, column, and group indices.

Coming back to our CorrStats class. There are two issues with the current implementation:

```python
CorrStats
```

Let's upgrade our CorrStats class to work on NumPy arrays and with an array wrapper:

```python
CorrStats
```

```python
>>> class CorrStats(vbt.Wrapping):
...     _expected_keys = vbt.Wrapping._expected_keys | {"obj1", "obj2"}  # (1)!
...
...     @classmethod
...     def from_objs(cls, obj1, obj2):  # (2)!
...         (obj1, obj2), wrapper = vbt.broadcast(
...             obj1, obj2, 
...             to_pd=False,
...             return_wrapper=True
...         )
...         return cls(wrapper, obj1, obj2)
...
...     def __init__(self, wrapper, obj1, obj2):
...         vbt.Wrapping.__init__(self, wrapper, obj1=obj1, obj2=obj2)
...
...         self._obj1 = vbt.to_2d_array(obj1)
...         self._obj2 = vbt.to_2d_array(obj2)
...
...     def indexing_func(self, pd_indexing_func, **kwargs):  # (3)!
...         wrapper_meta = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
...         new_wrapper = wrapper_meta["new_wrapper"]
...         row_idxs = wrapper_meta["row_idxs"]
...         col_idxs = wrapper_meta["col_idxs"]
...         return self.replace(
...             wrapper=new_wrapper,
...             obj1=self.obj1[row_idxs, :][:, col_idxs],
...             obj2=self.obj2[row_idxs, :][:, col_idxs]
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):  # (4)!
...         out = vbt.nb.nancorr_nb(self.obj1, self.obj2)
...         return self.wrapper.wrap_reduced(out)
...
...     def rolling_corr(self, window):
...         out = vbt.nb.rolling_corr_nb(
...             self.obj1, self.obj2, 
...             window, minp=window)
...         return self.wrapper.wrap(out)
```

```python
>>> class CorrStats(vbt.Wrapping):
...     _expected_keys = vbt.Wrapping._expected_keys | {"obj1", "obj2"}  # (1)!
...
...     @classmethod
...     def from_objs(cls, obj1, obj2):  # (2)!
...         (obj1, obj2), wrapper = vbt.broadcast(
...             obj1, obj2, 
...             to_pd=False,
...             return_wrapper=True
...         )
...         return cls(wrapper, obj1, obj2)
...
...     def __init__(self, wrapper, obj1, obj2):
...         vbt.Wrapping.__init__(self, wrapper, obj1=obj1, obj2=obj2)
...
...         self._obj1 = vbt.to_2d_array(obj1)
...         self._obj2 = vbt.to_2d_array(obj2)
...
...     def indexing_func(self, pd_indexing_func, **kwargs):  # (3)!
...         wrapper_meta = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
...         new_wrapper = wrapper_meta["new_wrapper"]
...         row_idxs = wrapper_meta["row_idxs"]
...         col_idxs = wrapper_meta["col_idxs"]
...         return self.replace(
...             wrapper=new_wrapper,
...             obj1=self.obj1[row_idxs, :][:, col_idxs],
...             obj2=self.obj2[row_idxs, :][:, col_idxs]
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):  # (4)!
...         out = vbt.nb.nancorr_nb(self.obj1, self.obj2)
...         return self.wrapper.wrap_reduced(out)
...
...     def rolling_corr(self, window):
...         out = vbt.nb.rolling_corr_nb(
...             self.obj1, self.obj2, 
...             window, minp=window)
...         return self.wrapper.wrap(out)
```

```python
_expected_keys=None
```

```python
pd_indexing_func
```

```python
CorrStats
```

As you might have noticed, we replaced the superclasses Configured and PandasIndexer with a single superclass Wrapping, which already inherits them both. Another change applies to the arguments taken by CorrStats: instead of taking two Pandas objects, it now takes wrapper of type ArrayWrapper along with the NumPy arrays obj1 and obj2. This has several benefits: we're keeping Pandas metadata consistent and managed by a single variable, while all actions are efficiently performed using NumPy alone. Whenever there is a need to present the findings, we can call ArrayWrapper.wrap_reduced and ArrayWrapper.wrap to transform them back into a Pandas format, which is done inside the methods CorrStats.corr and CorrStats.rolling_corr respectively.

```python
CorrStats
```

```python
wrapper
```

```python
obj1
```

```python
obj2
```

```python
CorrStats.corr
```

```python
CorrStats.rolling_corr
```

Since we don't want to force ourselves and the user to create an array wrapper manually, we also implemented the class method CorrStats.from_objs, which broadcasts both arrays and instantiates CorrStats. This way, we can provide array-like objects of any kind and CorrStats will automatically build the wrapper for us. Let's illustrate this by computing the correlation coefficient for df1 and df2, and then for df1 and parametrized df2:

```python
CorrStats.from_objs
```

```python
CorrStats
```

```python
CorrStats
```

```python
df1
```

```python
df2
```

```python
df1
```

```python
df2
```

```python
>>> df1.corrwith(df2)  # (1)!
a    0.3
b   -0.3
dtype: float64

>>> corrstats = CorrStats.from_objs(df1, df2)
>>> corrstats.corr()  # (2)!
a    0.3
b   -0.3
dtype: float64

>>> df2_sh = vbt.pd_acc.concat(
...     df2, df2.vbt.shuffle(seed=42), 
...     keys=['plain', 'shuffled'])
>>> df2_sh 
              plain    shuffled   
               a  b        a  b
2020-01-01     1  5        2  2
2020-01-02     2  4        5  4
2020-01-03     3  3        3  3
2020-01-04     4  2        1  5
2020-01-05     5  1        4  1

>>> df1.corrwith(df2_sh)  # (3)!
ValueError: cannot join with no overlapping index names

>>> corrstats = CorrStats.from_objs(df1, df2_sh)
>>> corrstats.corr()  # (4)!
plain     a    0.3
          b   -0.3
shuffled  a    0.4
          b   -0.9
dtype: float64
```

```python
>>> df1.corrwith(df2)  # (1)!
a    0.3
b   -0.3
dtype: float64

>>> corrstats = CorrStats.from_objs(df1, df2)
>>> corrstats.corr()  # (2)!
a    0.3
b   -0.3
dtype: float64

>>> df2_sh = vbt.pd_acc.concat(
...     df2, df2.vbt.shuffle(seed=42), 
...     keys=['plain', 'shuffled'])
>>> df2_sh 
              plain    shuffled   
               a  b        a  b
2020-01-01     1  5        2  2
2020-01-02     2  4        5  4
2020-01-03     3  3        3  3
2020-01-04     4  2        1  5
2020-01-05     5  1        4  1

>>> df1.corrwith(df2_sh)  # (3)!
ValueError: cannot join with no overlapping index names

>>> corrstats = CorrStats.from_objs(df1, df2_sh)
>>> corrstats.corr()  # (4)!
plain     a    0.3
          b   -0.3
shuffled  a    0.4
          b   -0.9
dtype: float64
```

Here's why we ditched Pandas in favor of Numba:

```python
>>> big_df1 = pd.DataFrame(np.random.uniform(size=(1000, 1000)))
>>> big_df2 = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_df1.rolling(10).corr(big_df2)  # (1)!
271 ms ± 13.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> corrstats = CorrStats.from_objs(big_df1, big_df2)
>>> %timeit corrstats.rolling_corr(10)  # (2)!
12 ms ± 50.5 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> big_df1 = pd.DataFrame(np.random.uniform(size=(1000, 1000)))
>>> big_df2 = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_df1.rolling(10).corr(big_df2)  # (1)!
271 ms ± 13.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> corrstats = CorrStats.from_objs(big_df1, big_df2)
>>> %timeit corrstats.rolling_corr(10)  # (2)!
12 ms ± 50.5 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Another addition we made concerns indexing. Since obj1 and obj2 are not regular Pandas objects anymore, we cannot simply apply pd_indexing_func on them. Instead, we can use the method ArrayWrapper.indexing_func_meta to get the rows, columns, and groups that this operation would select. We then apply those arrays on both NumPy objects. This approach is exceptionally useful because now we can select any data from the final shape:

```python
obj1
```

```python
obj2
```

```python
pd_indexing_func
```

```python
>>> corrstats = CorrStats.from_objs(df1, df2_sh)
>>> corrstats.loc['2020-01-02':'2020-01-05'].rolling_corr(3)  # (1)!
                         plain            shuffled
                   a         b         a         b
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03       NaN       NaN       NaN       NaN
2020-01-04 -0.327327  0.327327  0.327327 -0.981981
2020-01-05  0.500000 -0.240192 -0.654654 -0.960769
```

```python
>>> corrstats = CorrStats.from_objs(df1, df2_sh)
>>> corrstats.loc['2020-01-02':'2020-01-05'].rolling_corr(3)  # (1)!
                         plain            shuffled
                   a         b         a         b
2020-01-02       NaN       NaN       NaN       NaN
2020-01-03       NaN       NaN       NaN       NaN
2020-01-04 -0.327327  0.327327  0.327327 -0.981981
2020-01-05  0.500000 -0.240192 -0.654654 -0.960769
```

Note

Not all classes support indexing on rows. To make sure you can select rows, check whether the instance property column_only_select is False.

```python
column_only_select
```

That's how most high-tier classes in vectorbt are built. As a rule of thumb:

```python
wrapper
```

```python
wrapper.ndim
```

```python
from_
```

```python
__init__
```

```python
vbt.MA.run()
```

```python
vbt.MA()
```

### Base accessor¶

This subpackage also contains the accessor BaseAccessor, which exposes many basic operations to the end user and is being subclassed by all other accessors. It inherits Wrapping and thus we can do with it everything what we can do with our custom CorrStats class. Why calling an accessor a "base" accessor? Because it is the superclass of all other accessors in vectorbt and provides them with the core combining, reshaping, and indexing functions, such as BaseAccessor.to_2d_array to convert a Pandas object into a two-dimensional NumPy array.

```python
CorrStats
```

The "access" to the accessor is simple:

```python
>>> df.vbt
<vectorbtpro.accessors.Vbt_DFAccessor at 0x7fe3c19f7d68>

>>> df.vbt.to_2d_array()
array([[ 0,  5, 10, 15],
       [ 1,  6, 11, 16],
       [ 2,  7, 12, 17],
       [ 3,  8, 13, 18],
       [ 4,  9, 14, 19]])
```

```python
>>> df.vbt
<vectorbtpro.accessors.Vbt_DFAccessor at 0x7fe3c19f7d68>

>>> df.vbt.to_2d_array()
array([[ 0,  5, 10, 15],
       [ 1,  6, 11, 16],
       [ 2,  7, 12, 17],
       [ 3,  8, 13, 18],
       [ 4,  9, 14, 19]])
```

In the example above, Vbt_DFAccessor is the main accessor for DataFrames, and as you can see in its definition, BaseAccessor appears in its superclasses.

Probably the most interesting is the method BaseAccessor.combine, which allows for broadcasting and combining the current Pandas object with an arbitrary number of other array-like objects given the function combine_func (mainly using NumPy).

```python
combine_func
```

```python
>>> pd.Series([1, 2, 3]).vbt.combine(np.array([[4, 5, 6]]), np.add)
   0  1  2
0  5  6  7
1  6  7  8
2  7  8  9
```

```python
>>> pd.Series([1, 2, 3]).vbt.combine(np.array([[4, 5, 6]]), np.add)
   0  1  2
0  5  6  7
1  6  7  8
2  7  8  9
```

BaseAccessor implements a range of unary and binary magic methods using this method. For example, let's invoke BaseAccessor.__add__, which implements addition:

```python
BaseAccessor.__add__
```

```python
>>> pd.Series([1, 2, 3]) + np.array([[4, 5, 6]])  # (1)!
ValueError: Length of values (1) does not match length of index (3)

>>> pd.Series([1, 2, 3]).vbt + np.array([[4, 5, 6]])  # (2)!
   0  1  2
0  5  6  7
1  6  7  8
2  7  8  9
```

```python
>>> pd.Series([1, 2, 3]) + np.array([[4, 5, 6]])  # (1)!
ValueError: Length of values (1) does not match length of index (3)

>>> pd.Series([1, 2, 3]).vbt + np.array([[4, 5, 6]])  # (2)!
   0  1  2
0  5  6  7
1  6  7  8
2  7  8  9
```

```python
.vbt
```

```python
.vbt
```

Hint

To learn more about  methods, see A Guide to Python's Magic Methods.

All of these magic methods were added using class decorators. There are a lot of class decorators for all kind of things in vectorbt. Usually, they take a config and attach many attributes at once in some automated way.

## Generic¶

```python
flowchart TD;
    Analyzable -->|inherits| Wrapping;
    Analyzable -->|inherits| StatsBuilderMixin;
    Analyzable -->|inherits| PlotsBuilderMixin;
    GenericAccessor -->|inherits| BaseAccessor;
    GenericAccessor -->|inherits| Analyzable;
```

```python
flowchart TD;
    Analyzable -->|inherits| Wrapping;
    Analyzable -->|inherits| StatsBuilderMixin;
    Analyzable -->|inherits| PlotsBuilderMixin;
    GenericAccessor -->|inherits| BaseAccessor;
    GenericAccessor -->|inherits| Analyzable;
```

Subpackage generic is the computational core of vectorbt. It contains modules for processing and plotting time series and numeric data in general. More importantly: it implements an arsenal of Numba-compiled functions for accelerating and extending Pandas! Those functions are powering many corners of vectorbt, from indicators to portfolio analysis. But for now, let's focus on classes that could make our CorrStats class more powerful.

```python
CorrStats
```

### Builder mixins¶

Builder mixins are classes that, once subclassed by a class, allow to build a specific functionality from this class' attributes. There are two prominent members: StatsBuilderMixin and PlotsBuilderMixin. The former exposes the method StatsBuilderMixin.stats to compute various metrics, while the latter exposes the method PlotsBuilderMixin.plots to display various subplots. Both are subclassed by almost every class that can analyze data.

### Analyzing¶

Class Analyzable combines Wrapping and Builder mixins. It combines everything we introduced above to build a foundation for a seamless data analysis; that's why it's being subclassed by so many high-tier classes, such as Portfolio and Records.

What are we waiting for? Let's adapt our CorrStats class to become analyzable!

```python
CorrStats
```

```python
>>> class CorrStats(vbt.Analyzable):  # (1)!
...     _expected_keys = vbt.Analyzable._expected_keys | {"obj1", "obj2"}
...
...     @classmethod
...     def from_objs(cls, obj1, obj2):
...         (obj1, obj2), wrapper = vbt.broadcast(
...             obj1, obj2, 
...             to_pd=False,
...             return_wrapper=True
...         )
...         return cls(wrapper, obj1, obj2)
...
...     def __init__(self, wrapper, obj1, obj2):
...         vbt.Analyzable.__init__(self, wrapper, obj1=obj1, obj2=obj2)  # (2)!
...
...         self._obj1 = vbt.to_2d_array(obj1)
...         self._obj2 = vbt.to_2d_array(obj2)
...
...     def indexing_func(self, pd_indexing_func, **kwargs):
...         wrapper_meta = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
...         new_wrapper = wrapper_meta["new_wrapper"]
...         row_idxs = wrapper_meta["row_idxs"]
...         col_idxs = wrapper_meta["col_idxs"]
...         return self.replace(
...             wrapper=new_wrapper,
...             obj1=self.obj1[row_idxs, :][:, col_idxs],
...             obj2=self.obj2[row_idxs, :][:, col_idxs]
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         out = vbt.nb.nancorr_nb(self.obj1, self.obj2)
...         return self.wrapper.wrap_reduced(out)
...
...     def rolling_corr(self, window):
...         out = vbt.nb.rolling_corr_nb(
...             self.obj1, self.obj2, 
...             window, minp=window)
...         return self.wrapper.wrap(out)
...
...     _metrics = vbt.HybridConfig(  # (3)!
...         corr=dict(
...             title='Corr. Coefficient',
...             calc_func='corr'
...         )
...     )
...
...     _subplots = vbt.HybridConfig(  # (4)!
...          rolling_corr=dict(
...              title=vbt.Sub("Rolling Corr. Coefficient (window=$window)"),
...              plot_func=vbt.Sub('rolling_corr($window).vbt.plot'),  # (5)!
...              pass_trace_names=False
...          )
...     )
```

```python
>>> class CorrStats(vbt.Analyzable):  # (1)!
...     _expected_keys = vbt.Analyzable._expected_keys | {"obj1", "obj2"}
...
...     @classmethod
...     def from_objs(cls, obj1, obj2):
...         (obj1, obj2), wrapper = vbt.broadcast(
...             obj1, obj2, 
...             to_pd=False,
...             return_wrapper=True
...         )
...         return cls(wrapper, obj1, obj2)
...
...     def __init__(self, wrapper, obj1, obj2):
...         vbt.Analyzable.__init__(self, wrapper, obj1=obj1, obj2=obj2)  # (2)!
...
...         self._obj1 = vbt.to_2d_array(obj1)
...         self._obj2 = vbt.to_2d_array(obj2)
...
...     def indexing_func(self, pd_indexing_func, **kwargs):
...         wrapper_meta = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
...         new_wrapper = wrapper_meta["new_wrapper"]
...         row_idxs = wrapper_meta["row_idxs"]
...         col_idxs = wrapper_meta["col_idxs"]
...         return self.replace(
...             wrapper=new_wrapper,
...             obj1=self.obj1[row_idxs, :][:, col_idxs],
...             obj2=self.obj2[row_idxs, :][:, col_idxs]
...         )
...
...     @property
...     def obj1(self):
...         return self._obj1
...
...     @property
...     def obj2(self):
...         return self._obj2
...
...     def corr(self):
...         out = vbt.nb.nancorr_nb(self.obj1, self.obj2)
...         return self.wrapper.wrap_reduced(out)
...
...     def rolling_corr(self, window):
...         out = vbt.nb.rolling_corr_nb(
...             self.obj1, self.obj2, 
...             window, minp=window)
...         return self.wrapper.wrap(out)
...
...     _metrics = vbt.HybridConfig(  # (3)!
...         corr=dict(
...             title='Corr. Coefficient',
...             calc_func='corr'
...         )
...     )
...
...     _subplots = vbt.HybridConfig(  # (4)!
...          rolling_corr=dict(
...              title=vbt.Sub("Rolling Corr. Coefficient (window=$window)"),
...              plot_func=vbt.Sub('rolling_corr($window).vbt.plot'),  # (5)!
...              pass_trace_names=False
...          )
...     )
```

```python
Wrapping
```

```python
Analyzable
```

```python
Wrapping
```

```python
Analyzable
```

We changed a few things: replaced Wrapping with Analyzable, and added some metrics and subplots based on CorrStats.corr and CorrStats.rolling_corr. That's it! We can now pass arbitrary array-like objects to CorrStats.from_objs and it will return an instance that can be used to analyze the correlation between the objects, in particular using CorrStats.stats and CorrStats.plots:

```python
Wrapping
```

```python
Analyzable
```

```python
CorrStats.corr
```

```python
CorrStats.rolling_corr
```

```python
CorrStats.from_objs
```

```python
CorrStats.stats
```

```python
CorrStats.plots
```

```python
>>> corrstats = CorrStats.from_objs(df1, df2)
>>> corrstats.stats(column='a')  # (1)!
Corr. Coefficient    0.3
Name: a, dtype: object

>>> corrstats['a'].stats()  # (2)!
Corr. Coefficient    0.3
Name: a, dtype: object

>>> corrstats.plots(template_context=dict(window=3)).show()  # (3)!
```

```python
>>> corrstats = CorrStats.from_objs(df1, df2)
>>> corrstats.stats(column='a')  # (1)!
Corr. Coefficient    0.3
Name: a, dtype: object

>>> corrstats['a'].stats()  # (2)!
Corr. Coefficient    0.3
Name: a, dtype: object

>>> corrstats.plots(template_context=dict(window=3)).show()  # (3)!
```

```python
a
```

```python
a
```

There is nothing more satisfying than not having to write boilerplate code. Thanks to Analyzable, we can shift our focus entirely to analysis, while vectorbt takes care of everything else.

### Generic accessor¶

We don't have to look far to find a class that inherits Analyzable: class GenericAccessor extends the class BaseAccessor to deliver statistics and plots for any numeric data. It's a size-that-fits-all class with an objective to replicate, accelerate, and extend Pandas' core functionality. It implements in-house rolling, mapping, reducing, splitting, plotting, and many other kinds of methods, which can be used on any Series or DataFrame in the wild.

In a nutshell, GenericAccessor does the following:

Similarly to the Base accessor, the generic accessor uses class decorators and configs to attach many Numba-compiled and scikit-learn functions at once.

Usage is similar to CorrStats, except that we can call the generic accessor directly on Pandas objects since it's being directly subclassed by Vbt_DFAccessor!

```python
CorrStats
```

```python
>>> df.vbt.stats(column='a')
Start        2020-01-01 00:00:00
End          2020-01-05 00:00:00
Period           5 days 00:00:00
Count                          5
Mean                         2.0
Std                     1.581139
Min                            0
Median                       2.0
Max                            4
Min Index    2020-01-01 00:00:00
Max Index    2020-01-05 00:00:00
Name: a, dtype: object
```

```python
>>> df.vbt.stats(column='a')
Start        2020-01-01 00:00:00
End          2020-01-05 00:00:00
Period           5 days 00:00:00
Count                          5
Mean                         2.0
Std                     1.581139
Min                            0
Median                       2.0
Max                            4
Min Index    2020-01-01 00:00:00
Max Index    2020-01-05 00:00:00
Name: a, dtype: object
```

## Records¶

```python
flowchart TD;
    Records -.->|references| ColumnMapper;
    Records -.->|references| MappedArray;
    Records -->|inherits| Analyzable;
    MappedArray -.->|references| ColumnMapper;
    MappedArray -->|inherits| Analyzable;
```

```python
flowchart TD;
    Records -.->|references| ColumnMapper;
    Records -.->|references| MappedArray;
    Records -->|inherits| Analyzable;
    MappedArray -.->|references| ColumnMapper;
    MappedArray -->|inherits| Analyzable;
```

Records are structured arrays - a NumPy array that can hold different data types, just like a Pandas DataFrame. Records have one big advantage over DataFrames though: they are well understood by Numba, thus we can generate and use them efficiently. So, what's the catch? Records have no (index) labels and the API is very limited. We also learned that vectorbt doesn't like heterogenous data and prefers to work with multiple homogeneous arrays (remember how we need to split OHLC into O, H, L, and C?). Nevertheless, they take a very important place in our ecosystem - as containers for event data.

Trading is all about events: executing trades, aggregating them into positions, analyzing drawdowns, and much more. Each of such events is a complex piece of data that needs a container optimized for fast writes and reads, especially inside a Numba-compiled code (but please not a list of dictionaries, they are very inefficient). Structured arrays is exactly the data structure we need! Each event is a record that holds all the required information, such the column and row where it originally happened.

Because structured arrays are hard to analyze, there is a class just for this - Records! By subclassing Analyzable (suprise, surprise), it wraps a structured NumPy array and provides us with useful tools for its analysis. Every Records instance can be indexed just like a regular Pandas object and compute various metrics and plot graphs.

Let's generate the Drawdowns records for two columns of time series data:

```python
>>> dd_df = pd.DataFrame({
...     'a': [10, 11, 12, 11, 12, 13, 12],
...     'b': [14, 13, 12, 11, 12, 13, 14]
... }, index=vbt.date_range("2020", periods=7))
>>> drawdowns = dd_df.vbt.drawdowns
>>> drawdowns.readable
   Drawdown Id Column Start Index Valley Index  End Index  Start Value  \
0            0      a  2020-01-03   2020-01-04 2020-01-05         12.0   
1            1      a  2020-01-06   2020-01-07 2020-01-07         13.0   
2            0      b  2020-01-01   2020-01-04 2020-01-07         14.0   

   Valley Value  End Value     Status  
0          11.0       12.0  Recovered  
1          12.0       12.0     Active  
2          11.0       14.0  Recovered  

>>> drawdowns['b'].readable  # (1)!
   Drawdown Id Column Start Index Valley Index  End Index  Start Value  \
0            0      b  2020-01-01   2020-01-04 2020-01-07         14.0   

   Valley Value  End Value     Status  
0          11.0       14.0  Recovered
```

```python
>>> dd_df = pd.DataFrame({
...     'a': [10, 11, 12, 11, 12, 13, 12],
...     'b': [14, 13, 12, 11, 12, 13, 14]
... }, index=vbt.date_range("2020", periods=7))
>>> drawdowns = dd_df.vbt.drawdowns
>>> drawdowns.readable
   Drawdown Id Column Start Index Valley Index  End Index  Start Value  \
0            0      a  2020-01-03   2020-01-04 2020-01-05         12.0   
1            1      a  2020-01-06   2020-01-07 2020-01-07         13.0   
2            0      b  2020-01-01   2020-01-04 2020-01-07         14.0   

   Valley Value  End Value     Status  
0          11.0       12.0  Recovered  
1          12.0       12.0     Active  
2          11.0       14.0  Recovered  

>>> drawdowns['b'].readable  # (1)!
   Drawdown Id Column Start Index Valley Index  End Index  Start Value  \
0            0      b  2020-01-01   2020-01-04 2020-01-07         14.0   

   Valley Value  End Value     Status  
0          11.0       14.0  Recovered
```

```python
b
```

That's a lot of information! Each field is a regular NumPy array, so where do we get this rich information from? Maybe to your surprise, but the labels of the DataFrames above were auto-generated from the metadata that Drawdowns holds. This metadata is called a "field config" - a regular Config that describes each field (for instance, Drawdowns.field_config). This makes possible automating and enhancing the behavior of each field. Class Records, the base class to all records classes, has many methods to read and interpret this config.

```python
>>> drawdowns.status.values  # (1)!
array([1, 0, 1])

>>> drawdowns.get_apply_mapping_arr('status')  # (2)!
array(['Recovered', 'Active', 'Recovered'], dtype=object)
```

```python
>>> drawdowns.status.values  # (1)!
array([1, 0, 1])

>>> drawdowns.get_apply_mapping_arr('status')  # (2)!
array(['Recovered', 'Active', 'Recovered'], dtype=object)
```

### Column mapper¶

Records are one-dimensional structured NumPy arrays. Records from multiple columns are concatenated into a single array, so we need a mechanism to group them by column or group, for instance, to aggregate values column-wise. This is a non-trivial task because finding the records that correspond to a specific column requires searching all records, which is slow when done repeatedly. The task of the class ColumnMapper is to index all columns only once and cache the result (see ColumnMapper.col_map). A column mapper has at least two more advantages: it allows for grouping columns and enables efficient Indexing.

```python
>>> drawdowns.col_mapper.col_map  # (1)!
(array([0, 1, 2]), array([2, 1]))
```

```python
>>> drawdowns.col_mapper.col_map  # (1)!
(array([0, 1, 2]), array([2, 1]))
```

The produced column map means that the column a has two records at indices 0 and 1, while the column b is represented by only one record at index 2.

```python
a
```

```python
b
```

### Mapped arrays¶

If Records is our own DataFrame for events, then MappedArray is our own Series! Each field in records can be mapped into a mapped array. In fact, a mapped array is where most calculations take place. It's just like GenericAccessor, but with a totally different representation of data: one-dimensional and clustered instead of two-dimensional and column-wise. We can even seemingly convert between both representations. Why wouldn't we then simply convert a mapped array into a regular Series and do all the analysis there? There are several reasons:

Let's analyze the drawdown values of drawdowns:

```python
drawdowns
```

```python
>>> dd_ma = drawdowns.drawdown
>>> dd_ma
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff6d8514f98>

>>> dd_ma.values  # (1)!
array([-0.08333333, -0.07692308, -0.21428571])

>>> dd_ma.stats(column='a')
Start        2020-01-01 00:00:00
End          2020-01-07 00:00:00
Period           7 days 00:00:00
Count                          2
Mean                   -0.080128
Std                     0.004533
Min                    -0.083333
Median                 -0.080128
Max                    -0.076923
Min Index    2020-01-05 00:00:00
Max Index    2020-01-07 00:00:00
Name: a, dtype: object

>>> dd_ma.to_pd()  # (2)!
                   a         b
2020-01-01       NaN       NaN
2020-01-02       NaN       NaN
2020-01-03       NaN       NaN
2020-01-04       NaN       NaN
2020-01-05 -0.083333       NaN
2020-01-06       NaN       NaN
2020-01-07 -0.076923 -0.214286
```

```python
>>> dd_ma = drawdowns.drawdown
>>> dd_ma
<vectorbtpro.records.mapped_array.MappedArray at 0x7ff6d8514f98>

>>> dd_ma.values  # (1)!
array([-0.08333333, -0.07692308, -0.21428571])

>>> dd_ma.stats(column='a')
Start        2020-01-01 00:00:00
End          2020-01-07 00:00:00
Period           7 days 00:00:00
Count                          2
Mean                   -0.080128
Std                     0.004533
Min                    -0.083333
Median                 -0.080128
Max                    -0.076923
Min Index    2020-01-05 00:00:00
Max Index    2020-01-07 00:00:00
Name: a, dtype: object

>>> dd_ma.to_pd()  # (2)!
                   a         b
2020-01-01       NaN       NaN
2020-01-02       NaN       NaN
2020-01-03       NaN       NaN
2020-01-04       NaN       NaN
2020-01-05 -0.083333       NaN
2020-01-06       NaN       NaN
2020-01-07 -0.076923 -0.214286
```

```python
a
```

```python
b
```

Thanks to ColumnMapper and Analyzable, we can select rows and columns from a mapped array the same way as from records or any regular Pandas object:

```python
>>> dd_ma['b'].values
array([-0.21428571])
```

```python
>>> dd_ma['b'].values
array([-0.21428571])
```

## Summary¶

Kudos for following me all the way down here! The classes that we just covered build a strong foundation for data analysis with vectorbt; they implement design patterns that are encountered in most other places across the codebase, which makes them very easy to recognize and extend. In fact, the most hard-core class Portfolio is very similar to our CorrStats.

```python
CorrStats
```

You're now more than ready for using vectorbt, soldier

Python code

