# base module¶

Base class for working with data sources.

## BaseDataMixin class¶

```python
BaseDataMixin()
```

```python
BaseDataMixin()
```

Base mixin class for working with data.

Subclasses

### assert_has_feature method¶

```python
BaseDataMixin.assert_has_feature(
    feature
)
```

```python
BaseDataMixin.assert_has_feature(
    feature
)
```

Assert that feature exists.

### assert_has_symbol method¶

```python
BaseDataMixin.assert_has_symbol(
    symbol
)
```

```python
BaseDataMixin.assert_has_symbol(
    symbol
)
```

Assert that symbol exists.

### feature_wrapper property¶

Column wrapper.

### features property¶

List of features.

### get method¶

```python
BaseDataMixin.get(
    features=None,
    symbols=None,
    feature=None,
    symbol=None,
    **kwargs
)
```

```python
BaseDataMixin.get(
    features=None,
    symbols=None,
    feature=None,
    symbol=None,
    **kwargs
)
```

Get one or more features of one or more symbols of data.

### get_feature method¶

```python
BaseDataMixin.get_feature(
    feature,
    raise_error=False
)
```

```python
BaseDataMixin.get_feature(
    feature,
    raise_error=False
)
```

Get feature that match a feature index or label.

### get_feature_idx method¶

```python
BaseDataMixin.get_feature_idx(
    feature,
    raise_error=False
)
```

```python
BaseDataMixin.get_feature_idx(
    feature,
    raise_error=False
)
```

Return the index of a feature.

### get_symbol method¶

```python
BaseDataMixin.get_symbol(
    symbol,
    raise_error=False
)
```

```python
BaseDataMixin.get_symbol(
    symbol,
    raise_error=False
)
```

Get symbol that match a symbol index or label.

### get_symbol_idx method¶

```python
BaseDataMixin.get_symbol_idx(
    symbol,
    raise_error=False
)
```

```python
BaseDataMixin.get_symbol_idx(
    symbol,
    raise_error=False
)
```

Return the index of a symbol.

### has_feature method¶

```python
BaseDataMixin.has_feature(
    feature
)
```

```python
BaseDataMixin.has_feature(
    feature
)
```

Whether feature exists.

### has_multiple_keys class method¶

```python
BaseDataMixin.has_multiple_keys(
    keys
)
```

```python
BaseDataMixin.has_multiple_keys(
    keys
)
```

Check whether there are one or multiple keys.

### has_symbol method¶

```python
BaseDataMixin.has_symbol(
    symbol
)
```

```python
BaseDataMixin.has_symbol(
    symbol
)
```

Whether symbol exists.

### prepare_key class method¶

```python
BaseDataMixin.prepare_key(
    key
)
```

```python
BaseDataMixin.prepare_key(
    key
)
```

Prepare a key.

### select_feature_idxs method¶

```python
BaseDataMixin.select_feature_idxs(
    idxs,
    **kwargs
)
```

```python
BaseDataMixin.select_feature_idxs(
    idxs,
    **kwargs
)
```

Select one or more features by index.

Returns a new instance.

### select_features method¶

```python
BaseDataMixin.select_features(
    features,
    **kwargs
)
```

```python
BaseDataMixin.select_features(
    features,
    **kwargs
)
```

Select one or more features.

Returns a new instance.

### select_symbol_idxs method¶

```python
BaseDataMixin.select_symbol_idxs(
    idxs,
    **kwargs
)
```

```python
BaseDataMixin.select_symbol_idxs(
    idxs,
    **kwargs
)
```

Select one or more symbols by index.

Returns a new instance.

### select_symbols method¶

```python
BaseDataMixin.select_symbols(
    symbols,
    **kwargs
)
```

```python
BaseDataMixin.select_symbols(
    symbols,
    **kwargs
)
```

Select one or more symbols.

Returns a new instance.

### symbol_wrapper property¶

Symbol wrapper.

### symbols property¶

List of symbols.

## Data class¶

```python
Data(
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
Data(
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

Class that downloads, updates, and manages data coming from a data source.

Superclasses

Inherited members

Subclasses

### add method¶

```python
Data.add(
    key,
    data=None,
    **kwargs
)
```

```python
Data.add(
    key,
    data=None,
    **kwargs
)
```

Create a new Data instance with a new feature or symbol added to this instance.

Will try to determine the orientation automatically.

### add_column method¶

```python
Data.add_column(
    column,
    data=None,
    **kwargs
)
```

```python
Data.add_column(
    column,
    data=None,
    **kwargs
)
```

Create a new Data instance with a new column added to this instance.

### add_feature method¶

```python
Data.add_feature(
    feature,
    data=None,
    pull_feature=False,
    pull_kwargs=None,
    reuse_fetch_kwargs=True,
    run_kwargs=None,
    wrap_kwargs=None,
    merge_kwargs=None,
    **kwargs
)
```

```python
Data.add_feature(
    feature,
    data=None,
    pull_feature=False,
    pull_kwargs=None,
    reuse_fetch_kwargs=True,
    run_kwargs=None,
    wrap_kwargs=None,
    merge_kwargs=None,
    **kwargs
)
```

Create a new Data instance with a new feature added to this instance.

If data is None, uses Data.run(). If in addition pull_feature is True, uses Data.pull() instead.

```python
data
```

```python
pull_feature
```

### add_key method¶

```python
Data.add_key(
    key,
    data=None,
    **kwargs
)
```

```python
Data.add_key(
    key,
    data=None,
    **kwargs
)
```

Create a new Data instance with a new key added to this instance.

### add_symbol method¶

```python
Data.add_symbol(
    symbol,
    data=None,
    pull_kwargs=None,
    reuse_fetch_kwargs=True,
    merge_kwargs=None,
    **kwargs
)
```

```python
Data.add_symbol(
    symbol,
    data=None,
    pull_kwargs=None,
    reuse_fetch_kwargs=True,
    merge_kwargs=None,
    **kwargs
)
```

Create a new Data instance with a new symbol added to this instance.

If data is None, uses Data.pull().

```python
data
```

### align_columns class method¶

```python
Data.align_columns(
    data,
    missing=None,
    silence_warnings=None
)
```

```python
Data.align_columns(
    data,
    missing=None,
    silence_warnings=None
)
```

Align data to have the same columns.

See Data.align_index() for missing.

```python
missing
```

### align_data class method¶

```python
Data.align_data(
    data,
    last_index=None,
    delisted=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    silence_warnings=None
)
```

```python
Data.align_data(
    data,
    last_index=None,
    delisted=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    silence_warnings=None
)
```

Align data.

Removes any index duplicates, prepares the datetime index, and aligns the index and columns.

### align_index class method¶

```python
Data.align_index(
    data,
    missing=None,
    silence_warnings=None
)
```

```python
Data.align_index(
    data,
    missing=None,
    silence_warnings=None
)
```

Align data to have the same index.

The argument missing accepts the following values:

```python
missing
```

For defaults, see data.

### build_feature_config_doc class method¶

```python
Data.build_feature_config_doc(
    source_cls=None
)
```

```python
Data.build_feature_config_doc(
    source_cls=None
)
```

Build feature config documentation.

### check_dict_type class method¶

```python
Data.check_dict_type(
    arg,
    arg_name=None,
    dict_type=None
)
```

```python
Data.check_dict_type(
    arg,
    arg_name=None,
    dict_type=None
)
```

Check whether the argument conforms to a data dictionary.

### classes property¶

Key classes.

### column_stack class method¶

```python
Data.column_stack(
    *objs,
    wrapper_kwargs=None,
    **kwargs
)
```

```python
Data.column_stack(
    *objs,
    wrapper_kwargs=None,
    **kwargs
)
```

Stack multiple Data instances along columns.

Uses ArrayWrapper.column_stack() to stack the wrappers.

### column_type property¶

Return the column type.

### columns property¶

Columns.

Based on the default symbol wrapper.

### concat method¶

```python
Data.concat(
    keys=None,
    attach_classes=True,
    clean_index_kwargs=None,
    **kwargs
)
```

```python
Data.concat(
    keys=None,
    attach_classes=True,
    clean_index_kwargs=None,
    **kwargs
)
```

Concatenate keys along columns.

### data property¶

Data dictionary.

Has the type feature_dict for feature-oriented data or symbol_dict for symbol-oriented data.

### delisted property¶

Delisted flag per symbol of type symbol_dict.

### dict_type property¶

Return the dict type.

### dropna method¶

```python
Data.dropna(
    **kwargs
)
```

```python
Data.dropna(
    **kwargs
)
```

Drop missing values.

Keyword arguments are passed to Data.transform() and then to pd.Series.dropna or pd.DataFrame.dropna.

```python
pd.Series.dropna
```

```python
pd.DataFrame.dropna
```

### feature_classes property¶

Feature classes.

### feature_config class variable¶

Column config of Data.

```python
HybridConfig()
```

```python
HybridConfig()
```

Returns Data._feature_config, which gets (hybrid-) copied upon creation of each instance. Thus, changing this config won't affect the class.

```python
Data._feature_config
```

To change fields, you can either change the config in-place, override this property, or overwrite the instance variable Data._feature_config.

```python
Data._feature_config
```

### feature_oriented property¶

Whether data has features as keys.

### fetch class method¶

```python
Data.fetch(
    *args,
    **kwargs
)
```

```python
Data.fetch(
    *args,
    **kwargs
)
```

Exists for backward compatibility. Use Data.pull() instead.

### fetch_feature class method¶

```python
Data.fetch_feature(
    feature,
    **kwargs
)
```

```python
Data.fetch_feature(
    feature,
    **kwargs
)
```

Fetch a feature.

Can also return a dictionary that will be accessible in Data.returned_kwargs. If there are keyword arguments tz_localize, tz_convert, or freq in this dict, will pop them and use them to override global settings.

```python
tz_localize
```

```python
tz_convert
```

```python
freq
```

This is an abstract method - override it to define custom logic.

### fetch_kwargs property¶

Keyword arguments of type symbol_dict initially passed to Data.fetch_symbol().

### fetch_symbol class method¶

```python
Data.fetch_symbol(
    symbol,
    **kwargs
)
```

```python
Data.fetch_symbol(
    symbol,
    **kwargs
)
```

Fetch a symbol.

Can also return a dictionary that will be accessible in Data.returned_kwargs. If there are keyword arguments tz_localize, tz_convert, or freq in this dict, will pop them and use them to override global settings.

```python
tz_localize
```

```python
tz_convert
```

```python
freq
```

This is an abstract method - override it to define custom logic.

### fix_data_dict_type class method¶

```python
Data.fix_data_dict_type(
    data
)
```

```python
Data.fix_data_dict_type(
    data
)
```

Fix dict type for data.

### fix_dict_types_in_kwargs class method¶

```python
Data.fix_dict_types_in_kwargs(
    data_type,
    **kwargs
)
```

```python
Data.fix_dict_types_in_kwargs(
    data_type,
    **kwargs
)
```

Fix dict types in keyword arguments.

### freq property¶

Frequency.

Based on the default symbol wrapper.

### from_csv class method¶

```python
Data.from_csv(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_csv(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use CSVData to load data from CSV and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### from_data class method¶

```python
Data.from_data(
    data,
    columns_are_symbols=False,
    invert_data=False,
    single_key=True,
    classes=None,
    level_name=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    wrapper_kwargs=None,
    fetch_kwargs=None,
    returned_kwargs=None,
    last_index=None,
    delisted=None,
    silence_warnings=None,
    **kwargs
)
```

```python
Data.from_data(
    data,
    columns_are_symbols=False,
    invert_data=False,
    single_key=True,
    classes=None,
    level_name=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    wrapper_kwargs=None,
    fetch_kwargs=None,
    returned_kwargs=None,
    last_index=None,
    delisted=None,
    silence_warnings=None,
    **kwargs
)
```

Create a new Data instance from data.

Args

```python
data
```

```python
dict
```

```python
columns_are_symbols
```

```python
bool
```

```python
invert_data
```

```python
bool
```

```python
single_key
```

```python
bool
```

```python
classes
```

```python
level_name
```

```python
bool
```

```python
hashable
```

```python
iterable
```

```python
hashable
```

```python
tz_localize
```

```python
timezone_like
```

```python
tz_convert
```

```python
timezone_like
```

```python
missing_index
```

```python
str
```

```python
missing_columns
```

```python
str
```

```python
wrapper_kwargs
```

```python
dict
```

```python
fetch_kwargs
```

```python
returned_kwargs
```

```python
last_index
```

```python
delisted
```

```python
silence_warnings
```

```python
bool
```

```python
**kwargs
```

```python
__init__
```

For defaults, see data.

### from_data_str class method¶

```python
Data.from_data_str(
    data_str
)
```

```python
Data.from_data_str(
    data_str
)
```

Parse a Data instance from a string.

For example: YFData:BTC-USD or just BTC-USD where the data class is YFData by default.

```python
YFData:BTC-USD
```

```python
BTC-USD
```

### from_duckdb class method¶

```python
Data.from_duckdb(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_duckdb(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use DuckDBData to load data from a DuckDB database and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### from_feather class method¶

```python
Data.from_feather(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_feather(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use FeatherData to load data from Feather and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### from_hdf class method¶

```python
Data.from_hdf(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_hdf(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use HDFData to load data from HDF and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### from_parquet class method¶

```python
Data.from_parquet(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_parquet(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use ParquetData to load data from Parquet and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### from_sql class method¶

```python
Data.from_sql(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

```python
Data.from_sql(
    *args,
    fetch_kwargs=None,
    **kwargs
)
```

Use SQLData to load data from a SQL database and switch the class back to this class.

Use fetch_kwargs to provide keyword arguments that were originally used in fetching.

```python
fetch_kwargs
```

### get_base_setting class method¶

```python
Data.get_base_setting(
    *args,
    **kwargs
)
```

```python
Data.get_base_setting(
    *args,
    **kwargs
)
```

CustomData.get_setting with path_id="base".

```python
CustomData.get_setting
```

```python
path_id="base"
```

### get_base_settings class method¶

```python
Data.get_base_settings(
    *args,
    **kwargs
)
```

```python
Data.get_base_settings(
    *args,
    **kwargs
)
```

CustomData.get_settings with path_id="base".

```python
CustomData.get_settings
```

```python
path_id="base"
```

### get_feature_wrapper method¶

```python
Data.get_feature_wrapper(
    features=None,
    **kwargs
)
```

```python
Data.get_feature_wrapper(
    features=None,
    **kwargs
)
```

Get wrapper with features as columns.

### get_intersection_dict class method¶

```python
Data.get_intersection_dict(
    dct
)
```

```python
Data.get_intersection_dict(
    dct
)
```

Get sub-keys and corresponding sub-values that are the same for all keys.

### get_key_index class method¶

```python
Data.get_key_index(
    keys=None,
    level_name=None,
    feature_oriented=None
)
```

```python
Data.get_key_index(
    keys=None,
    level_name=None,
    feature_oriented=None
)
```

Get key index.

### get_key_wrapper method¶

```python
Data.get_key_wrapper(
    keys=None,
    attach_classes=True,
    clean_index_kwargs=None,
    group_by=None,
    **kwargs
)
```

```python
Data.get_key_wrapper(
    keys=None,
    attach_classes=True,
    clean_index_kwargs=None,
    group_by=None,
    **kwargs
)
```

Get wrapper with keys as columns.

If attach_classes is True, attaches Data.classes by stacking them over the keys using stack_indexes().

```python
attach_classes
```

Other keyword arguments are passed to the constructor of the wrapper.

### get_keys method¶

```python
Data.get_keys(
    dict_type
)
```

```python
Data.get_keys(
    dict_type
)
```

Get keys depending on the provided dict type.

### get_level_name class method¶

```python
Data.get_level_name(
    keys=None,
    level_name=None,
    feature_oriented=None
)
```

```python
Data.get_level_name(
    keys=None,
    level_name=None,
    feature_oriented=None
)
```

Get level name(s) for keys.

### get_symbol_wrapper method¶

```python
Data.get_symbol_wrapper(
    symbols=None,
    **kwargs
)
```

```python
Data.get_symbol_wrapper(
    symbols=None,
    **kwargs
)
```

Get wrapper with symbols as columns.

### has_base_setting class method¶

```python
Data.has_base_setting(
    *args,
    **kwargs
)
```

```python
Data.has_base_setting(
    *args,
    **kwargs
)
```

CustomData.has_setting with path_id="base".

```python
CustomData.has_setting
```

```python
path_id="base"
```

### has_base_settings class method¶

```python
Data.has_base_settings(
    *args,
    **kwargs
)
```

```python
Data.has_base_settings(
    *args,
    **kwargs
)
```

CustomData.has_settings with path_id="base".

```python
CustomData.has_settings
```

```python
path_id="base"
```

### has_key_dict class method¶

```python
Data.has_key_dict(
    arg,
    dict_type=None
)
```

```python
Data.has_key_dict(
    arg,
    dict_type=None
)
```

Check whether the argument contains any data dictionary.

### index property¶

Index.

Based on the default symbol wrapper.

### indexing_func method¶

```python
Data.indexing_func(
    *args,
    replace_kwargs=None,
    **kwargs
)
```

```python
Data.indexing_func(
    *args,
    replace_kwargs=None,
    **kwargs
)
```

Perform indexing on Data.

### invert method¶

```python
Data.invert(
    key_wrapper_kwargs=None,
    **kwargs
)
```

```python
Data.invert(
    key_wrapper_kwargs=None,
    **kwargs
)
```

Invert data and return a new instance.

### invert_data class method¶

```python
Data.invert_data(
    dct
)
```

```python
Data.invert_data(
    dct
)
```

Invert data by swapping keys and columns.

### items method¶

```python
Data.items(
    over='symbols',
    group_by=None,
    apply_group_by=False,
    keep_2d=False,
    key_as_index=False
)
```

```python
Data.items(
    over='symbols',
    group_by=None,
    apply_group_by=False,
    keep_2d=False,
    key_as_index=False
)
```

Iterate over columns (or groups if grouped and Wrapping.group_select is True), keys, features, or symbols. The respective mode can be selected with over.

```python
Wrapping.group_select
```

```python
over
```

See Wrapping.items() for iteration over columns. Iteration over keys supports group_by but doesn't support apply_group_by.

```python
group_by
```

```python
apply_group_by
```

### key_index property¶

Key index.

### key_wrapper property¶

Key wrapper.

### keys property¶

Keys in data.

Features if feature_dict and symbols if symbol_dict.

### last_index property¶

Last fetched index per symbol of type symbol_dict.

### level_name property¶

Level name(s) for keys.

Keys are symbols or features depending on the data dict type.

Must be a sequence if keys are tuples, otherwise a hashable. If False, no level names will be used.

### merge class method¶

```python
Data.merge(
    *datas,
    rename=None,
    **kwargs
)
```

```python
Data.merge(
    *datas,
    rename=None,
    **kwargs
)
```

Merge multiple Data instances.

Can merge both symbols and features. Data is overridden in the order as provided in datas.

```python
datas
```

### metrics class variable¶

Metrics supported by Data.

```python
HybridConfig(
    start_index=dict(
        title='Start Index',
        calc_func=<function Data.<lambda> at 0x15f5b94e0>,
        agg_func=None,
        tags='wrapper'
    ),
    end_index=dict(
        title='End Index',
        calc_func=<function Data.<lambda> at 0x15f5b9580>,
        agg_func=None,
        tags='wrapper'
    ),
    total_duration=dict(
        title='Total Duration',
        calc_func=<function Data.<lambda> at 0x15f5b9620>,
        apply_to_timedelta=True,
        agg_func=None,
        tags='wrapper'
    ),
    total_features=dict(
        title='Total Features',
        check_is_feature_oriented=True,
        calc_func=<function Data.<lambda> at 0x15f5b96c0>,
        agg_func=None,
        tags='data'
    ),
    total_symbols=dict(
        title='Total Symbols',
        check_is_symbol_oriented=True,
        calc_func=<function Data.<lambda> at 0x15f5b9760>,
        tags='data'
    ),
    null_counts=dict(
        title='Null Counts',
        calc_func=<function Data.<lambda> at 0x15f5b9800>,
        agg_func=<function Data.<lambda> at 0x15f5b98a0>,
        tags='data'
    )
)
```

```python
HybridConfig(
    start_index=dict(
        title='Start Index',
        calc_func=<function Data.<lambda> at 0x15f5b94e0>,
        agg_func=None,
        tags='wrapper'
    ),
    end_index=dict(
        title='End Index',
        calc_func=<function Data.<lambda> at 0x15f5b9580>,
        agg_func=None,
        tags='wrapper'
    ),
    total_duration=dict(
        title='Total Duration',
        calc_func=<function Data.<lambda> at 0x15f5b9620>,
        apply_to_timedelta=True,
        agg_func=None,
        tags='wrapper'
    ),
    total_features=dict(
        title='Total Features',
        check_is_feature_oriented=True,
        calc_func=<function Data.<lambda> at 0x15f5b96c0>,
        agg_func=None,
        tags='data'
    ),
    total_symbols=dict(
        title='Total Symbols',
        check_is_symbol_oriented=True,
        calc_func=<function Data.<lambda> at 0x15f5b9760>,
        tags='data'
    ),
    null_counts=dict(
        title='Null Counts',
        calc_func=<function Data.<lambda> at 0x15f5b9800>,
        agg_func=<function Data.<lambda> at 0x15f5b98a0>,
        tags='data'
    )
)
```

Returns Data._metrics, which gets (hybrid-) copied upon creation of each instance. Thus, changing this config won't affect the class.

```python
Data._metrics
```

To change metrics, you can either change the config in-place, override this property, or overwrite the instance variable Data._metrics.

```python
Data._metrics
```

### missing_columns property¶

Argument missing passed to Data.align_columns().

```python
missing
```

### missing_index property¶

Argument missing passed to Data.align_index().

```python
missing
```

### ndim property¶

Number of dimensions.

Based on the default symbol wrapper.

### override_feature_config_doc class method¶

```python
Data.override_feature_config_doc(
    __pdoc__,
    source_cls=None
)
```

```python
Data.override_feature_config_doc(
    __pdoc__,
    source_cls=None
)
```

Call this method on each subclass that overrides Data.feature_config.

### plot method¶

```python
Data.plot(
    column=None,
    feature=None,
    symbol=None,
    feature_map=None,
    plot_volume=None,
    base=None,
    **kwargs
)
```

```python
Data.plot(
    column=None,
    feature=None,
    symbol=None,
    feature_map=None,
    plot_volume=None,
    base=None,
    **kwargs
)
```

Plot either one feature of multiple symbols, or OHLC(V) of one symbol.

Args

```python
column
```

```python
hashable
```

Name of the feature or symbol to plot.

Depends on the data orientation.

```python
feature
```

```python
hashable
```

```python
symbol
```

```python
hashable
```

```python
feature_map
```

```python
sequence
```

```python
str
```

Dictionary mapping the feature names to OHLCV.

Applied only if OHLC(V) is plotted.

```python
plot_volume
```

```python
bool
```

Whether to plot volume beneath.

Applied only if OHLC(V) is plotted.

```python
base
```

```python
float
```

Rebase all series of a feature to a given initial base.

Note

The feature must contain prices.

Applied only if lines are plotted.

```python
kwargs
```

```python
dict
```

Usage

```python
>>> from vectorbtpro import *

>>> start = '2021-01-01 UTC'  # crypto is in UTC
>>> end = '2021-06-01 UTC'
>>> data = vbt.YFData.pull(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
```

```python
>>> from vectorbtpro import *

>>> start = '2021-01-01 UTC'  # crypto is in UTC
>>> end = '2021-06-01 UTC'
>>> data = vbt.YFData.pull(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
```

100%

100%

```python
>>> data.plot(feature='Close', base=1).show()
```

```python
>>> data.plot(feature='Close', base=1).show()
```

```python
>>> data.plot(symbol='BTC-USD').show()
```

```python
>>> data.plot(symbol='BTC-USD').show()
```

### plots_defaults property¶

Defaults for PlotsBuilderMixin.plots().

Merges PlotsBuilderMixin.plots_defaults and plots from data.

```python
plots
```

### prepare_dt class method¶

```python
Data.prepare_dt(
    obj,
    parse_dates=True,
    to_utc=True,
    remove_utc_tz=False
)
```

```python
Data.prepare_dt(
    obj,
    parse_dates=True,
    to_utc=True,
    remove_utc_tz=False
)
```

Prepare datetime index and columns.

If parse_dates is True, will try to convert any index and column with object data type into a datetime format using prepare_dt_index(). If parse_dates is a list or dict, will first check whether the name of the column is among the names that are in parse_dates.

```python
parse_dates
```

```python
parse_dates
```

```python
parse_dates
```

If to_utc is True or to_utc is "index" or to_utc is a sequence and index name is in this sequence, will localize/convert any datetime index to the UTC timezone. If to_utc is True or to_utc is "columns" or to_utc is a sequence and column name is in this sequence, will localize/convert any datetime column to the UTC timezone.

```python
to_utc
```

```python
to_utc
```

```python
to_utc
```

```python
to_utc
```

```python
to_utc
```

```python
to_utc
```

### prepare_dt_column class method¶

```python
Data.prepare_dt_column(
    sr,
    parse_dates=False,
    tz_localize=None,
    tz_convert=None,
    force_tz_convert=False,
    remove_tz=False
)
```

```python
Data.prepare_dt_column(
    sr,
    parse_dates=False,
    tz_localize=None,
    tz_convert=None,
    force_tz_convert=False,
    remove_tz=False
)
```

Prepare datetime column.

See Data.prepare_dt_index() for arguments.

### prepare_dt_index class method¶

```python
Data.prepare_dt_index(
    index,
    parse_dates=False,
    tz_localize=None,
    tz_convert=None,
    force_tz_convert=False,
    remove_tz=False
)
```

```python
Data.prepare_dt_index(
    index,
    parse_dates=False,
    tz_localize=None,
    tz_convert=None,
    force_tz_convert=False,
    remove_tz=False
)
```

Prepare datetime index.

If parse_dates is True, will try to convert the index with an object data type into a datetime format using prepare_dt_index().

```python
parse_dates
```

If tz_localize is not None, will localize a datetime-naive index into this timezone.

```python
tz_localize
```

If tz_convert is not None, will convert a datetime-aware index into this timezone. If force_tz_convert is True, will convert regardless of whether the index is datetime-aware.

```python
tz_convert
```

```python
force_tz_convert
```

### prepare_tzaware_index class method¶

```python
Data.prepare_tzaware_index(
    obj,
    tz_localize=None,
    tz_convert=None
)
```

```python
Data.prepare_tzaware_index(
    obj,
    tz_localize=None,
    tz_convert=None
)
```

Prepare a timezone-aware index of a Pandas object.

Uses Data.prepare_dt_index() with parse_dates=True and force_tz_convert=True.

```python
parse_dates=True
```

```python
force_tz_convert=True
```

For defaults, see data.

### pull class method¶

```python
Data.pull(
    keys=None,
    *,
    keys_are_features=None,
    features=None,
    symbols=None,
    classes=None,
    level_name=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    wrapper_kwargs=None,
    skip_on_error=None,
    silence_warnings=None,
    execute_kwargs=None,
    return_raw=False,
    **kwargs
)
```

```python
Data.pull(
    keys=None,
    *,
    keys_are_features=None,
    features=None,
    symbols=None,
    classes=None,
    level_name=None,
    tz_localize=None,
    tz_convert=None,
    missing_index=None,
    missing_columns=None,
    wrapper_kwargs=None,
    skip_on_error=None,
    silence_warnings=None,
    execute_kwargs=None,
    return_raw=False,
    **kwargs
)
```

Pull data.

Fetches each feature/symbol with Data.fetch_feature()/Data.fetch_symbol() and prepares it with Data.from_data().

Iteration over features/symbols is done using execute(). That is, it can be distributed and parallelized when needed.

Args

```python
keys
```

```python
hashable
```

```python
sequence
```

```python
hashable
```

```python
or dict
```

One or multiple keys.

Depending on keys_are_features will be set to features or symbols.

```python
keys_are_features
```

```python
features
```

```python
symbols
```

```python
keys_are_features
```

```python
bool
```

```python
keys
```

```python
features
```

```python
hashable
```

```python
sequence
```

```python
hashable
```

```python
or dict
```

One or multiple features.

If provided as a dictionary, will use keys as features and values as keyword arguments.

Note

Tuple is considered as a single feature (tuple is a hashable).

```python
symbols
```

```python
hashable
```

```python
sequence
```

```python
hashable
```

```python
or dict
```

One or multiple symbols.

If provided as a dictionary, will use keys as symbols and values as keyword arguments.

Note

Tuple is considered as a single symbol (tuple is a hashable).

```python
classes
```

See Data.classes.

Can be a hashable (single value), a dictionary (class names as keys and class values as values), or a sequence of such.

Note

Tuple is considered as a single class (tuple is a hashable).

```python
level_name
```

```python
bool
```

```python
hashable
```

```python
iterable
```

```python
hashable
```

```python
tz_localize
```

```python
any
```

```python
tz_convert
```

```python
any
```

```python
missing_index
```

```python
str
```

```python
missing_columns
```

```python
str
```

```python
wrapper_kwargs
```

```python
dict
```

```python
skip_on_error
```

```python
bool
```

```python
silence_warnings
```

```python
bool
```

Whether to silence all warnings.

Will also forward this argument to Data.fetch_feature()/Data.fetch_symbol() if in the signature.

```python
execute_kwargs
```

```python
dict
```

```python
return_raw
```

```python
bool
```

```python
**kwargs
```

Passed to Data.fetch_feature()/Data.fetch_symbol().

If two features/symbols require different keyword arguments, pass key_dict or feature_dict/symbol_dict for each argument.

For defaults, see data.

### realign method¶

```python
Data.realign(
    rule=None,
    *args,
    wrapper_meta=None,
    ffill=True,
    **kwargs
)
```

```python
Data.realign(
    rule=None,
    *args,
    wrapper_meta=None,
    ffill=True,
    **kwargs
)
```

Perform realigning on Data.

Looks for realign_func of each feature in Data.feature_config. If no function provided, resamples feature "open" with GenericAccessor.realign_opening() and other features with GenericAccessor.realign_closing().

```python
realign_func
```

### remove method¶

```python
Data.remove(
    keys,
    **kwargs
)
```

```python
Data.remove(
    keys,
    **kwargs
)
```

Create a new Data instance with one or more features or symbols removed from this instance.

Will try to determine the orientation automatically.

### remove_columns method¶

```python
Data.remove_columns(
    columns,
    **kwargs
)
```

```python
Data.remove_columns(
    columns,
    **kwargs
)
```

Create a new Data instance with one or more columns removed from this instance.

### remove_features method¶

```python
Data.remove_features(
    features,
    **kwargs
)
```

```python
Data.remove_features(
    features,
    **kwargs
)
```

Create a new Data instance with one or more features removed from this instance.

### remove_keys method¶

```python
Data.remove_keys(
    keys,
    **kwargs
)
```

```python
Data.remove_keys(
    keys,
    **kwargs
)
```

Create a new Data instance with one or more keys removed from this instance.

### remove_symbols method¶

```python
Data.remove_symbols(
    symbols,
    **kwargs
)
```

```python
Data.remove_symbols(
    symbols,
    **kwargs
)
```

Create a new Data instance with one or more symbols removed from this instance.

### rename method¶

```python
Data.rename(
    rename,
    to=None,
    **kwargs
)
```

```python
Data.rename(
    rename,
    to=None,
    **kwargs
)
```

Create a new Data instance with features or symbols renamed.

Will try to determine the orientation automatically.

### rename_columns method¶

```python
Data.rename_columns(
    rename,
    to=None,
    **kwargs
)
```

```python
Data.rename_columns(
    rename,
    to=None,
    **kwargs
)
```

Create a new Data instance with columns renamed.

### rename_features method¶

```python
Data.rename_features(
    rename,
    to=None,
    **kwargs
)
```

```python
Data.rename_features(
    rename,
    to=None,
    **kwargs
)
```

Create a new Data instance with features renamed.

### rename_in_dict class method¶

```python
Data.rename_in_dict(
    dct,
    rename
)
```

```python
Data.rename_in_dict(
    dct,
    rename
)
```

Rename keys in a dict.

### rename_keys method¶

```python
Data.rename_keys(
    rename,
    to=None,
    **kwargs
)
```

```python
Data.rename_keys(
    rename,
    to=None,
    **kwargs
)
```

Create a new Data instance with keys renamed.

### rename_symbols method¶

```python
Data.rename_symbols(
    rename,
    to=None,
    **kwargs
)
```

```python
Data.rename_symbols(
    rename,
    to=None,
    **kwargs
)
```

Create a new Data instance with symbols renamed.

### replace method¶

```python
Data.replace(
    **kwargs
)
```

```python
Data.replace(
    **kwargs
)
```

See Configured.replace().

Replaces the data's index and/or columns if they were changed in the wrapper.

### resample method¶

```python
Data.resample(
    *args,
    wrapper_meta=None,
    **kwargs
)
```

```python
Data.resample(
    *args,
    wrapper_meta=None,
    **kwargs
)
```

Perform resampling on Data.

Features "open", "high", "low", "close", "volume", "trade count", and "vwap" (case-insensitive) are recognized and resampled automatically.

Looks for resample_func of each feature in Data.feature_config. The function must accept the Data instance, object, and resampler.

```python
resample_func
```

### resolve_base_setting class method¶

```python
Data.resolve_base_setting(
    *args,
    **kwargs
)
```

```python
Data.resolve_base_setting(
    *args,
    **kwargs
)
```

CustomData.resolve_setting with path_id="base".

```python
CustomData.resolve_setting
```

```python
path_id="base"
```

### resolve_columns method¶

```python
Data.resolve_columns(
    columns,
    raise_error=True
)
```

```python
Data.resolve_columns(
    columns,
    raise_error=True
)
```

Return the columns of this instance that match the provided columns.

### resolve_features method¶

```python
Data.resolve_features(
    features,
    raise_error=True
)
```

```python
Data.resolve_features(
    features,
    raise_error=True
)
```

Return the features of this instance that match the provided features.

### resolve_key_arg method¶

```python
Data.resolve_key_arg(
    arg,
    k,
    arg_name,
    check_dict_type=True,
    template_context=None,
    is_kwargs=False
)
```

```python
Data.resolve_key_arg(
    arg,
    k,
    arg_name,
    check_dict_type=True,
    template_context=None,
    is_kwargs=False
)
```

Resolve argument.

### resolve_keys method¶

```python
Data.resolve_keys(
    keys,
    raise_error=True
)
```

```python
Data.resolve_keys(
    keys,
    raise_error=True
)
```

Return the keys of this instance that match the provided keys.

### resolve_keys_meta class method¶

```python
Data.resolve_keys_meta(
    keys=None,
    keys_are_features=None,
    features=None,
    symbols=None
)
```

```python
Data.resolve_keys_meta(
    keys=None,
    keys_are_features=None,
    features=None,
    symbols=None
)
```

Resolve metadata for keys.

### resolve_symbols method¶

```python
Data.resolve_symbols(
    symbols,
    raise_error=True
)
```

```python
Data.resolve_symbols(
    symbols,
    raise_error=True
)
```

Return the symbols of this instance that match the provided symbols.

### returned_kwargs property¶

Keyword arguments of type symbol_dict returned by Data.fetch_symbol().

### row_stack class method¶

```python
Data.row_stack(
    *objs,
    wrapper_kwargs=None,
    **kwargs
)
```

```python
Data.row_stack(
    *objs,
    wrapper_kwargs=None,
    **kwargs
)
```

Stack multiple Data instances along rows.

Uses ArrayWrapper.row_stack() to stack the wrappers.

### run method¶

```python
Data.run(
    func,
    *args,
    on_features=None,
    on_symbols=None,
    func_args=None,
    func_kwargs=None,
    magnet_kwargs=None,
    ignore_args=None,
    rename_args=None,
    location=None,
    prepend_location=None,
    unpack=False,
    concat=True,
    data_kwargs=None,
    silence_warnings=False,
    raise_errors=False,
    execute_kwargs=None,
    filter_results=True,
    raise_no_results=True,
    merge_func=None,
    merge_kwargs=None,
    template_context=None,
    return_keys=False,
    **kwargs
)
```

```python
Data.run(
    func,
    *args,
    on_features=None,
    on_symbols=None,
    func_args=None,
    func_kwargs=None,
    magnet_kwargs=None,
    ignore_args=None,
    rename_args=None,
    location=None,
    prepend_location=None,
    unpack=False,
    concat=True,
    data_kwargs=None,
    silence_warnings=False,
    raise_errors=False,
    execute_kwargs=None,
    filter_results=True,
    raise_no_results=True,
    merge_func=None,
    merge_kwargs=None,
    template_context=None,
    return_keys=False,
    **kwargs
)
```

Run a function on data.

Looks into the signature of the function and searches for arguments with the name data or those found among features or attributes.

```python
data
```

For example, the argument open will be substituted by Data.open.

```python
open
```

func can be one of the following:

```python
func
```

Use magnet_kwargs to provide keyword arguments that will be passed only if found in the signature of the function.

```python
magnet_kwargs
```

Use rename_args to rename arguments. For example, in Portfolio, data can be passed instead of close.

```python
rename_args
```

```python
close
```

Set unpack to True, "dict", or "frame" to use IndicatorBase.unpack(), IndicatorBase.to_dict(), and IndicatorBase.to_frame() respectively.

```python
unpack
```

Any argument in *args and **kwargs can be wrapped with run_func_dict/run_arg_dict to specify the value per function/argument name or index when func is iterable.

```python
*args
```

```python
**kwargs
```

```python
func
```

Multiple function calls are executed with execute().

### select method¶

```python
Data.select(
    keys,
    **kwargs
)
```

```python
Data.select(
    keys,
    **kwargs
)
```

Create a new Data instance with one or more features or symbols selected from this instance.

Will try to determine the orientation automatically.

### select_classes method¶

```python
Data.select_classes(
    key,
    **kwargs
)
```

```python
Data.select_classes(
    key,
    **kwargs
)
```

Select a feature or symbol from Data.classes.

### select_columns method¶

```python
Data.select_columns(
    columns,
    **kwargs
)
```

```python
Data.select_columns(
    columns,
    **kwargs
)
```

Create a new Data instance with one or more columns selected from this instance.

### select_delisted method¶

```python
Data.select_delisted(
    key,
    **kwargs
)
```

```python
Data.select_delisted(
    key,
    **kwargs
)
```

Select a feature or symbol from Data.delisted.

### select_feature_from_dict class method¶

```python
Data.select_feature_from_dict(
    feature,
    dct,
    **kwargs
)
```

```python
Data.select_feature_from_dict(
    feature,
    dct,
    **kwargs
)
```

Select the dictionary value belonging to a feature.

### select_feature_kwargs class method¶

```python
Data.select_feature_kwargs(
    feature,
    kwargs,
    **kwargs_
)
```

```python
Data.select_feature_kwargs(
    feature,
    kwargs,
    **kwargs_
)
```

Select the keyword arguments belonging to a feature.

### select_fetch_kwargs method¶

```python
Data.select_fetch_kwargs(
    key,
    **kwargs
)
```

```python
Data.select_fetch_kwargs(
    key,
    **kwargs
)
```

Select a feature or symbol from Data.fetch_kwargs.

### select_from_dict class method¶

```python
Data.select_from_dict(
    dct,
    keys,
    raise_error=False
)
```

```python
Data.select_from_dict(
    dct,
    keys,
    raise_error=False
)
```

Select keys from a dict.

### select_key_from_dict class method¶

```python
Data.select_key_from_dict(
    key,
    dct,
    dct_name='dct',
    dict_type=None,
    check_dict_type=True
)
```

```python
Data.select_key_from_dict(
    key,
    dct,
    dct_name='dct',
    dict_type=None,
    check_dict_type=True
)
```

Select the dictionary value belonging to a feature or symbol.

### select_key_kwargs class method¶

```python
Data.select_key_kwargs(
    key,
    kwargs,
    kwargs_name='kwargs',
    dict_type=None,
    check_dict_type=True
)
```

```python
Data.select_key_kwargs(
    key,
    kwargs,
    kwargs_name='kwargs',
    dict_type=None,
    check_dict_type=True
)
```

Select the keyword arguments belonging to a feature or symbol.

### select_keys method¶

```python
Data.select_keys(
    keys,
    **kwargs
)
```

```python
Data.select_keys(
    keys,
    **kwargs
)
```

Create a new Data instance with one or more keys selected from this instance.

### select_last_index method¶

```python
Data.select_last_index(
    key,
    **kwargs
)
```

```python
Data.select_last_index(
    key,
    **kwargs
)
```

Select a feature or symbol from Data.last_index.

### select_returned_kwargs method¶

```python
Data.select_returned_kwargs(
    key,
    **kwargs
)
```

```python
Data.select_returned_kwargs(
    key,
    **kwargs
)
```

Select a feature or symbol from Data.returned_kwargs.

### select_run_func_args class method¶

```python
Data.select_run_func_args(
    i,
    func_name,
    args
)
```

```python
Data.select_run_func_args(
    i,
    func_name,
    args
)
```

Select positional arguments that correspond to a runnable function index or name.

### select_run_func_kwargs class method¶

```python
Data.select_run_func_kwargs(
    i,
    func_name,
    kwargs
)
```

```python
Data.select_run_func_kwargs(
    i,
    func_name,
    kwargs
)
```

Select keyword arguments that correspond to a runnable function index or name.

### select_symbol_from_dict class method¶

```python
Data.select_symbol_from_dict(
    symbol,
    dct,
    **kwargs
)
```

```python
Data.select_symbol_from_dict(
    symbol,
    dct,
    **kwargs
)
```

Select the dictionary value belonging to a symbol.

### select_symbol_kwargs class method¶

```python
Data.select_symbol_kwargs(
    symbol,
    kwargs,
    **kwargs_
)
```

```python
Data.select_symbol_kwargs(
    symbol,
    kwargs,
    **kwargs_
)
```

Select the keyword arguments belonging to a symbol.

### set_base_settings class method¶

```python
Data.set_base_settings(
    *args,
    **kwargs
)
```

```python
Data.set_base_settings(
    *args,
    **kwargs
)
```

CustomData.set_settings with path_id="base".

```python
CustomData.set_settings
```

```python
path_id="base"
```

### shape property¶

Shape.

Based on the default symbol wrapper.

### shape_2d property¶

Shape as if the object was two-dimensional.

Based on the default symbol wrapper.

### single_feature property¶

Whether there is only one feature in Data.data.

### single_key property¶

Whether there is only one key in Data.data.

### single_symbol property¶

Whether there is only one symbol in Data.data.

### sql method¶

```python
Data.sql(
    query,
    dbcon=None,
    database=':memory:',
    db_config=None,
    alias='',
    params=None,
    other_objs=None,
    date_as_object=False,
    align_dtypes=True,
    squeeze=True,
    **kwargs
)
```

```python
Data.sql(
    query,
    dbcon=None,
    database=':memory:',
    db_config=None,
    alias='',
    params=None,
    other_objs=None,
    date_as_object=False,
    align_dtypes=True,
    squeeze=True,
    **kwargs
)
```

Run a SQL query on this instance using DuckDB.

First, connection gets established. Then, Data.get() gets invoked with **kwargs passed as keyword arguments and as_dict=True. Then, each returned object gets registered within the database. Finally, the query gets executed with duckdb.sql and the relation as a DataFrame gets returned. If squeeze is True, a DataFrame with one column will be converted into a Series.

```python
**kwargs
```

```python
as_dict=True
```

```python
duckdb.sql
```

```python
squeeze
```

### stats_defaults property¶

Defaults for StatsBuilderMixin.stats().

Merges StatsBuilderMixin.stats_defaults and stats from data.

```python
stats
```

### subplots class variable¶

Subplots supported by Data.

```python
HybridConfig(
    plot=RepEval(
        template='\n                if symbols is None:\n                    symbols = self.symbols\n                if not self.has_multiple_keys(symbols):\n                    symbols = [symbols]\n                [\n                    dict(\n                        check_is_not_grouped=True,\n                        plot_func="plot",\n                        plot_volume=False,\n                        symbol=s,\n                        title=s,\n                        pass_add_trace_kwargs=True,\n                        xaxis_kwargs=dict(rangeslider_visible=False, showgrid=True),\n                        yaxis_kwargs=dict(showgrid=True),\n                        tags="data",\n                    )\n                    for s in symbols\n                ]',
        context=dict(
            symbols=None
        ),
        strict=None,
        context_merge_kwargs=None,
        eval_id=None
    )
)
```

```python
HybridConfig(
    plot=RepEval(
        template='\n                if symbols is None:\n                    symbols = self.symbols\n                if not self.has_multiple_keys(symbols):\n                    symbols = [symbols]\n                [\n                    dict(\n                        check_is_not_grouped=True,\n                        plot_func="plot",\n                        plot_volume=False,\n                        symbol=s,\n                        title=s,\n                        pass_add_trace_kwargs=True,\n                        xaxis_kwargs=dict(rangeslider_visible=False, showgrid=True),\n                        yaxis_kwargs=dict(showgrid=True),\n                        tags="data",\n                    )\n                    for s in symbols\n                ]',
        context=dict(
            symbols=None
        ),
        strict=None,
        context_merge_kwargs=None,
        eval_id=None
    )
)
```

Returns Data._subplots, which gets (hybrid-) copied upon creation of each instance. Thus, changing this config won't affect the class.

```python
Data._subplots
```

To change subplots, you can either change the config in-place, override this property, or overwrite the instance variable Data._subplots.

```python
Data._subplots
```

### switch_class method¶

```python
Data.switch_class(
    new_cls,
    clear_fetch_kwargs=False,
    clear_returned_kwargs=False,
    **kwargs
)
```

```python
Data.switch_class(
    new_cls,
    clear_fetch_kwargs=False,
    clear_returned_kwargs=False,
    **kwargs
)
```

Switch the class of the data instance.

### symbol_classes property¶

Symbol classes.

### symbol_oriented property¶

Whether data has symbols as keys.

### to_csv method¶

```python
Data.to_csv(
    path_or_buf='.',
    ext='csv',
    mkdir_kwargs=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

```python
Data.to_csv(
    path_or_buf='.',
    ext='csv',
    mkdir_kwargs=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

Save data to CSV file(s).

Uses https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

If path_or_buf is a path to a directory, will save each feature/symbol to a separate file. If there's only one file, you can specify the file path via path_or_buf. If there are multiple files, use the same argument but wrap the multiple paths with key_dict.

```python
path_or_buf
```

```python
path_or_buf
```

### to_duckdb method¶

```python
Data.to_duckdb(
    connection=None,
    table=None,
    schema=None,
    catalog=None,
    write_format=None,
    write_path='.',
    write_options=None,
    mkdir_kwargs=None,
    to_utc=None,
    remove_utc_tz=True,
    if_exists='fail',
    connection_config=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    return_connection=False
)
```

```python
Data.to_duckdb(
    connection=None,
    table=None,
    schema=None,
    catalog=None,
    write_format=None,
    write_path='.',
    write_options=None,
    mkdir_kwargs=None,
    to_utc=None,
    remove_utc_tz=True,
    if_exists='fail',
    connection_config=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    return_connection=False
)
```

Save data to a DuckDB database.

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

If connection is None or a string, will resolve a connection with DuckDBData.resolve_connection(). It can additionally return the connection if return_connection is True or entire metadata (all passed arguments as feature_dict or symbol_dict). In this case, the engine won't be disposed by default.

```python
connection
```

```python
return_connection
```

If write_format is None and write_path is a directory (default), will persist each feature/symbol to a table (see https://duckdb.org/docs/guides/python/import_pandas). If catalog is not None, will make it default for this connection. If schema is not None, and it doesn't exist, will create a new schema in the current catalog and make it default for this connection. Any new table will be automatically created under this schema.

```python
write_format
```

```python
write_path
```

```python
catalog
```

```python
schema
```

If if_exists is "fail", will raise an error if a table with the same name already exists. If if_exists is "replace", will drop the existing table first. If if_exists is "append", will append the new table to the existing one.

```python
if_exists
```

```python
if_exists
```

```python
if_exists
```

If write_format is not None, it must be either "csv", "parquet", or "json". If write_path is a directory or has no suffix (meaning it's not a file), each feature/symbol will be saved to a separate file under that path and with the provided write_format as extension. The data will be saved using a COPY mechanism (see https://duckdb.org/docs/sql/statements/copy.html). To provide options to the write operation, pass them as a dictionary or an already formatted string (without brackets). For example, dict(compression="gzip") is same as "COMPRESSION 'gzip'".

```python
write_format
```

```python
write_path
```

```python
write_format
```

```python
COPY
```

```python
dict(compression="gzip")
```

For to_utc and remove_utc_tz, see Data.prepare_dt(). If to_utc is None, uses the corresponding setting of DuckDBData.

```python
to_utc
```

```python
remove_utc_tz
```

```python
to_utc
```

### to_feather method¶

```python
Data.to_feather(
    path_or_buf='.',
    mkdir_kwargs=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

```python
Data.to_feather(
    path_or_buf='.',
    mkdir_kwargs=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

Save data to Feather file(s) using PyArrow.

Uses https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_feather.html

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

If path_or_buf is a path to a directory, will save each feature/symbol to a separate file. If there's only one file, you can specify the file path via path_or_buf. If there are multiple files, use the same argument but wrap the multiple paths with key_dict.

```python
path_or_buf
```

```python
path_or_buf
```

### to_feature_oriented method¶

```python
Data.to_feature_oriented(
    **kwargs
)
```

```python
Data.to_feature_oriented(
    **kwargs
)
```

Convert this instance to the feature-oriented format.

Returns self if the data is already properly formatted.

### to_hdf method¶

```python
Data.to_hdf(
    path_or_buf='.',
    key=None,
    mkdir_kwargs=None,
    format='table',
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

```python
Data.to_hdf(
    path_or_buf='.',
    key=None,
    mkdir_kwargs=None,
    format='table',
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

Save data to an HDF file using PyTables.

Uses https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

If path_or_buf exists and it's a directory, will create inside it a file named after this class.

```python
path_or_buf
```

### to_parquet method¶

```python
Data.to_parquet(
    path_or_buf='.',
    mkdir_kwargs=None,
    partition_cols=None,
    partition_by=None,
    period_index_to='str',
    groupby_kwargs=None,
    keep_groupby_names=False,
    engine=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

```python
Data.to_parquet(
    path_or_buf='.',
    mkdir_kwargs=None,
    partition_cols=None,
    partition_by=None,
    period_index_to='str',
    groupby_kwargs=None,
    keep_groupby_names=False,
    engine=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    **kwargs
)
```

Save data to Parquet file(s) using PyArrow or FastParquet.

Uses https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

If path_or_buf is a path to a directory, will save each feature/symbol to a separate file. If there's only one file, you can specify the file path via path_or_buf. If there are multiple files, use the same argument but wrap the multiple paths with key_dict.

```python
path_or_buf
```

```python
path_or_buf
```

If partition_cols and partition_by are None, path_or_buf must be a file, otherwise it must be a directory. If partition_by is not None, will group the index by using ArrayWrapper.get_index_grouper() with **groupby_kwargs and put it inside partition_cols. In this case, partition_cols must be None.

```python
partition_cols
```

```python
partition_by
```

```python
path_or_buf
```

```python
partition_by
```

```python
**groupby_kwargs
```

```python
partition_cols
```

```python
partition_cols
```

### to_sql method¶

```python
Data.to_sql(
    engine=None,
    table=None,
    schema=None,
    to_utc=None,
    remove_utc_tz=True,
    attach_row_number=False,
    from_row_number=None,
    row_number_column=None,
    engine_config=None,
    dispose_engine=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    return_engine=False,
    **kwargs
)
```

```python
Data.to_sql(
    engine=None,
    table=None,
    schema=None,
    to_utc=None,
    remove_utc_tz=True,
    attach_row_number=False,
    from_row_number=None,
    row_number_column=None,
    engine_config=None,
    dispose_engine=None,
    check_dict_type=True,
    template_context=None,
    return_meta=False,
    return_engine=False,
    **kwargs
)
```

Save data to a SQL database using SQLAlchemy.

Uses https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html

Any argument can be provided per feature using feature_dict or per symbol using symbol_dict, depending on the format of the data dictionary.

Each feature/symbol gets saved to a separate table.

If engine is None or a string, will resolve an engine with SQLData.resolve_engine() and dispose it afterward if dispose_engine is None or True. It can additionally return the engine if return_engine is True or entire metadata (all passed arguments as feature_dict or symbol_dict). In this case, the engine won't be disposed by default.

```python
engine
```

```python
dispose_engine
```

```python
return_engine
```

If schema is not None and it doesn't exist, will create a new schema.

```python
schema
```

For to_utc and remove_utc_tz, see Data.prepare_dt(). If to_utc is None, uses the corresponding setting of SQLData.

```python
to_utc
```

```python
remove_utc_tz
```

```python
to_utc
```

### to_symbol_oriented method¶

```python
Data.to_symbol_oriented(
    **kwargs
)
```

```python
Data.to_symbol_oriented(
    **kwargs
)
```

Convert this instance to the symbol-oriented format.

Returns self if the data is already properly formatted.

### transform method¶

```python
Data.transform(
    transform_func,
    *args,
    per_feature=False,
    per_symbol=False,
    pass_frame=False,
    key_wrapper_kwargs=None,
    template_context=None,
    **kwargs
)
```

```python
Data.transform(
    transform_func,
    *args,
    per_feature=False,
    per_symbol=False,
    pass_frame=False,
    key_wrapper_kwargs=None,
    template_context=None,
    **kwargs
)
```

Transform data.

If one key (i.e., feature or symbol), passes the entire Series/DataFrame. If per_feature is True, passes the Series/DataFrame of each feature. If per_symbol is True, passes the Series/DataFrame of each symbol. If both are True, passes each feature and symbol combination as a Series if pass_frame is False or as a DataFrame with one column if pass_frame is True. If both are False, concatenates all features and symbols into a single DataFrame and calls transform_func on it. Then, splits the data by key and builds a new Data instance. Keyword arguments key_wrapper_kwargs are passed to Data.get_key_wrapper() to control, for example, attachment of classes.

```python
per_feature
```

```python
per_symbol
```

```python
pass_frame
```

```python
pass_frame
```

```python
transform_func
```

```python
key_wrapper_kwargs
```

After the transformation, the new data is aligned using Data.align_data().

Note

The returned object must have the same type and dimensionality as the input object.

Number of columns (i.e., features and symbols) and their names must stay the same. To remove columns, use either indexing or Data.select() (depending on the data orientation). To add new columns, use either column stacking or Data.merge().

Index, on the other hand, can be changed freely.

### try_fetch_feature class method¶

```python
Data.try_fetch_feature(
    feature,
    skip_on_error=False,
    silence_warnings=False,
    fetch_kwargs=None
)
```

```python
Data.try_fetch_feature(
    feature,
    skip_on_error=False,
    silence_warnings=False,
    fetch_kwargs=None
)
```

Try to fetch a feature using Data.fetch_feature().

### try_fetch_symbol class method¶

```python
Data.try_fetch_symbol(
    symbol,
    skip_on_error=False,
    silence_warnings=False,
    fetch_kwargs=None
)
```

```python
Data.try_fetch_symbol(
    symbol,
    skip_on_error=False,
    silence_warnings=False,
    fetch_kwargs=None
)
```

Try to fetch a symbol using Data.fetch_symbol().

### try_run class method¶

```python
Data.try_run(
    data,
    func_name,
    *args,
    raise_errors=False,
    silence_warnings=False,
    **kwargs
)
```

```python
Data.try_run(
    data,
    func_name,
    *args,
    raise_errors=False,
    silence_warnings=False,
    **kwargs
)
```

Try to run a function on data.

### try_update_feature method¶

```python
Data.try_update_feature(
    feature,
    skip_on_error=False,
    silence_warnings=False,
    update_kwargs=None
)
```

```python
Data.try_update_feature(
    feature,
    skip_on_error=False,
    silence_warnings=False,
    update_kwargs=None
)
```

Try to update a feature using Data.update_feature().

### try_update_symbol method¶

```python
Data.try_update_symbol(
    symbol,
    skip_on_error=False,
    silence_warnings=False,
    update_kwargs=None
)
```

```python
Data.try_update_symbol(
    symbol,
    skip_on_error=False,
    silence_warnings=False,
    update_kwargs=None
)
```

Try to update a symbol using Data.update_symbol().

### tz_convert property¶

Timezone to convert a datetime-aware to, which is initially passed to Data.pull().

### tz_localize property¶

Timezone to localize a datetime-naive index to, which is initially passed to Data.pull().

### update method¶

```python
Data.update(
    *,
    concat=True,
    skip_on_error=None,
    silence_warnings=None,
    execute_kwargs=None,
    return_raw=False,
    **kwargs
)
```

```python
Data.update(
    *,
    concat=True,
    skip_on_error=None,
    silence_warnings=None,
    execute_kwargs=None,
    return_raw=False,
    **kwargs
)
```

Update data.

Fetches new data for each feature/symbol using Data.update_feature()/Data.update_symbol().

Args

```python
concat
```

```python
bool
```

```python
skip_on_error
```

```python
bool
```

```python
silence_warnings
```

```python
bool
```

Whether to silence all warnings.

Will also forward this argument to Data.update_feature()/Data.update_symbol() if accepted by Data.fetch_feature()/Data.fetch_symbol().

```python
execute_kwargs
```

```python
dict
```

```python
return_raw
```

```python
bool
```

```python
**kwargs
```

Passed to Data.update_feature()/Data.update_symbol().

If two features/symbols require different keyword arguments, pass key_dict or feature_dict/symbol_dict for each argument.

Note

Returns a new Data instance instead of changing the data in place.

### update_classes method¶

```python
Data.update_classes(
    check_dict_type=True,
    **kwargs
)
```

```python
Data.update_classes(
    check_dict_type=True,
    **kwargs
)
```

Update Data.classes. Returns a new instance.

### update_feature method¶

```python
Data.update_feature(
    feature,
    **kwargs
)
```

```python
Data.update_feature(
    feature,
    **kwargs
)
```

Update a feature.

Can also return a dictionary that will be accessible in Data.returned_kwargs.

This is an abstract method - override it to define custom logic.

### update_fetch_kwargs method¶

```python
Data.update_fetch_kwargs(
    check_dict_type=True,
    **kwargs
)
```

```python
Data.update_fetch_kwargs(
    check_dict_type=True,
    **kwargs
)
```

Update Data.fetch_kwargs. Returns a new instance.

### update_returned_kwargs method¶

```python
Data.update_returned_kwargs(
    check_dict_type=True,
    **kwargs
)
```

```python
Data.update_returned_kwargs(
    check_dict_type=True,
    **kwargs
)
```

Update Data.returned_kwargs. Returns a new instance.

### update_symbol method¶

```python
Data.update_symbol(
    symbol,
    **kwargs
)
```

```python
Data.update_symbol(
    symbol,
    **kwargs
)
```

Update a symbol.

Can also return a dictionary that will be accessible in Data.returned_kwargs.

This is an abstract method - override it to define custom logic.

### use_feature_config_of method¶

```python
Data.use_feature_config_of(
    cls
)
```

```python
Data.use_feature_config_of(
    cls
)
```

Copy feature config from another Data class.

## DataWithFeatures class¶

```python
DataWithFeatures()
```

```python
DataWithFeatures()
```

Class exposes a read-only class property DataWithFeatures.field_config.

```python
DataWithFeatures.field_config
```

Subclasses

### feature_config function¶

Column config of ${cls_name}.

```python
${cls_name}
```

```python
${feature_config}
```

```python
${feature_config}
```

## MetaData class¶

```python
MetaData(
    *args,
    **kwargs
)
```

```python
MetaData(
    *args,
    **kwargs
)
```

Meta class that exposes a read-only class property StatsBuilderMixin.metrics.

```python
StatsBuilderMixin.metrics
```

Superclasses

```python
builtins.type
```

Inherited members

## MetaFeatures class¶

```python
MetaFeatures(
    *args,
    **kwargs
)
```

```python
MetaFeatures(
    *args,
    **kwargs
)
```

Meta class that exposes a read-only class property MetaFeatures.feature_config.

Superclasses

```python
builtins.type
```

Subclasses

### feature_config property¶

Column config.

## OHLCDataMixin class¶

```python
OHLCDataMixin()
```

```python
OHLCDataMixin()
```

Mixin class for working with OHLC data.

Superclasses

Inherited members

Subclasses

### close property¶

Close.

### daily_log_returns property¶

OHLCDataMixin.get_daily_log_returns() with default arguments.

### daily_returns property¶

OHLCDataMixin.get_daily_returns() with default arguments.

### drawdowns property¶

OHLCDataMixin.get_drawdowns() with default arguments.

### get_daily_log_returns method¶

```python
OHLCDataMixin.get_daily_log_returns(
    **kwargs
)
```

```python
OHLCDataMixin.get_daily_log_returns(
    **kwargs
)
```

Daily log returns.

### get_daily_returns method¶

```python
OHLCDataMixin.get_daily_returns(
    **kwargs
)
```

```python
OHLCDataMixin.get_daily_returns(
    **kwargs
)
```

Daily returns.

### get_drawdowns method¶

```python
OHLCDataMixin.get_drawdowns(
    **kwargs
)
```

```python
OHLCDataMixin.get_drawdowns(
    **kwargs
)
```

Generate drawdown records.

See Drawdowns.

### get_log_returns method¶

```python
OHLCDataMixin.get_log_returns(
    **kwargs
)
```

```python
OHLCDataMixin.get_log_returns(
    **kwargs
)
```

Log returns.

### get_returns method¶

```python
OHLCDataMixin.get_returns(
    **kwargs
)
```

```python
OHLCDataMixin.get_returns(
    **kwargs
)
```

Returns.

### get_returns_acc method¶

```python
OHLCDataMixin.get_returns_acc(
    **kwargs
)
```

```python
OHLCDataMixin.get_returns_acc(
    **kwargs
)
```

Return accessor of type ReturnsAccessor.

### has_any_ohlc property¶

Whether the instance has any of the OHLC features.

### has_any_ohlcv property¶

Whether the instance has any of the OHLCV features.

### has_ohlc property¶

Whether the instance has all the OHLC features.

### has_ohlcv property¶

Whether the instance has all the OHLCV features.

### high property¶

High.

### hlc3 property¶

HLC/3.

### log_returns property¶

OHLCDataMixin.get_log_returns() with default arguments.

### low property¶

Low.

### ohlc property¶

Return a OHLCDataMixin instance with the OHLC features only.

### ohlc4 property¶

OHLC/4.

### ohlcv property¶

Return a OHLCDataMixin instance with the OHLCV features only.

### open property¶

Open.

### returns property¶

OHLCDataMixin.get_returns() with default arguments.

### returns_acc property¶

OHLCDataMixin.get_returns_acc() with default arguments.

### trade_count property¶

Trade count.

### volume property¶

Volume.

### vwap property¶

VWAP.

## feature_dict class¶

```python
feature_dict(
    *args,
    **kwargs
)
```

```python
feature_dict(
    *args,
    **kwargs
)
```

Dict that contains features as keys.

Superclasses

```python
builtins.dict
```

Inherited members

## key_dict class¶

```python
key_dict(
    *args,
    **kwargs
)
```

```python
key_dict(
    *args,
    **kwargs
)
```

Dict that contains features or symbols as keys.

Superclasses

```python
builtins.dict
```

Inherited members

Subclasses

## run_arg_dict class¶

```python
run_arg_dict(
    *args,
    **kwargs
)
```

```python
run_arg_dict(
    *args,
    **kwargs
)
```

Dict that contains argument names as keys for Data.run().

Superclasses

```python
builtins.dict
```

Inherited members

## run_func_dict class¶

```python
run_func_dict(
    *args,
    **kwargs
)
```

```python
run_func_dict(
    *args,
    **kwargs
)
```

Dict that contains function names as keys for Data.run().

Superclasses

```python
builtins.dict
```

Inherited members

## symbol_dict class¶

```python
symbol_dict(
    *args,
    **kwargs
)
```

```python
symbol_dict(
    *args,
    **kwargs
)
```

Dict that contains symbols as keys.

Superclasses

```python
builtins.dict
```

Inherited members

