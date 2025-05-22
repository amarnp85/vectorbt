# hdf module¶

Module with HDFData.

## HDFData class¶

```python
HDFData(
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
HDFData(
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

Data class for fetching HDF data using PyTables.

Superclasses

Inherited members

### fetch_feature class method¶

```python
HDFData.fetch_feature(
    feature,
    **kwargs
)
```

```python
HDFData.fetch_feature(
    feature,
    **kwargs
)
```

Fetch the HDF object of a feature.

Uses HDFData.fetch_key().

### fetch_key class method¶

```python
HDFData.fetch_key(
    key,
    path=None,
    start=None,
    end=None,
    tz=None,
    start_row=None,
    end_row=None,
    chunk_func=None,
    **read_kwargs
)
```

```python
HDFData.fetch_key(
    key,
    path=None,
    start=None,
    end=None,
    tz=None,
    start_row=None,
    end_row=None,
    chunk_func=None,
    **read_kwargs
)
```

Fetch the HDF object of a feature or symbol.

Args

```python
key
```

```python
hashable
```

```python
path
```

```python
str
```

Path.

Will be resolved with HDFData.split_hdf_path().

If path is None, uses key as the path to the HDF file.

```python
path
```

```python
key
```

```python
start
```

```python
any
```

Start datetime.

Will extract the object's index and compare the index to the date. Will use the timezone of the object. See to_timestamp().

Note

Can only be used if the object was saved in the table format!

```python
end
```

```python
any
```

End datetime.

Will extract the object's index and compare the index to the date. Will use the timezone of the object. See to_timestamp().

Note

Can only be used if the object was saved in the table format!

```python
tz
```

```python
any
```

Target timezone.

See to_timezone().

```python
start_row
```

```python
int
```

Start row (inclusive).

Will use it when querying index as well.

```python
end_row
```

```python
int
```

End row (exclusive).

Will use it when querying index as well.

```python
chunk_func
```

```python
callable
```

Function to select and concatenate chunks from TableIterator.

```python
TableIterator
```

Gets called only if iterator or chunksize are set.

```python
iterator
```

```python
chunksize
```

```python
**read_kwargs
```

```python
pd.read_hdf
```

See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for other arguments.

For defaults, see custom.hdf in data.

```python
custom.hdf
```

### fetch_symbol class method¶

```python
HDFData.fetch_symbol(
    symbol,
    **kwargs
)
```

```python
HDFData.fetch_symbol(
    symbol,
    **kwargs
)
```

Load the HDF object for a symbol.

Uses HDFData.fetch_key().

### is_hdf_file class method¶

```python
HDFData.is_hdf_file(
    path
)
```

```python
HDFData.is_hdf_file(
    path
)
```

Return whether the path is an HDF file.

### match_path class method¶

```python
HDFData.match_path(
    path,
    match_regex=None,
    sort_paths=True,
    recursive=True,
    **kwargs
)
```

```python
HDFData.match_path(
    path,
    match_regex=None,
    sort_paths=True,
    recursive=True,
    **kwargs
)
```

Override FileData.match_path to return a list of HDF paths (path to file + key) matching a path.

```python
FileData.match_path
```

### split_hdf_path class method¶

```python
HDFData.split_hdf_path(
    path,
    key=None
)
```

```python
HDFData.split_hdf_path(
    path,
    key=None
)
```

Split the path to an HDF object into the path to the file and the key.

### update_feature method¶

```python
HDFData.update_feature(
    feature,
    **kwargs
)
```

```python
HDFData.update_feature(
    feature,
    **kwargs
)
```

Update data of a feature.

Uses HDFData.update_key() with key_is_feature=True.

```python
key_is_feature=True
```

### update_key method¶

```python
HDFData.update_key(
    key,
    key_is_feature=False,
    **kwargs
)
```

```python
HDFData.update_key(
    key,
    key_is_feature=False,
    **kwargs
)
```

Update data of a feature or symbol.

### update_symbol method¶

```python
HDFData.update_symbol(
    symbol,
    **kwargs
)
```

```python
HDFData.update_symbol(
    symbol,
    **kwargs
)
```

Update data for a symbol.

Uses HDFData.update_key() with key_is_feature=False.

```python
key_is_feature=False
```

## HDFKeyNotFoundError class¶

```python
HDFKeyNotFoundError(
    *args,
    **kwargs
)
```

```python
HDFKeyNotFoundError(
    *args,
    **kwargs
)
```

Gets raised if the key to an HDF object could not be found.

Superclasses

```python
builtins.BaseException
```

```python
builtins.Exception
```

## HDFPathNotFoundError class¶

```python
HDFPathNotFoundError(
    *args,
    **kwargs
)
```

```python
HDFPathNotFoundError(
    *args,
    **kwargs
)
```

Gets raised if the path to an HDF file could not be found.

Superclasses

```python
builtins.BaseException
```

```python
builtins.Exception
```

