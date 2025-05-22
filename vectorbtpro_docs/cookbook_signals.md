# Signals¶

Question

Learn more in Signal development tutorial.

## Cleaning¶

Only two arrays can be cleaned at a time, for more arrays write a custom Numba function that does the job.

```python
@njit
def custom_clean_nb(long_en, long_ex, short_en, short_ex):
    new_long_en = np.full_like(long_en, False)
    new_long_ex = np.full_like(long_ex, False)
    new_short_en = np.full_like(short_en, False)
    new_short_ex = np.full_like(short_ex, False)

    for col in range(long_en.shape[1]):  # (1)!
        position = 0  # (2)!
        for i in range(long_en.shape[0]):  # (3)!
            if long_en[i, col] and position != 1:
                new_long_en[i, col] = True  # (4)!
                position = 1
            elif short_en[i, col] and position != -1:
                new_short_en[i, col] = True
                position = -1
            elif long_ex[i, col] and position == 1:
                new_long_ex[i, col] = True
                position = 0
            elif short_ex[i, col] and position == -1:
                new_short_ex[i, col] = True
                position = 0

    return new_long_en, new_long_ex, new_short_en, new_short_ex
```

```python
@njit
def custom_clean_nb(long_en, long_ex, short_en, short_ex):
    new_long_en = np.full_like(long_en, False)
    new_long_ex = np.full_like(long_ex, False)
    new_short_en = np.full_like(short_en, False)
    new_short_ex = np.full_like(short_ex, False)

    for col in range(long_en.shape[1]):  # (1)!
        position = 0  # (2)!
        for i in range(long_en.shape[0]):  # (3)!
            if long_en[i, col] and position != 1:
                new_long_en[i, col] = True  # (4)!
                position = 1
            elif short_en[i, col] and position != -1:
                new_short_en[i, col] = True
                position = -1
            elif long_ex[i, col] and position == 1:
                new_long_ex[i, col] = True
                position = 0
            elif short_ex[i, col] and position == -1:
                new_short_ex[i, col] = True
                position = 0

    return new_long_en, new_long_ex, new_short_en, new_short_ex
```

Tip

Convert each input array to NumPy with arr = vbt.to_2d_array(df) and then each output array back to Pandas with new_df = df.vbt.wrapper.wrap(arr).

```python
arr = vbt.to_2d_array(df)
```

```python
new_df = df.vbt.wrapper.wrap(arr)
```

