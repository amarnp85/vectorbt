# Datetime¶

VBT loves flexibility, and so it allows us to construct various datetime-related objects from human-readable strings.

## Timestamps¶

Timestamps represent a single point in time, similar to a datetime object in Python's datetime module, but with enhanced functionality for data analysis and manipulation.

```python
datetime
```

```python
vbt.timestamp()  # (1)!
vbt.utc_timestamp()  # (2)!
vbt.local_timestamp()  # (3)!
vbt.timestamp(tz="America/New_York")  # (4)!
vbt.timestamp("1 Jul 2020")  # (5)!
vbt.timestamp("7 days ago")  # (6)!
```

```python
vbt.timestamp()  # (1)!
vbt.utc_timestamp()  # (2)!
vbt.local_timestamp()  # (3)!
vbt.timestamp(tz="America/New_York")  # (4)!
vbt.timestamp("1 Jul 2020")  # (5)!
vbt.timestamp("7 days ago")  # (6)!
```

```python
pd.Timestamp.now()
```

```python
pd.Timestamp.now(tz="utc")
```

```python
pd.Timestamp.now(tz="tzlocal()")
```

```python
pd.Timestamp.now(tz="New_York/America")
```

```python
pd.Timestamp("2020-07-01")
```

```python
pd.Timestamp.now() - pd.Timedelta(days=7)
```

## Timezones¶

Timezones can be used in timestamps, making it a powerful tool for global time-based data analysis.

```python
vbt.timezone()  # (1)!
vbt.timezone("utc")  # (2)!
vbt.timezone("America/New_York")  # (3)!
vbt.timezone("+0500")  # (4)!
```

```python
vbt.timezone()  # (1)!
vbt.timezone("utc")  # (2)!
vbt.timezone("America/New_York")  # (3)!
vbt.timezone("+0500")  # (4)!
```

## Timedeltas¶

Timedeltas deal with continuous time spans and precise time differences. They are commonly used for adding or subtracting durations from timestamps, or for measuring the difference between two timestamps.

```python
vbt.timedelta()  # (1)!
vbt.timedelta("7 days")  # (2)!
vbt.timedelta("weekly")
vbt.timedelta("Y", approximate=True)  # (3)!
```

```python
vbt.timedelta()  # (1)!
vbt.timedelta("7 days")  # (2)!
vbt.timedelta("weekly")
vbt.timedelta("Y", approximate=True)  # (3)!
```

```python
pd.Timedelta(nanoseconds=1)
```

```python
pd.Timedelta(days=7)
```

## Date offsets¶

Date offsets handle calendar-specific offsets (e.g., adding a month, skipping weekends with business days). They are commonly used for calendar-aware adjustments and recurring periods.

```python
vbt.offset("Y")  # (1)!
vbt.offset("YE")  # (2)!
vbt.offset("weekstart")  # (3)!
vbt.offset("monday")
vbt.offset("july")  # (4)!
```

```python
vbt.offset("Y")  # (1)!
vbt.offset("YE")  # (2)!
vbt.offset("weekstart")  # (3)!
vbt.offset("monday")
vbt.offset("july")  # (4)!
```

```python
pd.offsets.YearBegin()
```

```python
pd.offsets.YearEnd()
```

```python
pd.offsets.Week(weekday=0)
```

```python
pd.offsets.YearBegin(month=7)
```

