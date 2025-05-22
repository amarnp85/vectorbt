# Portfolio¶

Portfolio refers to any combination of financial assets held by a trader. In the world of vectorbt, "portfolio" is a multidimensional structure capable of simulating and tracking multiple independent but also dependent portfolio instances. The main function of a portfolio is to apply a trading logic on a set of inputs to simulate a realistic trading environment, also referred to as "simulation". The outputs of such a simulation are orders and other information that can be used by the user in assessing the portfolio's performance, also referred to as "reconstruction" or "post-analysis". Both phases are isolated in nature, which enables various interesting use cases for quantitative analysis and data science.

The main class concerned with simulating and analyzing portfolios (i.e., with the actual backtesting) is Portfolio, which is a regular Python class subclassing Analyzable and having a range of Numba-compiled functions at its disposal. It's built similarly to other analyzable classes in the way that it has diverse class methods for instantiation from a range of inputs (such as Portfolio.from_signals taking signals), it's a state-full class capable of wrapping and indexing any Pandas-like objects contained inside it, and it can compute metrics and display (sub-)plots for quick introspection of the stored data.

## Simulation¶

So, what's a simulation? It's just a sophisticated loop!

A typical simulation in vectorbt takes some inputs (such as signals), gradually iterates over their rows (time steps in the real world) using a for-loop, and at each row, runs the trading logic by issuing and executing orders, and updating the current state of the trading environment such as the cash balance and position size. If we think about it, it's the exact same way we would approach algorithmic trading in reality: at each minute/hour/day (= row), decide what to do (= trading logic), and place an order if you decided to change your position in the market.

Now, let's talk about execution. The core of the vectorbt's backtesting engine is fully Numba-compiled for best performance. The functionality of the engine is distributed across many functions in an entire sub-package - portfolio.nb, ranging from core order execution commands to calculation of P&L in trade records. Remember that those functions aren't meant to be used directly (unless specifically desired) but are used by Python functions higher in the stack that know how to properly pre-process their input data and post-process the output data.

In the following parts, we'll discuss order execution and processing, and gradually implement a collection of simple pipelines to better illustrate various simulation concepts.

## Primitive commands¶

Remember that vectorbt is an exceptionally raw backtester: it's primary commands are "buy"  and "sell"  This means that any strategy that can be translated into a set of those commands is also supported out of the box. This also means that more complex orders such as limit and SL orders must be implemented manually. In contrast to other backtesting frameworks where processing is monolothic and functionality is written in an object-oriented manner, Numba forces vectorbt to implement most of the functionality in a procedural manner.

Info

Even though Numba supports OOP by compiling Python classes with @jitclass, they are treated as functions, must be statically typed, and have performance drawbacks that don't allow us to jump on the wagon just yet.

```python
@jitclass
```

Functions related to order execution are primarily located in portfolio.nb.core. The functions implementing our primary two commands are buy_nb and sell_nb. Among the requested size and price of an order, the primary input of each of these functions is the current account state of type AccountState, which contains the current cash balance, position size, and other information about the current environment. Whenever we buy or sell something, the function creates and returns a new state of the same type. Furthermore, it returns an order result of type OrderResult, which contains the filled size, price adjusted with slippage, transaction fee, order side, status information on whether the order succeeded or failed, and valuable information about why it failed.

### Buying¶

The buy operation consists of two distinct operations: "long-buy" implemented by long_buy_nb and "short-buy" implemented by short_buy_nb. The first one opens or increases a long position, while the second one decreases a short position. By chaining these two operations, we can reverse a short position, which is done automatically by buy_nb: it checks the position we're in (if any), and calls the responsible function.

Let's say we have $100 available and want to buy 1 share at the price of $15:

```python
>>> from vectorbtpro import *

>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,  # (1)!
...     locked_cash=0.0,  # (2)!
...     free_cash=100.0  # (3)!
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state,
...     size=1.0,
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=1.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=85.0,
    position=1.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=85.0
)
```

```python
>>> from vectorbtpro import *

>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,  # (1)!
...     locked_cash=0.0,  # (2)!
...     free_cash=100.0  # (3)!
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state,
...     size=1.0,
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=1.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=85.0,
    position=1.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=85.0
)
```

The returned state indicates that we spent $15 and increased our position by 1 share. The order result contains details about the executed order: we bought 1 share for $15, with no transaction fees. Since order side and status are of named tuple type OrderSide and OrderStatus respectively, we can query the meaning behind those numbers as follows:

```python
>>> vbt.pf_enums.OrderSide._fields[order_result.side]
'Buy'

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Filled'
```

```python
>>> vbt.pf_enums.OrderSide._fields[order_result.side]
'Buy'

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Filled'
```

Info

If any value is -1 and cannot be found in the named tuple, the information is unavailable.

```python
-1
```

Now, based on the new state, let's execute a transaction that uses up the remaining cash:

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state,  # (1)!
...     size=np.inf,  # (2)!
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.666666666666667,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=0.0,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state,  # (1)!
...     size=np.inf,  # (2)!
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.666666666666667,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=0.0,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

Since vectorbt was originally tailored to cryptocurrency and fractional shares, the default behavior is buying as much as possible (here 5.67), even if the amount is below of that requested. But what happens if we wanted to have the entire share instead? Let's specify the size granularity of 1, indicating that only integer amounts should be allowed:

```python
5.67
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state,
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=10.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state,
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=10.0
)
```

This has bought exactly 6 shares. Given the new account state, let's run the same transaction again:

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     new_account_state,  # (1)!
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=1,
    status_info=5
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    free_cash=10.0
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     new_account_state,  # (1)!
...     size=np.inf,
...     price=15.0,
...     size_granularity=1
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=1,
    status_info=5
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=10.0,
    position=6.0,
    debt=0.0,
    free_cash=10.0
)
```

The account state remains unchanged, while so many NaNs in the order result hint at a failed order. Let's query the meaning behind the status and status information numbers using OrderStatus and OrderStatusInfo named tuple respectively:

```python
>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Ignored'

>>> vbt.pf_enums.OrderStatusInfo._fields[order_result.status_info]
'SizeZero'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]  # (1)!
'Size is zero'
```

```python
>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Ignored'

>>> vbt.pf_enums.OrderStatusInfo._fields[order_result.status_info]
'SizeZero'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]  # (1)!
'Size is zero'
```

Here, the status "Size is zero" means that by considering our cash balance and after applying the size granularity, the (potentially) filled order size is zero, thus the order should be ignored. Ignored orders have no effect on the trading environment and are simply, well, ignored. But sometimes, when the user has specific requirements and vectorbt cannot execute them, the status will become "Rejected", indicating that the request could not be fulfilled and an error can be thrown if wanted.

For example, let's try to buy more than possible:

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=1000.0, 
...     price=15.0,
...     allow_partial=False
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=12
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=100.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=100.0
)

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]
'Final size is less than requested'
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=1000.0, 
...     price=15.0,
...     allow_partial=False
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=12
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=100.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=100.0
)

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]
'Final size is less than requested'
```

There are many other parameters to control the execution. Let's use 50% of cash, and apply 1% in fees and slippage:

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0,
...     fees=0.01,  # (1)!
...     slippage=0.01,  # (2)!
...     percent=0.5  # (3)!
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=3.2676534980230696,
    price=15.15,
    fees=0.4950495049504937,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=50.0,
    position=3.2676534980230696,
    debt=0.0,
    locked_cash=0.0,
    free_cash=50.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0,
...     fees=0.01,  # (1)!
...     slippage=0.01,  # (2)!
...     percent=0.5  # (3)!
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=3.2676534980230696,
    price=15.15,
    fees=0.4950495049504937,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=50.0,
    position=3.2676534980230696,
    debt=0.0,
    locked_cash=0.0,
    free_cash=50.0
)
```

```python
fixed_fees
```

The final fees and the price adjusted with the slippage are reflected in the order result.

Whenever we place an order, we can specify any price. Thus, it may sometimes happen that the provided price is (by mistake of the user) higher than the highest price of that bar or lower than the lowest price of that bar. Also, if the user wanted the price to be closing, and he specified a slippage, this would also be quite unrealistic. To avoid such mistakes, the function performs an OHLC check. For this, we need to specify the price_area of type PriceArea: with the price boundaries, and what should be done if a boundary violation happens via price_area_vio_mode of type PriceAreaVioMode:

```python
price_area
```

```python
price_area_vio_mode
```

```python
>>> price_area = vbt.pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state,
...     size=np.inf,
...     price=np.inf,
...     price_area=price_area,
...     price_area_vio_mode=vbt.pf_enums.PriceAreaVioMode.Error
... )
ValueError: Adjusted order price is above the highest price
```

```python
>>> price_area = vbt.pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state,
...     size=np.inf,
...     price=np.inf,
...     price_area=price_area,
...     price_area_vio_mode=vbt.pf_enums.PriceAreaVioMode.Error
... )
ValueError: Adjusted order price is above the highest price
```

### Selling¶

The sell operation consists of two distinct operations: "long-sell" implemented by long_sell_nb and "short-sell" implemented by short_sell_nb. The first one decreases a long position, while the second one opens or increases a short position. By chaining these two operations, we can reverse a long position, which is done automatically by sell_nb: it checks the position we're in (if any), and calls the responsible function.

The function for selling takes the same arguments but uses them in the opposite direction. Let's remove 2 shares from a position of 10 shares:

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state,
...     size=2.0,
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=2.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=30.0,
    position=8.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=30.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state,
...     size=2.0,
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=2.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=30.0,
    position=8.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=30.0
)
```

The size in the order result remains positive but the side has changed from 0 to 1:

```python
>>> vbt.pf_enums.OrderSide._fields[order_result.side]
'Sell'
```

```python
>>> vbt.pf_enums.OrderSide._fields[order_result.side]
'Sell'
```

### Shorting¶

Shorting is a regular sell operation with sell_nb, but with one exception: it now involves the debt as well as the locked cash balance. Whenever we short, we are borrowing shares and selling them to buyers willing to pay the market price. This operation increases the cash balance and turns the position size negative. It also registers the received cash amount as a debt, and subtracts it from the free cash balance. Whenever we buy some shares back, the debt decreases proportionally to the value of the shares bought back, while the free cash might increase depending upon whether the price was higher or lower than the average short-selling price. Whenever we cover the short position entirely, the debt becomes zero and the free cash balance returns to the same level as the regular cash balance.

Note

You shouldn't treat debt as an absolute amount of cash you owe since you owe shares, not cash; it's used for calculating the average leverage and entry price of the short position, which is then used to calculate the change in the free cash balance with each trade.

To borrow any shares, we need a positive free cash balance to be used as a collateral. The exact amount of free cash needed for a shorting operation depends on margin; by default, we need the same amount of funds available in our margin account as the value of to-be-borrowed shares. For example, if we short a stock and the new position had a value of $100, we would be required to have the $100 that came from the short sale plus an additional $100 in cash, for a total of $200, which depending on the definition is either a 100% (before sale) or 200% (after sale) initial margin requirement. Maintenance margin and liquidation checks are the responsibility of the user (for now).

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf,  # (1)!
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=200.0,
    position=-6.666666666666667,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf,  # (1)!
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=200.0,
    position=-6.666666666666667,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

Info

Infinity is a special value in vectorbt and usually means "go as far as you can".

Here's what happened. First, we've converted all the available free cash ($100) into the locked one such that it becomes the collateral of our shorting operation. Because the default leverage is 1, we've borrowed the same value in shares ($100), which has been added to the regular cash balance as well registered as a debt. This value corresponds to (minus = borrowed) 6.67 shares. But here's the problem: since we've doubled the regular cash balance, it may be used by other assets. To avoid that, all operations are done strictly on the free cash balance! But it's zero right now, so how do we buy back the borrowed shares? Remember that the debt and locked cash represent the total amount of cash that we used in the first place; by adding both to the free cash, we get our cash limit for the current buy operation, which nicely matches the regular cash when dealing with only one asset.

To change the margin, we can use the argument leverage. For example, setting it to 2 will allow us to borrow twice as many shares as can be covered by the current free cash:

```python
leverage
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf,
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=13.333333333333334,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=300.0,
    position=-13.333333333333334,
    debt=200.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf,
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=13.333333333333334,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=300.0,
    position=-13.333333333333334,
    debt=200.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

The debt-to-locked-cash ratio is now 2, which corresponds to the leverage that we specified.

Info

We can specify a different leverage for each short-sell order, even in the same position.

Let's try to run the same operation again, but now on the new account state:

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=np.inf,
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=6
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=300.0,
    position=-13.333333333333334,
    debt=200.0,
    locked_cash=100.0,
    free_cash=0.0
)

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]
'Not enough cash'
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=np.inf,
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=np.nan,
    price=np.nan,
    fees=np.nan,
    side=-1,
    status=2,
    status_info=6
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=300.0,
    position=-13.333333333333334,
    debt=200.0,
    locked_cash=100.0,
    free_cash=0.0
)

>>> vbt.pf_enums.OrderStatus._fields[order_result.status]
'Rejected'

>>> vbt.pf_enums.status_info_desc[order_result.status_info]
'Not enough cash'
```

We see that vectorbt prevents the free cash balance to become negative.

To order any quantity possible, we can use the unlimited leverage:

```python
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=1000, 
...     price=15.0, 
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=1000.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=15100.0,
    position=-1000.0,
    debt=15000.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=1000, 
...     price=15.0, 
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=1000.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=15100.0,
    position=-1000.0,
    debt=15000.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

What's the effective leverage of this operation?

```python
>>> new_account_state.debt / new_account_state.locked_cash
150.0
```

```python
>>> new_account_state.debt / new_account_state.locked_cash
150.0
```

If we had to calculate the current portfolio value, it would still default to the initial cash since no transaction costs were involved and no additional trades were made:

```python
>>> new_account_state.cash + new_account_state.position * order_result.price
100.0
```

```python
>>> new_account_state.cash + new_account_state.position * order_result.price
100.0
```

As we see, the positive cash balance and the negative position size keep the total value in balance. Now, let's illustrate buying back some shares using buy_nb. First, we'll borrow 10 shares with 2x leverage and sell them for the price of $10 per share:

```python
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=10.0, 
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=250.0,
    position=-10.0,
    debt=150.0,
    locked_cash=75.0,
    free_cash=25.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=10.0, 
...     price=15.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=250.0,
    position=-10.0,
    debt=150.0,
    locked_cash=75.0,
    free_cash=25.0
)
```

Let's buy back 5 shares for the price of $30 per share (my condolences):

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=30.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=30.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=100.0,
    position=-5.0,
    debt=75.0,
    locked_cash=37.5,
    free_cash=-12.5
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=30.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=30.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=100.0,
    position=-5.0,
    debt=75.0,
    locked_cash=37.5,
    free_cash=-12.5
)
```

We executed the order for $150, which have been deducted from the regular cash balance. The position has been reduced by half, to 5 borrowed shares. Along with the position, the debt and locked cash have also been reduced by half. Given the absolute amount of released debt ($75), we can then compute the P&L, which is the total spent cash subtracted from the total released debt, or -$75. But this operation has also released some locked cash - $37.5, such that the amount of cash added back to our free cash balance is -$37.5, which makes it negative. A negative free cash means that we won't be able to buy any other assets apart from reducing any short positions and thus potentially releasing additional funds (i.e., profits as well as losses are shared among all assets within the same group with cash sharing). Even though we've got a negative free cash, we can still buy back more shares because the sum of debt, locked_cash, and free_cash is greater than zero.

```python
debt
```

```python
locked_cash
```

```python
free_cash
```

Not only the debt and locked cash can be used to compute the effective leverage, but we can also derive the average entry price of the entire position:

```python
>>> new_account_state2.debt / abs(new_account_state2.position)
15.0
```

```python
>>> new_account_state2.debt / abs(new_account_state2.position)
15.0
```

Let's say instead of jumping, the price has dipped to $10 per share (my congratulations!):

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=10.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=200.0,
    position=-5.0,
    debt=75.0,
    locked_cash=37.5,
    free_cash=87.5
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=10.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=200.0,
    position=-5.0,
    debt=75.0,
    locked_cash=37.5,
    free_cash=87.5
)
```

We see that the debt and locked cash have decreased to the same level as previously (because we've bought back the same relative amount of shares), but the free cash balance is now $87.5, netting $25 in profit! Again, the calculation is simple: we just take the total amount of spent cash ($5 * 10 = $50) and subtract it from the total released debt (0.5 * $150 = $75) to get the P&L. By adding the P&L, the released locked cash (0.5 * $75 = $37.5), and the current free cash ($25) together, we get the new free cash of $87.5 immediately available for all other assets to use.

Let's compute the equity to validate the profit:

```python
>>> st0 = account_state
>>> st1 = new_account_state2
>>> avg_entry_price = st1.debt / abs(st1.position)  # (1)!
>>> new_equity = st1.cash + st1.position * avg_entry_price
>>> new_equity - st0.cash
25.0
```

```python
>>> st0 = account_state
>>> st1 = new_account_state2
>>> avg_entry_price = st1.debt / abs(st1.position)  # (1)!
>>> new_equity = st1.cash + st1.position * avg_entry_price
>>> new_equity - st0.cash
25.0
```

Let's close out the open short position using the same price:

```python
>>> order_result, new_account_state3 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state2, 
...     size=5.0, 
...     price=10.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state3)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

```python
>>> order_result, new_account_state3 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state2, 
...     size=5.0, 
...     price=10.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state3)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

The free cash balance equals to the regular cash balance, and we are debt-free! Additionally, the last two operations have brought us $50 in profit, or (15 - 10) * 10 = $50.

Finally, let's try to close out the position using an astronomically high price!

```python
>>> order_result, new_account_state3 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state2, 
...     size=5.0, 
...     price=100.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=2.0,
    price=100.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state3)
AccountState(
    cash=0.0,
    position=-3.0,
    debt=45.0,
    locked_cash=22.5,
    free_cash=-67.5
)
```

```python
>>> order_result, new_account_state3 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state2, 
...     size=5.0, 
...     price=100.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=2.0,
    price=100.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state3)
AccountState(
    cash=0.0,
    position=-3.0,
    debt=45.0,
    locked_cash=22.5,
    free_cash=-67.5
)
```

We could buy back only 2 shares out of the remaining 5. If we try the same operation again, we would get the "Not enough cash" message because debt + locked_cash + free_cash is below or equal to zero. We also witness the regular cash balance going to zero, which means that we've exhausted all of our capital; but we shouldn't rely on it to make trading decisions! If any other asset buys shares with leverage, regular cash may go negative, which doesn't necessarily mean that we have no cash left - only free cash (with debt and locked cash when covering shorts) provide us with the right information.

```python
debt + locked_cash + free_cash
```

### Leverage¶

Even though vectorbt allows setting an arbitrary (even an infinite) cash and ordering as many shares as required by the user, this constellation is associated with some drawbacks: an infinite cash leads to an infinite portfolio value, which makes certain operations on that value impossible, such as the conversion from a target percentage into a target amount of shares; but also, the more cash we have, the smaller are potential contributions of the positions to the portfolio value, thus lowering the magnitude of the portfolio returns. What we need though is to multiply those contributions without inflating the cash balance, which can be effectively done using leverage.

Leverage involves borrowing additional funds to buy shares. In main contrast to shorting, leverage is applied to long positions and borrows cash instead of shares. The underlying mechanism is quite similar though. First, we multiply the available free cash by leverage. Then, we derive the order value and the fraction of it to be borrowed. Finally, we move the borrowed cash to debt while declaring a part of the free cash as the collateral of the operation and moving it to locked_cash. But since the locked cash must be spent to buy a part of the shares, it changes the way the effective leverage can be calculated from debt / locked_cash to debt / locked_cash + 1.

```python
leverage
```

```python
debt
```

```python
locked_cash
```

```python
debt / locked_cash
```

```python
debt / locked_cash + 1
```

Let's say we have $100 in our margin account and want to buy $200 worth of shares. As we learned previously, we can specify an infinite leverage and vectorbt will derive the effective leverage internally for us:

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=20, 
...     price=10.0,
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=-100.0,
    position=20.0,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=20, 
...     price=10.0,
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=-100.0,
    position=20.0,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

We can see that $100 were deducted from our free cash balance and additional $100 were borrowed, thus bringing the effective leverage to 2:

```python
>>> new_account_state.debt / new_account_state.locked_cash + 1
2.0
```

```python
>>> new_account_state.debt / new_account_state.locked_cash + 1
2.0
```

Buying 10 shares instead would use no leverage since the transaction can be entirely covered by the free cash, even if the leverage is infinity:

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=10.0,
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=0.0,
    position=10.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=10.0,
...     leverage=np.inf
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=0.0,
    position=10.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

Is there a way to use only a subset of our own free cash while borrowing the rest? Yes! The command buy_nb takes the argument leverage_mode of the type LeverageMode, which supports two modes: "lazy" and "eager" leveraging. The first mode is the default one and enables leverage only if the quantity to be bought cannot be fulfilled using own resources. The second mode enables leverage for any quantity and requires the leverage to be set explicitly, that is, an infinite leverage would raise an error.

```python
leverage_mode
```

Note

Shorting supports "lazy" leveraging only.

Let's buy 10 shares with a 3x leverage:

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=10.0,
...     leverage=3,
...     leverage_mode=vbt.pf_enums.LeverageMode.Eager
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=0.0,
    position=10.0,
    debt=66.66666666666667,
    locked_cash=33.333333333333336,
    free_cash=66.66666666666666
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=10.0,
...     leverage=3,
...     leverage_mode=vbt.pf_enums.LeverageMode.Eager
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=10.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=0.0,
    position=10.0,
    debt=66.66666666666667,
    locked_cash=33.333333333333336,
    free_cash=66.66666666666666
)
```

We've used only $33.3 from our free cash balance as the collateral to borrow additional $66.6, making a total of $100 that were spent to buy the desired quantity.

Now, how do we repay the debt? When selling, the debt and locked cash balances decrease proportionally to the number of shares sold, which is exactly the same procedure as during (partially-) closing a short position. The main difference is in the calculation of the P&L: we take the total of the released debt and locked cash and subtract them from the cash retrieved from the operation.

First, we'll use a 2x leverage to buy 10 shares for the price of $20 per share:

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=20.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=20.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=-100.0,
    position=10.0,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

```python
>>> order_result, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=20.0,
...     leverage=2
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10,
    price=20.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=-100.0,
    position=10.0,
    debt=100.0,
    locked_cash=100.0,
    free_cash=0.0
)
```

Let's sell 5 shares for the price of $5 per share (my condolences):

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=5.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=5.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=-75.0,
    position=5.0,
    debt=50.0,
    locked_cash=50.0,
    free_cash=-25.0
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=5.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=5.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=-75.0,
    position=5.0,
    debt=50.0,
    locked_cash=50.0,
    free_cash=-25.0
)
```

We've retrieved $25 from this operation, which have been added to the regular cash balance. The debt and locked cash have been cut in half because the half of the leveraged position has been closed out. The P&L of this operation is the gained cash ($25) minus the released debt ($50) and locked cash ($50), which makes it a loss of $75. By adding this number to the released locked cash, we arrive at the new free cash of -$25 - a change that is propagated to all other assets using the same cash balance and thus preventing them to open or increase their positions.

Another way of calculating the P&L using the equity:

```python
>>> st0 = account_state
>>> st1 = new_account_state2
>>> avg_entry_price = (st1.debt + st1.locked_cash) / abs(st1.position)  # (1)!
>>> new_equity = st1.cash + st1.position * avg_entry_price
>>> new_equity - st0.cash
-75.0
```

```python
>>> st0 = account_state
>>> st1 = new_account_state2
>>> avg_entry_price = (st1.debt + st1.locked_cash) / abs(st1.position)  # (1)!
>>> new_equity = st1.cash + st1.position * avg_entry_price
>>> new_equity - st0.cash
-75.0
```

Now, let's say instead of dipping, the price has jumped to $40 per share (my congratulations!):

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=40.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=40.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=100.0,
    position=5.0,
    debt=50.0,
    locked_cash=50.0,
    free_cash=150.0
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=40.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=40.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=100.0,
    position=5.0,
    debt=50.0,
    locked_cash=50.0,
    free_cash=150.0
)
```

We've retrieved $200 from this operation, which have been added to the regular cash balance. The debt and locked cash have been cut in half because the half of the leveraged position has been closed out. The P&L of this operation is the gained cash ($200) minus the released debt ($50) and locked cash ($50), which makes it a profit of $100. By adding this number to the released locked cash, we arrive at the new free cash of $150, which can be now used by all other assets sharing the same balance.

Let's close out the remaining position using the same price:

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=40.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=40.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=300.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=300.0
)
```

```python
>>> order_result, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=5.0, 
...     price=40.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=5.0,
    price=40.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=300.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=300.0
)
```

We've made a profit of $200, which is just the same as we had used our own cash:

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=200.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=200.0
... )
>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=20.0
... )
>>> _, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=10.0, 
...     price=40.0
... )
>>> new_account_state2.free_cash - account_state.free_cash
200.0
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=200.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=200.0
... )
>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     size=10, 
...     price=20.0
... )
>>> _, new_account_state2 = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     size=10.0, 
...     price=40.0
... )
>>> new_account_state2.free_cash - account_state.free_cash
200.0
```

### Symmetry¶

Long and short positions behave symmetrically. For example, let's open two opposite positions using an infinite size and 10x leverage and close them out with a price difference of $5 per share that favors the current position:

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )

>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     direction=vbt.pf_enums.Direction.LongOnly,
...     size=np.inf, 
...     price=10.0,
...     leverage=10
... )
>>> _, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     direction=vbt.pf_enums.Direction.LongOnly,
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(new_account_state)
AccountState(
    cash=600.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=600.0
)

>>> _, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     direction=vbt.pf_enums.Direction.ShortOnly,
...     size=np.inf, 
...     price=10.0,
...     leverage=10
... )
>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     direction=vbt.pf_enums.Direction.ShortOnly,
...     size=np.inf, 
...     price=5.0
... )
>>> vbt.pprint(new_account_state)
AccountState(
    cash=600.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=600.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0
... )

>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=account_state, 
...     direction=vbt.pf_enums.Direction.LongOnly,
...     size=np.inf, 
...     price=10.0,
...     leverage=10
... )
>>> _, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=new_account_state, 
...     direction=vbt.pf_enums.Direction.LongOnly,
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(new_account_state)
AccountState(
    cash=600.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=600.0
)

>>> _, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     direction=vbt.pf_enums.Direction.ShortOnly,
...     size=np.inf, 
...     price=10.0,
...     leverage=10
... )
>>> _, new_account_state = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     direction=vbt.pf_enums.Direction.ShortOnly,
...     size=np.inf, 
...     price=5.0
... )
>>> vbt.pprint(new_account_state)
AccountState(
    cash=600.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=600.0
)
```

### Reversing¶

Positions in vectorbt can be reversed with a single order. To reverse a position, the direction argument should stay at its default - Direction.Both. Let's start with a position of 10 shares, reverse it to the maximum extent in the short direction, and then reverse it to the maximum extent again but in the opposite (long) direction:

```python
Direction.Both
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=300.0,
    position=-10.0,
    debt=150.0,
    locked_cash=150.0,
    free_cash=0.0
)

>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=0.0,
    position=10.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=300.0,
    position=-10.0,
    debt=150.0,
    locked_cash=150.0,
    free_cash=0.0
)

>>> order_result, new_account_state2 = vbt.pf_nb.buy_nb(
...     account_state=new_account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=20.0,
    price=15.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state2)
AccountState(
    cash=0.0,
    position=10.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=0.0
)
```

Both operations are symmetric in nature and cancel each other out by a repetitive call, thus ultimately we've arrived at our initial state of the account.

### Closing¶

To close out a position and to avoid its reversal, we can either specify the exact size, or the size of infinity and the current direction via the direction argument of the type Direction. For example, if we're in a long position and specified the long-only direction, the position won't be reversed:

```python
direction
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     direction=vbt.pf_enums.Direction.LongOnly
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0, 
...     direction=vbt.pf_enums.Direction.LongOnly
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

Note

Using the buy_nb and sell_nb commands guarantees to execute the order in the long and short direction respectively.

We can also use the commands that are guaranteed to execute within the current position and not open an opposite one: long_sell_nb for long positions and short_buy_nb for short positions. They don't require the argument direction at all, just the size of infinity:

```python
direction
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.long_sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

```python
>>> account_state = vbt.pf_enums.AccountState(
...     cash=0.0,
...     position=10.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=0.0
... )
>>> order_result, new_account_state = vbt.pf_nb.long_sell_nb(
...     account_state=account_state, 
...     size=np.inf, 
...     price=15.0
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=10.0,
    price=15.0,
    fees=0.0,
    side=1,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_account_state)
AccountState(
    cash=150.0,
    position=0.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=150.0
)
```

### Pipeline/1¶

Even by using just these two essential commands, we can already build our own backtesting pipeline of arbitrary complexity and flexibility. As said before, a simulation is just a loop that iterates over timestamps. Let's create a simplified pipeline that puts $1 into Bitcoin each time it discovers a Golden Cross entry signal, and sells $1 otherwise. What we want is one number: the final value of the portfolio.

```python
>>> @njit
... def pipeline_1_nb(close, entries, exits, init_cash=100):
...     account_state = vbt.pf_enums.AccountState(  # (1)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         locked_cash=0.0,
...         free_cash=float(init_cash)
...     )
...     for i in range(close.shape[0]):
...         if entries[i]:
...             _, account_state = vbt.pf_nb.buy_nb(  # (2)!
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...         if exits[i]:
...             _, account_state = vbt.pf_nb.sell_nb(
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...     return account_state.cash + account_state.position * close[-1]  # (3)!

>>> data = vbt.YFData.pull("BTC-USD", end="2022-01-01")
>>> sma_50 = vbt.talib("SMA").run(data.get("Close"), 50)
>>> sma_200 = vbt.talib("SMA").run(data.get("Close"), 200)
>>> entries = sma_50.real_crossed_above(sma_200)
>>> exits = sma_50.real_crossed_below(sma_200)

>>> pipeline_1_nb(
...     data.get("Close").values, 
...     entries.values, 
...     exits.values
... )
210.71073253390762
```

```python
>>> @njit
... def pipeline_1_nb(close, entries, exits, init_cash=100):
...     account_state = vbt.pf_enums.AccountState(  # (1)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         locked_cash=0.0,
...         free_cash=float(init_cash)
...     )
...     for i in range(close.shape[0]):
...         if entries[i]:
...             _, account_state = vbt.pf_nb.buy_nb(  # (2)!
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...         if exits[i]:
...             _, account_state = vbt.pf_nb.sell_nb(
...                 account_state=account_state,
...                 size=1 / close[i],
...                 price=close[i]
...             )
...     return account_state.cash + account_state.position * close[-1]  # (3)!

>>> data = vbt.YFData.pull("BTC-USD", end="2022-01-01")
>>> sma_50 = vbt.talib("SMA").run(data.get("Close"), 50)
>>> sma_200 = vbt.talib("SMA").run(data.get("Close"), 200)
>>> entries = sma_50.real_crossed_above(sma_200)
>>> exits = sma_50.real_crossed_below(sma_200)

>>> pipeline_1_nb(
...     data.get("Close").values, 
...     entries.values, 
...     exits.values
... )
210.71073253390762
```

Hint

Adding a suffix _nb to indicate a Numba-compiled function is not necessary but still a good convention in vectorbt.

```python
_nb
```

We can validate the pipeline using one of the preset simulation methods:

```python
>>> vbt.Portfolio.from_orders(
...     data.get("Close"), 
...     size=entries.astype(int) - exits.astype(int), 
...     size_type="value"
... ).final_value
210.71073253390762
```

```python
>>> vbt.Portfolio.from_orders(
...     data.get("Close"), 
...     size=entries.astype(int) - exits.astype(int), 
...     size_type="value"
... ).final_value
210.71073253390762
```

## Order execution¶

Using the primitive commands is fun as long as we exactly know the direction of the order and are sure that the provided arguments make sense. But very often, we have to deal with more complex requirements such as target percentages, which change the order direction depending on the current value. In addition, the commands do not validate their arguments; for example, there won't be any error thrown in a case when the user accidentally passes a negative order price. But also, we need a better representation of an order - it's a bad practice of passing all the parameters such as slippage as keyword arguments.

All the checks and other pre-processing procedures are happening in the function execute_order_nb. The first input to this function is an order execution state of type ExecState, which contains the same information as an account state we saw above, but with additional information on the current valuation. The second input is a named tuple of type Order representing an order. The third argument is the price area, which we are also already familiar with.

### Order¶

An order in vectorbt is represented by a named tuple. Named tuples are alternatives to data classes in both the Python and Numba world; they are very efficient and lightweight data structures that can be easily constructed and processed. Let's create an instance of an order:

```python
>>> order = vbt.pf_enums.Order()
>>> vbt.pprint(order)
Order(
    size=np.inf,
    price=np.inf,
    size_type=0,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode=0,
    reject_prob=0.0,
    price_area_vio_mode=0,
    allow_partial=True,
    raise_reject=False,
    log=False
)
```

```python
>>> order = vbt.pf_enums.Order()
>>> vbt.pprint(order)
Order(
    size=np.inf,
    price=np.inf,
    size_type=0,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=np.nan,
    max_size=np.nan,
    size_granularity=np.nan,
    leverage=1.0,
    leverage_mode=0,
    reject_prob=0.0,
    price_area_vio_mode=0,
    allow_partial=True,
    raise_reject=False,
    log=False
)
```

The tuple allows for attribute access through the dot notation:

```python
>>> order.direction
2
```

```python
>>> order.direction
2
```

Other than this, it behaves just like any other tuple in Python:

```python
>>> order[3]
2

>>> tuple(order)  # (1)!
(inf,
 inf,
 0,
 2,
 0.0,
 0.0,
 0.0,
 nan,
 nan,
 nan,
 1.0,
 0,
 0.0,
 0,
 True,
 False,
 False)
```

```python
>>> order[3]
2

>>> tuple(order)  # (1)!
(inf,
 inf,
 0,
 2,
 0.0,
 0.0,
 0.0,
 nan,
 nan,
 nan,
 1.0,
 0,
 0.0,
 0,
 True,
 False,
 False)
```

One issue that we still have to address when working with Numba are default arguments: although we can construct a new tuple solely with default arguments in Numba as we did above, if we want to override some values, they must be located strictly on the left in that tuple's definition. Otherwise, we must explicitly provide all the default arguments located before them:

```python
>>> @njit
... def create_order_nb():
...     return vbt.pf_enums.Order()  # (1)!

>>> create_order_nb()
Order(size=inf, price=inf, ...)

>>> @njit
... def create_order_nb(size, price):
...     return vbt.pf_enums.Order(size=size, price=price)  # (2)!

>>> create_order_nb(1, 15)
Order(size=1, price=15, ...)

>>> @njit
... def create_order_nb(size, price, direction):
...     return vbt.pf_enums.Order(size=size, price=price, direction=direction)  # (3)!

>>> create_order_nb(1, 15, 2)
Failed in nopython mode pipeline (step: nopython frontend)
```

```python
>>> @njit
... def create_order_nb():
...     return vbt.pf_enums.Order()  # (1)!

>>> create_order_nb()
Order(size=inf, price=inf, ...)

>>> @njit
... def create_order_nb(size, price):
...     return vbt.pf_enums.Order(size=size, price=price)  # (2)!

>>> create_order_nb(1, 15)
Order(size=1, price=15, ...)

>>> @njit
... def create_order_nb(size, price, direction):
...     return vbt.pf_enums.Order(size=size, price=price, direction=direction)  # (3)!

>>> create_order_nb(1, 15, 2)
Failed in nopython mode pipeline (step: nopython frontend)
```

Another issue are data types. In the example above where we provided integer size and price, Numba had no issues processing them. But as soon as we create such as order in a loop and one of the arguments is a float instead of an integer provided previously, Numba will throw an error because it cannot unify data types anymore. Thus, we should cast all arguments to their target data types before constructing an order.

Both issues are solved by using the function order_nb:

```python
>>> @njit
... def create_order_nb(size, price, direction):
...     return vbt.pf_nb.order_nb(size=size, price=price, direction=direction)

>>> create_order_nb(1, 15, 2)
Order(size=1.0, price=15.0, ..., direction=2, ...)
```

```python
>>> @njit
... def create_order_nb(size, price, direction):
...     return vbt.pf_nb.order_nb(size=size, price=price, direction=direction)

>>> create_order_nb(1, 15, 2)
Order(size=1.0, price=15.0, ..., direction=2, ...)
```

Notice how the size and price arguments were automatically cast to floats.

Hint

Use order_nb instead of Order whenever possible.

To create an order that closes out the current position, we can conveniently use close_position_nb:

```python
>>> vbt.pf_nb.close_position_nb(15)  # (1)!
Order(size=0.0, price=15.0, size_type=6, ...)
```

```python
>>> vbt.pf_nb.close_position_nb(15)  # (1)!
Order(size=0.0, price=15.0, size_type=6, ...)
```

```python
TargetAmount
```

### Validation¶

Having constructed the order, execute_order_nb will check the arguments of that order for correct data types and values. For example, let's try passing a negative price:

```python
>>> exec_state = vbt.pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> vbt.pf_nb.execute_order_nb(
...     exec_state,
...     vbt.pf_nb.order_nb(price=-15)
... )
ValueError: order.price must be finite and 0 or greater
```

```python
>>> exec_state = vbt.pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> vbt.pf_nb.execute_order_nb(
...     exec_state,
...     vbt.pf_nb.order_nb(price=-15)
... )
ValueError: order.price must be finite and 0 or greater
```

### Price resolution¶

After validating the inputs, execute_order_nb uses them to decide which command to run: buy or sell. But first, it has to do some preprocessing.

Even though vectorbt isn't associated with any particular data schema and can run on tick data just as well as on bar data, it still gives us an option to provide the current candle (price_area) for validation and resolution reasons. In such a case, it will consider the passed order price as a price point located within four price bounds: the opening, high, low, and closing price. Since order execution must happen strictly within those bounds, setting order price to -np.inf and np.inf will replace it by the opening and closing price respectively. Hence, next time, when you see any default price being np.inf, just know that it means the close price

```python
price_area
```

```python
-np.inf
```

```python
np.inf
```

```python
np.inf
```

```python
>>> price_area = vbt.pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (1)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=-np.inf),
...     price_area=price_area
... )
>>> order_result.price
10.0

>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (2)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=np.inf),
...     price_area=price_area
... )
>>> order_result.price
12.0

>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (3)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=np.inf)
... )
>>> order_result.price
nan
```

```python
>>> price_area = vbt.pf_enums.PriceArea(
...     open=10,
...     high=14,
...     low=8,
...     close=12
... )
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (1)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=-np.inf),
...     price_area=price_area
... )
>>> order_result.price
10.0

>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (2)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=np.inf),
...     price_area=price_area
... )
>>> order_result.price
12.0

>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(  # (3)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=np.inf)
... )
>>> order_result.price
nan
```

```python
np.nan
```

### Size type conversion¶

Our primitive commands accept only a size in the number of shares, thus we have to convert any size type defined in SizeType to Amount. Different size types require different information for conversion; for example, TargetAmount requires to know the current position size, while Value also requires to know the current valuation price.

```python
Amount
```

```python
TargetAmount
```

```python
Value
```

Let's execute an order such that the new position has 3 shares:

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=3, 
...         size_type=vbt.pf_enums.SizeType.TargetAmount
...     ),
...     price_area=price_area
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=3.0,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=64.0,
    position=3.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=64.0,
    val_price=15.0,
    value=100.0
)
```

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=3, 
...         size_type=vbt.pf_enums.SizeType.TargetAmount
...     ),
...     price_area=price_area
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=3.0,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=64.0,
    position=3.0,
    debt=0.0,
    locked_cash=0.0,
    free_cash=64.0,
    val_price=15.0,
    value=100.0
)
```

Since we're not in the market, vectorbt used buy_nb to buy 3 shares. If we were in the market with 10 shares, it would have used sell_nb to sell 7 shares.

### Direction¶

Function order_nb takes the argument direction for two reasons: resolve the direction of the order based on the sign of the argument size, and decide on whether to reverse the position or just to close out it. When the direction is LongOnly or Both, a positive size means buying and a negative size means selling. When the direction is ShortOnly though, the exact opposite happens: a positive size means selling and a negative size means buying. This is because a positive size is associated with increasing a position, which means buying to increase a long position and selling to increase a short position. For example, if the direction was ShortOnly and the size was a negative infinity, any short position would be closed out and any long position would be enlarged.

```python
direction
```

```python
size
```

```python
LongOnly
```

```python
Both
```

```python
ShortOnly
```

```python
ShortOnly
```

### Valuation¶

Speaking about the valuation price, it's the latest available price at the time of decision-making, or the price used to calculate the portfolio value. In many simulation methods, valuation price defaults to the order price, but sometimes it makes more sense to use the open or previous close price for the conversion step. The separation of the valuation and order price enables us to introduce a time gap between order placement and its execution. This is important because, in reality, not always an order can be executed right away.

Let's order 100% of the portfolio value:

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=1.0, 
...         size_type=vbt.pf_enums.SizeType.TargetPercent
...     ),
...     price_area=price_area
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=20.0,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=20.0,
    val_price=15.0,
    value=100.0
)
```

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=1.0, 
...         size_type=vbt.pf_enums.SizeType.TargetPercent
...     ),
...     price_area=price_area
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=12.0,
    fees=0.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=20.0,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=20.0,
    val_price=15.0,
    value=100.0
)
```

Why haven't we spent the entire cash? Because to convert the target percentage into the target amount of shares, vectorbt used the provided order execution state with val_price of $15 and value of $100, which produced 100 / 15 = 6.67. The closer the valuation price is to the order price, the closer the calculation result would be to the target requirement.

```python
val_price
```

```python
value
```

```python
100 / 15 = 6.67
```

By default, if we want to place multiple orders within the same bar (for example, in pairs trading), vectorbt wouldn't adjust the portfolio value after each order. This is because it assumes that we made our trading decisions way before order execution and adjusting the value would affect those decisions. But also, an order has only a marginal immediate effect on the value, for example, because of a commission. To force vectorbt to update the valuation price and value itself, we can enable update_value:

```python
update_value
```

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=1.0, 
...         size_type=vbt.pf_enums.SizeType.TargetPercent,
...         fixed_fees=10,
...         slippage=0.01
...     ),
...     price_area=price_area,
...     update_value=True
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=12.120000000000001,
    fees=10.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=9.199999999999989,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=9.199999999999989,
    val_price=12.120000000000001,
    value=90.0
)
```

```python
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(
...         size=1.0, 
...         size_type=vbt.pf_enums.SizeType.TargetPercent,
...         fixed_fees=10,
...         slippage=0.01
...     ),
...     price_area=price_area,
...     update_value=True
... )
>>> vbt.pprint(order_result)
OrderResult(
    size=6.666666666666667,
    price=12.120000000000001,
    fees=10.0,
    side=0,
    status=0,
    status_info=-1
)
>>> vbt.pprint(new_exec_state)
ExecState(
    cash=9.199999999999989,
    position=6.666666666666667,
    debt=0.0,
    locked_cash=0.0,
    free_cash=9.199999999999989,
    val_price=12.120000000000001,
    value=90.0
)
```

Notice how the new valuation price has been set to the close price adjusted with the slippage while the value has decreased by the fixed commission. Any new order placed after this one would use the updated value and thus probably produce a different outcome.

Note

Use this feature only if you can control the order in which orders appear within a bar and when you have intra-bar data.

### Pipeline/2¶

Let's create another simplified pipeline that orders given a target percentage array. In particular, we'll keep 50% of the portfolio value in shares, and rebalance monthly. We'll calculate the portfolio value based on the open price at the beginning of each bar, and order at the end of each bar (to keep things realistic). Also, we'll fill asset value and portfolio value arrays to later plot the allocation at each bar.

```python
>>> @njit
... def pipeline_2_nb(open, close, target_pct, init_cash=100):
...     asset_value_out = np.empty(close.shape, dtype=float_)  # (1)!
...     value_out = np.empty(close.shape, dtype=float_)
...     exec_state = vbt.pf_enums.ExecState(  # (2)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         locked_cash=0.0,
...         free_cash=float(init_cash),
...         val_price=np.nan,
...         value=np.nan
...     )
...
...     for i in range(close.shape[0]):
...         if not np.isnan(target_pct[i]):  # (3)!
...             val_price = open[i]
...             value = exec_state.cash + val_price * exec_state.position  # (4)!
...
...             exec_state = vbt.pf_enums.ExecState(  # (5)!
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 locked_cash=exec_state.locked_cash,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             order = vbt.pf_nb.order_nb(  # (6)!
...                 size=target_pct[i],
...                 price=close[i],
...                 size_type=vbt.pf_enums.SizeType.TargetPercent
...             )
...             _, exec_state = vbt.pf_nb.execute_order_nb(  # (7)!
...                 exec_state=exec_state,
...                 order=order
...             )
...
...         asset_value_out[i] = exec_state.position * close[i]  # (8)!
...         value_out[i] = exec_state.cash + exec_state.position * close[i]
...         
...     return asset_value_out, value_out
```

```python
>>> @njit
... def pipeline_2_nb(open, close, target_pct, init_cash=100):
...     asset_value_out = np.empty(close.shape, dtype=float_)  # (1)!
...     value_out = np.empty(close.shape, dtype=float_)
...     exec_state = vbt.pf_enums.ExecState(  # (2)!
...         cash=float(init_cash),
...         position=0.0,
...         debt=0.0,
...         locked_cash=0.0,
...         free_cash=float(init_cash),
...         val_price=np.nan,
...         value=np.nan
...     )
...
...     for i in range(close.shape[0]):
...         if not np.isnan(target_pct[i]):  # (3)!
...             val_price = open[i]
...             value = exec_state.cash + val_price * exec_state.position  # (4)!
...
...             exec_state = vbt.pf_enums.ExecState(  # (5)!
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 locked_cash=exec_state.locked_cash,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             order = vbt.pf_nb.order_nb(  # (6)!
...                 size=target_pct[i],
...                 price=close[i],
...                 size_type=vbt.pf_enums.SizeType.TargetPercent
...             )
...             _, exec_state = vbt.pf_nb.execute_order_nb(  # (7)!
...                 exec_state=exec_state,
...                 order=order
...             )
...
...         asset_value_out[i] = exec_state.position * close[i]  # (8)!
...         value_out[i] = exec_state.cash + exec_state.position * close[i]
...         
...     return asset_value_out, value_out
```

```python
np.empty
```

```python
np.nan
```

Let's run the pipeline on our Bitcoin data:

```python
>>> symbol_wrapper = data.get_symbol_wrapper()  # (1)!
>>> target_pct = symbol_wrapper.fill()
>>> target_pct.vbt.set(0.5, every="M", inplace=True)  # (2)!

>>> asset_value, value = pipeline_2_nb(
...     data.get("Open").values, 
...     data.get("Close").values, 
...     target_pct.values
... )
>>> asset_value = symbol_wrapper.wrap(asset_value)  # (3)!
>>> value = symbol_wrapper.wrap(value)
>>> allocations = (asset_value / value).rename(None)  # (4)!
>>> allocations.vbt.scatterplot(trace_kwargs=dict(
...     marker=dict(
...         color=allocations, 
...         colorscale="Temps", 
...         size=3,
...         cmin=0.3,
...         cmid=0.5,
...         cmax=0.7
...     )
... )).show()
```

```python
>>> symbol_wrapper = data.get_symbol_wrapper()  # (1)!
>>> target_pct = symbol_wrapper.fill()
>>> target_pct.vbt.set(0.5, every="M", inplace=True)  # (2)!

>>> asset_value, value = pipeline_2_nb(
...     data.get("Open").values, 
...     data.get("Close").values, 
...     target_pct.values
... )
>>> asset_value = symbol_wrapper.wrap(asset_value)  # (3)!
>>> value = symbol_wrapper.wrap(value)
>>> allocations = (asset_value / value).rename(None)  # (4)!
>>> allocations.vbt.scatterplot(trace_kwargs=dict(
...     marker=dict(
...         color=allocations, 
...         colorscale="Temps", 
...         size=3,
...         cmin=0.3,
...         cmid=0.5,
...         cmax=0.7
...     )
... )).show()
```

```python
data.wrapper
```

```python
0.5
```

Hint

Each point represents a revaluation at the end of each bar.

We see that allocations are being regularly pulled back to the target level of 50%.

Let's validate the pipeline using Portfolio.from_orders:

```python
>>> pf = vbt.Portfolio.from_orders(
...     data, 
...     size=target_pct, 
...     size_type="targetpercent"
... )
>>> pf.allocations.vbt.scatterplot(trace_kwargs=dict(
...     marker=dict(
...         color=allocations, 
...         colorscale="Temps", 
...         size=3,
...         cmin=0.3,
...         cmid=0.5,
...         cmax=0.7
...     )
... )).show()
```

```python
>>> pf = vbt.Portfolio.from_orders(
...     data, 
...     size=target_pct, 
...     size_type="targetpercent"
... )
>>> pf.allocations.vbt.scatterplot(trace_kwargs=dict(
...     marker=dict(
...         color=allocations, 
...         colorscale="Temps", 
...         size=3,
...         cmin=0.3,
...         cmid=0.5,
...         cmax=0.7
...     )
... )).show()
```

One of the biggest advantages of using vectorbt is that you can run your minimalistic trading environment in any Python function, even inside objective functions of machine learning models! There is no need to trigger the entire backtesting pipeline as a script or any other complex process like most backtesting frameworks force us to do

## Order processing¶

Order execution takes an order instruction and translates it into a buy or sell operation. The responsibility of the user is to do something with the returned order execution state and result; mostly, we want to post-process and append each successful order to some list for later analysis - that's where order and log records come into play. Furthermore, we may want to raise an error if an order has been rejected and a certain flag in the requirements is present. All of this is ensured by process_order_nb.

### Order records¶

Order records is a structured NumPy array of the data type order_dt containing information on each successful order. Each order in this array is assumed to be completed, that is, you should view an order as a trade in the vectorbt's world. Since we're dealing with Numba, we cannot and should not use lists and other inefficient data structures for storing such complex information. Given that orders have fields with variable data types, the best data structure is a record array, which is a regular NumPy array with a complex data type and that behaves similarly to a Pandas DataFrame.

Since any NumPy array is a non-appendable structure, we should initialize an empty array of a sufficient size, and gradually fill it with new information. For this, we need a counter - a simple integer - that points to an index of the next record to be written.

Info

Actually, you can append to a NumPy array, but it will create a new array. Don't try this at home

Let's create an array with two order records and a counter:

```python
>>> order_records = np.empty(2, dtype=vbt.pf_enums.order_dt)
>>> order_count = 0
```

```python
>>> order_records = np.empty(2, dtype=vbt.pf_enums.order_dt)
>>> order_count = 0
```

We shouldn't access this array just yet because it contains memory garbage, thus it requires the user to manually set all the values in the array, and should be used with caution.

```python
>>> order_records
array([(4585679916398730403, ..., 4583100142070297783),
       (4582795628349012822, ..., 4576866499094039639)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

```python
>>> order_records
array([(4585679916398730403, ..., 4583100142070297783),
       (4582795628349012822, ..., 4576866499094039639)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

Let's execute an order using execute_order_nb at the 678th bar, and fill the first record in the array:

```python
>>> exec_state = vbt.pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0)
... )
>>> if order_result.status == vbt.pf_enums.OrderStatus.Filled:  # (1)!
...     order_records["id"][order_count] = order_count  # (2)!
...     order_records["col"][order_count] = 0
...     order_records["idx"][order_count] = 678  # (3)!
...     order_records["size"][order_count] = order_result.size
...     order_records["price"][order_count] = order_result.price
...     order_records["fees"][order_count] = order_result.fees
...     order_records["side"][order_count] = order_result.side
...     order_count += 1

>>> order_records[0]
(0, 0, 678, 6.66666667, 15., 0., 0)

>>> order_count
1
```

```python
>>> exec_state = vbt.pf_enums.ExecState(
...     cash=100.0,
...     position=0.0,
...     debt=0.0,
...     locked_cash=0.0,
...     free_cash=100.0,
...     val_price=15.0,
...     value=100.0
... )
>>> order_result, new_exec_state = vbt.pf_nb.execute_order_nb(
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0)
... )
>>> if order_result.status == vbt.pf_enums.OrderStatus.Filled:  # (1)!
...     order_records["id"][order_count] = order_count  # (2)!
...     order_records["col"][order_count] = 0
...     order_records["idx"][order_count] = 678  # (3)!
...     order_records["size"][order_count] = order_result.size
...     order_records["price"][order_count] = order_result.price
...     order_records["fees"][order_count] = order_result.fees
...     order_records["side"][order_count] = order_result.side
...     order_count += 1

>>> order_records[0]
(0, 0, 678, 6.66666667, 15., 0., 0)

>>> order_count
1
```

Note

When writing to an element of a record field, first select the field, and then the index.

At the next bar, we'll reverse the position and fill the second record:

```python
>>> order_result, new_exec_state2 = vbt.pf_nb.execute_order_nb(
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0)
... )
>>> if order_result.status == vbt.pf_enums.OrderStatus.Filled:
...     order_records["id"][order_count] = order_count  # (1)!
...     order_records["col"][order_count] = 0
...     order_records["idx"][order_count] = 679
...     order_records["size"][order_count] = order_result.size
...     order_records["price"][order_count] = order_result.price
...     order_records["fees"][order_count] = order_result.fees
...     order_records["side"][order_count] = order_result.side
...     order_count += 1

>>> order_records[1]
(1, 0, 679, 13.33333333, 16., 0., 1)

>>> order_count
2
```

```python
>>> order_result, new_exec_state2 = vbt.pf_nb.execute_order_nb(
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0)
... )
>>> if order_result.status == vbt.pf_enums.OrderStatus.Filled:
...     order_records["id"][order_count] = order_count  # (1)!
...     order_records["col"][order_count] = 0
...     order_records["idx"][order_count] = 679
...     order_records["size"][order_count] = order_result.size
...     order_records["price"][order_count] = order_result.price
...     order_records["fees"][order_count] = order_result.fees
...     order_records["side"][order_count] = order_result.side
...     order_count += 1

>>> order_records[1]
(1, 0, 679, 13.33333333, 16., 0., 1)

>>> order_count
2
```

Here are the order records that we've populated:

```python
>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

```python
>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

But instead of setting each of these records manually, we can use process_order_nb to do it for us! We just need to do one little adjustment: both the order records and the counter must be provided per column since vectorbt primarily works on multi-columnar data. This means that the order records array must become a two-dimensional array and the counter constant must become a one-dimensional array (both with only one column in our example):

```python
>>> order_records = np.empty((2, 1), dtype=vbt.pf_enums.order_dt)
>>> order_counts = np.full(1, 0, dtype=int_)

>>> order_result1, new_exec_state1 = vbt.pf_nb.process_order_nb(
...     0, 0, 678,  # (1)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0),
...     order_records=order_records,
...     order_counts=order_counts
... )
>>> order_result2, new_exec_state2 = vbt.pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0),
...     order_records=order_records,
...     order_counts=order_counts
... )

>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})

>>> order_counts
array([2])
```

```python
>>> order_records = np.empty((2, 1), dtype=vbt.pf_enums.order_dt)
>>> order_counts = np.full(1, 0, dtype=int_)

>>> order_result1, new_exec_state1 = vbt.pf_nb.process_order_nb(
...     0, 0, 678,  # (1)!
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0),
...     order_records=order_records,
...     order_counts=order_counts
... )
>>> order_result2, new_exec_state2 = vbt.pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0),
...     order_records=order_records,
...     order_counts=order_counts
... )

>>> order_records
array([(0, 0, 678,  6.66666667, 15., 0., 0),
       (1, 0, 679, 13.33333333, 16., 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})

>>> order_counts
array([2])
```

Such filled order records will become the backbone of the post-analysis phase.

### Log records¶

Log records have the data type log_dt and are similar to order records, but with a few key differences: they are saved irrespective of whether the order has been filled, and they also contain information on the current execution state, the order request, and the new execution state. This way, we can completely and post-factum track down issues related to order processing.

```python
>>> order_records = np.empty((2, 1), dtype=vbt.pf_enums.order_dt)
>>> order_counts = np.full(1, 0, dtype=int_)
>>> log_records = np.empty((2, 1), dtype=vbt.pf_enums.log_dt)
>>> log_counts = np.full(1, 0, dtype=int_)

>>> order_result1, new_exec_state1 = vbt.pf_nb.process_order_nb(
...     0, 0, 678,
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0, log=True),  # (1)!
...     order_records=order_records,
...     order_counts=order_counts,
...     log_records=log_records,
...     log_counts=log_counts
... )
>>> order_result2, new_exec_state2 = vbt.pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0, log=True),
...     order_records=order_records,
...     order_counts=order_counts,
...     log_records=log_records,
...     log_counts=log_counts
... )

>>> log_records
array([[(0, 0, 0, 678, ..., 0., 15., 100., 0)],
       [(1, 0, 0, 679, ..., 0., 15., 100., 1)]],
      dtype={'names':['id','group',...,'res_status_info','order_id'], ...})
```

```python
>>> order_records = np.empty((2, 1), dtype=vbt.pf_enums.order_dt)
>>> order_counts = np.full(1, 0, dtype=int_)
>>> log_records = np.empty((2, 1), dtype=vbt.pf_enums.log_dt)
>>> log_counts = np.full(1, 0, dtype=int_)

>>> order_result1, new_exec_state1 = vbt.pf_nb.process_order_nb(
...     0, 0, 678,
...     exec_state=exec_state,
...     order=vbt.pf_nb.order_nb(size=np.inf, price=15.0, log=True),  # (1)!
...     order_records=order_records,
...     order_counts=order_counts,
...     log_records=log_records,
...     log_counts=log_counts
... )
>>> order_result2, new_exec_state2 = vbt.pf_nb.process_order_nb(
...     0, 0, 679,
...     exec_state=new_exec_state,
...     order=vbt.pf_nb.order_nb(size=-np.inf, price=16.0, log=True),
...     order_records=order_records,
...     order_counts=order_counts,
...     log_records=log_records,
...     log_counts=log_counts
... )

>>> log_records
array([[(0, 0, 0, 678, ..., 0., 15., 100., 0)],
       [(1, 0, 0, 679, ..., 0., 15., 100., 1)]],
      dtype={'names':['id','group',...,'res_status_info','order_id'], ...})
```

Note

Logging costs performance and memory. Use only when really needed.

### Pipeline/3¶

Let's extend the last pipeline to independently process an arbitrary number of columns, and gradually fill order records. This way, we can backtest multiple parameter combinations by taking advantage of multidimensionality!

```python
>>> @njit
... def pipeline_3_nb(open, close, target_pct, init_cash=100):
...     order_records = np.empty(close.shape, dtype=vbt.pf_enums.order_dt)  # (1)!
...     order_counts = np.full(close.shape[1], 0, dtype=int_)
...
...     for col in range(close.shape[1]):  # (2)!
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             if not np.isnan(target_pct[i, col]):  # (3)!
...                 val_price = open[i, col]
...                 value = exec_state.cash + val_price * exec_state.position
...
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     locked_cash=exec_state.locked_cash,
...                     free_cash=exec_state.free_cash,
...                     val_price=val_price,
...                     value=value
...                 )
...                 order = vbt.pf_nb.order_nb(
...                     size=target_pct[i, col],
...                     price=close[i, col],
...                     size_type=vbt.pf_enums.SizeType.TargetPercent
...                 )
...                 _, exec_state = vbt.pf_nb.process_order_nb(
...                     col, col, i,  # (4)!
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)  # (4)!
```

```python
>>> @njit
... def pipeline_3_nb(open, close, target_pct, init_cash=100):
...     order_records = np.empty(close.shape, dtype=vbt.pf_enums.order_dt)  # (1)!
...     order_counts = np.full(close.shape[1], 0, dtype=int_)
...
...     for col in range(close.shape[1]):  # (2)!
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             if not np.isnan(target_pct[i, col]):  # (3)!
...                 val_price = open[i, col]
...                 value = exec_state.cash + val_price * exec_state.position
...
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     locked_cash=exec_state.locked_cash,
...                     free_cash=exec_state.free_cash,
...                     val_price=val_price,
...                     value=value
...                 )
...                 order = vbt.pf_nb.order_nb(
...                     size=target_pct[i, col],
...                     price=close[i, col],
...                     size_type=vbt.pf_enums.SizeType.TargetPercent
...                 )
...                 _, exec_state = vbt.pf_nb.process_order_nb(
...                     col, col, i,  # (4)!
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)  # (4)!
```

```python
close
```

Info

We are flattening (repartitioning) order records because most records are left unfilled, thus unnecessarily taking memory. By flattening, we're effectively compressing them without losing any information because each record already tracks the column it's supposed to be in.

Our pipeline now expects all arrays to be two-dimensional. Let's test three value combinations of the parameter every, which controls the re-allocation periodicity. For this, we need to expand all arrays to have the same number of columns as the parameter combinations.

```python
every
```

```python
>>> every = pd.Index(["M", "Q", "Y"], name="every")

>>> open = data.get("Open").vbt.tile(3, keys=every)  # (1)!
>>> close = data.get("Close").vbt.tile(3, keys=every)
>>> close
every                                 M             Q             Y
Date                                                               
2014-09-17 00:00:00+00:00    457.334015    457.334015    457.334015
2014-09-18 00:00:00+00:00    424.440002    424.440002    424.440002
2014-09-19 00:00:00+00:00    394.795990    394.795990    394.795990
...                                 ...           ...           ...
2021-12-29 00:00:00+00:00  46444.710938  46444.710938  46444.710938
2021-12-30 00:00:00+00:00  47178.125000  47178.125000  47178.125000
2021-12-31 00:00:00+00:00  46306.445312  46306.445312  46306.445312

[2663 rows x 3 columns]

>>> target_pct = symbol_wrapper.fill().vbt.tile(3, keys=every)
>>> target_pct.vbt.set(0.5, every="M", columns=["M"], inplace=True)  # (2)!
>>> target_pct.vbt.set(0.5, every="Q", columns=["Q"], inplace=True)
>>> target_pct.vbt.set(0.5, every="Y", columns=["Y"], inplace=True)

>>> order_records = pipeline_3_nb(
...     open.values, 
...     close.values, 
...     target_pct.values
... )
>>> order_records
array([( 0, 0,   14, 1.29056570e-01,   383.61499023, 0., 0),  << first column
       ( 1, 0,   45, 1.00206092e-02,   325.74899292, 0., 0),
       ( 2, 0,   75, 7.10912824e-03,   379.24499512, 0., 1),
       ...
       (84, 0, 2571, 7.79003416e-04, 48116.94140625, 0., 0),
       (85, 0, 2602, 3.00678739e-03, 61004.40625   , 0., 1),
       (86, 0, 2632, 6.84410394e-04, 57229.828125  , 0., 0),
       ( 0, 1,   13, 1.32947604e-01,   386.94400024, 0., 0),  << second column
       ( 1, 1,  105, 1.16132613e-02,   320.19299316, 0., 0),
       ( 2, 1,  195, 1.83187063e-02,   244.22399902, 0., 0),
       ...
       (27, 1, 2478, 7.74416872e-03, 35040.8359375 , 0., 0),
       (28, 1, 2570, 2.08567037e-03, 43790.89453125, 0., 1),
       (29, 1, 2662, 1.72637091e-03, 46306.4453125 , 0., 1),
       ( 0, 2,  105, 1.60816173e-01,   320.19299316, 0., 0),  << third column
       ( 1, 2,  470, 2.34573523e-02,   430.56698608, 0., 1),
       ( 2, 2,  836, 3.81744650e-02,   963.74298096, 0., 1),
       ...
       ( 5, 2, 1931, 2.83026812e-02,  7193.59912109, 0., 1),
       ( 6, 2, 2297, 3.54188390e-02, 29001.72070312, 0., 1),
       ( 7, 2, 2662, 1.14541249e-02, 46306.4453125 , 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

```python
>>> every = pd.Index(["M", "Q", "Y"], name="every")

>>> open = data.get("Open").vbt.tile(3, keys=every)  # (1)!
>>> close = data.get("Close").vbt.tile(3, keys=every)
>>> close
every                                 M             Q             Y
Date                                                               
2014-09-17 00:00:00+00:00    457.334015    457.334015    457.334015
2014-09-18 00:00:00+00:00    424.440002    424.440002    424.440002
2014-09-19 00:00:00+00:00    394.795990    394.795990    394.795990
...                                 ...           ...           ...
2021-12-29 00:00:00+00:00  46444.710938  46444.710938  46444.710938
2021-12-30 00:00:00+00:00  47178.125000  47178.125000  47178.125000
2021-12-31 00:00:00+00:00  46306.445312  46306.445312  46306.445312

[2663 rows x 3 columns]

>>> target_pct = symbol_wrapper.fill().vbt.tile(3, keys=every)
>>> target_pct.vbt.set(0.5, every="M", columns=["M"], inplace=True)  # (2)!
>>> target_pct.vbt.set(0.5, every="Q", columns=["Q"], inplace=True)
>>> target_pct.vbt.set(0.5, every="Y", columns=["Y"], inplace=True)

>>> order_records = pipeline_3_nb(
...     open.values, 
...     close.values, 
...     target_pct.values
... )
>>> order_records
array([( 0, 0,   14, 1.29056570e-01,   383.61499023, 0., 0),  << first column
       ( 1, 0,   45, 1.00206092e-02,   325.74899292, 0., 0),
       ( 2, 0,   75, 7.10912824e-03,   379.24499512, 0., 1),
       ...
       (84, 0, 2571, 7.79003416e-04, 48116.94140625, 0., 0),
       (85, 0, 2602, 3.00678739e-03, 61004.40625   , 0., 1),
       (86, 0, 2632, 6.84410394e-04, 57229.828125  , 0., 0),
       ( 0, 1,   13, 1.32947604e-01,   386.94400024, 0., 0),  << second column
       ( 1, 1,  105, 1.16132613e-02,   320.19299316, 0., 0),
       ( 2, 1,  195, 1.83187063e-02,   244.22399902, 0., 0),
       ...
       (27, 1, 2478, 7.74416872e-03, 35040.8359375 , 0., 0),
       (28, 1, 2570, 2.08567037e-03, 43790.89453125, 0., 1),
       (29, 1, 2662, 1.72637091e-03, 46306.4453125 , 0., 1),
       ( 0, 2,  105, 1.60816173e-01,   320.19299316, 0., 0),  << third column
       ( 1, 2,  470, 2.34573523e-02,   430.56698608, 0., 1),
       ( 2, 2,  836, 3.81744650e-02,   963.74298096, 0., 1),
       ...
       ( 5, 2, 1931, 2.83026812e-02,  7193.59912109, 0., 1),
       ( 6, 2, 2297, 3.54188390e-02, 29001.72070312, 0., 1),
       ( 7, 2, 2662, 1.14541249e-02, 46306.4453125 , 0., 1)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

### Wrapping¶

This output is exactly what Portfolio requires as input: order records with a couple of other arguments can be used to reconstruct the simulation state, including the regular cash balance and the position size at each time step. The reconstructed state can be used to model the equity curve, then returns, and then the accompanying metrics such as the old but gold Sharpe ratio. So, what how do we construct a portfolio? Instead of using any class method, we'll pass the data directly to the class. For this, only three arguments are required: a wrapper, a close series, and order records. Ideally, we should also supply arguments that were used during the simulation, such as the initial cash:

```python
>>> vbt.phelp(vbt.Portfolio)
Portfolio.__init__(
    self,
    wrapper,
    order_records,
    *,
    close,
    open=None,
    high=None,
    low=None,
    log_records=None,
    cash_sharing=False,
    init_cash='auto',
    ...
)
```

```python
>>> vbt.phelp(vbt.Portfolio)
Portfolio.__init__(
    self,
    wrapper,
    order_records,
    *,
    close,
    open=None,
    high=None,
    low=None,
    log_records=None,
    cash_sharing=False,
    init_cash='auto',
    ...
)
```

Hint

Notice *? It means any argument after order_records must be provided as a keyword argument.

```python
*
```

```python
order_records
```

Let's do the wrapping part:

```python
>>> pf = vbt.Portfolio(
...     close.vbt.wrapper,
...     order_records,
...     open=open,
...     close=close,
...     init_cash=100  # (1)!
... )
```

```python
>>> pf = vbt.Portfolio(
...     close.vbt.wrapper,
...     order_records,
...     open=open,
...     close=close,
...     init_cash=100  # (1)!
... )
```

We can now use the portfolio the same way as if we simulated it with any preset method:

```python
>>> pf.sharpe_ratio
every
MS    1.267804
Q     1.309943
Y     1.393820
Name: sharpe_ratio, dtype: float64
```

```python
>>> pf.sharpe_ratio
every
MS    1.267804
Q     1.309943
Y     1.393820
Name: sharpe_ratio, dtype: float64
```

## Flexible indexing¶

The issue of bringing all arrays to the same shape as we did above is that it unnecessarily consumes memory: even though the only array that has different data in each column is target_pct, we have almost tripled memory consumption by having to expand other arrays like close. Imagine how expensive would it be having to align dozens of such array-like arguments

```python
target_pct
```

```python
close
```

Flexible indexing allows us to overcome this alignment step and to access each element of an array solely based on its shape. For example, there is no need to tile close three times if each row stays the same for each column - we can simply return the same row element irrespective of the column being queried. The same goes for a one-dimensional array with elements per column - return the same column element for each row. The only requirement is that the array must have one dimension if it should broadcast against rows or columns, and two dimensions if it should broadcast against rows and columns. Any scalars should be transformed into one of the above formats, otherwise we'll be greeted with an ugly Numba error.

```python
close
```

For the actual indexing, we can use the following Numba-compiled functions:

Let's demonstrate their use in different scenarios:

```python
>>> per_row_arr = np.array([1, 2, 3])
>>> per_col_arr = np.array([4, 5])
>>> per_elem_arr = np.array([
...     [6, 7],
...     [8, 9],
...     [10, 11]
... ])

>>> vbt.flex_select_1d_pr_nb(per_row_arr, 2)  # (1)!
3

>>> vbt.flex_select_1d_pc_nb(per_col_arr, 1)  # (2)!
5

>>> vbt.flex_select_nb(per_elem_arr, 2, 1)  # (3)!
11
```

```python
>>> per_row_arr = np.array([1, 2, 3])
>>> per_col_arr = np.array([4, 5])
>>> per_elem_arr = np.array([
...     [6, 7],
...     [8, 9],
...     [10, 11]
... ])

>>> vbt.flex_select_1d_pr_nb(per_row_arr, 2)  # (1)!
3

>>> vbt.flex_select_1d_pc_nb(per_col_arr, 1)  # (2)!
5

>>> vbt.flex_select_nb(per_elem_arr, 2, 1)  # (3)!
11
```

One-dimensional indexing functions are suited only for arguments that are one-dimensional by design, such as initial capital, which makes only sense to be provided per column, not per element. But what if the user should also be able to pass per_row_arr or per_col_arr as fully-broadcast arrays? In this case, the user needs to expand both arrays to two dimensions according to the NumPy's broadcasting rules and use exclusively flex_select_nb. The reason for this is that Numba isn't flexible enough to permit doing operations on both one-dimensional and two-dimensional arrays, so we must decide on the indexing function beforehand.

```python
per_row_arr
```

```python
per_col_arr
```

```python
>>> per_row_arr_2d = per_row_arr[:, None]  # (1)!
>>> per_row_arr_2d
array([[1],
       [2],
       [3]])

>>> vbt.flex_select_nb(per_row_arr_2d, 2, 1)
3

>>> per_col_arr_2d = per_col_arr[None]  # (2)!
>>> per_col_arr_2d
array([[4, 5]])

>>> vbt.flex_select_nb(per_col_arr_2d, 2, 1)
5
```

```python
>>> per_row_arr_2d = per_row_arr[:, None]  # (1)!
>>> per_row_arr_2d
array([[1],
       [2],
       [3]])

>>> vbt.flex_select_nb(per_row_arr_2d, 2, 1)
3

>>> per_col_arr_2d = per_col_arr[None]  # (2)!
>>> per_col_arr_2d
array([[4, 5]])

>>> vbt.flex_select_nb(per_col_arr_2d, 2, 1)
5
```

```python
(3,)
```

```python
(3, 1)
```

```python
(3,)
```

```python
(1, 3)
```

This yields the same results as if we had aligned the arrays prior to indexing (= memory expensive):

```python
>>> target_shape = (3, 2)

>>> vbt.broadcast_array_to(per_row_arr, target_shape[0])[2]
3

>>> vbt.broadcast_array_to(per_col_arr, target_shape[1])[1]
5

>>> vbt.broadcast_array_to(per_row_arr_2d, target_shape)[2, 1]
3

>>> vbt.broadcast_array_to(per_col_arr_2d, target_shape)[2, 1]
5
```

```python
>>> target_shape = (3, 2)

>>> vbt.broadcast_array_to(per_row_arr, target_shape[0])[2]
3

>>> vbt.broadcast_array_to(per_col_arr, target_shape[1])[1]
5

>>> vbt.broadcast_array_to(per_row_arr_2d, target_shape)[2, 1]
3

>>> vbt.broadcast_array_to(per_col_arr_2d, target_shape)[2, 1]
5
```

Hint

If you're not sure whether a flexible array will be indexed correctly, try broadcasting it with NumPy!

### Rotational indexing¶

But what happens if the index is out of bounds? Let's say we iterate over 6 columns but an array has data only for 3. In such a case, vectorbt can rotate the index and return the first element in the array for the fourth column, the second element for the fifth column, and so on:

```python
>>> vbt.flex_select_1d_pr_nb(per_row_arr, 100, rotate_rows=True)  # (1)!
2

>>> vbt.flex_select_1d_pc_nb(per_col_arr, 100, rotate_cols=True)  # (2)!
4

>>> vbt.flex_select_nb(per_elem_arr, 100, 100, rotate_rows=True, rotate_cols=True)
8
```

```python
>>> vbt.flex_select_1d_pr_nb(per_row_arr, 100, rotate_rows=True)  # (1)!
2

>>> vbt.flex_select_1d_pc_nb(per_col_arr, 100, rotate_cols=True)  # (2)!
4

>>> vbt.flex_select_nb(per_elem_arr, 100, 100, rotate_rows=True, rotate_cols=True)
8
```

If you think that this is crazy, and you would rather have an error shown: rotational indexing is very useful when it comes to testing multiple assets and parameter combinations. Without it (default), we would need to tile the asset DataFrame by the number of parameter combinations, but with it, we could have just passed the data without tiling and thus wasting memory. But also, in many places, vectorbt ensures that all arrays can broadcast against each other nicely anyway.

### Pipeline/4¶

Let's adapt the previous pipeline for flexible indexing. Since usually we don't know which one of the passed arrays has the full shape, and sometimes there is no array with the full shape at all, we need to introduce another argument - target_shape - to provide the full shape for our loops to iterate over. We'll also experiment with rotational indexing, which isn't supported by any of the preset simulation methods.

```python
target_shape
```

```python
>>> @njit
... def pipeline_4_nb(
...     target_shape, 
...     open, 
...     close, 
...     target_pct, 
...     init_cash=100,
...     rotate_cols=False
... ):
...     init_cash_ = vbt.to_1d_array_nb(np.asarray(init_cash))  # (1)!
...     open_ = vbt.to_2d_array_nb(np.asarray(open))  # (2)!
...     close_ = vbt.to_2d_array_nb(np.asarray(close))
...     target_pct_ = vbt.to_2d_array_nb(np.asarray(target_pct))
...     order_records = np.empty(target_shape, dtype=vbt.pf_enums.order_dt)
...     order_counts = np.full(target_shape[1], 0, dtype=int_)
...
...     for col in range(target_shape[1]):
...         init_cash_elem = vbt.flex_select_1d_pc_nb(
...             init_cash_, col, rotate_cols=rotate_cols)  # (3)!
...
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash_elem),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash_elem),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(target_shape[0]):
...             open_elem = vbt.flex_select_nb(
...                 open_, i, col, rotate_cols=rotate_cols)  # (4)!
...             close_elem = vbt.flex_select_nb(
...                 close_, i, col, rotate_cols=rotate_cols)
...             target_pct_elem = vbt.flex_select_nb(
...                 target_pct_, i, col, rotate_cols=rotate_cols)
...
...             if not np.isnan(target_pct_elem):
...                 value = exec_state.cash + open_elem * exec_state.position
...
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     locked_cash=exec_state.locked_cash,
...                     free_cash=exec_state.free_cash,
...                     val_price=open_elem,
...                     value=value
...                 )
...                 order = vbt.pf_nb.order_nb(
...                     size=target_pct_elem,
...                     price=close_elem,
...                     size_type=vbt.pf_enums.SizeType.TargetPercent
...                 )
...                 _, exec_state = vbt.pf_nb.process_order_nb(
...                     col, col, i,
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

```python
>>> @njit
... def pipeline_4_nb(
...     target_shape, 
...     open, 
...     close, 
...     target_pct, 
...     init_cash=100,
...     rotate_cols=False
... ):
...     init_cash_ = vbt.to_1d_array_nb(np.asarray(init_cash))  # (1)!
...     open_ = vbt.to_2d_array_nb(np.asarray(open))  # (2)!
...     close_ = vbt.to_2d_array_nb(np.asarray(close))
...     target_pct_ = vbt.to_2d_array_nb(np.asarray(target_pct))
...     order_records = np.empty(target_shape, dtype=vbt.pf_enums.order_dt)
...     order_counts = np.full(target_shape[1], 0, dtype=int_)
...
...     for col in range(target_shape[1]):
...         init_cash_elem = vbt.flex_select_1d_pc_nb(
...             init_cash_, col, rotate_cols=rotate_cols)  # (3)!
...
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash_elem),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash_elem),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(target_shape[0]):
...             open_elem = vbt.flex_select_nb(
...                 open_, i, col, rotate_cols=rotate_cols)  # (4)!
...             close_elem = vbt.flex_select_nb(
...                 close_, i, col, rotate_cols=rotate_cols)
...             target_pct_elem = vbt.flex_select_nb(
...                 target_pct_, i, col, rotate_cols=rotate_cols)
...
...             if not np.isnan(target_pct_elem):
...                 value = exec_state.cash + open_elem * exec_state.position
...
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=exec_state.cash,
...                     position=exec_state.position,
...                     debt=exec_state.debt,
...                     locked_cash=exec_state.locked_cash,
...                     free_cash=exec_state.free_cash,
...                     val_price=open_elem,
...                     value=value
...                 )
...                 order = vbt.pf_nb.order_nb(
...                     size=target_pct_elem,
...                     price=close_elem,
...                     size_type=vbt.pf_enums.SizeType.TargetPercent
...                 )
...                 _, exec_state = vbt.pf_nb.process_order_nb(
...                     col, col, i,
...                     exec_state=exec_state,
...                     order=order,
...                     order_records=order_records,
...                     order_counts=order_counts
...                 )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

```python
1d
```

```python
open[i, col]
```

```python
1d
```

Thanks to flexible indexing, we can now use all arrays without tiling:

```python
>>> target_shape = vbt.broadcast_shapes(  # (1)!
...     data.get("Open").values.shape,
...     data.get("Close").values.shape,
...     target_pct.values.shape
... )
>>> target_shape
(2663, 3)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     data.get("Open").values,
...     data.get("Close").values,
...     target_pct.values
... )
>>> len(order_records)
125
```

```python
>>> target_shape = vbt.broadcast_shapes(  # (1)!
...     data.get("Open").values.shape,
...     data.get("Close").values.shape,
...     target_pct.values.shape
... )
>>> target_shape
(2663, 3)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     data.get("Open").values,
...     data.get("Close").values,
...     target_pct.values
... )
>>> len(order_records)
125
```

This also allows us to provide target percentages as a constant to re-allocate at each bar! Since constants don't have any implications on the targe shape, we only need to broadcast the price shapes:

```python
>>> target_shape = vbt.broadcast_shapes(
...     data.get("Open").values.shape,
...     data.get("Close").values.shape
... )
>>> target_shape
(2663,)

>>> target_shape = vbt.to_2d_shape(target_shape)  # (1)!
>>> target_shape
(2663, 1)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     data.get("Open").values,
...     data.get("Close").values,
...     0.5
... )
len(order_records)
2663
```

```python
>>> target_shape = vbt.broadcast_shapes(
...     data.get("Open").values.shape,
...     data.get("Close").values.shape
... )
>>> target_shape
(2663,)

>>> target_shape = vbt.to_2d_shape(target_shape)  # (1)!
>>> target_shape
(2663, 1)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     data.get("Open").values,
...     data.get("Close").values,
...     0.5
... )
len(order_records)
2663
```

This operation has generated the same number of orders as we have elements in the data:

```python
>>> np.product(symbol_wrapper.shape_2d)
2663
```

```python
>>> np.product(symbol_wrapper.shape_2d)
2663
```

To demonstrate rotational indexing, let's pull multiple symbols and perform the simulation without having to tile or change them in any way:

```python
>>> mult_data = vbt.YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     end="2022-01-01",
...     missing_index="drop"
... )
```

```python
>>> mult_data = vbt.YFData.pull(
...     ["BTC-USD", "ETH-USD"], 
...     end="2022-01-01",
...     missing_index="drop"
... )
```

Symbol 2/2

Symbol 2/2

```python
>>> mult_symbol_wrapper = mult_data.get_symbol_wrapper()
>>> mult_target_pct = pd.concat([
...     mult_symbol_wrapper.fill().vbt.set(0.5, every=every[i])
...     for i in range(len(every))
... ], axis=1, keys=every)  # (1)!

>>> target_shape = vbt.broadcast_shapes(
...     vbt.tile_shape(mult_data.get("Open").values.shape, len(every)),  # (2)!
...     vbt.tile_shape(mult_data.get("Close").values.shape, len(every)), 
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 6)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     mult_data.get("Open").values,  # (3)!
...     mult_data.get("Close").values,
...     mult_target_pct.values,
...     rotate_cols=True
... )
>>> len(order_records)
142
```

```python
>>> mult_symbol_wrapper = mult_data.get_symbol_wrapper()
>>> mult_target_pct = pd.concat([
...     mult_symbol_wrapper.fill().vbt.set(0.5, every=every[i])
...     for i in range(len(every))
... ], axis=1, keys=every)  # (1)!

>>> target_shape = vbt.broadcast_shapes(
...     vbt.tile_shape(mult_data.get("Open").values.shape, len(every)),  # (2)!
...     vbt.tile_shape(mult_data.get("Close").values.shape, len(every)), 
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 6)

>>> order_records = pipeline_4_nb(
...     target_shape,
...     mult_data.get("Open").values,  # (3)!
...     mult_data.get("Close").values,
...     mult_target_pct.values,
...     rotate_cols=True
... )
>>> len(order_records)
142
```

```python
open
```

```python
close
```

Without rotation, we would have got an "IndexError: index out of bounds" error as the number of columns in the target shape is bigger than that in the price arrays.

## Grouping¶

Using groups, we can put multiple columns to the same backtesting basket

Generally, a group consists of a number of columns that are part of a single portfolio entity and should be backtested as a single whole. Very often, we use groups to share capital among multiple columns, but we can also use groups to bind columns on some logical level. During a simulation, it's our responsibility to make use of grouping. For example, even though process_order_nb requires a group index, it uses it just for filling log records and nothing else. But after the simulation, vectorbt has many tools at its disposal to enable us in aggregating and analyzing various information per group, such as portfolio value.

Groups can be constructed and provided in two ways: as group lengths and as a group map. The former is easier to handle, marginally faster, and requires the columns to be split into monolithic groups, while the latter allows the columns of a group to be distributed arbitrarily, and is generally a more flexible option. Group lengths is the format primarily used by simulation methods (since asset columns, in contrast to parameter columns, are usually located next to each other), while group maps are predominantly used by generic functions specialized in pre- and post-analysis. Both formats can be easily generated by a Grouper instance.

### Group lengths¶

Let's create a custom column index with 5 assets, and put them into 2 groups. Since group lengths work on monolithic groups only, assets in each group must be next to each other:

```python
>>> columns = pd.Index(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"])
>>> mono_grouper = vbt.Grouper(columns, group_by=[0, 0, 0, 1, 1])
>>> mono_grouper.get_group_lens()  # (1)!
array([3, 2])
```

```python
>>> columns = pd.Index(["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"])
>>> mono_grouper = vbt.Grouper(columns, group_by=[0, 0, 0, 1, 1])
>>> mono_grouper.get_group_lens()  # (1)!
array([3, 2])
```

The first element in the returned array is the number of columns with the label 0, and the second element is the number of columns with the label 1.

```python
0
```

```python
1
```

Hint

Grouper doesn't care if we pass a list of integers or a sequence of strings - it will convert everything into a Pandas Index and treat it as group labels. They don't have to be alphanumerically sorted.

If we create discrete groups, the generation will fail:

```python
>>> dist_grouper = vbt.Grouper(columns, group_by=[0, 1, 0, 1, 1])
>>> dist_grouper.get_group_lens()
ValueError: group_by must form monolithic groups
```

```python
>>> dist_grouper = vbt.Grouper(columns, group_by=[0, 1, 0, 1, 1])
>>> dist_grouper.get_group_lens()
ValueError: group_by must form monolithic groups
```

Now, how do we define logic per group? Here's a template:

```python
>>> group_lens = mono_grouper.get_group_lens()

>>> group_end_idxs = np.cumsum(group_lens)  # (1)!
>>> group_start_idxs = group_end_idxs - group_lens  # (2)!

>>> for group in range(len(group_lens)):  # (3)!
...     from_col = group_start_idxs[group]
...     to_col = group_end_idxs[group]
...     # (4)!
...
...     for col in range(from_col, to_col):  # (5)!
...         pass  # (6)!
```

```python
>>> group_lens = mono_grouper.get_group_lens()

>>> group_end_idxs = np.cumsum(group_lens)  # (1)!
>>> group_start_idxs = group_end_idxs - group_lens  # (2)!

>>> for group in range(len(group_lens)):  # (3)!
...     from_col = group_start_idxs[group]
...     to_col = group_end_idxs[group]
...     # (4)!
...
...     for col in range(from_col, to_col):  # (5)!
...         pass  # (6)!
```

### Group map¶

Group map is a tuple of two arrays:

Thus, a group map makes distributed groups inherently monolithic, such that we can work with any possible group distribution:

```python
>>> mono_grouper.get_group_map()
(array([0, 1, 2, 3, 4]), array([3, 2]))

>>> dist_grouper.get_group_map()
(array([0, 2, 1, 3, 4]), array([2, 3]))
```

```python
>>> mono_grouper.get_group_map()
(array([0, 1, 2, 3, 4]), array([3, 2]))

>>> dist_grouper.get_group_map()
(array([0, 2, 1, 3, 4]), array([2, 3]))
```

In the second example, the first two (2) column indices in the first array belong to the first group, while the remaining three (3) column indices belong to the second group.

```python
2
```

```python
3
```

Here's a template for working with a group map:

```python
>>> group_map = dist_grouper.get_group_map()

>>> group_idxs, group_lens = group_map
>>> group_start_idxs = np.cumsum(group_lens) - group_lens  # (1)!

>>> for group in range(len(group_lens)):
...     group_len = group_lens[group]
...     start_idx = group_start_idxs[group]
...     col_idxs = group_idxs[start_idx : start_idx + group_len]  # (2)!
...     # (3)!
... 
...     for k in range(len(col_idxs)):  # (4)!
...         col = col_idxs[k]
...         # (5)!
```

```python
>>> group_map = dist_grouper.get_group_map()

>>> group_idxs, group_lens = group_map
>>> group_start_idxs = np.cumsum(group_lens) - group_lens  # (1)!

>>> for group in range(len(group_lens)):
...     group_len = group_lens[group]
...     start_idx = group_start_idxs[group]
...     col_idxs = group_idxs[start_idx : start_idx + group_len]  # (2)!
...     # (3)!
... 
...     for k in range(len(col_idxs)):  # (4)!
...         col = col_idxs[k]
...         # (5)!
```

### Call sequence¶

When sharing capital between two or more assets, we sometimes want to process one column before the others. This makes most sense, for example, in cases where we need to exit positions before opening new ones to release funds for them. If we look at the templates for both grouping formats above, we can precisely identify where the column processing order should be changed: in the for-loop that iterates over columns. But how do we programmatically change this order? Here comes a call sequence into play.

Call sequence is an array of column indices in the order of their processing. For example, if the third column should be processed first, the first column second, and the second column third, the call sequence would be [2, 0, 1]. That is, we are always moving from left to right in the call sequence and pick the current column index. Such a design has one immense benefit: we can use another array, such as with potential order values, to (arg-)sort the call sequence.

```python
[2, 0, 1]
```

The sorting is done by the function insert_argsort_nb, which takes an array with values to sort by and an array of indices, and sorts the indices in-place using insertion sort in the order the values appear in the first array. This sorting algorithm is best suited for smaller arrays and does not require any additional memory space - perfect for groups with assets!

Let's say we have three assets: one not in position, one in a short position, and one in a long position. We want to close all positions in the order such that assets that should be sold are processed first. Otherwise, we wouldn't have cash from exiting the long position to close out the short position. For this, we will first use approx_order_value_nb to approximate the order value of each operation:

```python
>>> position = np.array([0.0, -10.0, 10.0])
>>> val_price = np.array([10.0, 25.0, 15.0])
>>> debt = np.array([0.0, 100.0, 0.0])
>>> locked_cash = np.array([0.0, 100.0, 0.0])
>>> order_value = np.empty(3, dtype=float_)

>>> for col in range(len(position)):
...     exec_state = vbt.pf_enums.ExecState(
...         cash=200.0,  # (1)!
...         position=position[col],  # (2)!
...         debt=debt[col],
...         locked_cash=locked_cash[col],
...         free_cash=0.0,
...         val_price=val_price[col],
...         value=100.0
...     )
...     order_value[col] = vbt.pf_nb.approx_order_value_nb(
...         exec_state=exec_state,
...         size=0.,
...         size_type=vbt.pf_enums.SizeType.TargetAmount,
...         direction=vbt.pf_enums.Direction.Both
...     )

>>> order_value  # (3)!
array([0., 50., -150.])
```

```python
>>> position = np.array([0.0, -10.0, 10.0])
>>> val_price = np.array([10.0, 25.0, 15.0])
>>> debt = np.array([0.0, 100.0, 0.0])
>>> locked_cash = np.array([0.0, 100.0, 0.0])
>>> order_value = np.empty(3, dtype=float_)

>>> for col in range(len(position)):
...     exec_state = vbt.pf_enums.ExecState(
...         cash=200.0,  # (1)!
...         position=position[col],  # (2)!
...         debt=debt[col],
...         locked_cash=locked_cash[col],
...         free_cash=0.0,
...         val_price=val_price[col],
...         value=100.0
...     )
...     order_value[col] = vbt.pf_nb.approx_order_value_nb(
...         exec_state=exec_state,
...         size=0.,
...         size_type=vbt.pf_enums.SizeType.TargetAmount,
...         direction=vbt.pf_enums.Direction.Both
...     )

>>> order_value  # (3)!
array([0., 50., -150.])
```

We see that the second column would require approx. $50 in cash and the third column would bring approx. $150 in cash to close out the position. Let's create a call sequence and sort it by the order value:

```python
>>> from vectorbtpro.utils.array_ import insert_argsort_nb

>>> call_seq = np.array([0, 1, 2])  # (1)!
>>> insert_argsort_nb(order_value, call_seq)
>>> call_seq
array([2, 0, 1])
```

```python
>>> from vectorbtpro.utils.array_ import insert_argsort_nb

>>> call_seq = np.array([0, 1, 2])  # (1)!
>>> insert_argsort_nb(order_value, call_seq)
>>> call_seq
array([2, 0, 1])
```

Note

Both the order value and the call sequence are sorted in-place!

We can then modify the for-loop to iterate over the call sequence instead:

```python
>>> for k in range(len(call_seq)):  # (1)!
...     c = call_seq[k]
...     col = from_col + c

>>> for k in range(len(call_seq)):  # (2)!
...     c = call_seq[k]
...     col = col_idxs[c]
```

```python
>>> for k in range(len(call_seq)):  # (1)!
...     c = call_seq[k]
...     col = from_col + c

>>> for k in range(len(call_seq)):  # (2)!
...     c = call_seq[k]
...     col = col_idxs[c]
```

Hint

A good practice is to keep a consistent naming of variables. Here, we're using k to denote an index in the call sequence, c to denote a column index within a group, and col to denote a global column index.

```python
k
```

```python
c
```

```python
col
```

### Pipeline/5¶

Let's upgrade our previous pipeline to rebalance groups of assets. To better illustrate how important is sorting by order value when rebalancing multi-asset portfolios, we'll introduce another argument auto_call_seq to switch between sorting and not sorting. We will use group lengths as the grouping format of choice because of its simplicity. Also note that now we have to keep a lot of position-related information in arrays rather than constants since they exist in relation to columns rather than groups. In addition, as we already know how to fill order records, let's track the allocation at each bar instead.

```python
auto_call_seq
```

```python
>>> @njit
... def pipeline_5_nb(
...     target_shape,  # (1)!
...     group_lens,  # (2)!
...     open,
...     close, 
...     target_pct,  # (3)!
...     init_cash=100,
...     auto_call_seq=True,
...     rotate_cols=False
... ):
...     init_cash_ = vbt.to_1d_array_nb(np.asarray(init_cash))
...     open_ = vbt.to_2d_array_nb(np.asarray(open))
...     close_ = vbt.to_2d_array_nb(np.asarray(close))
...     target_pct_ = vbt.to_2d_array_nb(np.asarray(target_pct))
...     alloc = np.empty(target_shape, dtype=float_)  # (4)!
...
...     group_end_idxs = np.cumsum(group_lens)
...     group_start_idxs = group_end_idxs - group_lens
...
...     for group in range(len(group_lens)):  # (5)!
...         group_len = group_lens[group]
...         from_col = group_start_idxs[group]
...         to_col = group_end_idxs[group]
...
...         # (6)!
...         init_cash_elem = vbt.flex_select_1d_pc_nb(
...             init_cash_, group, rotate_cols=rotate_cols)
...     
...         last_position = np.full(group_len, 0.0, dtype=float_)  # (7)!
...         last_debt = np.full(group_len, 0.0, dtype=float_)
...         last_locked_cash = np.full(group_len, 0.0, dtype=float_)
...         cash_now = float(init_cash_elem)
...         free_cash_now = float(init_cash_elem)
...
...         order_value = np.empty(group_len, dtype=float_)  # (8)!
...         call_seq = np.empty(group_len, dtype=int_)
... 
...         for i in range(target_shape[0]):  # (9)!
...             # (10)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * open_elem  # (11)!
...         
...             # (12)!
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_nb(
...                     target_pct_, i, col, rotate_cols=rotate_cols)
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=cash_now,
...                     position=last_position[c],
...                     locked_cash=last_locked_cash[c],
...                     debt=last_debt[c],
...                     free_cash=free_cash_now,
...                     val_price=open_elem,
...                     value=value_now,
...                 )
...                 order_value[c] = vbt.pf_nb.approx_order_value_nb(  # (13)!
...                     exec_state=exec_state,
...                     size=target_pct_elem,
...                     size_type=vbt.pf_enums.SizeType.TargetPercent,
...                     direction=vbt.pf_enums.Direction.Both
...                 )
...                 call_seq[c] = c  # (14)!
... 
...             if auto_call_seq:
...                 vbt.pf_nb.insert_argsort_nb(order_value, call_seq)  # (15)!
... 
...             for k in range(len(call_seq)):  # (16)!
...                 c = call_seq[k]  # (17)!
...                 col = from_col + c  # (18)!
...
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_nb(
...                     target_pct_, i, col, rotate_cols=rotate_cols)
...
...                 if not np.isnan(target_pct_elem):  # (19)!
...                     order = vbt.pf_nb.order_nb(
...                         size=target_pct_elem,
...                         price=close_elem,
...                         size_type=vbt.pf_enums.SizeType.TargetPercent,
...                         direction=vbt.pf_enums.Direction.Both
...                     )
...                     exec_state = vbt.pf_enums.ExecState(
...                         cash=cash_now,
...                         position=last_position[c],
...                         debt=last_debt[c],
...                         locked_cash=last_locked_cash[c],
...                         free_cash=free_cash_now,
...                         val_price=open_elem,
...                         value=value_now,
...                     )
...                     _, new_exec_state = vbt.pf_nb.process_order_nb(
...                         group=group,
...                         col=col,
...                         i=i,
...                         exec_state=exec_state,
...                         order=order
...                     )
...                     cash_now = new_exec_state.cash
...                     free_cash_now = new_exec_state.free_cash
...                     value_now = new_exec_state.value
...                     last_position[c] = new_exec_state.position
...                     last_debt[c] = new_exec_state.debt
...                     last_locked_cash[c] = new_exec_state.locked_cash
...
...             # (20)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * close_elem
...
...             # (21)!
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 alloc[i, col] = last_position[c] * close_elem / value_now
... 
...     return alloc
```

```python
>>> @njit
... def pipeline_5_nb(
...     target_shape,  # (1)!
...     group_lens,  # (2)!
...     open,
...     close, 
...     target_pct,  # (3)!
...     init_cash=100,
...     auto_call_seq=True,
...     rotate_cols=False
... ):
...     init_cash_ = vbt.to_1d_array_nb(np.asarray(init_cash))
...     open_ = vbt.to_2d_array_nb(np.asarray(open))
...     close_ = vbt.to_2d_array_nb(np.asarray(close))
...     target_pct_ = vbt.to_2d_array_nb(np.asarray(target_pct))
...     alloc = np.empty(target_shape, dtype=float_)  # (4)!
...
...     group_end_idxs = np.cumsum(group_lens)
...     group_start_idxs = group_end_idxs - group_lens
...
...     for group in range(len(group_lens)):  # (5)!
...         group_len = group_lens[group]
...         from_col = group_start_idxs[group]
...         to_col = group_end_idxs[group]
...
...         # (6)!
...         init_cash_elem = vbt.flex_select_1d_pc_nb(
...             init_cash_, group, rotate_cols=rotate_cols)
...     
...         last_position = np.full(group_len, 0.0, dtype=float_)  # (7)!
...         last_debt = np.full(group_len, 0.0, dtype=float_)
...         last_locked_cash = np.full(group_len, 0.0, dtype=float_)
...         cash_now = float(init_cash_elem)
...         free_cash_now = float(init_cash_elem)
...
...         order_value = np.empty(group_len, dtype=float_)  # (8)!
...         call_seq = np.empty(group_len, dtype=int_)
... 
...         for i in range(target_shape[0]):  # (9)!
...             # (10)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * open_elem  # (11)!
...         
...             # (12)!
...             for c in range(group_len):
...                 col = from_col + c
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_nb(
...                     target_pct_, i, col, rotate_cols=rotate_cols)
...                 exec_state = vbt.pf_enums.ExecState(
...                     cash=cash_now,
...                     position=last_position[c],
...                     locked_cash=last_locked_cash[c],
...                     debt=last_debt[c],
...                     free_cash=free_cash_now,
...                     val_price=open_elem,
...                     value=value_now,
...                 )
...                 order_value[c] = vbt.pf_nb.approx_order_value_nb(  # (13)!
...                     exec_state=exec_state,
...                     size=target_pct_elem,
...                     size_type=vbt.pf_enums.SizeType.TargetPercent,
...                     direction=vbt.pf_enums.Direction.Both
...                 )
...                 call_seq[c] = c  # (14)!
... 
...             if auto_call_seq:
...                 vbt.pf_nb.insert_argsort_nb(order_value, call_seq)  # (15)!
... 
...             for k in range(len(call_seq)):  # (16)!
...                 c = call_seq[k]  # (17)!
...                 col = from_col + c  # (18)!
...
...                 open_elem = vbt.flex_select_nb(
...                     open_, i, col, rotate_cols=rotate_cols)
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 target_pct_elem = vbt.flex_select_nb(
...                     target_pct_, i, col, rotate_cols=rotate_cols)
...
...                 if not np.isnan(target_pct_elem):  # (19)!
...                     order = vbt.pf_nb.order_nb(
...                         size=target_pct_elem,
...                         price=close_elem,
...                         size_type=vbt.pf_enums.SizeType.TargetPercent,
...                         direction=vbt.pf_enums.Direction.Both
...                     )
...                     exec_state = vbt.pf_enums.ExecState(
...                         cash=cash_now,
...                         position=last_position[c],
...                         debt=last_debt[c],
...                         locked_cash=last_locked_cash[c],
...                         free_cash=free_cash_now,
...                         val_price=open_elem,
...                         value=value_now,
...                     )
...                     _, new_exec_state = vbt.pf_nb.process_order_nb(
...                         group=group,
...                         col=col,
...                         i=i,
...                         exec_state=exec_state,
...                         order=order
...                     )
...                     cash_now = new_exec_state.cash
...                     free_cash_now = new_exec_state.free_cash
...                     value_now = new_exec_state.value
...                     last_position[c] = new_exec_state.position
...                     last_debt[c] = new_exec_state.debt
...                     last_locked_cash[c] = new_exec_state.locked_cash
...
...             # (20)!
...             value_now = cash_now
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 value_now += last_position[c] * close_elem
...
...             # (21)!
...             for c in range(group_len):
...                 col = from_col + c
...                 close_elem = vbt.flex_select_nb(
...                     close_, i, col, rotate_cols=rotate_cols)
...                 alloc[i, col] = last_position[c] * close_elem / value_now
... 
...     return alloc
```

```python
target_shape
```

```python
target_shape
```

```python
target_shape
```

```python
c
```

```python
col
```

```python
np.nan
```

Wow, this went complex really fast!

But it's not that complex as it may appear. We took a bunch of columns and split them into groups. Then, for each group, we defined a mini-pipeline that applies our logic on the columns within this group only, acting as a single portfolio unit. At the beginning of each bar, we calculate the portfolio value, and build a call sequence that re-arranges columns by their order value. We then iterate over this sequence and execute an order in each column. Finally, at the end of each bar, we again re-calculate the portfolio value and write the real allocation of each asset to the output array. The best in this pipeline is that it closely mimics how preset simulation methods work in vectorbt, and it's one of the most flexible pieces of code you can actually write!

Let's allocate 70% to BTC and 30% to ETH, and rebalance on a monthly basis:

```python
>>> mult_target_pct = mult_symbol_wrapper.fill()
>>> mult_target_pct.vbt.set([[0.7, 0.3]], every="M", inplace=True)
>>> grouper = vbt.Grouper(mult_symbol_wrapper.columns, group_by=True)
>>> group_lens = grouper.get_group_lens()

>>> target_shape = vbt.broadcast_shapes(
...     mult_data.get("Open").values.shape,
...     mult_data.get("Close").values.shape,
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 2)

>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values,
...     mult_target_pct.values
... )
>>> alloc = mult_symbol_wrapper.wrap(alloc)
>>> alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

```python
>>> mult_target_pct = mult_symbol_wrapper.fill()
>>> mult_target_pct.vbt.set([[0.7, 0.3]], every="M", inplace=True)
>>> grouper = vbt.Grouper(mult_symbol_wrapper.columns, group_by=True)
>>> group_lens = grouper.get_group_lens()

>>> target_shape = vbt.broadcast_shapes(
...     mult_data.get("Open").values.shape,
...     mult_data.get("Close").values.shape,
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 2)

>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values,
...     mult_target_pct.values
... )
>>> alloc = mult_symbol_wrapper.wrap(alloc)
>>> alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

Info

As you might have noticed, some allocations do not quite sum to 100%. This is because we used the open price for group valuation and decision-making, while the actual orders were executed using the close price. By the way, it's a bad sign when everything aligns perfectly - this could mean that your simulation is too ideal for the real world.

And here's the same procedure but without sorting the call sequence array:

```python
>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values,
...     mult_target_pct.values,
...     auto_call_seq=False
... )
>>> alloc = mult_symbol_wrapper.wrap(alloc)
>>> alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

```python
>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values,
...     mult_target_pct.values,
...     auto_call_seq=False
... )
>>> alloc = mult_symbol_wrapper.wrap(alloc)
>>> alloc.vbt.plot(
...    trace_kwargs=dict(stackgroup="one"),
...    use_gl=False
... ).show()
```

As we see, some rebalancing steps couldn't be completed at all because long operations were executed before short operations, leaving them without the required funds.

The biggest advantage of this pipeline is in its flexibility: we can turn off grouping via group_by=False to run the entire logic per column (each group will contain only one column). We can also test multiple weight combinations via multiple groups, without having to tile the pricing data thanks to rotational indexing. This, for example, cannot be done even with Portfolio.from_orders

```python
group_by=False
```

```python
>>> groups = pd.Index([0, 0, 1, 1], name="group")
>>> target_alloc = pd.Index([0.7, 0.3, 0.5, 0.5], name="target_alloc")

>>> final_columns = vbt.stack_indexes((  # (1)!
...     groups,
...     target_alloc, 
...     vbt.tile_index(mult_symbol_wrapper.columns, 2)
... ))
>>> final_wrapper = mult_symbol_wrapper.replace(  # (2)!
...     columns=final_columns,
...     group_by="group"
... )
>>> mult_target_pct = final_wrapper.fill(group_by=False)  # (3)!
>>> mult_target_pct.vbt.set(target_alloc.values[None], every="M", inplace=True)
>>> group_lens = final_wrapper.grouper.get_group_lens()

>>> n_groups = final_wrapper.grouper.get_group_count()
>>> target_shape = vbt.broadcast_shapes(
...     vbt.tile_shape(mult_data.get("Open").values.shape, n_groups),  # (4)!
...     vbt.tile_shape(mult_data.get("Close").values.shape, n_groups), 
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 4)

>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     mult_target_pct.values,
...     rotate_cols=True
... )
>>> alloc = mult_target_pct.vbt.wrapper.wrap(alloc)
>>> alloc
group                                       0                   1          
target_alloc                    0.7       0.3       0.5       0.5
symbol                      BTC-USD   ETH-USD   BTC-USD   ETH-USD
Date                                                             
2017-11-09 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-10 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-11 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-12 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-13 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
...                             ...       ...       ...       ...
2021-12-27 00:00:00+00:00  0.703817  0.296183  0.504464  0.495536
2021-12-28 00:00:00+00:00  0.703452  0.296548  0.504026  0.495974
2021-12-29 00:00:00+00:00  0.708035  0.291965  0.509543  0.490457
2021-12-30 00:00:00+00:00  0.706467  0.293533  0.507650  0.492350
2021-12-31 00:00:00+00:00  0.704346  0.295654  0.505099  0.494901

[1514 rows x 4 columns]
```

```python
>>> groups = pd.Index([0, 0, 1, 1], name="group")
>>> target_alloc = pd.Index([0.7, 0.3, 0.5, 0.5], name="target_alloc")

>>> final_columns = vbt.stack_indexes((  # (1)!
...     groups,
...     target_alloc, 
...     vbt.tile_index(mult_symbol_wrapper.columns, 2)
... ))
>>> final_wrapper = mult_symbol_wrapper.replace(  # (2)!
...     columns=final_columns,
...     group_by="group"
... )
>>> mult_target_pct = final_wrapper.fill(group_by=False)  # (3)!
>>> mult_target_pct.vbt.set(target_alloc.values[None], every="M", inplace=True)
>>> group_lens = final_wrapper.grouper.get_group_lens()

>>> n_groups = final_wrapper.grouper.get_group_count()
>>> target_shape = vbt.broadcast_shapes(
...     vbt.tile_shape(mult_data.get("Open").values.shape, n_groups),  # (4)!
...     vbt.tile_shape(mult_data.get("Close").values.shape, n_groups), 
...     mult_target_pct.values.shape
... )
>>> target_shape
(1514, 4)

>>> alloc = pipeline_5_nb(
...     target_shape,
...     group_lens,
...     mult_data.get("Open").values,
...     mult_data.get("Close").values, 
...     mult_target_pct.values,
...     rotate_cols=True
... )
>>> alloc = mult_target_pct.vbt.wrapper.wrap(alloc)
>>> alloc
group                                       0                   1          
target_alloc                    0.7       0.3       0.5       0.5
symbol                      BTC-USD   ETH-USD   BTC-USD   ETH-USD
Date                                                             
2017-11-09 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-10 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-11 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-12 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
2017-11-13 00:00:00+00:00  0.000000  0.000000  0.000000  0.000000
...                             ...       ...       ...       ...
2021-12-27 00:00:00+00:00  0.703817  0.296183  0.504464  0.495536
2021-12-28 00:00:00+00:00  0.703452  0.296548  0.504026  0.495974
2021-12-29 00:00:00+00:00  0.708035  0.291965  0.509543  0.490457
2021-12-30 00:00:00+00:00  0.706467  0.293533  0.507650  0.492350
2021-12-31 00:00:00+00:00  0.704346  0.295654  0.505099  0.494901

[1514 rows x 4 columns]
```

```python
open
```

```python
close
```

## Contexts¶

Sometimes, there is a need to create a simulation method that takes a user-defined function and calls it to make some trading decision. Such a UDF would require access to the simulation's state (such as the current position size and direction) and other information, which could quickly involve dozens of variables. Remember that we cannot do full-scale OOP in Numba, thus we have to pass data using primitive containers such as tuples. But usage of variable positional arguments or a regular tuple would be quite cumbersome for the user because accessing each field can only be done using an integer index or tuple unpacking. To ease this burden, we usually pass such information in form of a named tuple, often referred to as a (simulation) "context".

### Pipeline/6¶

Let's create a very basic pipeline that iterates over rows and columns, and, at each element, calls a UDF to get an order and execute it!

First, we need to answer the following question: "What information would a UDF need?" Mostly, we just include everything we have:

```python
>>> SimContext = namedtuple("SimContext", [
...     "open",  # (1)!
...     "high",
...     "low",
...     "close",
...     "init_cash",
...     "col",  # (2)!
...     "i",
...     "price_area",  # (3)!
...     "exec_state"
... ])
```

```python
>>> SimContext = namedtuple("SimContext", [
...     "open",  # (1)!
...     "high",
...     "low",
...     "close",
...     "init_cash",
...     "col",  # (2)!
...     "i",
...     "price_area",  # (3)!
...     "exec_state"
... ])
```

And here's our pipeline that takes and calls an order function:

```python
>>> @njit
... def pipeline_6_nb(
...     open, high, low, close, 
...     order_func_nb, order_args=(), 
...     init_cash=100
... ):
...     order_records = np.empty(close.shape, dtype=vbt.pf_enums.order_dt)
...     order_counts = np.full(close.shape[1], 0, dtype=int_)
...
...     for col in range(close.shape[1]):
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             val_price = open[i, col]
...             value = exec_state.cash + val_price * exec_state.position
...
...             price_area = vbt.pf_enums.PriceArea(
...                 open[i, col],
...                 high[i, col],
...                 low[i, col],
...                 close[i, col]
...             )
...             exec_state = vbt.pf_enums.ExecState(
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 locked_cash=exec_state.locked_cash,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             sim_ctx = SimContext(  # (1)!
...                 open=open,
...                 high=high,
...                 low=low,
...                 close=close,
...                 init_cash=init_cash,
...                 col=col,
...                 i=i,
...                 price_area=price_area,
...                 exec_state=exec_state
...             )
...             order = order_func_nb(sim_ctx, *order_args)  # (2)!
...             _, exec_state = vbt.pf_nb.process_order_nb(
...                 col, col, i,
...                 exec_state=exec_state,
...                 order=order,
...                 price_area=price_area,
...                 order_records=order_records,
...                 order_counts=order_counts
...             )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

```python
>>> @njit
... def pipeline_6_nb(
...     open, high, low, close, 
...     order_func_nb, order_args=(), 
...     init_cash=100
... ):
...     order_records = np.empty(close.shape, dtype=vbt.pf_enums.order_dt)
...     order_counts = np.full(close.shape[1], 0, dtype=int_)
...
...     for col in range(close.shape[1]):
...         exec_state = vbt.pf_enums.ExecState(
...             cash=float(init_cash),
...             position=0.0,
...             debt=0.0,
...             locked_cash=0.0,
...             free_cash=float(init_cash),
...             val_price=np.nan,
...             value=np.nan
...         )
...
...         for i in range(close.shape[0]):
...             val_price = open[i, col]
...             value = exec_state.cash + val_price * exec_state.position
...
...             price_area = vbt.pf_enums.PriceArea(
...                 open[i, col],
...                 high[i, col],
...                 low[i, col],
...                 close[i, col]
...             )
...             exec_state = vbt.pf_enums.ExecState(
...                 cash=exec_state.cash,
...                 position=exec_state.position,
...                 debt=exec_state.debt,
...                 locked_cash=exec_state.locked_cash,
...                 free_cash=exec_state.free_cash,
...                 val_price=val_price,
...                 value=value
...             )
...             sim_ctx = SimContext(  # (1)!
...                 open=open,
...                 high=high,
...                 low=low,
...                 close=close,
...                 init_cash=init_cash,
...                 col=col,
...                 i=i,
...                 price_area=price_area,
...                 exec_state=exec_state
...             )
...             order = order_func_nb(sim_ctx, *order_args)  # (2)!
...             _, exec_state = vbt.pf_nb.process_order_nb(
...                 col, col, i,
...                 exec_state=exec_state,
...                 order=order,
...                 price_area=price_area,
...                 order_records=order_records,
...                 order_counts=order_counts
...             )
...         
...     return vbt.nb.repartition_nb(order_records, order_counts)
```

Let's write our own order function that generates orders based on signals:

```python
>>> @njit  # (1)!
... def signal_order_func_nb(c, entries, exits):
...     if entries[c.i, c.col] and c.exec_state.position == 0:  # (2)!
...         return vbt.pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return vbt.pf_nb.close_position_nb()
...     return vbt.pf_nb.order_nothing_nb()

>>> broadcasted_args = vbt.broadcast(  # (3)!
...     dict(
...         open=data.get("Open").values,
...         high=data.get("High").values,
...         low=data.get("Low").values,
...         close=data.get("Close").values,
...         entries=entries.values,  # (4)!
...         exits=exits.values
...     ),
...     min_ndim=2
... )

>>> pipeline_6_nb(
...     broadcasted_args["open"],
...     broadcasted_args["high"], 
...     broadcasted_args["low"], 
...     broadcasted_args["close"], 
...     signal_order_func_nb,
...     order_args=(
...         broadcasted_args["entries"],
...         broadcasted_args["exits"]
...     )
... )
array([( 0, 0,  300, 0.34786966,   287.46398926, 0., 0),
       ( 1, 0,  362, 0.34786966,   230.64399719, 0., 1),
       ( 2, 0,  406, 0.26339233,   304.61801147, 0., 0),
       ( 3, 0, 1290, 0.26339233,  6890.52001953, 0., 1),
       ( 4, 0, 1680, 0.33210511,  5464.86669922, 0., 0),
       ( 5, 0, 1865, 0.33210511,  9244.97265625, 0., 1),
       ( 6, 0, 1981, 0.31871477,  9633.38671875, 0., 0),
       ( 7, 0, 2016, 0.31871477,  6681.06298828, 0., 1),
       ( 8, 0, 2073, 0.2344648 ,  9081.76171875, 0., 0),
       ( 9, 0, 2467, 0.2344648 , 35615.87109375, 0., 1),
       (10, 0, 2555, 0.17333543, 48176.34765625, 0., 0)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

```python
>>> @njit  # (1)!
... def signal_order_func_nb(c, entries, exits):
...     if entries[c.i, c.col] and c.exec_state.position == 0:  # (2)!
...         return vbt.pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return vbt.pf_nb.close_position_nb()
...     return vbt.pf_nb.order_nothing_nb()

>>> broadcasted_args = vbt.broadcast(  # (3)!
...     dict(
...         open=data.get("Open").values,
...         high=data.get("High").values,
...         low=data.get("Low").values,
...         close=data.get("Close").values,
...         entries=entries.values,  # (4)!
...         exits=exits.values
...     ),
...     min_ndim=2
... )

>>> pipeline_6_nb(
...     broadcasted_args["open"],
...     broadcasted_args["high"], 
...     broadcasted_args["low"], 
...     broadcasted_args["close"], 
...     signal_order_func_nb,
...     order_args=(
...         broadcasted_args["entries"],
...         broadcasted_args["exits"]
...     )
... )
array([( 0, 0,  300, 0.34786966,   287.46398926, 0., 0),
       ( 1, 0,  362, 0.34786966,   230.64399719, 0., 1),
       ( 2, 0,  406, 0.26339233,   304.61801147, 0., 0),
       ( 3, 0, 1290, 0.26339233,  6890.52001953, 0., 1),
       ( 4, 0, 1680, 0.33210511,  5464.86669922, 0., 0),
       ( 5, 0, 1865, 0.33210511,  9244.97265625, 0., 1),
       ( 6, 0, 1981, 0.31871477,  9633.38671875, 0., 0),
       ( 7, 0, 2016, 0.31871477,  6681.06298828, 0., 1),
       ( 8, 0, 2073, 0.2344648 ,  9081.76171875, 0., 0),
       ( 9, 0, 2467, 0.2344648 , 35615.87109375, 0., 1),
       (10, 0, 2555, 0.17333543, 48176.34765625, 0., 0)],
      dtype={'names':['id','col','idx','size','price','fees','side'], ...})
```

```python
@njit
```

We just created our own shallow Portfolio.from_order_func functionality, neat!

Your homework is to extend this pipeline with flexible indexing

## Performance¶

In terms of performance, Numba code is often a roller coaster

Numba is a just-in-time (JIT) compiler that analyzes and optimizes code, and finally uses the LLVM compiler library to generate a machine code version of a Python function to be compiled. But sometimes, even if the function looks efficient on paper, Numba may generate a suboptimal machine code because of some variables or their types not interacting optimally. In such a case, the code may still run very fast compared to a similar implementation with Python or even to another JIT compiler, but there is a lot of space for improvement that may be hard to discover, even for experienced users. There are even cases where switching the lines in which variables are defined suddenly and unexpectedly has a negative/positive effect on performance.

Apart from official tips, there are some of the best practices you should always keep in mind when designing and optimizing Numba-compiled functions:

```python
np.cumsum
```

```python
-
```

```python
+
```

```python
*
```

```python
/
```

```python
args
```

```python
*args
```

```python
fastmath=True
```

Hint

As a rule of thumb: the simpler is the code, the easier it becomes for Numba to analyze and optimize it.

### Benchmarking¶

To benchmark a simulator, we can use the timeit module. If possible, create some sample data of a sufficient size, and prepare for the worst-case scenario where orders are issued and executed at each single time step to benchmark the full load. Also, make sure to run tests all the way during the simulator's development to track the evolution of its execution time and stability.

Note

Generation of sample data and preparation of other inputs must be done prior to benchmarking.

Let's generate 1-minute random OHLC data for one year using RandomOHLCData:

```python
>>> test_data = vbt.RandomOHLCData.pull(
...     start="2020-01-01", 
...     end="2021-01-01",
...     timeframe="1min",  # (1)!
...     std=0.0001,  # (2)!
...     symmetric=True  # (3)!
... )
>>> test_data.resample("1d").plot().show()  # (4)!
```

```python
>>> test_data = vbt.RandomOHLCData.pull(
...     start="2020-01-01", 
...     end="2021-01-01",
...     timeframe="1min",  # (1)!
...     std=0.0001,  # (2)!
...     symmetric=True  # (3)!
... )
>>> test_data.resample("1d").plot().show()  # (4)!
```

Then, we need to prepare all the data, which includes filling signals such that there is at least one order at each bar (our worst-case scenario for performance and memory):

```python
>>> test_open = test_data.get("Open").values[:, None]  # (2)!
>>> test_high = test_data.get("High").values[:, None]
>>> test_low = test_data.get("Low").values[:, None]
>>> test_close = test_data.get("Close").values[:, None]
>>> test_entries = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_exits = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_entries[0::2] = True  # (3)!
>>> test_exits[1::2] = True  # (4)!
>>> del test_data  # (5)!

>>> test_entries.shape
(527041, 1)
```

```python
>>> test_open = test_data.get("Open").values[:, None]  # (2)!
>>> test_high = test_data.get("High").values[:, None]
>>> test_low = test_data.get("Low").values[:, None]
>>> test_close = test_data.get("Close").values[:, None]
>>> test_entries = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_exits = np.full(test_data.get_symbol_wrapper().shape, False)[:, None]
>>> test_entries[0::2] = True  # (3)!
>>> test_exits[1::2] = True  # (4)!
>>> del test_data  # (5)!

>>> test_entries.shape
(527041, 1)
```

Each of the arrays is 527,041 data points long.

So, how is our simulator performing on this data?

```python
>>> %%timeit  # (1)!
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
79.4 ms ± 290 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
>>> %%timeit  # (1)!
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
79.4 ms ± 290 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
timeit
```

80 milliseconds to generate half a million orders on Apple M1, not bad!

To better illustrate how only a minor change can impact performance, we will create a new order function that also creates a zero-sized empty array:

```python
>>> @njit
... def subopt_signal_order_func_nb(c, entries, exits):
...     _ = np.empty(0)  # (1)!
...
...     if entries[c.i, c.col] and c.exec_state.position == 0:
...         return vbt.pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return vbt.pf_nb.close_position_nb()
...     return vbt.pf_nb.order_nothing_nb()

>>> %%timeit
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     subopt_signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
130 ms ± 675 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
>>> @njit
... def subopt_signal_order_func_nb(c, entries, exits):
...     _ = np.empty(0)  # (1)!
...
...     if entries[c.i, c.col] and c.exec_state.position == 0:
...         return vbt.pf_nb.order_nb()
...     if exits[c.i, c.col] and c.exec_state.position > 0:
...         return vbt.pf_nb.close_position_nb()
...     return vbt.pf_nb.order_nothing_nb()

>>> %%timeit
>>> pipeline_6_nb(
...     test_open, 
...     test_high, 
...     test_low, 
...     test_close, 
...     subopt_signal_order_func_nb,
...     order_args=(
...         test_entries, 
...         test_exits
...     )
... )
130 ms ± 675 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

As we see, creating an empty array at each bar has slowed down the execution by more than 50%. And this is a very important lesson to learn: create arrays outside of loops and only once!

### Auto-parallelization¶

Because of path dependencies (= the current state depends on the previous one), we cannot parallelize the loop that iterates over rows (= time). But here's the deal: since vectorbt allows us to define a multi-columnar backtesting logic, we can parallelize the loop that iterates over columns or groups of columns, given that those columns or groups of columns are independent of each other - all using Numba alone. By the way, this is one of the primary reasons why vectorbt loves two-dimensional data layouts so much.

Automatic parallelization with Numba cannot be simpler: just replace range that you want to parallelize with numba.prange, and instruct Numba to parallelize the function by passing parallel=True to the @njit decorator. This will (try to) execute the code in the loop simultaneously by multiple parallel threads. You can read more about automatic parallelization with Numba here and about the available threading layers here. On MacBook Air (M1, 2020), turning on parallelization reduces the processing time by 2-3 times on average. Usually, a simple arithmetic-heavy code without creating any arrays can be better parallelized than a complex vectorization-heavy code.

```python
range
```

```python
numba.prange
```

```python
parallel=True
```

```python
@njit
```

Important

You can modify the same array from multiple threads, as done by countless functions in vectorbt. Just make sure that multiple threads (columns, in our case) aren't modifying the same elements and data in general!

Here's a small example of a function that computes the expanding maximum on two-dimensional data, without and with automatic parallelization:

```python
>>> arr = np.random.uniform(size=(1000000, 10))

>>> @njit
... def expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=float_)
...     for col in range(arr.shape[1]):
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit expanding_max_nb(arr)
40.7 ms ± 558 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> @njit(parallel=True)  # (1)!
... def parallel_expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=float_)
...     for col in prange(arr.shape[1]):  # (2)!
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit parallel_expanding_max_nb(arr)
26.6 ms ± 437 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

```python
>>> arr = np.random.uniform(size=(1000000, 10))

>>> @njit
... def expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=float_)
...     for col in range(arr.shape[1]):
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit expanding_max_nb(arr)
40.7 ms ± 558 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> @njit(parallel=True)  # (1)!
... def parallel_expanding_max_nb(arr):
...     out = np.empty_like(arr, dtype=float_)
...     for col in prange(arr.shape[1]):  # (2)!
...         maxv = -np.inf
...         for i in range(arr.shape[0]):
...             if arr[i, col] > maxv:
...                 maxv = arr[i, col]
...             out[i, col] = maxv
...     return out

>>> %timeit parallel_expanding_max_nb(arr)
26.6 ms ± 437 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

It's your turn: enable automatic parallelization of columns in the sixth pipeline and benchmark it! Just don't forget to reduce the number of rows and increase the number of columns.

### Caching¶

Even if we had optimized the simulation pipeline for the best-possible performance, the actual compilation step would take a huge chunk of that time savings away. However, the good news is that Numba doesn't have to re-compile the function the second time it's executed, given that we passed the same argument types (not data!). This means that we need to wait only once if we want to test the same function on many parameter combinations, at the same Python runtime. Sadly, if only one argument differs in type, or we've restarted the Python runtime, Numba has to compile again.

But luckily, Numba gives us a mechanism to avoid re-compilation even if we've restarted the runtime, called caching. To enable caching, just pass cache=True to the @njit decorator.

```python
cache=True
```

```python
@njit
```

Important

Avoid turning on caching for functions that take complex, user-defined data, such as (named) tuples and other functions. This may lead to some hidden bugs and kernel crashes if the data changes during the next runtime. Also make sure that your function doesn't use global variables. For example, the fifth pipeline is perfectly cacheable, while the sixth pipeline is not cacheable, or maybe could be if order_func_nb was cacheable as well.

```python
order_func_nb
```

Make sure to define any cached function inside a Python file rather than in a notebook cell since it must have a clear filepath such that it can be introspected by Numba. To invalidate the cache, go to the directory where the function resides and remove the __pycache__ directory. You can do that by running rm -rf __pycache__ from your terminal.

```python
__pycache__
```

```python
rm -rf __pycache__
```

Hint

A good practice is to invalidate the cache every time you change the code of a cached function to avoid potential side effects. Also, disable caching for the entire time of developing a function and turn it only on once the function has been fully implemented.

### AOT compilation¶

Using ahead-of-time compilation, we can compile a function only once and get no compilation overhead at runtime. Although this feature of Numba isn't widely used in vectorbt because it would restrict us from passing input data flexibly, we can make use of it in cases where we know the argument types in advance. Let's pre-compile our fifth pipeline!

For this, we have to specify the signature of a function explicitly. You can read more about it in the types reference.

```python
>>> from numba.pycc import CC
>>> cc = CC('pipeline_5')  # (1)!

>>> sig = "f8[:, :](" \ # (2)!
...       "UniTuple(i8, 2), " \ # (3)!
...       "i8[:], " \ # (4)!
...       "f8[:, :], " \ # (5)!
...       "f8[:, :], " \
...       "f8[:, :], " \
...       "f8[:], " \ # (6)!
...       "b1, " \ # (7)!
...       "b1" \
...       ")"

>>> cc.export('pipeline_5_nb', sig)(pipeline_5_nb)   # (8)!
>>> cc.compile() # (9)!
```

```python
>>> from numba.pycc import CC
>>> cc = CC('pipeline_5')  # (1)!

>>> sig = "f8[:, :](" \ # (2)!
...       "UniTuple(i8, 2), " \ # (3)!
...       "i8[:], " \ # (4)!
...       "f8[:, :], " \ # (5)!
...       "f8[:, :], " \
...       "f8[:, :], " \
...       "f8[:], " \ # (6)!
...       "b1, " \ # (7)!
...       "b1" \
...       ")"

>>> cc.export('pipeline_5_nb', sig)(pipeline_5_nb)   # (8)!
>>> cc.compile() # (9)!
```

```python
target_shape
```

```python
group_lens
```

```python
open
```

```python
close
```

```python
target_pct
```

```python
init_cash
```

```python
auto_call_seq
```

```python
rotate_cols
```

This has generated an extension module named pipeline_5. On macOS, the actual filename is pipeline_5.cpython-37m-darwin.so. We can then import the module like a regular Python module and run the function pipeline_5_nb of that module:

```python
pipeline_5
```

```python
pipeline_5.cpython-37m-darwin.so
```

```python
pipeline_5_nb
```

```python
>>> import pipeline_5

>>> mult_alloc = pipeline_5.pipeline_5_nb(
...     target_shape,
...     group_lens,
...     vbt.to_2d_array(mult_data.get("Open")).astype("f8", copy=False),  # (1)!
...     vbt.to_2d_array(mult_data.get("Close")).astype("f8", copy=False),
...     vbt.to_2d_array(mult_target_pct).astype("f8", copy=False),
...     vbt.to_1d_array(100).astype("f8", copy=False),
...     auto_call_seq=True,  # (2)!
...     rotate_cols=True
... )
```

```python
>>> import pipeline_5

>>> mult_alloc = pipeline_5.pipeline_5_nb(
...     target_shape,
...     group_lens,
...     vbt.to_2d_array(mult_data.get("Open")).astype("f8", copy=False),  # (1)!
...     vbt.to_2d_array(mult_data.get("Close")).astype("f8", copy=False),
...     vbt.to_2d_array(mult_target_pct).astype("f8", copy=False),
...     vbt.to_1d_array(100).astype("f8", copy=False),
...     auto_call_seq=True,  # (2)!
...     rotate_cols=True
... )
```

That was lightning fast!

Important

You should ensure that the provided arguments exactly match the registered signature, otherwise you may get errors that may be very difficult to debug. For example, while setting init_cash to 100 would yield an "index is out of bounds" error, casting the array to integer would make all allocations zero!

```python
init_cash
```

```python
100
```

## Summary¶

We've covered in detail a lot of components of a typical simulator in vectorbt. Simulation is the primary step in backtesting of a trading strategy, and by mastering it you'll gain some hard-core skills that can be applied just in any place of the vectorbt's rich Numba ecosystem.

One of the most important takeaways from this documentation piece is that implementing a custom simulator is as easy (or as difficult) as any other Numba-compiled function, and there is no point in using the preset simulation methods such as Portfolio.from_signals if you can produce the same results, achieve a multifold performance gain, be able to use rotational indexing, caching, and AOT compilation, by designing your own pipeline from scratch. After all, it's just a bunch of loops that gradually move over the shape of a matrix, execute orders, update the state of the simulation, and write some output data. Everything else is up to your imagination

Python code

