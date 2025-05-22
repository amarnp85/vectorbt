# Integrations¶

PortfolioOptimizer integrates nicely with various third-party libraries.

## PyPortfolioOpt¶

PyPortfolioOpt is a library that implements portfolio optimization methods, including classical efficient frontier techniques and Black-Litterman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.

PyPortfolioOpt implements a range of optimization methods that are very easy to use. The optimization procedure consists of several distinct steps (some of them may be skipped depending on the optimizer):

```python
pfopt.expected_returns
```

```python
pfopt.risk_models
```

```python
pypfopt.efficient_frontier
```

```python
pypfopt.base_optimizer
```

For example, let's perform the mean-variance optimization (MVO) for maximum Sharpe:

```python
>>> from pypfopt.expected_returns import mean_historical_return
>>> from pypfopt.risk_models import CovarianceShrinkage
>>> from pypfopt.efficient_frontier import EfficientFrontier

>>> expected_returns = mean_historical_return(data.get("Close"))
>>> cov_matrix = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> optimizer = EfficientFrontier(expected_returns, cov_matrix)
>>> weights = optimizer.max_sharpe()
>>> weights
OrderedDict([('ADAUSDT', 0.1166001117223088),
             ('BNBUSDT', 0.0),
             ('BTCUSDT', 0.0),
             ('ETHUSDT', 0.8833998882776911),
             ('XRPUSDT', 0.0)])
```

```python
>>> from pypfopt.expected_returns import mean_historical_return
>>> from pypfopt.risk_models import CovarianceShrinkage
>>> from pypfopt.efficient_frontier import EfficientFrontier

>>> expected_returns = mean_historical_return(data.get("Close"))
>>> cov_matrix = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> optimizer = EfficientFrontier(expected_returns, cov_matrix)
>>> weights = optimizer.max_sharpe()
>>> weights
OrderedDict([('ADAUSDT', 0.1166001117223088),
             ('BNBUSDT', 0.0),
             ('BTCUSDT', 0.0),
             ('ETHUSDT', 0.8833998882776911),
             ('XRPUSDT', 0.0)])
```

### Parsing¶

Thanks to an outstanding work done by @robertmartin8, the entire codebase of PyPortfolioOpt (with a few exceptions) has consistent argument and function namings, such that we can build a semantic web of functions acting as inputs to other functions. Why this is important? Because the user just needs to provide the target function (let's say, EfficientFrontier.max_sharpe), and we can programmatically figure out the entire call stack having the pricing data alone! And if the user passes any additional keyword arguments, we can check which functions from the stack accept those arguments and automatically pass them.

```python
EfficientFrontier.max_sharpe
```

For the example above, the web looks like this:

```python
flowchart TD;
    id1["User"]
    id2["mean_historical_return"]
    id3["CovarianceShrinkage.ledoit_wolf"]
    id4["EfficientFrontier"]

    id1 -->|"prices"| id2;
    id1 -->|"prices"| id3;
    id2 -->|"expected_returns"| id4;
    id3 -->|"cov_matrix"| id4;
```

```python
flowchart TD;
    id1["User"]
    id2["mean_historical_return"]
    id3["CovarianceShrinkage.ledoit_wolf"]
    id4["EfficientFrontier"]

    id1 -->|"prices"| id2;
    id1 -->|"prices"| id3;
    id2 -->|"expected_returns"| id4;
    id3 -->|"cov_matrix"| id4;
```

(Reload the page if the diagram doesn't show up)

And here comes vectorbt into play. First, it implements a function resolve_pypfopt_func_kwargs that takes an arbitrary PyPortfolioOpt function, and resolves its arguments. Whenever an argument passed by the user has been matched with an argument in the function's signature, it marks this argument to be passed to the function. Let's try it out on expected returns:

```python
>>> from vectorbtpro.portfolio.pfopt.base import resolve_pypfopt_func_kwargs

>>> vbt.phelp(mean_historical_return)  # (1)!
mean_historical_return(
    prices,
    returns_data=False,
    compounding=True,
    frequency=252,
    log_returns=False
):
    Calculate annualised mean (daily) historical return from input (daily) asset prices.
    Use ``compounding`` to toggle between the default geometric mean (CAGR) and the
    arithmetic mean.

>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     mean_historical_return, 
...     prices=data.get("Close"),  # (2)!
...     freq="1h",  # (3)!
...     year_freq="365d",
...     other_arg=100  # (4)!
... )))
{
    'prices': <pandas.core.frame.DataFrame object at 0x7f9428052c50 of shape (8767, 5)>,
    'returns_data': False,
    'compounding': True,
    'frequency': 8760.0,
    'log_returns': False
}
```

```python
>>> from vectorbtpro.portfolio.pfopt.base import resolve_pypfopt_func_kwargs

>>> vbt.phelp(mean_historical_return)  # (1)!
mean_historical_return(
    prices,
    returns_data=False,
    compounding=True,
    frequency=252,
    log_returns=False
):
    Calculate annualised mean (daily) historical return from input (daily) asset prices.
    Use ``compounding`` to toggle between the default geometric mean (CAGR) and the
    arithmetic mean.

>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     mean_historical_return, 
...     prices=data.get("Close"),  # (2)!
...     freq="1h",  # (3)!
...     year_freq="365d",
...     other_arg=100  # (4)!
... )))
{
    'prices': <pandas.core.frame.DataFrame object at 0x7f9428052c50 of shape (8767, 5)>,
    'returns_data': False,
    'compounding': True,
    'frequency': 8760.0,
    'log_returns': False
}
```

```python
freq
```

```python
year_freq
```

```python
frequency
```

And now let's run it on EfficientFrontier:

```python
EfficientFrontier
```

```python
>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     EfficientFrontier, 
...     prices=data.get("Close")
... )))
{
    'expected_returns': <pandas.core.series.Series object at 0x7f9479927128 of shape (5,)>,
    'cov_matrix': <pandas.core.frame.DataFrame object at 0x7f94280528d0 of shape (5, 5)>,
    'weight_bounds': (
        0,
        1
    ),
    'solver': None,
    'verbose': False,
    'solver_options': None
}
```

```python
>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     EfficientFrontier, 
...     prices=data.get("Close")
... )))
{
    'expected_returns': <pandas.core.series.Series object at 0x7f9479927128 of shape (5,)>,
    'cov_matrix': <pandas.core.frame.DataFrame object at 0x7f94280528d0 of shape (5, 5)>,
    'weight_bounds': (
        0,
        1
    ),
    'solver': None,
    'verbose': False,
    'solver_options': None
}
```

We see that vectorbt magically resolved arguments expected_returns and cov_matrix using resolve_pypfopt_expected_returns and resolve_pypfopt_cov_matrix respectively. If we provided those two arguments manually, vectorbt would use them right away. We can also provide those arguments as strings to change the function with which they are generated:

```python
expected_returns
```

```python
cov_matrix
```

```python
>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     EfficientFrontier, 
...     prices=data.get("Close"),
...     expected_returns="ema_historical_return",
...     cov_matrix="sample_cov"
... )))
{
    'expected_returns': <pandas.core.series.Series object at 0x7f9428044cf8 of shape (5,)>,
    'cov_matrix': <pandas.core.frame.DataFrame object at 0x7f942805bf60 of shape (5, 5)>,
    'weight_bounds': (
        0,
        1
    ),
    'solver': None,
    'verbose': False,
    'solver_options': None
}
```

```python
>>> print(vbt.prettify(resolve_pypfopt_func_kwargs(
...     EfficientFrontier, 
...     prices=data.get("Close"),
...     expected_returns="ema_historical_return",
...     cov_matrix="sample_cov"
... )))
{
    'expected_returns': <pandas.core.series.Series object at 0x7f9428044cf8 of shape (5,)>,
    'cov_matrix': <pandas.core.frame.DataFrame object at 0x7f942805bf60 of shape (5, 5)>,
    'weight_bounds': (
        0,
        1
    ),
    'solver': None,
    'verbose': False,
    'solver_options': None
}
```

### Auto-optimization¶

Knowing how to parse and resolve function arguments, vectorbt implements a function pypfopt_optimize, which takes user requirements and translates them into function calls. The usage of this function cannot be easier!

```python
>>> vbt.pypfopt_optimize(prices=data.get("Close"))
{'ADAUSDT': 0.1166,
 'BNBUSDT': 0.0,
 'BTCUSDT': 0.0,
 'ETHUSDT': 0.8834,
 'XRPUSDT': 0.0}
```

```python
>>> vbt.pypfopt_optimize(prices=data.get("Close"))
{'ADAUSDT': 0.1166,
 'BNBUSDT': 0.0,
 'BTCUSDT': 0.0,
 'ETHUSDT': 0.8834,
 'XRPUSDT': 0.0}
```

In short, pypfopt_optimize first resolves the optimizer using resolve_pypfopt_optimizer, which, in turn, triggers a waterfall of argument resolutions by parsing arguments, including the calculation of the expected returns and the risk model quantifying asset risk. Then, it adds objectives and constraints to the optimizer instance. Finally, it calls the target metric method (such as max_sharpe) or custom convex/non-convex objective using the same parsing procedure as we did above. If wanted, it can also translate continuous weights into discrete ones using DiscreteAllocation.

```python
max_sharpe
```

```python
DiscreteAllocation
```

Since multiple PyPortfolioOpt functions can require the same argument that has to be pre-computed yet, pypfopt_optimize deploys a built-in caching mechanism. Additionally, if any of the arguments weren't used, it throws a warning (which can be mitigated by setting silence_warnings to True) stating that an argument wasn't required by any function in the call stack.

```python
silence_warnings
```

Below, we will demonstrate various optimizations done both using PyPortfolioOpt and vectorbt. Optimizing a long/short portfolio to minimise total variance:

```python
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(None, S, weight_bounds=(-1, 1))
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', -0.01118),
             ('BNBUSDT', 0.09695),
             ('BTCUSDT', 0.9624),
             ('ETHUSDT', -0.10516),
             ('XRPUSDT', 0.05699)])
```

```python
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(None, S, weight_bounds=(-1, 1))
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', -0.01118),
             ('BNBUSDT', 0.09695),
             ('BTCUSDT', 0.9624),
             ('ETHUSDT', -0.10516),
             ('XRPUSDT', 0.05699)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns=None,
...     weight_bounds=(-1, 1),
...     target="min_volatility"
... )
{'ADAUSDT': -0.01118,
 'BNBUSDT': 0.09695,
 'BTCUSDT': 0.9624,
 'ETHUSDT': -0.10516,
 'XRPUSDT': 0.05699}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns=None,
...     weight_bounds=(-1, 1),
...     target="min_volatility"
... )
{'ADAUSDT': -0.01118,
 'BNBUSDT': 0.09695,
 'BTCUSDT': 0.9624,
 'ETHUSDT': -0.10516,
 'XRPUSDT': 0.05699}
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

Optimizing a portfolio to maximise the Sharpe ratio, subject to direction constraints:

```python
directions = ["long", "long", "long", "short", "short"]
direction_mapper = dict(zip(data.symbols, directions))
```

```python
directions = ["long", "long", "long", "short", "short"]
direction_mapper = dict(zip(data.symbols, directions))
```

```python
>>> from pypfopt.expected_returns import capm_return

>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
>>> for symbol, direction in direction_mapper.items():
...     idx = data.symbols.index(symbol)
...     if direction == "long":
...         ef.add_constraint(lambda w, _idx=idx: w[_idx] >= 0)
...     if direction == "short":
...         ef.add_constraint(lambda w, _idx=idx: w[_idx] <= 0)
>>> ef.max_sharpe()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('BTCUSDT', 0.26614),
             ('ETHUSDT', 0.433),
             ('BNBUSDT', 0.30086),
             ('XRPUSDT', 0.0),
             ('ADAUSDT', 0.0)])
```

```python
>>> from pypfopt.expected_returns import capm_return

>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
>>> for symbol, direction in direction_mapper.items():
...     idx = data.symbols.index(symbol)
...     if direction == "long":
...         ef.add_constraint(lambda w, _idx=idx: w[_idx] >= 0)
...     if direction == "short":
...         ef.add_constraint(lambda w, _idx=idx: w[_idx] <= 0)
>>> ef.max_sharpe()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('BTCUSDT', 0.26614),
             ('ETHUSDT', 0.433),
             ('BNBUSDT', 0.30086),
             ('XRPUSDT', 0.0),
             ('ADAUSDT', 0.0)])
```

```python
>>> constraints = []
>>> for symbol, direction in direction_mapper.items():
...     idx = data.symbols.index(symbol)
...     if direction == "long":
...         constraints.append(lambda w, _idx=idx: w[_idx] >= 0)
...     if direction == "short":
...         constraints.append(lambda w, _idx=idx: w[_idx] <= 0)
>>> vbt.pypfopt_optimize(  
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     cov_matrix="ledoit_wolf",
...     target="max_sharpe",
...     weight_bounds=(-1, 1),
...     constraints=constraints,
... )
{'BTCUSDT': 0.26614,
 'ETHUSDT': 0.433,
 'BNBUSDT': 0.30086,
 'XRPUSDT': 0.0,
 'ADAUSDT': 0.0}
```

```python
>>> constraints = []
>>> for symbol, direction in direction_mapper.items():
...     idx = data.symbols.index(symbol)
...     if direction == "long":
...         constraints.append(lambda w, _idx=idx: w[_idx] >= 0)
...     if direction == "short":
...         constraints.append(lambda w, _idx=idx: w[_idx] <= 0)
>>> vbt.pypfopt_optimize(  
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     cov_matrix="ledoit_wolf",
...     target="max_sharpe",
...     weight_bounds=(-1, 1),
...     constraints=constraints,
... )
{'BTCUSDT': 0.26614,
 'ETHUSDT': 0.433,
 'BNBUSDT': 0.30086,
 'XRPUSDT': 0.0,
 'ADAUSDT': 0.0}
```

Optimizing a portfolio to maximise the Sharpe ratio, subject to sector constraints:

```python
>>> sector_mapper = {
...     "ADAUSDT": "DeFi",
...     "BNBUSDT": "DeFi",
...     "BTCUSDT": "Payment",
...     "ETHUSDT": "DeFi",
...     "XRPUSDT": "Payment"
... }
>>> sector_lower = {
...     "DeFi": 0.75
... }
>>> sector_upper = {}
```

```python
>>> sector_mapper = {
...     "ADAUSDT": "DeFi",
...     "BNBUSDT": "DeFi",
...     "BTCUSDT": "Payment",
...     "ETHUSDT": "DeFi",
...     "XRPUSDT": "Payment"
... }
>>> sector_lower = {
...     "DeFi": 0.75
... }
>>> sector_upper = {}
```

```python
>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
>>> adausdt_index = ef.tickers.index("ADAUSDT")
>>> ef.add_constraint(lambda w: w[adausdt_index] == 0.10)
>>> ef.max_sharpe()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.1),
             ('BNBUSDT', 0.2772),
             ('BTCUSDT', 0.0524),
             ('ETHUSDT', 0.3728),
             ('XRPUSDT', 0.1976)])
```

```python
>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
>>> adausdt_index = ef.tickers.index("ADAUSDT")
>>> ef.add_constraint(lambda w: w[adausdt_index] == 0.10)
>>> ef.max_sharpe()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.1),
             ('BNBUSDT', 0.2772),
             ('BTCUSDT', 0.0524),
             ('ETHUSDT', 0.3728),
             ('XRPUSDT', 0.1976)])
```

```python
>>> adausdt_index = list(sector_mapper.keys()).index("ADAUSDT")
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     sector_mapper=sector_mapper,
...     sector_lower=sector_lower,
...     sector_upper=sector_upper,
...     constraints=[lambda w: w[adausdt_index] == 0.10]
... )
{'ADAUSDT': 0.1,
 'BNBUSDT': 0.2772,
 'BTCUSDT': 0.0524,
 'ETHUSDT': 0.3728,
 'XRPUSDT': 0.1976}
```

```python
>>> adausdt_index = list(sector_mapper.keys()).index("ADAUSDT")
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     sector_mapper=sector_mapper,
...     sector_lower=sector_lower,
...     sector_upper=sector_upper,
...     constraints=[lambda w: w[adausdt_index] == 0.10]
... )
{'ADAUSDT': 0.1,
 'BNBUSDT': 0.2772,
 'BTCUSDT': 0.0524,
 'ETHUSDT': 0.3728,
 'XRPUSDT': 0.1976}
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

```python
EfficientFrontier.max_sharpe
```

Optimizing a portfolio to maximise return for a given risk, subject to sector constraints, with an L2 regularisation objective:

```python
>>> from pypfopt.objective_functions import L2_reg

>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
>>> ef.add_objective(L2_reg, gamma=0.1)
>>> ef.efficient_risk(0.15)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.26004),
             ('BNBUSDT', 0.24466),
             ('BTCUSDT', 0.10778),
             ('ETHUSDT', 0.2453),
             ('XRPUSDT', 0.14222)])
```

```python
>>> from pypfopt.objective_functions import L2_reg

>>> mu = capm_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
>>> ef.add_objective(L2_reg, gamma=0.1)
>>> ef.efficient_risk(0.15)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.26004),
             ('BNBUSDT', 0.24466),
             ('BTCUSDT', 0.10778),
             ('ETHUSDT', 0.2453),
             ('XRPUSDT', 0.14222)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     sector_mapper=sector_mapper,
...     sector_lower=sector_lower,
...     sector_upper=sector_upper,
...     objectives=["L2_reg"],  # (2)!
...     gamma=0.1,  # (3)!
...     target="efficient_risk",
...     target_volatility=0.15  # (4)!
... )
{'ADAUSDT': 0.26004,
 'BNBUSDT': 0.24466,
 'BTCUSDT': 0.10778,
 'ETHUSDT': 0.2453,
 'XRPUSDT': 0.14222}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     sector_mapper=sector_mapper,
...     sector_lower=sector_lower,
...     sector_upper=sector_upper,
...     objectives=["L2_reg"],  # (2)!
...     gamma=0.1,  # (3)!
...     target="efficient_risk",
...     target_volatility=0.15  # (4)!
... )
{'ADAUSDT': 0.26004,
 'BNBUSDT': 0.24466,
 'BTCUSDT': 0.10778,
 'ETHUSDT': 0.2453,
 'XRPUSDT': 0.14222}
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

```python
pypfopt.objective_functions
```

Optimizing along the mean-semivariance frontier:

```python
>>> from pypfopt import EfficientSemivariance
>>> from pypfopt.expected_returns import returns_from_prices

>>> mu = capm_return(data.get("Close"))
>>> returns = returns_from_prices(data.get("Close"))
>>> returns = returns.dropna()
>>> es = EfficientSemivariance(mu, returns)
>>> es.efficient_return(0.01)
>>> weights = es.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.0),
             ('BNBUSDT', 0.0),
             ('BTCUSDT', 1.0),
             ('ETHUSDT', 0.0),
             ('XRPUSDT', 0.0)])
```

```python
>>> from pypfopt import EfficientSemivariance
>>> from pypfopt.expected_returns import returns_from_prices

>>> mu = capm_return(data.get("Close"))
>>> returns = returns_from_prices(data.get("Close"))
>>> returns = returns.dropna()
>>> es = EfficientSemivariance(mu, returns)
>>> es.efficient_return(0.01)
>>> weights = es.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.0),
             ('BNBUSDT', 0.0),
             ('BTCUSDT', 1.0),
             ('ETHUSDT', 0.0),
             ('XRPUSDT', 0.0)])
```

```python
>>> vbt.pypfopt_optimize(
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     optimizer="efficient_semivariance",  # (1)!
...     target="efficient_return",
...     target_return=0.01
... )
{'ADAUSDT': 0.0, 'BNBUSDT': 0.0, 'BTCUSDT': 1, 'ETHUSDT': 0.0, 'XRPUSDT': 0.0}
```

```python
>>> vbt.pypfopt_optimize(
...     prices=data.get("Close"),
...     expected_returns="capm_return",
...     optimizer="efficient_semivariance",  # (1)!
...     target="efficient_return",
...     target_return=0.01
... )
{'ADAUSDT': 0.0, 'BNBUSDT': 0.0, 'BTCUSDT': 1, 'ETHUSDT': 0.0, 'XRPUSDT': 0.0}
```

```python
EfficientSemivariance
```

```python
returns
```

Minimizing transaction costs:

```python
>>> initial_weights = np.array([1 / len(data.symbols)] * len(data.symbols))
```

```python
>>> initial_weights = np.array([1 / len(data.symbols)] * len(data.symbols))
```

```python
>>> from pypfopt.objective_functions import transaction_cost

>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_objective(transaction_cost, w_prev=initial_weights, k=0.001)
>>> ef.add_objective(L2_reg, gamma=0.05)
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.16025),
             ('BNBUSDT', 0.2),
             ('BTCUSDT', 0.27241),
             ('ETHUSDT', 0.2),
             ('XRPUSDT', 0.16734)])
```

```python
>>> from pypfopt.objective_functions import transaction_cost

>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.add_objective(transaction_cost, w_prev=initial_weights, k=0.001)
>>> ef.add_objective(L2_reg, gamma=0.05)
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.16025),
             ('BNBUSDT', 0.2),
             ('BTCUSDT', 0.27241),
             ('ETHUSDT', 0.2),
             ('XRPUSDT', 0.16734)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     objectives=["transaction_cost", "L2_reg"],
...     w_prev=initial_weights, 
...     k=0.001,
...     gamma=0.05,
...     target="min_volatility"
... )
{'ADAUSDT': 0.16025,
 'BNBUSDT': 0.2,
 'BTCUSDT': 0.27241,
 'ETHUSDT': 0.2,
 'XRPUSDT': 0.16734}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     objectives=["transaction_cost", "L2_reg"],
...     w_prev=initial_weights, 
...     k=0.001,
...     gamma=0.05,
...     target="min_volatility"
... )
{'ADAUSDT': 0.16025,
 'BNBUSDT': 0.2,
 'BTCUSDT': 0.27241,
 'ETHUSDT': 0.2,
 'XRPUSDT': 0.16734}
```

```python
mean_historical_return
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

Custom convex objective:

```python
>>> import cvxpy as cp

>>> def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
...     log_sum = cp.sum(cp.log(w))
...     var = cp.quad_form(w, cov_matrix)
...     return var - k * log_sum
```

```python
>>> import cvxpy as cp

>>> def logarithmic_barrier_objective(w, cov_matrix, k=0.1):
...     log_sum = cp.sum(cp.log(w))
...     var = cp.quad_form(w, cov_matrix)
...     return var - k * log_sum
```

```python
>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.3))
>>> ef.convex_objective(logarithmic_barrier_objective, cov_matrix=S, k=0.001)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.12214),
             ('BNBUSDT', 0.22175),
             ('BTCUSDT', 0.3),
             ('ETHUSDT', 0.21855),
             ('XRPUSDT', 0.13756)])
```

```python
>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 0.3))
>>> ef.convex_objective(logarithmic_barrier_objective, cov_matrix=S, k=0.001)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.12214),
             ('BNBUSDT', 0.22175),
             ('BTCUSDT', 0.3),
             ('ETHUSDT', 0.21855),
             ('XRPUSDT', 0.13756)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     weight_bounds=(0.01, 0.3),
...     k=0.001,
...     target=logarithmic_barrier_objective  # (2)!
... )
{'ADAUSDT': 0.12214,
 'BNBUSDT': 0.22175,
 'BTCUSDT': 0.3,
 'ETHUSDT': 0.21855,
 'XRPUSDT': 0.13756}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     weight_bounds=(0.01, 0.3),
...     k=0.001,
...     target=logarithmic_barrier_objective  # (2)!
... )
{'ADAUSDT': 0.12214,
 'BNBUSDT': 0.22175,
 'BTCUSDT': 0.3,
 'ETHUSDT': 0.21855,
 'XRPUSDT': 0.13756}
```

```python
mean_historical_return
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

```python
logarithmic_barrier_objective
```

```python
cov_matrix
```

Custom non-convex objective:

```python
>>> def deviation_risk_parity(w, cov_matrix):
...     cov_matrix = np.asarray(cov_matrix)
...     n = cov_matrix.shape[0]
...     rp = (w * (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
...     return cp.sum_squares(rp - 1 / n).value
```

```python
>>> def deviation_risk_parity(w, cov_matrix):
...     cov_matrix = np.asarray(cov_matrix)
...     n = cov_matrix.shape[0]
...     rp = (w * (cov_matrix @ w)) / cp.quad_form(w, cov_matrix)
...     return cp.sum_squares(rp - 1 / n).value
```

```python
>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.17421),
             ('BNBUSDT', 0.19933),
             ('BTCUSDT', 0.2515),
             ('ETHUSDT', 0.1981),
             ('XRPUSDT', 0.17686)])
```

```python
>>> mu = mean_historical_return(data.get("Close"))
>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> ef = EfficientFrontier(mu, S)
>>> ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.17421),
             ('BNBUSDT', 0.19933),
             ('BTCUSDT', 0.2515),
             ('ETHUSDT', 0.1981),
             ('XRPUSDT', 0.17686)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     target=deviation_risk_parity,  # (2)!
...     target_is_convex=False
... )
{'ADAUSDT': 0.17421,
 'BNBUSDT': 0.19933,
 'BTCUSDT': 0.2515,
 'ETHUSDT': 0.1981,
 'XRPUSDT': 0.17686}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     target=deviation_risk_parity,  # (2)!
...     target_is_convex=False
... )
{'ADAUSDT': 0.17421,
 'BNBUSDT': 0.19933,
 'BTCUSDT': 0.2515,
 'ETHUSDT': 0.1981,
 'XRPUSDT': 0.17686}
```

```python
mean_historical_return
```

```python
CovarianceShrinkage.ledoit_wolf
```

```python
EfficientFrontier
```

```python
deviation_risk_parity
```

```python
cov_matrix
```

Black-Litterman Allocation (read more):

```python
>>> sp500_data = vbt.YFData.pull(
...     "^GSPC", 
...     start=data.wrapper.index[0], 
...     end=data.wrapper.index[-1]
... )
>>> market_caps = data.get("Close") * data.get("Volume")
>>> viewdict = {
...     "ADAUSDT": 0.20, 
...     "BNBUSDT": -0.30, 
...     "BTCUSDT": 0, 
...     "ETHUSDT": -0.2, 
...     "XRPUSDT": 0.15
... }
```

```python
>>> sp500_data = vbt.YFData.pull(
...     "^GSPC", 
...     start=data.wrapper.index[0], 
...     end=data.wrapper.index[-1]
... )
>>> market_caps = data.get("Close") * data.get("Volume")
>>> viewdict = {
...     "ADAUSDT": 0.20, 
...     "BNBUSDT": -0.30, 
...     "BTCUSDT": 0, 
...     "ETHUSDT": -0.2, 
...     "XRPUSDT": 0.15
... }
```

```python
>>> from pypfopt.black_litterman import (
...     market_implied_risk_aversion,
...     market_implied_prior_returns,
...     BlackLittermanModel
... )

>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> delta = market_implied_risk_aversion(sp500_data.get("Close"))
>>> prior = market_implied_prior_returns(market_caps.iloc[-1], delta, S)
>>> bl = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
>>> rets = bl.bl_returns()
>>> ef = EfficientFrontier(rets, S)
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.0),
             ('BNBUSDT', 0.06743),
             ('BTCUSDT', 0.89462),
             ('ETHUSDT', 0.0),
             ('XRPUSDT', 0.03795)])
```

```python
>>> from pypfopt.black_litterman import (
...     market_implied_risk_aversion,
...     market_implied_prior_returns,
...     BlackLittermanModel
... )

>>> S = CovarianceShrinkage(data.get("Close")).ledoit_wolf()
>>> delta = market_implied_risk_aversion(sp500_data.get("Close"))
>>> prior = market_implied_prior_returns(market_caps.iloc[-1], delta, S)
>>> bl = BlackLittermanModel(S, pi=prior, absolute_views=viewdict)
>>> rets = bl.bl_returns()
>>> ef = EfficientFrontier(rets, S)
>>> ef.min_volatility()
>>> weights = ef.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.0),
             ('BNBUSDT', 0.06743),
             ('BTCUSDT', 0.89462),
             ('ETHUSDT', 0.0),
             ('XRPUSDT', 0.03795)])
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="bl_returns",  # (2)!
...     market_prices=sp500_data.get("Close"),
...     market_caps=market_caps.iloc[-1],
...     absolute_views=viewdict,
...     target="min_volatility"
... )
{'ADAUSDT': 0.0,
 'BNBUSDT': 0.06743,
 'BTCUSDT': 0.89462,
 'ETHUSDT': 0.0,
 'XRPUSDT': 0.03795}
```

```python
>>> vbt.pypfopt_optimize(  # (1)!
...     prices=data.get("Close"),
...     expected_returns="bl_returns",  # (2)!
...     market_prices=sp500_data.get("Close"),
...     market_caps=market_caps.iloc[-1],
...     absolute_views=viewdict,
...     target="min_volatility"
... )
{'ADAUSDT': 0.0,
 'BNBUSDT': 0.06743,
 'BTCUSDT': 0.89462,
 'ETHUSDT': 0.0,
 'XRPUSDT': 0.03795}
```

```python
mean_historical_return
```

```python
EfficientFrontier
```

```python
BlackLittermanModel
```

```python
delta
```

```python
prior
```

```python
bl_returns
```

Hierarchical Risk Parity (read more):

```python
>>> from pypfopt import HRPOpt

>>> rets = returns_from_prices(data.get("Close"))
>>> hrp = HRPOpt(rets)
>>> hrp.optimize()
>>> weights = hrp.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.19415),
             ('BNBUSDT', 0.24834),
             ('BTCUSDT', 0.22827),
             ('ETHUSDT', 0.15217),
             ('XRPUSDT', 0.17707)])
```

```python
>>> from pypfopt import HRPOpt

>>> rets = returns_from_prices(data.get("Close"))
>>> hrp = HRPOpt(rets)
>>> hrp.optimize()
>>> weights = hrp.clean_weights()
>>> weights
OrderedDict([('ADAUSDT', 0.19415),
             ('BNBUSDT', 0.24834),
             ('BTCUSDT', 0.22827),
             ('ETHUSDT', 0.15217),
             ('XRPUSDT', 0.17707)])
```

```python
>>> vbt.pypfopt_optimize(
...     prices=data.get("Close"),  # (1)!
...     optimizer="hrp",
...     target="optimize"
... )
{'ADAUSDT': 0.19415,
 'BNBUSDT': 0.24834,
 'BTCUSDT': 0.22827,
 'ETHUSDT': 0.15217,
 'XRPUSDT': 0.17707}
```

```python
>>> vbt.pypfopt_optimize(
...     prices=data.get("Close"),  # (1)!
...     optimizer="hrp",
...     target="optimize"
... )
{'ADAUSDT': 0.19415,
 'BNBUSDT': 0.24834,
 'BTCUSDT': 0.22827,
 'ETHUSDT': 0.15217,
 'XRPUSDT': 0.17707}
```

### Argument groups¶

In cases where two functions require an argument with the same name but you want to pass different values to them, pass the argument as an instance of pfopt_func_dict where keys should be functions or their names, and values should be different argument values:

```python
>>> vbt.pypfopt_optimize(  
...     prices=data.get("Close"),
...     expected_returns="bl_returns",  
...     market_prices=sp500_data.get("Close"),
...     market_caps=market_caps.iloc[-1],
...     absolute_views=viewdict,
...     target="min_volatility",
...     cov_matrix=vbt.pfopt_func_dict({
...         "EfficientFrontier": "sample_cov",  # (1)!
...         "_def": "ledoit_wolf"  # (2)!
...     })
... )
{'ADAUSDT': 0.0,
 'BNBUSDT': 0.05013,
 'BTCUSDT': 0.91912,
 'ETHUSDT': 0.0,
 'XRPUSDT': 0.03075}
```

```python
>>> vbt.pypfopt_optimize(  
...     prices=data.get("Close"),
...     expected_returns="bl_returns",  
...     market_prices=sp500_data.get("Close"),
...     market_caps=market_caps.iloc[-1],
...     absolute_views=viewdict,
...     target="min_volatility",
...     cov_matrix=vbt.pfopt_func_dict({
...         "EfficientFrontier": "sample_cov",  # (1)!
...         "_def": "ledoit_wolf"  # (2)!
...     })
... )
{'ADAUSDT': 0.0,
 'BNBUSDT': 0.05013,
 'BTCUSDT': 0.91912,
 'ETHUSDT': 0.0,
 'XRPUSDT': 0.03075}
```

```python
EfficientFrontier
```

### Periodically¶

So, why does vectorbt implement a special parser for PyPortfolioOpt instead of using the original, modularly-built API of the library?

Having a single function that rules them all makes it much easier to use as an optimization function. For example, vectorbt uses sensible defaults for expected returns and other variables, and knows exactly where those variables should be used. In addition, being able to pass arbitrary keyword arguments and letting vectorbt distribute them over functions enables easier testing of multiple argument combinations using groups.

Let's demonstrate this by using PortfolioOptimizer.from_pypfopt, which uses pypfopt_optimize as optimize_func. Optimize for the maximum Sharpe in the previous week:

```python
optimize_func
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_pypfopt(
...     prices=data.get("Close"),
...     every="W"
... )

>>> pfo.plot().show()
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_pypfopt(
...     prices=data.get("Close"),
...     every="W"
... )

>>> pfo.plot().show()
```

And here's how easy it's to test multiple combinations of the argument target:

```python
target
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_pypfopt(
...     prices=data.get("Close"),
...     every="W",
...     target=vbt.Param([
...         "max_sharpe", 
...         "min_volatility", 
...         "max_quadratic_utility"
...     ])
... )

>>> pfo.plot(column="min_volatility").show()  # (1)!
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_pypfopt(
...     prices=data.get("Close"),
...     every="W",
...     target=vbt.Param([
...         "max_sharpe", 
...         "min_volatility", 
...         "max_quadratic_utility"
...     ])
... )

>>> pfo.plot(column="min_volatility").show()  # (1)!
```

```python
column
```

Group 3/3

Group 3/3

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
target
max_sharpe               2.779042
min_volatility           1.862926
max_quadratic_utility    2.352667
Name: sharpe_ratio, dtype: float64
```

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
target
max_sharpe               2.779042
min_volatility           1.862926
max_quadratic_utility    2.352667
Name: sharpe_ratio, dtype: float64
```

We see that optimizing for the maximum Sharpe also yields the highest out-of-sample Sharpe. Great!

#### Manually¶

We can also wrap the pypfopt_optimize function manually, for example, to do some data preprocessing or weight postprocessing:

```python
>>> def optimize_func(prices, index_slice, **kwargs):
...     period_prices = prices.iloc[index_slice]
...     return vbt.pypfopt_optimize(prices=period_prices, **kwargs)

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     data.symbol_wrapper,
...     optimize_func,
...     prices=data.get("Close"),
...     index_slice=vbt.Rep("index_slice"),
...     every="W"
... )
```

```python
>>> def optimize_func(prices, index_slice, **kwargs):
...     period_prices = prices.iloc[index_slice]
...     return vbt.pypfopt_optimize(prices=period_prices, **kwargs)

>>> pfo = vbt.PortfolioOptimizer.from_optimize_func(
...     data.symbol_wrapper,
...     optimize_func,
...     prices=data.get("Close"),
...     index_slice=vbt.Rep("index_slice"),
...     every="W"
... )
```

## Riskfolio-Lib¶

Riskfolio-Lib is a library for making quantitative strategic asset allocation or portfolio optimization in Python made in Peru 🇵🇪. Its objective is to help students, academics and practitioners to build investment portfolios based on mathematically complex models with low effort. It is built on top of cvxpy and closely integrated with pandas data structures.

Similarly to PyPortfolioOpt, Riskfolio-Lib also implements a range of portfolio optimization tools. A common optimization procedure consists of the following steps:

```python
Portfolio
```

```python
stats
```

```python
optimization
```

For example, let's perform the mean-variance optimization (MVO) for maximum Sharpe:

```python
>>> import riskfolio as rp

>>> returns = data.get("Close").vbt.to_returns()

>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist",
...     method_cov="hist",
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT       XRPUSDT
weights  0.207779  1.043621e-08  0.336897  0.455323  3.650466e-09
```

```python
>>> import riskfolio as rp

>>> returns = data.get("Close").vbt.to_returns()

>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist",
...     method_cov="hist",
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT       XRPUSDT
weights  0.207779  1.043621e-08  0.336897  0.455323  3.650466e-09
```

Hint

Why doesn't assets_stats return anything? Because it calculates mu and cov and overrides the portfolio attributes port.mu and port.cov in place.

```python
assets_stats
```

```python
mu
```

```python
cov
```

```python
port.mu
```

```python
port.cov
```

### Parsing¶

Using the way above to produce a vector of weights from vectors of returns is great, but dividing the optimization procedure into a set of different function calls makes it harder to parameterize. What we need is just one function that can express an arbitrary Riskfolio-Lib setup, preferably defined using keyword arguments alone. To make one function out of many, we need to know the inputs of each function, the outputs, and how those functions play together. Thanks to the consistent naming of arguments and functions in Riskfolio-Lib (kudos to @dcajasn!), but also to the entire collection of tutorials, we can crystallize out the order of function calls depending on the optimization task.

For example, the optimization method Portfolio.optimization with the model "Classic" would require the statistic method Portfolio.assets_stats to be called first. The model "FM" would require the statistic methods Portfolio.assets_stats and Portfolio.factors_stats to be called. If the user also provided constraints, then we would additionally need to pre-process them by the respective constraints function.

Let's say we've established the call stack, how do we distribute arguments over the functions? We can read the signature of each function:

```python
>>> from vectorbtpro.utils.parsing import get_func_arg_names

>>> get_func_arg_names(port.assets_stats)
['method_mu', 'method_cov', 'd']
```

```python
>>> from vectorbtpro.utils.parsing import get_func_arg_names

>>> get_func_arg_names(port.assets_stats)
['method_mu', 'method_cov', 'd']
```

If the user passes an argument called method_mu, it should be passed to this function and to any other function listing this argument as it mostly like means the same thing. To resolve the arguments that need to be passed to the respective Riskfolio-Lib function, there is a convenient function resolve_riskfolio_func_kwargs:

```python
method_mu
```

```python
>>> from vectorbtpro.portfolio.pfopt.base import resolve_riskfolio_func_kwargs

>>> resolve_riskfolio_func_kwargs(
...     port.assets_stats,
...     method_mu="hist",
...     method_cov="hist",
...     model="Classic"
... )
{'method_mu': 'hist', 'method_cov': 'hist'}
```

```python
>>> from vectorbtpro.portfolio.pfopt.base import resolve_riskfolio_func_kwargs

>>> resolve_riskfolio_func_kwargs(
...     port.assets_stats,
...     method_mu="hist",
...     method_cov="hist",
...     model="Classic"
... )
{'method_mu': 'hist', 'method_cov': 'hist'}
```

In a case where any of the arguments need to be overridden for one particular function only, we can provide func_kwargs dictionary with functions as keys and keyword arguments as values:

```python
func_kwargs
```

```python
>>> resolve_riskfolio_func_kwargs(
...     port.assets_stats,
...     method_mu="hist",
...     method_cov="hist",
...     model="Classic",
...     func_kwargs=dict(
...         assets_stats=dict(method_mu="ewma1"),
...         optimization=dict(model="BL")
...     )
... )
{'method_mu': 'ewma1', 'method_cov': 'hist'}
```

```python
>>> resolve_riskfolio_func_kwargs(
...     port.assets_stats,
...     method_mu="hist",
...     method_cov="hist",
...     model="Classic",
...     func_kwargs=dict(
...         assets_stats=dict(method_mu="ewma1"),
...         optimization=dict(model="BL")
...     )
... )
{'method_mu': 'ewma1', 'method_cov': 'hist'}
```

This way, we can let vectorbt distribute the arguments, but still reserve the possibility of doing this manually using func_kwargs.

```python
func_kwargs
```

### Auto-optimization¶

Knowing how to parse and resolve function arguments, vectorbt once again implements a function that can take a single set of keyword arguments and translate them into an optimization procedure - riskfolio_optimize. This function is as easy to use as the one for PyPortfolioOpt!

```python
>>> vbt.riskfolio_optimize(returns)
{'ADAUSDT': 0.20777948652846492,
 'BNBUSDT': 1.0435918170753283e-08,
 'BTCUSDT': 0.33689720861500716,
 'ETHUSDT': 0.45532329077024425,
 'XRPUSDT': 3.6503655112892327e-09}
```

```python
>>> vbt.riskfolio_optimize(returns)
{'ADAUSDT': 0.20777948652846492,
 'BNBUSDT': 1.0435918170753283e-08,
 'BTCUSDT': 0.33689720861500716,
 'ETHUSDT': 0.45532329077024425,
 'XRPUSDT': 3.6503655112892327e-09}
```

Under the hood, it first resolves the portfolio class by reading the argument port_cls, and then creates a new portfolio instance by passing any keyword arguments that match the signature of its constructor method __init__. After this, it determines the optimization method by reading the argument opt_method, which is "optimization" by default. Knowing the optimization method and the model (provided via the argument model), it can figure out which statistic methods prior to the optimization should be executed and in what order. The names of those statistic methods are saved in stats_methods, unless they are already provided by the user. The next step is the resolution of any asset classes, constraints, and views, and translating them into keyword arguments that can be consumed by the methods that follow in the call stack. For instance, asset classes are pre-processed using the function resolve_asset_classes, which allows the user to pass asset_classes using a range of convenient formats otherwise not supported by Riskfolio-Lib. Having all keyword arguments ready, the function executes the statistic methods (if any), and finally, the optimization method. It then returns the weights as a dictionary with the columns (i.e., asset names) from the returns array as keys.

```python
port_cls
```

```python
__init__
```

```python
opt_method
```

```python
"optimization"
```

```python
model
```

```python
stats_methods
```

```python
asset_classes
```

Below, we will demonstrate various optimizations done both using Riskfolio-Lib and vectorbt. Ulcer Index Portfolio Optimization for Mean Risk (notebook):

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic", 
...     rm="UCI", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
>>> w.T
              ADAUSDT       BNBUSDT  BTCUSDT  ETHUSDT       XRPUSDT
weights  4.421983e-11  1.922731e-11   0.8343   0.1657  9.143250e-11
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic", 
...     rm="UCI", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
>>> w.T
              ADAUSDT       BNBUSDT  BTCUSDT  ETHUSDT       XRPUSDT
weights  4.421983e-11  1.922731e-11   0.8343   0.1657  9.143250e-11
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94,
...     model="Classic", 
...     rm="UCI", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 4.4219828299615346e-11,
 'BNBUSDT': 1.9227309961407513e-11,
 'BTCUSDT': 0.8342998038068898,
 'ETHUSDT': 0.16570019603823058,
 'XRPUSDT': 9.143250192538338e-11}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94,
...     model="Classic", 
...     rm="UCI", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 4.4219828299615346e-11,
 'BNBUSDT': 1.9227309961407513e-11,
 'BTCUSDT': 0.8342998038068898,
 'ETHUSDT': 0.16570019603823058,
 'XRPUSDT': 9.143250192538338e-11}
```

Worst Case Mean Variance Portfolio Optimization using box and elliptical uncertainty sets (notebook):

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> port.wc_stats(
...     box="s", 
...     ellip="s", 
...     q=0.05, 
...     n_sim=3000, 
...     window=3, 
...     dmu=0.1, 
...     dcov=0.1, 
...     seed=0
... )
>>> w = port.wc_optimization(
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     Umu="box", 
...     Ucov="box"
... )
>>> w.T
              ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT       XRPUSDT
weights  8.434620e-11  4.298850e-11  0.385894  0.614106  4.185089e-11
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> port.wc_stats(
...     box="s", 
...     ellip="s", 
...     q=0.05, 
...     n_sim=3000, 
...     window=3, 
...     dmu=0.1, 
...     dcov=0.1, 
...     seed=0
... )
>>> w = port.wc_optimization(
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     Umu="box", 
...     Ucov="box"
... )
>>> w.T
              ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT       XRPUSDT
weights  8.434620e-11  4.298850e-11  0.385894  0.614106  4.185089e-11
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     opt_method="wc",  # (1)!
...     method_mu="hist", 
...     method_cov="hist", 
...     box="s", 
...     ellip="s", 
...     q=0.05, 
...     n_sim=3000, 
...     window=3, 
...     dmu=0.1, 
...     dcov=0.1, 
...     seed=0,
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     Umu="box", 
...     Ucov="box"
... )
{'ADAUSDT': 8.434620227152581e-11,
 'BNBUSDT': 4.2988498616065945e-11,
 'BTCUSDT': 0.38589404919778153,
 'ETHUSDT': 0.6141059506330329,
 'XRPUSDT': 4.185089189555317e-11}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     opt_method="wc",  # (1)!
...     method_mu="hist", 
...     method_cov="hist", 
...     box="s", 
...     ellip="s", 
...     q=0.05, 
...     n_sim=3000, 
...     window=3, 
...     dmu=0.1, 
...     dcov=0.1, 
...     seed=0,
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     Umu="box", 
...     Ucov="box"
... )
{'ADAUSDT': 8.434620227152581e-11,
 'BNBUSDT': 4.2988498616065945e-11,
 'BTCUSDT': 0.38589404919778153,
 'ETHUSDT': 0.6141059506330329,
 'XRPUSDT': 4.185089189555317e-11}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     func_kwargs=dict(  # (1)!
...         assets_stats=dict(
...             opt_method="wc",
...             method_mu="hist", 
...             method_cov="hist"
...         ),
...         wc_stats=dict(
...             box="s", 
...             ellip="s", 
...             q=0.05, 
...             n_sim=3000, 
...             window=3, 
...             dmu=0.1, 
...             dcov=0.1, 
...             seed=0
...         ),
...         wc_optimization=dict(
...             obj="Sharpe", 
...             rf=0, 
...             l=0, 
...             Umu="box", 
...             Ucov="box"
...         )
...     )
... )
{'ADAUSDT': 8.434620227152581e-11,
 'BNBUSDT': 4.2988498616065945e-11,
 'BTCUSDT': 0.38589404919778153,
 'ETHUSDT': 0.6141059506330329,
 'XRPUSDT': 4.185089189555317e-11}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     func_kwargs=dict(  # (1)!
...         assets_stats=dict(
...             opt_method="wc",
...             method_mu="hist", 
...             method_cov="hist"
...         ),
...         wc_stats=dict(
...             box="s", 
...             ellip="s", 
...             q=0.05, 
...             n_sim=3000, 
...             window=3, 
...             dmu=0.1, 
...             dcov=0.1, 
...             seed=0
...         ),
...         wc_optimization=dict(
...             obj="Sharpe", 
...             rf=0, 
...             l=0, 
...             Umu="box", 
...             Ucov="box"
...         )
...     )
... )
{'ADAUSDT': 8.434620227152581e-11,
 'BNBUSDT': 4.2988498616065945e-11,
 'BTCUSDT': 0.38589404919778153,
 'ETHUSDT': 0.6141059506330329,
 'XRPUSDT': 4.185089189555317e-11}
```

```python
stats
```

```python
optimization
```

Mean Variance Portfolio with Short Weights (notebook):

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.sht = True
>>> port.uppersht = 0.3
>>> port.upperlng = 1.3
>>> port.budget = 1.0
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic", 
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
weights  0.295482 -2.109934e-07  0.456143  0.548375 -0.299999
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.sht = True
>>> port.uppersht = 0.3
>>> port.upperlng = 1.3
>>> port.budget = 1.0
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94
... )
>>> w = port.optimization(
...     model="Classic", 
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT   BTCUSDT   ETHUSDT   XRPUSDT
weights  0.295482 -2.109934e-07  0.456143  0.548375 -0.299999
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     sht=True,
...     uppersht=0.3,
...     upperlng=1.3,
...     budget=1.0,
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.2954820993653493,
 'BNBUSDT': -2.1099344275128538e-07,
 'BTCUSDT': 0.45614303697962627,
 'ETHUSDT': 0.5483745379125106,
 'XRPUSDT': -0.2999994632634474}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     sht=True,
...     uppersht=0.3,
...     upperlng=1.3,
...     budget=1.0,
...     method_mu="hist", 
...     method_cov="hist", 
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.2954820993653493,
 'BNBUSDT': -2.1099344275128538e-07,
 'BTCUSDT': 0.45614303697962627,
 'ETHUSDT': 0.5483745379125106,
 'XRPUSDT': -0.2999994632634474}
```

Constraints on Assets (notebook):

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist",
...     d=0.94
... )
>>> asset_classes = {"Assets": returns.columns.tolist()}
>>> asset_classes = pd.DataFrame(asset_classes)
>>> constraints = {  # (1)!
...     "Disabled": [False, False],
...     "Type": ["All Assets", "Assets"],
...     "Set": ["", ""],
...     "Position": ["", "BTCUSDT"],
...     "Sign": [">=", "<="],
...     'Weight': [0.1, 0.15],
...     "Type Relative": ["", ""],
...     "Relative Set": ["", ""],
...     "Relative": ["", ""],
...     "Factor": ["", ""],
... }
>>> constraints = pd.DataFrame(constraints)
>>> A, B = rp.assets_constraints(constraints, asset_classes)
>>> port.ainequality = A
>>> port.binequality = B
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT  BNBUSDT  BTCUSDT   ETHUSDT  XRPUSDT
weights  0.181443      0.1     0.15  0.468557      0.1
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist",
...     d=0.94
... )
>>> asset_classes = {"Assets": returns.columns.tolist()}
>>> asset_classes = pd.DataFrame(asset_classes)
>>> constraints = {  # (1)!
...     "Disabled": [False, False],
...     "Type": ["All Assets", "Assets"],
...     "Set": ["", ""],
...     "Position": ["", "BTCUSDT"],
...     "Sign": [">=", "<="],
...     'Weight': [0.1, 0.15],
...     "Type Relative": ["", ""],
...     "Relative Set": ["", ""],
...     "Relative": ["", ""],
...     "Factor": ["", ""],
... }
>>> constraints = pd.DataFrame(constraints)
>>> A, B = rp.assets_constraints(constraints, asset_classes)
>>> port.ainequality = A
>>> port.binequality = B
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT  BNBUSDT  BTCUSDT   ETHUSDT  XRPUSDT
weights  0.181443      0.1     0.15  0.468557      0.1
```

```python
BTCUSDT
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     constraints=[{  # (1)!
...         "Type": "All Assets",
...         "Sign": ">=",
...         "Weight": 0.1
...     }, {
...         "Type": "Assets",
...         "Position": "BTCUSDT",
...         "Sign": "<=",
...         "Weight": 0.15
...     }],
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.181442978888792,
 'BNBUSDT': 0.10000000609450148,
 'BTCUSDT': 0.1499998352568763,
 'ETHUSDT': 0.4685571774444982,
 'XRPUSDT': 0.10000000231533206}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     constraints=[{  # (1)!
...         "Type": "All Assets",
...         "Sign": ">=",
...         "Weight": 0.1
...     }, {
...         "Type": "Assets",
...         "Position": "BTCUSDT",
...         "Sign": "<=",
...         "Weight": 0.15
...     }],
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.181442978888792,
 'BNBUSDT': 0.10000000609450148,
 'BTCUSDT': 0.1499998352568763,
 'ETHUSDT': 0.4685571774444982,
 'XRPUSDT': 0.10000000231533206}
```

```python
asset_classes
```

Constraints on Asset Classes (notebook):

```python
>>> tags = [
...     "Smart contracts",
...     "Smart contracts",
...     "Payments",
...     "Smart contracts",
...     "Payments"
... ]
```

```python
>>> tags = [
...     "Smart contracts",
...     "Smart contracts",
...     "Payments",
...     "Smart contracts",
...     "Payments"
... ]
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist",
...     d=0.94
... )
>>> asset_classes = {
...     "Assets": returns.columns.tolist(),
...     "Tags": tags
... }
>>> asset_classes = pd.DataFrame(asset_classes)
>>> constraints = {  # (1)!
...     "Disabled": [False],
...     "Type": ["Classes"],
...     "Set": ["Tags"],
...     "Position": ["Smart contracts"],
...     "Sign": [">="],
...     'Weight': [0.8],
...     "Type Relative": [""],
...     "Relative Set": [""],
...     "Relative": [""],
...     "Factor": [""],
... }
>>> constraints = pd.DataFrame(constraints)
>>> A, B = rp.assets_constraints(constraints, asset_classes)
>>> port.ainequality = A
>>> port.binequality = B
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT  BTCUSDT   ETHUSDT       XRPUSDT
weights  0.227839  5.856725e-10      0.2  0.572161  1.852774e-10
```

```python
>>> port = rp.Portfolio(returns=returns)
>>> port.assets_stats(
...     method_mu="hist", 
...     method_cov="hist",
...     d=0.94
... )
>>> asset_classes = {
...     "Assets": returns.columns.tolist(),
...     "Tags": tags
... }
>>> asset_classes = pd.DataFrame(asset_classes)
>>> constraints = {  # (1)!
...     "Disabled": [False],
...     "Type": ["Classes"],
...     "Set": ["Tags"],
...     "Position": ["Smart contracts"],
...     "Sign": [">="],
...     'Weight': [0.8],
...     "Type Relative": [""],
...     "Relative Set": [""],
...     "Relative": [""],
...     "Factor": [""],
... }
>>> constraints = pd.DataFrame(constraints)
>>> A, B = rp.assets_constraints(constraints, asset_classes)
>>> port.ainequality = A
>>> port.binequality = B
>>> w = port.optimization(
...     model="Classic",
...     rm="MV",
...     obj="Sharpe",
...     rf=0,
...     l=0,
...     hist=True
... )
>>> w.T
          ADAUSDT       BNBUSDT  BTCUSDT   ETHUSDT       XRPUSDT
weights  0.227839  5.856725e-10      0.2  0.572161  1.852774e-10
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     asset_classes={"Tags": tags},
...     constraints=[{
...         "Type": "Classes",
...         "Set": "Tags",
...         "Position": "Smart contracts",
...         "Sign": ">=",
...         "Weight": 0.8
...     }],
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.22783907021563807,
 'BNBUSDT': 5.856745345006487e-10,
 'BTCUSDT': 0.19999999471503008,
 'ETHUSDT': 0.5721609342983793,
 'XRPUSDT': 1.852779892852209e-10}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     method_mu="hist", 
...     method_cov="hist", 
...     asset_classes={"Tags": tags},
...     constraints=[{
...         "Type": "Classes",
...         "Set": "Tags",
...         "Position": "Smart contracts",
...         "Sign": ">=",
...         "Weight": 0.8
...     }],
...     d=0.94,
...     rm="MV", 
...     obj="Sharpe", 
...     rf=0, 
...     l=0, 
...     hist=True
... )
{'ADAUSDT': 0.22783907021563807,
 'BNBUSDT': 5.856745345006487e-10,
 'BTCUSDT': 0.19999999471503008,
 'ETHUSDT': 0.5721609342983793,
 'XRPUSDT': 1.852779892852209e-10}
```

Nested Clustered Optimization (NCO) (notebook):

```python
>>> port = rp.HCPortfolio(returns=returns)
>>> w = port.optimization(
...     model="NCO",
...     codependence="pearson",
...     covariance="hist",
...     obj="MinRisk",
...     rm="MV",
...     rf=0,
...     l=2,
...     linkage="ward",
...     max_k=10,
...     leaf_order=True
... )
>>> w.T
              ADAUSDT  BNBUSDT   BTCUSDT       ETHUSDT   XRPUSDT
weights  6.402583e-09  0.05898  0.911545  3.331509e-09  0.029475
```

```python
>>> port = rp.HCPortfolio(returns=returns)
>>> w = port.optimization(
...     model="NCO",
...     codependence="pearson",
...     covariance="hist",
...     obj="MinRisk",
...     rm="MV",
...     rf=0,
...     l=2,
...     linkage="ward",
...     max_k=10,
...     leaf_order=True
... )
>>> w.T
              ADAUSDT  BNBUSDT   BTCUSDT       ETHUSDT   XRPUSDT
weights  6.402583e-09  0.05898  0.911545  3.331509e-09  0.029475
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     port_cls="HCPortfolio",  # (1)!
...     model="NCO",
...     codependence="pearson",
...     covariance="hist",
...     obj="MinRisk",
...     rm="MV",
...     rf=0,
...     l=2,
...     linkage="ward",
...     max_k=10,
...     leaf_order=True
... )
{'ADAUSDT': 6.402581338827853e-09,
 'BNBUSDT': 0.05897978986842499,
 'BTCUSDT': 0.9115447868616637,
 'ETHUSDT': 3.3315084935084894e-09,
 'XRPUSDT': 0.029475413535821407}
```

```python
>>> vbt.riskfolio_optimize(
...     returns,
...     port_cls="HCPortfolio",  # (1)!
...     model="NCO",
...     codependence="pearson",
...     covariance="hist",
...     obj="MinRisk",
...     rm="MV",
...     rf=0,
...     l=2,
...     linkage="ward",
...     max_k=10,
...     leaf_order=True
... )
{'ADAUSDT': 6.402581338827853e-09,
 'BNBUSDT': 0.05897978986842499,
 'BTCUSDT': 0.9115447868616637,
 'ETHUSDT': 3.3315084935084894e-09,
 'XRPUSDT': 0.029475413535821407}
```

Note

If you're getting the message "The problem doesn't have a solution with actual input parameters" when using the "MOSEK" solver, make sure to install and activate MOSEK. Also, you can try out "ECOS".

### Periodically¶

As mentioned earlier, having one function that rules them all is not only easier to operate with, but its main purpose is to be parameterized and used in rebalancing with PortfolioOptimizer. In particular, the optimization function above is used by the method PortfolioOptimizer.from_riskfolio, which calls it periodically. Let's demonstrate its power by optimizing for the maximum Sharpe in the previous week:

```python
>>> pfo = vbt.PortfolioOptimizer.from_riskfolio(
...     returns=returns,
...     every="W"
... )

>>> pfo.plot().show()
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_riskfolio(
...     returns=returns,
...     every="W"
... )

>>> pfo.plot().show()
```

What about parameters? We can wrap any (also nested) argument with Param to test multiple parameter combinations. Let's test various maximum BTCUSDT weights to see if constraints are working properly:

```python
BTCUSDT
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_riskfolio(
...     returns=returns,
...     constraints=[{
...         "Type": "Assets",
...         "Position": "BTCUSDT",
...         "Sign": "<=",
...         "Weight": vbt.Param([0.1, 0.2, 0.3], name="BTCUSDT_maxw")  # (1)!
...     }],
...     every="W",
...     param_search_kwargs=dict(incl_types=list)  # (2)!
... )
```

```python
>>> pfo = vbt.PortfolioOptimizer.from_riskfolio(
...     returns=returns,
...     constraints=[{
...         "Type": "Assets",
...         "Position": "BTCUSDT",
...         "Sign": "<=",
...         "Weight": vbt.Param([0.1, 0.2, 0.3], name="BTCUSDT_maxw")  # (1)!
...     }],
...     every="W",
...     param_search_kwargs=dict(incl_types=list)  # (2)!
... )
```

Group 3/3

Group 3/3

```python
>>> pfo.allocations.groupby("BTCUSDT_maxw").max()
symbol        ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
BTCUSDT_maxw                                             
0.1               1.0      1.0      0.1      1.0      1.0
0.2               1.0      1.0      0.2      1.0      1.0
0.3               1.0      1.0      0.3      1.0      1.0
```

```python
>>> pfo.allocations.groupby("BTCUSDT_maxw").max()
symbol        ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT
BTCUSDT_maxw                                             
0.1               1.0      1.0      0.1      1.0      1.0
0.2               1.0      1.0      0.2      1.0      1.0
0.3               1.0      1.0      0.3      1.0      1.0
```

Works flawlessly

## Universal portfolios¶

Note

To install this package, first install vectorbtpro and then universal-portfolios, not them together. Since its dependency versions are quite strict, you may want to ignore its dependencies altogether by running pip install -U universal-portfolios --no-deps.

```python
pip install -U universal-portfolios --no-deps
```

The purpose of Universal Portfolios is to put together different Online Portfolio Selection (OLPS) algorithms and provide unified tools for their analysis.

In contrast to PyPortfolioOpt, where weights are generated based on a specific range in time, the purpose of OLPS is to choose portfolio weights in every period to maximize its final wealth. That is, the generated weights have always the shape of the original array.

Let's take a look at the uniform allocation (UCRP):

```python
>>> from universal import tools, algos

>>> with vbt.WarningsFiltered():  # (1)!
...     algo = algos.CRP()
...     algo_result = algo.run(data.get("Close"))

>>> algo_result.weights
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT  CASH
Open time                                                                   
2020-01-01 00:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-01-01 01:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-01-01 02:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
...                            ...      ...      ...      ...      ...   ...
2020-12-31 21:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-12-31 22:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-12-31 23:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0

[8767 rows x 6 columns]
```

```python
>>> from universal import tools, algos

>>> with vbt.WarningsFiltered():  # (1)!
...     algo = algos.CRP()
...     algo_result = algo.run(data.get("Close"))

>>> algo_result.weights
symbol                     ADAUSDT  BNBUSDT  BTCUSDT  ETHUSDT  XRPUSDT  CASH
Open time                                                                   
2020-01-01 00:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-01-01 01:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-01-01 02:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
...                            ...      ...      ...      ...      ...   ...
2020-12-31 21:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-12-31 22:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0
2020-12-31 23:00:00+00:00      0.2      0.2      0.2      0.2      0.2   0.0

[8767 rows x 6 columns]
```

As we see, Universal Portfolios has generated and allocated weights at each single timestamp, which is pretty unrealistic because rebalancing that frequently is unsustainable in practice, unless the frequency of data is low. Additionally, iterating over this amount of data with this library is usually quite slow.

To account for this, we should downsample the pricing array to a longer time frame, and then upsample back to the original index. Let's try this out on the DynamicCRP algorithm by downsampling to the daily frequency first:

```python
DynamicCRP
```

```python
>>> with vbt.WarningsFiltered():
...     algo = algos.DynamicCRP(
...         n=30, 
...         min_history=7, 
...         metric='sharpe', 
...         alpha=0.01
...     )
...     algo_result = algo.run(data.get("Close").resample("D").last())
...     down_weights = algo_result.weights

>>> down_weights
symbol                          ADAUSDT       BNBUSDT   BTCUSDT       ETHUSDT  \
Open time                                                                       
2020-01-01 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
2020-01-02 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
2020-01-03 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
...                                 ...           ...       ...           ...   
2020-12-29 00:00:00+00:00  8.475716e-09  8.176270e-09  0.664594  8.274986e-09   
2020-12-30 00:00:00+00:00  0.000000e+00  0.000000e+00  0.656068  0.000000e+00   
2020-12-31 00:00:00+00:00  0.000000e+00  0.000000e+00  0.655105  0.000000e+00   

symbol                          XRPUSDT  
Open time                                
2020-01-01 00:00:00+00:00  2.000000e-01  
2020-01-02 00:00:00+00:00  2.000000e-01  
2020-01-03 00:00:00+00:00  2.000000e-01  
...                                 ...  
2020-12-29 00:00:00+00:00  9.004152e-09  
2020-12-30 00:00:00+00:00  0.000000e+00  
2020-12-31 00:00:00+00:00  0.000000e+00  

[366 rows x 5 columns]
```

```python
>>> with vbt.WarningsFiltered():
...     algo = algos.DynamicCRP(
...         n=30, 
...         min_history=7, 
...         metric='sharpe', 
...         alpha=0.01
...     )
...     algo_result = algo.run(data.get("Close").resample("D").last())
...     down_weights = algo_result.weights

>>> down_weights
symbol                          ADAUSDT       BNBUSDT   BTCUSDT       ETHUSDT  \
Open time                                                                       
2020-01-01 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
2020-01-02 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
2020-01-03 00:00:00+00:00  2.000000e-01  2.000000e-01  0.200000  2.000000e-01   
...                                 ...           ...       ...           ...   
2020-12-29 00:00:00+00:00  8.475716e-09  8.176270e-09  0.664594  8.274986e-09   
2020-12-30 00:00:00+00:00  0.000000e+00  0.000000e+00  0.656068  0.000000e+00   
2020-12-31 00:00:00+00:00  0.000000e+00  0.000000e+00  0.655105  0.000000e+00   

symbol                          XRPUSDT  
Open time                                
2020-01-01 00:00:00+00:00  2.000000e-01  
2020-01-02 00:00:00+00:00  2.000000e-01  
2020-01-03 00:00:00+00:00  2.000000e-01  
...                                 ...  
2020-12-29 00:00:00+00:00  9.004152e-09  
2020-12-30 00:00:00+00:00  0.000000e+00  
2020-12-31 00:00:00+00:00  0.000000e+00  

[366 rows x 5 columns]
```

Notice how the calculation still takes a considerable amount of time, even by reducing the total number of re-allocation timestamps by 24 times.

Let's bring the weights back to the original time frame:

```python
>>> weights = down_weights.vbt.realign(
...     data.wrapper.index,
...     freq="1h",
...     source_rbound=True,  # (1)!
...     target_rbound=True,
...     ffill=False  # (2)!
... )
>>> weights
symbol                     ADAUSDT  BNBUSDT   BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                              
2020-01-01 00:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-01-01 01:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-01-01 02:00:00+00:00      NaN      NaN       NaN      NaN      NaN
...                            ...      ...       ...      ...      ...
2020-12-31 21:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-12-31 22:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-12-31 23:00:00+00:00      0.0      0.0  0.655105      0.0      0.0

[8766 rows x 5 columns]
```

```python
>>> weights = down_weights.vbt.realign(
...     data.wrapper.index,
...     freq="1h",
...     source_rbound=True,  # (1)!
...     target_rbound=True,
...     ffill=False  # (2)!
... )
>>> weights
symbol                     ADAUSDT  BNBUSDT   BTCUSDT  ETHUSDT  XRPUSDT
Open time                                                              
2020-01-01 00:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-01-01 01:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-01-01 02:00:00+00:00      NaN      NaN       NaN      NaN      NaN
...                            ...      ...       ...      ...      ...
2020-12-31 21:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-12-31 22:00:00+00:00      NaN      NaN       NaN      NaN      NaN
2020-12-31 23:00:00+00:00      0.0      0.0  0.655105      0.0      0.0

[8766 rows x 5 columns]
```

This array can now be used in simulation.

To simplify the workflow introduced above, vectorbt implements a class method PortfolioOptimizer.from_universal_algo, which triggers the entire simulation with Universal Portfolios and, after done, picks the allocations at specific dates from the resulting DataFrame. By default, it picks the timestamps of non-NA, non-repeating weights. The method itself takes the algorithm (algo) and the pricing data (S). The former can take many value types: from the name or instance of the algorithm class (must be a subclass of universal.algo.Algo), to the actual result of the algorithm (of type universal.result.AlgoResult).

```python
algo
```

```python
S
```

```python
universal.algo.Algo
```

```python
universal.result.AlgoResult
```

Let's run the same algorithm as above, but now using PortfolioOptimizer. We will also test multiple value combinations for n:

```python
n
```

```python
>>> with vbt.WarningsFiltered():
...     down_pfo = vbt.PortfolioOptimizer.from_universal_algo(
...         "DynamicCRP",
...         data.get("Close").resample("D").last(),
...         n=vbt.Param([7, 14, 30, 90]), 
...         min_history=7, 
...         metric='sharpe', 
...         alpha=0.01
...     )

>>> down_pfo.plot(column=90).show()
```

```python
>>> with vbt.WarningsFiltered():
...     down_pfo = vbt.PortfolioOptimizer.from_universal_algo(
...         "DynamicCRP",
...         data.get("Close").resample("D").last(),
...         n=vbt.Param([7, 14, 30, 90]), 
...         min_history=7, 
...         metric='sharpe', 
...         alpha=0.01
...     )

>>> down_pfo.plot(column=90).show()
```

Group 4/4

Group 4/4

We can then upsample the optimizer back to the original time frame by constructing an instance of Resampler and passing it to PortfolioOptimizer.resample:

```python
>>> resampler = vbt.Resampler(
...     down_pfo.wrapper.index, 
...     data.wrapper.index, 
...     target_freq="1h"
... )
>>> pfo = down_pfo.resample(resampler)
```

```python
>>> resampler = vbt.Resampler(
...     down_pfo.wrapper.index, 
...     data.wrapper.index, 
...     target_freq="1h"
... )
>>> pfo = down_pfo.resample(resampler)
```

Note

An allocation at the end of a daily bar will be placed at the end of the first hourly bar on that day, which may be undesired if the allocation uses any information from that daily bar. To account for this, calculate and use the right bounds of both indexes with Resampler.get_rbound_index.

And finally, use the new optimizer in a simulation:

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
n
7     2.913174
14    3.456085
30    3.276883
90    2.176654
Name: sharpe_ratio, dtype: float64
```

```python
>>> pf = pfo.simulate(data, freq="1h")

>>> pf.sharpe_ratio
n
7     2.913174
14    3.456085
30    3.276883
90    2.176654
Name: sharpe_ratio, dtype: float64
```

### Custom algorithm¶

Let's create our own mean-reversion algorithm using Universal Portfolios. The idea is that badly performing stocks will revert to its mean and have higher returns than those above their mean.

```python
>>> from universal.algo import Algo

>>> class MeanReversion(Algo):
...     PRICE_TYPE = 'log'
...     
...     def __init__(self, n):
...         self.n = n
...         super().__init__(min_history=n)
...     
...     def init_weights(self, cols):
...         return pd.Series(np.zeros(len(cols)), cols)
...     
...     def step(self, x, last_b, history):
...         ma = history.iloc[-self.n:].mean()
...         delta = x - ma
...         w = np.maximum(-delta, 0.)
...         return w / sum(w)

>>> with vbt.WarningsFiltered():
...     pfo = vbt.PortfolioOptimizer.from_universal_algo(
...         MeanReversion,
...         data.get("Close").resample("D").last(),  # (1)!
...         n=30,  # (2)!
...         every="W"  # (3)!
...     )

>>> pfo.plot().show()
```

```python
>>> from universal.algo import Algo

>>> class MeanReversion(Algo):
...     PRICE_TYPE = 'log'
...     
...     def __init__(self, n):
...         self.n = n
...         super().__init__(min_history=n)
...     
...     def init_weights(self, cols):
...         return pd.Series(np.zeros(len(cols)), cols)
...     
...     def step(self, x, last_b, history):
...         ma = history.iloc[-self.n:].mean()
...         delta = x - ma
...         w = np.maximum(-delta, 0.)
...         return w / sum(w)

>>> with vbt.WarningsFiltered():
...     pfo = vbt.PortfolioOptimizer.from_universal_algo(
...         MeanReversion,
...         data.get("Close").resample("D").last(),  # (1)!
...         n=30,  # (2)!
...         every="W"  # (3)!
...     )

>>> pfo.plot().show()
```

Now it's your turn: create and implement a simple optimization strategy that would make sense in the real world - you'd be amazed how complex and interesting some strategies can become after starting with something really basic

Python code  Notebook

