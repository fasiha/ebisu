from pprint import pprint
from scipy.special import gammaln
from ebisu import logsumexp
from dataclasses import dataclass
from typing import Any, Optional
from colorama import Fore, Style
from scipy.optimize import minimize_scalar
import numpy as np
from scipy.stats import gamma as gammarv, lognorm, weibull_min
import pylab as plt

plt.ion()

data = [
    [1, 1, 0.2],
    [0, 1, 22.9],
    [0, 1, 21.5],
    [1, 1, 75.0],
    [0, 1, 119.2],
    [1, 1, 24.6],
    [1, 1, 44.3],
    [1, 1, 230.3],
    [0, 1, 205.2],
    [1, 1, 25.4],
    [1, 1, 94.6],
    [1, 1, 167.9],
    [0, 1, 216.0],
    [1, 1, 23.7],
    [1, 1, 120.5],
    [1, 1, 216.8],
    [1, 1, 502.6],
    [0, 1, 1272.9],
    [1, 1, 143.6],
    [1, 1, 409.4],
    [1, 1, 720.0],
    [1, 1, 1215.1],
    [1, 1, 2242.0],
    [1, 1, 2554.1],
]


@dataclass
class Model():
  logWeights: Any
  samples: Any
  weights: Optional[Any] = None


def normLogW(logWeights):
  ws = np.exp(logWeights)
  return ws / sum(ws)


def predict(model: Model, hoursElapsed: float) -> float:
  logps = (-hoursElapsed / model.samples) * np.log(2)
  ws = model.weights if model.weights is not None else 1 / len(model.logWeights)
  return sum(ws * np.exp(logps))


def fitWeibull(model: Model):
  m1 = sum(model.weights * model.samples)
  m2 = sum(model.weights * model.samples**2)
  from scipy.special import gamma
  resK = minimize_scalar(
      lambda k: abs(m2 - (m1 / gamma(1 + 1 / k))**2 * gamma(1 + 2 / k)),
      bracket=[.1, 1],
      method='golden')
  resLambda = m1 / gamma(1 + 1 / resK.x)
  return (resK.x, resLambda, weibull_min(resK.x, scale=resLambda))


def update(model: Model, hoursElapsed: float, correct: int | bool) -> Model:
  logps = (-hoursElapsed / model.samples) * np.log(2)
  model.logWeights += logps if correct else np.expm1(-np.expm1(logps))
  model.weights = normLogW(model.logWeights)
  return model


def halflife(model: Model) -> float:
  ws = model.weights if model.weights is not None else 1 / len(model.logWeights)
  res = minimize_scalar(
      lambda h: abs(0.5 - np.sum(ws * np.exp2(-h / model.samples))),
      bracket=[1, 1000],
      method='golden',
      tol=.1,
      options=dict(maxiter=50))
  return res.x


def run(data, model):
  oldHl = halflife(model)
  print(f'{oldHl=}')
  for idx, (correct, total, hoursElapsed) in enumerate(data):
    assert total == 1 and (correct == 0 or correct == 1)  # only handle Bernoulli case
    correctStr = f'{"" if correct else Fore.RED}{correct=}{Style.RESET_ALL}'

    expectedP = predict(model, hoursElapsed)
    model = update(model, hoursElapsed, correct)
    newHl = halflife(model)
    fit = fitWeibull(model)
    print(
        f'{idx=:2d}, {hoursElapsed=:6.1f}, p={expectedP:.2f}, {correctStr}/{total=}, hl={newHl:.2f}, (k,l)={fit[0]:.2f},{fit[1]:.2f}'
    )
  return model


size = 10_000_000
k, l = .25, 20
wrv = weibull_min(k, scale=l)
mkmodel = lambda: Model(logWeights=np.zeros(size), samples=wrv.rvs(size))

m1 = run([[0, 1, 20], [1, 1, 4]], mkmodel())
m2 = run([[1, 1, 4]], mkmodel())
"""
The thing with models like Ebisu v2 where the unknown parameter is assumed to be static is,
failures and successes shift its estimate around and you never lose old data.

We could downplay the impact of this somehow but.

Whereas with a model where you have multiple weighted sub-models whose weights you update.
That feels more evolutionary.
"""

from sympy.stats import Weibull, density, E, variance
from sympy import Symbol, simplify, S

z = Symbol("z")
X = Weibull("x", l, k)

print(simplify(E(2**(-1 / X))).evalf())
