from dataclasses import dataclass
from typing import Any
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


size = 1_000_000
# a,b = 1.0, 1.0 # loosest, probably will be unstable
a, b = 2.0, 2.0  # looser
a, b = 4.0, 4.0  # tighter
hlSamples = 10**(gammarv.rvs(a, scale=1 / b, size=size))
expGamma = Model(logWeights=np.zeros(size), samples=hlSamples)

mu, s2 = np.log(10), 2
hlSamplesLnorm = lognorm.rvs(np.sqrt(s2), scale=np.exp(mu), size=size)
logNorm = Model(logWeights=np.zeros(size), samples=hlSamplesLnorm)

plt.figure()
plt.hist(hlSamplesLnorm, bins=100, density=True, alpha=0.5, range=(1, 1000), label='logNormal')
plt.hist(hlSamples, bins=100, density=True, alpha=0.5, range=(1, 1000), label='logGamma')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.legend()


def normLogW(logWeights):
  ws = np.exp(logWeights)
  return ws / sum(ws)


def predict(model: Model, hoursElapsed: float) -> float:
  logps = (-hoursElapsed / model.samples) * np.log(2)
  ws = normLogW(model.logWeights)
  return sum(ws * np.exp(logps))


def update(model: Model, hoursElapsed: float, correct: int | bool) -> Model:
  logps = (-hoursElapsed / model.samples) * np.log(2)
  model.logWeights += logps if correct else np.expm1(-np.expm1(logps))
  return model


def halflife(model: Model) -> float:
  ws = normLogW(model.logWeights)
  res = minimize_scalar(
      lambda h: abs(0.5 - np.sum(ws * np.exp2(-h / model.samples))),
      bracket=[1, 1000],
      method='golden',
      tol=.1,
      options=dict(maxiter=50))
  return res.x


for idx, (correct, total, hoursElapsed) in enumerate(data):
  assert total == 1 and (correct == 0 or correct == 1)  # only handle Bernoulli case

  expectedP = predict(expGamma, hoursElapsed)
  expGamma = update(expGamma, hoursElapsed, correct)
  newHl = halflife(expGamma)

  correctStr = f'{"" if correct else Fore.RED}{correct=}{Style.RESET_ALL}'
  print(
      f'{idx=:2d}, {hoursElapsed=:6.1f}, p={expectedP:.2f}, {correctStr}/{total=}, hl={newHl:.2f}')

  expectedP = predict(logNorm, hoursElapsed)
  logNorm = update(logNorm, hoursElapsed, correct)
  newHl = halflife(logNorm)
  print(f'                             p={expectedP:.2f}, hl={newHl:.2f}')
