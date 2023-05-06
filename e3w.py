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


size = 1_000_000
# a,b = 1.0, 1.0 # loosest, probably will be unstable
a, b = 2.0, 2.0  # looser
a, b = 4.0, 4.0  # tighter

hlSamplesExpGamma = 10**(gammarv.rvs(a, scale=1 / b, size=size))
expGamma = Model(logWeights=np.zeros(size), samples=hlSamplesExpGamma)

mu, s2 = np.log(10), 2
hlSamplesLnorm = lognorm.rvs(np.sqrt(s2), scale=np.exp(mu), size=size)
logNorm = Model(logWeights=np.zeros(size), samples=hlSamplesLnorm)

k, l = .25, 1
hlSamplesWeibull = weibull_min.rvs(k, scale=l, size=size)
weibully = Model(logWeights=np.zeros(size), samples=hlSamplesWeibull)


def expGammaPdf(z: np.ndarray, k: float, theta: float):
  from scipy.special import gamma
  return (np.log(z) / np.log(10))**(k - 1) * np.exp(-np.log(z) / (theta * np.log(10))) / (
      theta**k * np.log(10) * (z) * gamma(k))


h = np.logspace(-1, 6)
plt.figure()
plt.loglog(h, lognorm.pdf(h, np.sqrt(s2), scale=np.exp(mu)), label='logNorm')
plt.loglog(h, expGammaPdf(h, a, 1 / b), label='expGamma')
plt.loglog(h, weibull_min.pdf(h, k, scale=l), label='Weibull')
plt.xlabel('hours')
plt.legend()

plt.figure()
range = (1, 10000)
plt.hist(hlSamplesLnorm, bins=100, density=True, alpha=0.25, range=range, label='logNormal')
plt.hist(hlSamplesExpGamma, bins=100, density=True, alpha=0.25, range=range, label='exp10Gamma')
plt.hist(hlSamplesWeibull, bins=100, density=True, alpha=0.25, range=range, label='Weibull')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel('hours')
plt.legend()


def normLogW(logWeights):
  ws = np.exp(logWeights)
  return ws / sum(ws)


def predict(model: Model, hoursElapsed: float) -> float:
  logps = (-hoursElapsed / model.samples) * np.log(2)
  ws = model.weights if model.weights is not None else 1 / len(model.logWeights)
  return sum(ws * np.exp(logps))


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
  print(f'                       logN, p={expectedP:.2f}, hl={newHl:.2f}')

  expectedP = predict(weibully, hoursElapsed)
  weibully = update(weibully, hoursElapsed, correct)
  newHl = halflife(weibully)
  print(f'                    Weibull, p={expectedP:.2f}, hl={newHl:.2f}')
