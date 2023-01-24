from typing import Optional
import ebisu
import numpy as np
from ebisu.ebisu import HOURS_PER_MILLISECONDS

from ebisu.ebisuHelpers import timeMs

Model = ebisu.Model


def _predictRecallMonteCarlo(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    size: int = 100_000,
) -> list[dict]:
  from scipy.stats import beta as betarv
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS

  ret: list[float] = []

  n = len(model.pred.hs)
  trace = np.exp2(-elapsedHours / np.array(model.pred.hs))

  wmaxMean = model.pred.wmaxMean
  maxVarAlphaBeta = _betaMeanToAlphaBeta(wmaxMean)
  maxVar = _betaToVar(*maxVarAlphaBeta)
  for var in np.linspace(1, 0.1, 10) * maxVar:
    thisAlphaBeta = _meanVarToBeta(wmaxMean, var)
    wmaxs = betarv.rvs(*thisAlphaBeta, size=size)
    ws = np.exp(np.arange(n) / (n - 1) * np.log(wmaxs[:, np.newaxis]))
    # `ws.shape == (size, n)`
    pRecalls = np.max(ws * trace, axis=1)
    pRecall = np.mean(pRecalls)
    pRecallStd = np.std(pRecalls)
    ret.append(dict(mean=pRecall, std=pRecallStd, wmaxStd=np.sqrt(var)))
  return ret


def _betaToVar(a: float, b: float) -> float:
  return a * b / ((a + b)**2 * (a + b + 1))


def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def _betaMeanToAlphaBeta(wmaxMean: float) -> tuple[float, float]:
  # find (α, β) such that Beta(α, β) is lowest variance AND α>=2, β>=2

  # following are solutions for μ = a/(a+b) such that a=2 and b=2, which will
  # be the maximum-var solution (proof by handwaving)
  b = -2 + 2 / wmaxMean  # beta = 2
  if b >= 2:
    return (2.0, min(b, 10 + np.log10(b)))
    # I mean sure β might be 300 (for e.g., `h=1` hour) but let's cap this for sanity

  a = -2 * wmaxMean / (wmaxMean - 1)  # alpha = 2
  if a >= 2:
    return (min(a, 10 + np.log10(a)), 2.0)
  raise Exception('unable to find prior')


wmaxMean = 0.5
hoursElapsed = 13.0

m = ebisu.initModel(wmaxMean=wmaxMean, now=0)
print('ebisu approx', ebisu.predictRecall(m, now=3600e3 * hoursElapsed, logDomain=False))

maxVar = _betaToVar(*_betaMeanToAlphaBeta(m.pred.wmaxMean))
print('ebisu analytical', ebisu.predictRecallBayesian(m, maxVar, now=3600e3 * hoursElapsed))
res = _predictRecallMonteCarlo(m, now=3600e3 * hoursElapsed)
# pprint(res)
print('\n'.join([", ".join([f"{k}={v:.4f}" for k, v in x.items()]) for x in res]))

import pylab as plt

plt.ion()

plt.scatter([x["wmaxStd"] for x in res], [x["std"] for x in res], label='orig')
p2 = np.polyfit([x["wmaxStd"] for x in res], [x["std"] for x in res], 2)
p1 = np.polyfit([x["wmaxStd"] for x in res], [x["std"] for x in res], 1)
xs = np.linspace(*plt.xlim())
plt.plot(xs, np.polyval(p2, xs), label='quad')
plt.plot(xs, np.polyval(p1, xs), label='lin')
