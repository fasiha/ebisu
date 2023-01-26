from typing import Optional
import ebisu
import numpy as np
from ebisu.ebisu import HOURS_PER_MILLISECONDS

from ebisu.ebisuHelpers import timeMs
import pylab as plt

plt.ion()

Model = ebisu.Model


def _betaToVar(a: float, b: float) -> float:
  return a * b / ((a + b)**2 * (a + b + 1))


def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


hoursElapseds = 4**(np.arange(5))
fig, axs = plt.subplots(1, len(hoursElapseds))
fig2, axs2 = plt.subplots(1, len(hoursElapseds))
for hidx, hoursElapsed in enumerate(hoursElapseds):
  wmaxVars = np.logspace(-4, np.log10(0.25), 101)
  approxs = []
  for wmaxMean in np.logspace(-2, np.log10(0.9), 9):  # or np.linspace(0.1, 0.9, 9):

    a, b = _meanVarToBeta(wmaxMean, wmaxVars)

    idx = (a > 0) & (b > 0)
    thisVars = wmaxVars[idx]

    m = ebisu.initModel(wmaxMean=wmaxMean, now=0)
    now = 3600e3 * hoursElapsed
    approx = ebisu.predictRecall(m, now=now, logDomain=False)
    approxs.append(approx)

    exact = np.array([ebisu.predictRecallBayesian(m, wmaxVar, now=now) for wmaxVar in thisVars])
    (handle,) = axs[hidx].semilogx(thisVars, exact, label=f'{wmaxMean=:.1g}')
    (handle2,) = axs2[hidx].semilogx(thisVars, exact / approx, label=f'{wmaxMean=:.1g}')

    idx2 = (a[idx] >= 2) & (b[idx] >= 2)
    axs[hidx].semilogx(thisVars[idx2], exact[idx2], linewidth=3, color=handle.get_color())
    axs2[hidx].semilogx(
        thisVars[idx2], exact[idx2] / approx, linewidth=3, color=handle2.get_color())

  axs[hidx].scatter(
      wmaxVars[0] * np.ones_like(approxs),
      approxs,
      marker='o',
      color='black',
      label='predictRecall')

  axs[hidx].set_xlabel('wmaxVar')
  axs2[hidx].set_xlabel('wmaxVar')
  if hidx == 0:
    axs[hidx].set_ylabel('Probability of recall')
    axs2[hidx].set_ylabel('exact/approx')

  axs[hidx].set_title(f'{hoursElapsed} h')
  axs[hidx].set_ylim((0, 1))

  axs2[hidx].set_title(f'{hoursElapsed} h')
  axs2[hidx].set_ylim((0.2, 1.05))
# plt.tight_layout()
# plt.legend()

for a in range(1, 10):
  b = a
  wmaxMean = a / (a + b)  # will be 0.5
  wmaxVar = a * b / ((a + b)**2 * (a + b + 1))  # definition
  m = ebisu.initModel(wmaxMean=wmaxMean, now=0)
  now = 3600e3
  approx = ebisu.predictRecall(m, now=now, logDomain=False)
  exact = ebisu.predictRecallBayesian(m, wmaxVar, now=now)
  print(f'{wmaxVar=:0.4f} {approx=:0.4f}, {exact=:0.4f}')


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

# maxVar = _betaToVar(*_betaMeanToAlphaBeta(m.pred.wmaxMean))
# print('ebisu analytical', ebisu.predictRecallBayesian(m, maxVar, now=3600e3 * hoursElapsed))
# res = _predictRecallMonteCarlo(m, now=3600e3 * hoursElapsed)
# print('\n'.join([", ".join([f"{k}={v:.4f}" for k, v in x.items()]) for x in res]))

# plt.figure()
# plt.scatter([x["wmaxStd"] for x in res], [x["std"] for x in res], label='orig')
# p2 = np.polyfit([x["wmaxStd"] for x in res], [x["std"] for x in res], 2)
# p1 = np.polyfit([x["wmaxStd"] for x in res], [x["std"] for x in res], 1)
# xs = np.linspace(*plt.xlim())
# plt.plot(xs, np.polyval(p2, xs), label='quad')
# plt.plot(xs, np.polyval(p1, xs), label='lin')

m = ebisu.initModel(.5, now=0)
#
plt.figure()
hs = np.logspace(-1, 3)
for wmaxMean in [0.1, .25, .5, .75, .9]:
  ps = [ebisu.predictRecallBayesian(m, now=3600e3 * h, wmaxMean=wmaxMean) for h in hs]
  plt.plot(hs, ps, label=f'μ={wmaxMean:.2g}')
plt.legend()


foo = lambda h: ebisu.predictRecallBayesian(ebisu.initModel(initHlMean=h, now=0), now=3600e3 * h)
hs = np.logspace(-1, 3, 101)
plt.figure()
plt.semilogx(hs, [foo(h) for h in hs])

