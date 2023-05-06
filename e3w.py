from colorama import Fore, Style
from scipy.optimize import minimize_scalar
import numpy as np
from scipy.stats import gamma as gammarv, lognorm
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


def plotter(samples, a, b, range=None):
  fig, axs = plt.subplots(2, 1)
  axs[0].hist(10**(samples), bins=100, range=range, density=True)
  axs[1].hist((samples), bins=100, density=True)
  fig.suptitle(f'{a=}, {b=}')
  axs[0].set_xscale('log')
  axs[0].set_yscale('log')
  return fig, axs


size = 1_000_000
# a,b = 1.0, 1.0 # loosest, probably will be unstable
a, b = 2.0, 2.0  # looser
a, b = 4.0, 4.0  # tighter
logHlSamples = gammarv.rvs(a, scale=1 / b, size=size)
hlSamples = 10**(logHlSamples)
logWeights = np.zeros(size)

fig, axs = plotter(logHlSamples, a, b)

mu, s2 = np.log(10), 2
hlSamplesLnorm = lognorm.rvs(np.sqrt(s2), scale=np.exp(mu), size=size)
logWeightsLnorm = np.zeros(size)

plt.figure()
plt.hist(hlSamplesLnorm, bins=100, density=True, alpha=0.5, range=[1, 1000], label='logNormal')
plt.hist(hlSamples, bins=100, density=True, alpha=0.5, range=[1, 1000], label='logGamma')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.legend()


def normLogW(logWeights):
  ws = np.exp(logWeights)
  return ws / sum(ws)


for idx, (correct, total, hoursElapsed) in enumerate(data):
  logps = (-hoursElapsed / hlSamples) * np.log(2)
  ws = normLogW(logWeights)
  expectedP = np.exp(sum(ws * logps))
  logWeights += logps if correct else np.expm1(-np.expm1(logps))

  ws = normLogW(logWeights)
  res = minimize_scalar(
      lambda h: abs(0.5 - np.sum(ws * np.exp2(-h / hlSamples))),
      bracket=[1, 1000],
      method='golden',
      tol=.1,
      options=dict(maxiter=50))

  correctStr = f'{"" if correct else Fore.RED}{correct=}{Style.RESET_ALL}'
  print(
      f'{idx=:2d}, {hoursElapsed=:6.1f}, p={expectedP:.2f}, {correctStr}/{total=}, hl={res.x:.2f}')

  logps = (-hoursElapsed / hlSamplesLnorm) * np.log(2)
  ws = normLogW(logWeightsLnorm)
  expectedP = np.exp(sum(ws * logps))
  logWeightsLnorm += logps if correct else np.expm1(-np.expm1(logps))

  ws = normLogW(logWeightsLnorm)
  res = minimize_scalar(
      lambda h: abs(0.5 - np.sum(ws * np.exp2(-h / hlSamplesLnorm))),
      bracket=[1, 1000],
      method='golden',
      tol=.1,
      options=dict(maxiter=50))
  print(f'                             p={expectedP:.2f}, hl={res.x:.2f}')
