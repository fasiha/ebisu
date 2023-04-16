import ebisu
import numpy as np

initHl = 10
pow = 14
hls = np.logspace(-1, 5, 20)
ws = ebisu.ebisu._halflifeToFinalWeight(initHl, hls, pow)
ws = ws / np.sum(ws)

ps = 2**(-initHl / hls)

wplaw = lambda vec, ws, pow: np.sum(ws / np.sum(ws) * vec**pow)**(1 / pow)
plaw = lambda vec, pow: np.mean(vec**pow)**(1 / pow)
normcetin = lambda vec, k, eps: np.mean(((vec)**2 + eps)**(k / 2))
norm = lambda vec, p: np.mean(vec**p)**(1 / p)
bol = lambda vec, a: np.sum(vec * np.exp(a * vec)) / np.sum(np.exp(a * vec))
# bolw = lambda vec, w, a: np.sum(vec * w * np.exp(a * w * vec)) / np.sum(w * np.exp(a * vec))
mel = lambda vec, a: 1 / a * np.log(np.mean(np.exp(a * vec)))
egreedy = lambda vec, e: e * np.mean(vec) + (1 - e) * np.max(vec)

powerlaw = np.sum(ws * ps**pow)**(1 / pow)

from scipy.optimize import minimize_scalar


def _halflifeToFinalWeight(halflife, hs, f):
  n = len(hs)

  v = 2**(-halflife / np.array(hs))
  target = (0.5)
  n = len(hs)
  ivec = (np.arange(n) / (n - 1))

  xToWs = lambda x: np.exp(x)**ivec

  def optlog(logwfinal):
    ws = xToWs(logwfinal)
    return abs(target - f(v, ws))

  res = minimize_scalar(optlog, bounds=[np.log(1e-13), 0])
  assert res.success
  return (xToWs(res.x)).tolist()


wb = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: bol(vec * ws, 4))
bol(ps * wb, 4)

ivec = (np.arange(len(hls)) / (len(hls) - 1))

weps = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: egreedy(vec * ws, .1))
print(egreedy(ps * weps, .1))

egreedy(weps * 2**(-initHl / hls), .1)

import pylab as plt

plt.ion()
t = np.logspace(-3, 6, 1001)
plt.figure()
plt.loglog(t, [egreedy(weps * 2**(-x / hls), .1) for x in t], label="e-greedy")

ws = ebisu.ebisu._halflifeToFinalWeight(initHl, hls, pow)
plt.loglog(t, [wplaw(2**(-x / hls), ws, pow) for x in t], label=f"wpow{pow}")

wb = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: bol(vec * ws, 2))
plt.loglog(t, [bol(2**(-x / hls) * wb, 2) for x in t], label="bol2")

wm = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: mel(ws * vec, 2))
plt.loglog(t, [mel(wm * 2**(-x / hls), 2) for x in t], label="mel2")

wm = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: mel(ws * vec, 5))
plt.loglog(t, [mel(wm * 2**(-x / hls), 5) for x in t], label="mel5")

wmax = _halflifeToFinalWeight(initHl, hls, lambda vec, ws: np.max(ws * vec))
plt.loglog(t, [np.max(wmax * 2**(-x / hls)) for x in t], label="max")

plt.legend()

plt.figure()
for pow in [.2, .5, .7, 1, 1.5, 2, 3]:
  wmax = (.01**ivec**pow).tolist()
  plt.loglog(t, [np.max(2**(-x / hls) * wmax) for x in t], label=f"{pow}")

plt.legend()

plt.figure()
for last in [.001, .01, .1, .5]:
  wmax = (last**ivec).tolist()
  plt.loglog(t, [np.max(2**(-x / hls) * wmax) for x in t], label=f"{last}")

plt.legend()

plt.figure()
for firstHalflife in [1, 5, 7.5, 8.5]:
  m = ebisu.initModel(halflife=10, now=0, firstHalflife=firstHalflife, n=20)
  plt.loglog(
      t, [ebisu.predictRecall(m, now=h * 3600e3, logDomain=False) for h in t],
      label=f'{firstHalflife=}')
plt.legend()
plt.grid()