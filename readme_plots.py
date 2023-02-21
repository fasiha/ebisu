import re
import numpy as np
import ebisu
import pylab as plt

from ebisu.ebisu import _meanVarToGamma

plt.ion()
norm = lambda v: np.array(v) / np.sum(v)

hs = np.logspace(0, 3, 4)
hl = 100
power = 4
ws = ebisu.ebisu._halflifeToFinalWeight(hl, hs, power)
# ws = norm(ws)

m = ebisu.initModel(
    power=power,
    weightsHalflifeGammas=[(w, _meanVarToGamma(h, (h * .5)**2)) for w, h in zip(ws, hs)],
    now=0)

ts = np.logspace(-2, 4, 501)

styles = [('-', 1.5), ('--', 1.5), ('-.', 2), ('-', 2.5), ('--', 2.5)]
plt.figure()
for w, h, style in zip(ws, hs, styles):
  ret, = plt.plot(
      ts,
      np.exp2(-ts[:, np.newaxis] / h) * w,
      linewidth=style[1],
      linestyle=style[0],
      label=f'{h=:,g} hours')
  plt.plot(ts[0], w, marker='o', color=ret.get_color())

pRecalls = [ebisu.predictRecall(m, now=3600e3 * t, logDomain=False) for t in ts]
plt.plot(ts, pRecalls, 'k:', alpha=0.6, linewidth=4, label=f'Prob. Recall, q={power}')

plt.grid()
plt.legend()

plt.xlabel('hours since last review')
plt.ylabel('recall probability')
plt.title('Power law recall probability from ensemble of exponentials')

plt.xlim([-10, 210])
plt.tight_layout()

plt.savefig('leaky-integrators-precall.png', dpi=300)
plt.savefig('leaky-integrators-precall.svg')

plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.xlim([1e-2 * .5, 2 * 1e4])
plt.ylim([.008, 1.33])
ax = plt.gca()
fixup = lambda vec: [re.sub(r'.0$', '', f'{x:,}') for x in vec]
ax.set_xticklabels(fixup(ax.get_xticks()))
ax.set_yticklabels(fixup(ax.get_yticks()))

plt.savefig('leaky-integrators-precall-loglog.png', dpi=300)
plt.savefig('leaky-integrators-precall-loglog.svg')
