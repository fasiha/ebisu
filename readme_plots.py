import numpy as np
import pylab as plt
import ebisu

plt.ion()

hs = np.logspace(0, 4, 5)
ws = ebisu.ebisu._makeWs(len(hs), 0.1, 'exp')

plt.figure()
plt.plot(hs, ws, 'o-')
plt.xlabel('halflife (hours)')
plt.ylabel('weight (unitless)')
plt.suptitle('Weights per halflife (wMax=0.1)')
plt.title('wmaxMean=0.1, hmax=1e4, n=5')
plt.ylim([0, np.max(plt.ylim())])
plt.grid()
plt.savefig('leaky-integrators-weights.png', dpi=300)
plt.savefig('leaky-integrators-weights.svg')

ts = np.linspace(0, 10000, 100001)

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

arr = np.exp2(-ts[:, np.newaxis] * (1 / hs[np.newaxis, :])) * ws
plt.plot(
    ts, np.max(arr, axis=1), linewidth=4, color='black', alpha=0.66, linestyle=':', label='pRecall')
plt.legend()
plt.xlabel('hours since last review')
plt.ylabel('recall probability')
plt.suptitle('Overall recall probability')
plt.title('wmaxMean=0.1, hmax=1e4, n=5')
plt.grid()
plt.xlim([-10, 260])

plt.savefig('leaky-integrators-precall.png', dpi=300)
plt.savefig('leaky-integrators-precall.svg')

plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.ylim([.05, 1.2])
plt.xlim([1e-1, 1e4])
plt.savefig('leaky-integrators-precall-loglog.png', dpi=300)
plt.savefig('leaky-integrators-precall-loglog.svg')
