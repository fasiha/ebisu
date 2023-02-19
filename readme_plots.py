import re
import numpy as np
import ebisu
import pylab as plt

plt.ion()

hs = np.logspace(0, 4, 5)
hl = 2
ws = ebisu.ebisu._halflifeToFinalWeight(hl, hs)

plt.figure()
plt.plot(hs, ws, 'o-')
plt.xlabel('halflife (hours)')
plt.ylabel('weight (unitless)')
plt.suptitle(f'Weights per halflife')
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
plt.title('Overall recall probability')
plt.grid()
plt.xlim([-10, 220])
plt.tight_layout()

plt.savefig('leaky-integrators-precall.png', dpi=300)
plt.savefig('leaky-integrators-precall.svg')

plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.ylim([.05, 1.2])
plt.xlim([1e-1, 1e4])

plt.hlines([0.5], np.min(plt.xlim()), hl, color=(.1, .1, .1), alpha=0.4, linestyles='dotted')
plt.vlines([hl], np.min(plt.ylim()), 0.5, color=(.1, .1, .1), alpha=0.4, linestyles='dotted')

ax = plt.gca()
fixup = lambda vec: [re.sub(r'.0$', '', f'{x:,}') for x in vec]
ax.set_xticklabels(fixup(ax.get_xticks()))
ax.set_yticklabels(fixup(ax.get_yticks()))

plt.tight_layout()
plt.savefig('leaky-integrators-precall-loglog.png', dpi=300)
plt.savefig('leaky-integrators-precall-loglog.svg')
