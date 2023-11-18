import pylab as plt  # type:ignore
import numpy as np
import ebisu.ebisu2beta as ebisu2
import ebisu
import re

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.']
sizes = [1, 1, 1.5, 1.5, 2]

model = ebisu.initModel(10, firstWeight=0.5)
ts = np.logspace(-1, 6, 501)

plt.figure()

plt.plot(ts, [ebisu.predictRecall(model, t) for t in ts], 'k:', label=f'Ensemble', linewidth=4)

for idx, atom in enumerate(model):
  tup = (atom.alpha, atom.beta, atom.time)
  w = 2**atom.log2weight
  # w = 2**atom.log2weight / (2**model[0].log2weight)
  plt.plot(
      ts, [w * ebisu2.predictRecall(tup, t, exact=True) for t in ts],
      label=f'{atom.time:,g} hours, weight {100*w:.1f}%',
      color=colors[idx],
      linestyle=linestyles[idx % 3],
      linewidth=sizes[idx % 5])

plt.legend()
plt.xlabel('Hours since last review')
plt.ylabel('Recall probability')
plt.title('Power law recall from exponential ensemble')

plt.xlim([-5, 205])

plt.savefig('figures/power-linear.png', dpi=300)
plt.savefig('figures/power-linear.svg')

plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.ylim([.008, 1.2])
plt.xlim([.06, 4e5])

plt.ylim([.002, 2])
plt.xlim([.06, 9e5])

plt.legend(loc='lower left')

fixup = lambda vec: [re.sub(r'.0$', '', f'{x:,}') for x in vec]
ax = plt.gca()
ax.set_xticklabels(fixup(ax.get_xticks()))
ax.set_yticklabels(fixup(ax.get_yticks()))

t = ax.text(
    .4,
    1.5,
    "Full recall",
    ha="center",
    va="center",
    size=10,
    alpha=0.5,
    bbox=dict(boxstyle="darrow,pad=0.3", ec='black', lw=.5, alpha=0.5, fc="lightgreen"))

t = ax.text(
    1e3,
    .5,
    "              Power law            ",
    rotation=-30,
    ha="center",
    va="center",
    size=10,
    alpha=0.5,
    horizontalalignment='center',
    bbox=dict(boxstyle="darrow,pad=0.3", ec='black', lw=.5, alpha=0.5, fc="lightblue"))
t = ax.text(
    6e5,
    .01,
    "Exponential",
    rotation=-70,
    ha="center",
    va="center",
    size=10,
    alpha=0.5,
    bbox=dict(boxstyle="darrow,pad=0.3", ec='black', lw=.5, alpha=0.5, fc="yellow"))

plt.savefig('figures/power-log.png', dpi=300)
plt.savefig('figures/power-log.svg')
