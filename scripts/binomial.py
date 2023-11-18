import pylab as plt  # type:ignore
import numpy as np
import ebisu.ebisu2beta as ebisu2

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.']

model = (3, 3, 1)
ts = np.linspace(2, 24 * 4) / 24

h = lambda k, n=1: [ebisu2.modelToPercentileDecay(ebisu2.updateRecall(model, k, n, t)) for t in ts]

fig, ax = plt.subplots(1, 2)
ax[0].plot(ts, h(0), color=colors[0], linestyle=linestyles[0], label='0 out of 1')
ax[0].plot(ts, h(1), color=colors[1], linestyle=linestyles[1], label='1 out of 1')

ax[1].plot(ts, h(0, 2), color=colors[2], linestyle=linestyles[0], label='0 out of 2')
ax[1].plot(ts, h(1, 2), color=colors[3], linestyle=linestyles[1], label='1 out of 2')
ax[1].plot(ts, h(2, 2), color=colors[4], linestyle=linestyles[2], label='2 out of 2', linewidth=2)

for a in ax:
  a.hlines([1], 0, ts[-1], ['red'], linewidth=3, alpha=0.3)

ylim = max(ax[1].get_ylim())
for a in ax:
  a.set_ylim([0, ylim])
  a.legend()
  a.set_xlabel('Quiz time (days)')
ax[0].set_ylabel('Mean posterior halflife (days)')

fig.suptitle(f'Comparing binary and binomial quizzes (old halflife 1 day, α=β={model[0]})')
fig.tight_layout()

plt.savefig('figures/binaryBinomial.png', dpi=300)
plt.savefig('figures/binaryBinomial.svg')
