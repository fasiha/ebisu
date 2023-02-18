import ebisu
import numpy as np
import pylab as plt

plt.ion()

initHl = 7.0
elapseds = np.linspace(0, 30, 31)[1:]


def helper(stdScale: float, result: int, n=1) -> list[float]:
  m = ebisu.initModel(
      weightsHalflifeGammas=[(1.0, ebisu.ebisu._meanVarToGamma(initHl, initHl * stdScale))], now=0)
  return [
      ebisu.hoursForRecallDecay(ebisu.updateRecall(m, result, n, now=3600e3 * elapsed))
      for elapsed in elapseds
  ]


plt.figure()
lines = plt.plot(elapseds, helper(1.5, 1), 'x-', label='Pass; high σ')
lines += plt.plot(elapseds, helper(.5, 1), 'x--', label='Pass; low σ')

plt.plot(elapseds, helper(1.5, 0), 'o-', color=lines[0].get_color(), label='Fail; high σ')
plt.plot(elapseds, helper(0.5, 0), 'o--', color=lines[1].get_color(), label='Fail; low σ')

plt.hlines([7], 0, elapseds[-1], ['red'], linewidth=3, alpha=0.5)

plt.grid()
plt.xlabel('Quiz time (hours after last seen)')
plt.ylabel('Posterior halflife (hours)')
plt.title('Mean posterior halflife (previously 7 hours)')
plt.legend()
plt.tight_layout()
plt.savefig('binom-updates.png', dpi=300)
plt.savefig('binom-updates.svg')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.']
fig, ax = plt.subplots(1, 2)
ax[0].plot(elapseds, helper(0.5, 1), color=colors[0], linestyle=linestyles[0], label='1 out of 1')
ax[0].plot(elapseds, helper(0.5, 0), color=colors[1], linestyle=linestyles[1], label='0 out of 1')
ax[0].hlines([7], 0, elapseds[-1], ['red'], linewidth=3, alpha=0.5)

ax[1].plot(
    elapseds, helper(0.5, 2, 2), color=colors[2], linestyle=linestyles[0], label='2 out of 2')
ax[1].plot(
    elapseds, helper(0.5, 1, 2), color=colors[3], linestyle=linestyles[1], label='1 out of 2')
ax[1].plot(
    elapseds, helper(0.5, 0, 2), color=colors[4], linestyle=linestyles[2], label='0 out of 2')
ax[1].hlines([7], 0, elapseds[-1], ['red'], linewidth=3, alpha=0.5)

ylim = ax[1].get_ylim()
ax[0].set_ylim(*ylim)
for a in ax:
  a.legend()
  a.grid()
  a.set_xlabel('Quiz time (hours)')
ax[0].set_ylabel('Mean posterior halflife (hours)')

fig.suptitle('Comparing binary to n=2 binomial quizzes (old halflife 7 hours)')
fig.tight_layout()

plt.savefig('binom-n2-updates.png', dpi=300)
plt.savefig('binom-n2-updates.svg')
