import numpy as np
import pylab as plt  # type:ignore
from scipy.stats import beta  # type:ignore

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()
remmax = lambda v: v / max(v)
linestyles = ['--', '-', '-.', ':']

ps = np.linspace(0, 1, 501)
plt.figure()
for i, ab in enumerate([1.25, 2, 5, 10]):
  plt.plot(ps, remmax(beta.pdf(ps, ab, ab)), linestyle=linestyles[i], label=f'α=β={ab}')

plt.legend()
plt.xlabel('Recall probability at halflife')
plt.ylabel('Probability of recall probability')
plt.title('Confidence in recall probability after one half-life')
plt.savefig('figures/betas.png', dpi=300)
plt.savefig('figures/betas.svg')
