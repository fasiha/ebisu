import numpy as np
import pylab as plt  # type:ignore
from scipy.stats import beta as betarv  # type:ignore

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()


def generatePis(deltaT, alpha=10.0, beta=10.0):
  piT = betarv.rvs(alpha, beta, size=50 * 1000)
  piT2 = piT**deltaT
  plt.hist(piT2, bins=20, label='δ={}'.format(deltaT), alpha=0.25, density=True)


[generatePis(p) for p in [0.3, 1., 3.]]
plt.xlabel('Recall probability')
plt.ylabel('Probability of recall probability')
plt.title('Histograms of $p_t^δ$ for different δ')
plt.legend(loc=0)
plt.savefig('figures/pidelta.svg')
plt.savefig('figures/pidelta.png', dpi=150)
plt.show()
