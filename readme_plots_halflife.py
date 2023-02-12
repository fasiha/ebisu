import ebisu
import numpy as np
import pylab as plt

plt.ion()

# m = ebisu.initModel(2.0, finalHalflife=1e4, n=1, now=0)
m = ebisu.initModel(2.0, finalHalflife=1e4, n=5, now=0)
t = np.logspace(-2, 5, 201)

mc = [ebisu.predictRecallMonteCarlo(m, now=3600e3 * h, size=1_000, logDomain=False) for h in t]
semibayes = [ebisu.predictRecallSemiBayesian(m, now=3600e3 * h, logDomain=False) for h in t]
approx = [ebisu.predictRecall(m, now=3600e3 * h, logDomain=False) for h in t]

plt.figure()
plt.loglog(t, approx, label='Approx')
plt.loglog(t, mc, label='Monte Carlo')
plt.loglog(t, semibayes, label='Semi Bayes', linestyle='--')

plt.ylim([.09, 1.2])
plt.legend()
