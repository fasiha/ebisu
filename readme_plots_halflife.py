import re
import ebisu
import numpy as np
import pylab as plt

from ebisu.ebisu import _meanVarToGamma

plt.ion()

n = 5
weights = np.logspace(0, -.5, n)
means = np.logspace(0, 4, n)

m = ebisu.initModel(
    weightsHalflifeGammas=[(w, _meanVarToGamma(m, (m * .5)**2)) for w, m in zip(weights, means)],
    now=0)

if not True:
  args = [((168.33800531142657, 10.746849757942703), 0.0),
          ((266.2011448623055, 6.7716803079048615), 0.0),
          ((161.96002614662905, 1.6221086461114889), 0.0),
          ((74.47528111402488, 0.33831814181165487), 0.0),
          ((34.657557668290885, 0.07114847384043546), 0.0),
          ((16.555840625352076, 0.015481775849237909), 0.0),
          ((8.143284096070321, 0.0034885949137291503), 0.0),
          ((4.30243057061622, 0.0006968135515118837), -0.03790720622799234),
          ((2.835570579436257, 0.00016664726176917382), -0.25872832608891866),
          ((4.0, 4e-05), -2.353669592990435)]
  m = ebisu.initModel(weightsHalflifeGammas=[(np.exp2(w), ab) for ab, w in args], now=0)

t = np.logspace(-2, 5, 601)

mc: list[float] = []
mcErr: list[float] = []

extra = dict()
for h in t:
  res = ebisu.predictRecallMonteCarlo(m, now=3600e3 * h, size=10_000, logDomain=False, extra=extra)
  mc.append(res)
  mcErr.append(extra['std'])

semibayes = [ebisu.predictRecallSemiBayesian(m, now=3600e3 * h, logDomain=False) for h in t]
approx = [ebisu.predictRecall(m, now=3600e3 * h, logDomain=False) for h in t]

plt.figure()
plt.plot(t, mc, label='Monte Carlo', linewidth=5, linestyle=':')
plt.plot(t, semibayes, label='Semi Bayes', alpha=.9)
plt.plot(t, approx, label='Full Approx', alpha=0.6, linewidth=3)

plt.gca().set_xscale("log")
plt.gca().set_yscale("log")

if not True:
  plt.ylim([.09, 1.2])
  plt.xlim([.05, 10e3])

  plt.xlabel('hours since last review')
  plt.ylabel('recall probability')
  plt.title('Overall recall probability')
  plt.legend()
  plt.grid()

  ax = plt.gca()
  fixupx = lambda vec: [re.sub(r'.0$', '', f'{x:,}') for x in vec]
  ax.set_xticklabels(fixupx(ax.get_xticks()))

  fixupy = lambda vec: [re.sub(r'.0$', '', f'{x:.1f}') for x in vec]
  ys = np.linspace(.1, 1, 10)
  ax.set_yticks(ys)
  ax.set_yticklabels(fixupy(ys))

  plt.tight_layout()
  plt.savefig('predictRecall-approx.png', dpi=300)
  plt.savefig('predictRecall-approx.svg')

plt.figure()
plt.semilogx(t, np.array(semibayes) / np.array(mc), label='SemiBayes vs MC')
plt.semilogx(t, np.array(approx) / np.array(mc), label='FullApprox vs MC')
plt.legend()
plt.grid()

relerr = lambda act, exp: np.abs((np.array(act) - np.array(exp)) / np.array(exp))
plt.figure()
plt.loglog(t, relerr(semibayes, mc), label='SemiBayes vs MC')
plt.loglog(t, relerr(approx, mc), label='FullApprox vs MC')
plt.legend()
plt.grid()
