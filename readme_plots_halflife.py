import re
import ebisu
import numpy as np
import pylab as plt

from ebisu.gammaDistribution import meanVarToGamma

plt.ion()

norm = lambda v: np.array(v) / np.sum(v)

hs = np.logspace(0, 4, 5)
hl = 100
power = 4
ws = ebisu.ebisu._halflifeToFinalWeight(hl, hs, power)
ws = norm(ws)

m = ebisu.initModel(
    power=power,
    weightsHalflifeGammas=[(w, meanVarToGamma(h, (h * .5)**2)) for w, h in zip(ws, hs)],
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

t = np.logspace(-2, 5, 201)

mc: list[float] = []
mcErr: list[float] = []

extra = dict()
for h in t:
  res = ebisu.predictRecallMonteCarlo(m, now=3600e3 * h, size=100_000, logDomain=False, extra=extra)
  mc.append(res)
  mcErr.append(extra['std'])

semibayes = [
    ebisu.predictRecallSemiBayesian(m, now=3600e3 * h, innerPower=False, logDomain=False) for h in t
]
semibayes2 = [
    ebisu.predictRecallSemiBayesian(m, now=3600e3 * h, innerPower=True, logDomain=False) for h in t
]
approx = [ebisu.predictRecall(m, now=3600e3 * h, logDomain=False) for h in t]

fig, axs = plt.subplots(2, 1)
axs[0].semilogx(t, mc, label='Monte Carlo', linewidth=5, linestyle=':')
l1 = axs[0].semilogx(t, semibayes, label='Semi Bayes 1', alpha=.9)
l2 = axs[0].semilogx(t, semibayes2, label='Semi Bayes 2', linestyle='--')
l3 = axs[0].semilogx(t, approx, label='Full Approx', alpha=0.4, linewidth=3)

axs[1].hlines(1, *axs[0].get_xlim(), linewidth=4, alpha=0.25, color='black')

axs[1].semilogx(
    t,
    np.array(semibayes) / np.array(mc),
    alpha=.9,
    color=l1[0].get_color(),
    label='Semi Bayes 1 / Monte Carlo')
axs[1].semilogx(
    t,
    np.array(semibayes2) / np.array(mc),
    # alpha=.9,
    linestyle='--',
    color=l2[0].get_color(),
    label='Semi Bayes 2 / Monte Carlo')
axs[1].semilogx(
    t,
    np.array(approx) / np.array(mc),
    alpha=0.4,
    linewidth=3,
    color=l3[0].get_color(),
    label='Full Approx / Monte Carlo')
axs[1].set_xlim(axs[0].get_xlim())

for a in axs:
  a.legend()
  a.grid(True)
axs[0].set_ylabel('recall probability')
axs[1].set_ylabel('ratio')
axs[1].set_xlabel('hours since last review')
axs[1].set_ylim(.95, 1.13)
fig.suptitle('Overall recall probability')

fixupx = lambda vec: [re.sub(r'.0$', '', f'{x:,}') for x in vec]
axs[0].set_xticklabels(fixupx(axs[0].get_xticks()))
axs[1].set_xticklabels(fixupx(axs[1].get_xticks()))

plt.tight_layout()
plt.savefig('predictRecall-approx.png', dpi=300)
plt.savefig('predictRecall-approx.svg')
