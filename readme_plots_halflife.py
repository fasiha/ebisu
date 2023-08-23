import ebisu3beta
import ebisu2beta as ebisu2

import numpy as np
import pylab as plt

plt.style.use('ggplot')
plt.ion()
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec

ts = np.logspace(1, 6, 1001)
w1 = 0.75
nAtoms = 4
m = ebisu3beta.initModel(100, 2.0, n=nAtoms, w1=w1, now=0)

hours = [
    49.3,
    50.4,
    118.4,
    91.3,
    76.2,
    192.3,
    119.2,
    386.9,
    299.3,
    419.9,
    1331.3,
    5076.4,
]
models = [
    [
        dict(w=0.44, h=114.98, a=2, b=2),
        dict(w=0.37, h=578.9, a=2, b=2),
        dict(w=0.13, h=3179.11, a=2, b=2),
        dict(w=0.044, h=17799.68, a=2, b=2),
        dict(w=0.015, h=100016.89, a=2, b=2)
    ],
    [
        dict(w=0.39, h=130.61, a=2, b=2),
        dict(w=0.4, h=595.8, a=2, b=2),
        dict(w=0.15, h=3196.31, a=2, b=2),
        dict(w=0.05, h=17816.93, a=2, b=2),
        dict(w=0.017, h=100034.15, a=2, b=2)
    ],
    [
        dict(w=0.18, h=231.94, a=2, b=2),
        dict(w=0.48, h=696.64, a=2, b=2),
        dict(w=0.23, h=3297.66, a=2, b=2),
        dict(w=0.083, h=17918.4, a=2, b=2),
        dict(w=0.028, h=100135.64, a=2, b=2)
    ],
    [
        dict(w=0.15, h=260.29, a=2, b=2),
        dict(w=0.48, h=726.75, a=2, b=2),
        dict(w=0.25, h=3328.71, a=2, b=2),
        dict(w=0.089, h=17949.66, a=2, b=2),
        dict(w=0.031, h=100166.94, a=2, b=2)
    ],
    [
        dict(w=0.14, h=284.65, a=2, b=2),
        dict(w=0.48, h=752.02, a=2, b=2),
        dict(w=0.26, h=3354.65, a=2, b=2),
        dict(w=0.095, h=17975.76, a=2, b=2),
        dict(w=0.032, h=100193.08, a=2, b=2)
    ],
    [
        dict(w=0.11, h=340.32, a=2, b=2),
        dict(w=0.47, h=813.44, a=2, b=2),
        dict(w=0.28, h=3419.53, a=2, b=2),
        dict(w=0.11, h=18041.49, a=2, b=2),
        dict(w=0.037, h=100258.96, a=2, b=2)
    ],
    [
        dict(w=0.093, h=378.55, a=2, b=2),
        dict(w=0.46, h=852.54, a=2, b=2),
        dict(w=0.3, h=3459.95, a=2, b=2),
        dict(w=0.11, h=18082.28, a=2, b=2),
        dict(w=0.04, h=100299.83, a=2, b=2)
    ],
    [
        dict(w=0.035, h=711.63, a=2, b=2),
        dict(w=0.35, h=1177.45, a=2, b=2),
        dict(w=0.38, h=3789.38, a=2, b=2),
        dict(w=0.17, h=18413.48, a=2, b=2),
        dict(w=0.061, h=100631.4, a=2, b=2)
    ],
    [
        dict(w=0.03, h=804.48, a=2, b=2),
        dict(w=0.33, h=1271.89, a=2, b=2),
        dict(w=0.39, h=3889.36, a=2, b=2),
        dict(w=0.18, h=18515.56, a=2, b=2),
        dict(w=0.066, h=100733.93, a=2, b=2)
    ],
    [
        dict(w=0.024, h=931.9, a=2, b=2),
        dict(w=0.3, h=1401.9, a=2, b=2),
        dict(w=0.41, h=4028.52, a=2, b=2),
        dict(w=0.2, h=18658.5, a=2, b=2),
        dict(w=0.073, h=100877.7, a=2, b=2)
    ],
    [
        dict(w=0.0066, h=2069.56, a=2, b=2),
        dict(w=0.14, h=2501.96, a=2.1, b=2.1),
        dict(w=0.42, h=5150.34, a=2, b=2),
        dict(w=0.31, h=19794.93, a=2, b=2),
        dict(w=0.12, h=102017.9, a=2, b=2)
    ],
    [
        dict(w=0.00091, h=6302.71, a=2, b=2),
        dict(w=0.027, h=6596.96, a=2.1, b=2.1),
        dict(w=0.24, h=9353.48, a=2.1, b=2.1),
        dict(w=0.47, h=24099.54, a=2, b=2),
        dict(w=0.26, h=106359.26, a=2, b=2)
    ],
]

plt.figure()

predBayes = [
    ebisu3beta.predictRecall(m, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts
]
predApprox = [
    ebisu3beta.predictRecallApprox(m, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts
]
plt.semilogx(ts, np.array(predApprox) / np.array(predBayes), label='Init')
"""
[
        dict(w=0.44, h=114.98, a=2, b=2),
        dict(w=0.37, h=578.9, a=2, b=2),
        dict(w=0.13, h=3179.11, a=2, b=2),
        dict(w=0.044, h=17799.68, a=2, b=2),
        dict(w=0.015, h=100016.89, a=2, b=2)
    ],
"""
prevHour = None
for (model, hour) in zip(models, hours):
  ws = [m['w'] for m in model]
  m.log2weights = np.log2(np.array(ws) / sum(ws))
  m.models = [(m['a'], m['b'], m['h']) for m in model]

  due = ebisu3beta.hoursForRecallDecay(m, 0.8)
  predBayes = [
      ebisu3beta.predictRecall(m, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts
  ]
  predApprox = [
      ebisu3beta.predictRecallApprox(m, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts
  ]

  if prevHour is None or hour > prevHour * 1.5:
    plt.semilogx(ts, np.array(predApprox) / np.array(predBayes), label=f'Due {due:.0f} h')
    prevHour = hour

plt.legend()
plt.title('Ratio of approx/Bayesian recall probability, mixture-of-Betas')
plt.xlabel('Elapsed time (hours)')
plt.ylabel('approxRredictRecall(h) / predictRecall(h)')
plt.savefig('betas-approx-recall.png', dpi=300)

###
import utils
import ebisu
from ankiCompare import convertAnkiResultToBinomial

df = utils.sqliteToDf('collection.anki2', True)
print(f'loaded SQL data, {len(df)} rows')

train, TEST_TRAIN = utils.traintest(df, noPerfectCardsInTraining=False)
print(f'split flashcards into train/test, {len(train)} cards in train set')

card, = [t for t in train if t.key == 1300038030426]
now = 0

bModel = ebisu3beta.initModel(100, 2.0, n=nAtoms, w1=w1, now=now)
gModel = ebisu.initModel(halflife=100, now=now, power=1, n=nAtoms, stdScale=1, w1=w1)

betaCompare = np.array([
    (ebisu3beta.predictRecall(bModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
     ebisu3beta.predictRecallApprox(bModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False))
    for t in ts
])
gammaCompare = np.array([(
    ebisu.predictRecallApprox(gModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
    ebisu.predictRecall(gModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
) for t in ts])

bDue = ebisu3beta.hoursForRecallDecay(bModel, .8)
gDue = ebisu.hoursForRecallDecay(gModel, .8)

fig, (a1, a2) = plt.subplots(2, 1)
a1.semilogx(ts, betaCompare[:, 1] / betaCompare[:, 0], label=f'Init: {bDue:.0f} h')
a2.semilogx(ts, gammaCompare[:, 0] / gammaCompare[:, 1], label=f'Init: {gDue:.0f} h')

prevHour = None
for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
  resultArgs = convertAnkiResultToBinomial(ankiResult, 'approx')
  now += elapsedTime * MILLISECONDS_PER_HOUR
  bModel = ebisu3beta.updateRecall(
      bModel, **resultArgs, now=now, updateThreshold=.99, weightThreshold=0.49)
  gModel = ebisu.updateRecall(
      gModel, **resultArgs, now=now, updateThreshold=.99, weightThreshold=0.49)

  if prevHour is None or elapsedTime > prevHour * 1.5:
    prevHour = elapsedTime
    bDue = ebisu3beta.hoursForRecallDecay(bModel, 0.8)
    gDue = ebisu.hoursForRecallDecay(gModel, 0.8)

    betaCompare = np.array([
        (ebisu3beta.predictRecall(bModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
         ebisu3beta.predictRecallApprox(
             bModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False)) for t in ts
    ])
    gammaCompare = np.array([(
        ebisu.predictRecallApprox(gModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
        ebisu.predictRecall(gModel, now=now + t * MILLISECONDS_PER_HOUR, logDomain=False),
    ) for t in ts])
    a1.semilogx(ts, betaCompare[:, 1] / betaCompare[:, 0], label=f'Due {bDue:.0f} h')
    a2.semilogx(ts, gammaCompare[:, 0] / gammaCompare[:, 1], label=f'Due {gDue:.0f} h')

a1.legend()
a2.legend()
fig.suptitle('Predict recall, approx/exact')
a1.set_ylabel('beta mixture')
a2.set_ylabel('gamma mixture')
a2.set_xlabel('hours since review')

##
from ebisu import gammaPredictRecall

bModel = ebisu3beta.initModel(100, 2.0, n=nAtoms, w1=w1, now=0)
gModel = ebisu.initModel(halflife=100, now=0, power=1, n=nAtoms, stdScale=1, w1=w1)

psBeta = np.array([[ebisu2.predictRecall(m, x, exact=True) for x in ts] for m in bModel.models])
weightedPs = np.exp2(bModel.log2weights)[:, np.newaxis] * psBeta

psGamma = np.array(
    [[gammaPredictRecall(a, b, x, logDomain=False) for x in ts] for a, b in gModel.halflifeGammas])
weightedPsGamma = np.exp2(gModel.log2weights)[:, np.newaxis] * psGamma

fig, (a1, a2) = plt.subplots(2, 1)
a1.loglog(
    ts,
    [ebisu3beta.predictRecall(bModel, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts],
    'k-')
a1.loglog(ts, weightedPs.T)

a2.loglog(
    ts,
    [ebisu.predictRecallApprox(gModel, now=t * MILLISECONDS_PER_HOUR, logDomain=False) for t in ts],
    'k-')
a2.loglog(ts, weightedPsGamma.T)
a2.set_ylim((1e-3, 2))
a1.set_ylim((1e-3, 2))
