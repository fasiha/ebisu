"""
Needs numpy, scipy, pandas, matplotlib, tqdm, and of course ebisu.
"""

import json
from tqdm import tqdm  #type:ignore
from scipy.optimize import minimize_scalar  # type:ignore
from scipy.stats import beta as betarv, binom as binomrv  # type:ignore
from scipy.special import beta as betafn  # type:ignore
import pylab as plt  # type:ignore
import numpy as np

import ebisu.ebisu2beta as ebisu2
from utils import binomialLogProbabilityFocal, convertAnkiResultToBinomial, noisyLogProbabilityFocal, printableList, sqliteToDf, traintest, clipclim

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()


def predictRecall(model, elapsed):
  delta = elapsed / model[-1]
  l = np.log2(1 + delta)
  return ebisu2.predictRecall(model, l * model[-1], exact=True)


def confirmMath(VIZ=False):
  alpha, beta, t = 2, 2, 10
  elapsed = 15

  bs = betarv.rvs(alpha, beta, size=100_000)

  delta = elapsed / t
  prior = bs**np.log2(1 + delta)

  p = np.linspace(0, 1, 1001)[1:]
  l = np.log2(1 + delta)
  pdf = p**(alpha / l - 1) * (1 - p**(1 / l))**(beta - 1) / l / betafn(alpha, beta)

  if VIZ:
    plt.figure()
    plt.hist(prior, bins=50, density=True, alpha=0.5)
    plt.plot(p, pdf)

  def g(x, delta=3):
    return x**(np.log2(1 + delta))

  def ginv(y, delta=3):
    return y**(1 / (np.log2(1 + delta)))

  assert (np.isclose(g(ginv(.3)), .3))
  assert (np.isclose(g(ginv(1.3)), 1.3))

  results = ([
      np.mean(prior),
      betafn(alpha + l, beta) / betafn(alpha, beta),
      predictRecall((alpha, beta, t), elapsed)
  ])
  assert np.allclose(results, results[0], rtol=1e-2)


confirmMath()


def updateRecall(model, result, total, elapsed, q0=None):
  delta = elapsed / model[-1]
  l = np.log2(1 + delta)
  return ebisu2.updateRecall(model, result, total, tnow=l * model[-1], rebalance=False, q0=q0)


def modelToPercentileDecay(model, percentile=0.5):
  logLeft, logRight = 0, 0
  counter = 0
  while predictRecall(model, 10**logLeft) <= percentile:
    logLeft -= 1
    counter += 1
    if counter >= 20:
      raise Exception('unable to find left bound')

  counter = 0
  while predictRecall(model, 10**logRight) >= percentile:
    logRight += 1
    counter += 1
    if counter >= 20:
      raise Exception('unable to find right bound')

  res = minimize_scalar(
      lambda h: abs(percentile - predictRecall(model, h)), bounds=[10**logLeft, 10**logRight])
  assert res.success
  return res.x


MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def printDetails(cards, initModels, modelsDb, logLikDb, outfile=None):
  # key: (card integer, model number, quiz number)
  if outfile:
    print(f'Writing details to {outfile}')
  with open(outfile, 'w') if outfile else None as outfile:
    for cardNum, card in tqdm(enumerate(cards), total=len(cards)):
      sumLls = [
          sum([ll
               for k, ll in logLikDb.items()
               if k[0] == cardNum and k[1] == modelNum])
          for modelNum in range(len(initModels))
      ]
      print(f'{cardNum}, key={card.key}, lls={printableList(sumLls)}', file=outfile)
      numQuizzes = len([ll for k, ll in logLikDb.items() if k[0] == cardNum and k[1] == 0])

      lls = []
      hls = []
      ps = []
      for quizNum in range(numQuizzes):
        lls.append([logLikDb[(cardNum, modelNum, quizNum)] for modelNum in range(len(initModels))])
        hls.append([
            printableList([
                modelToPercentileDecay(modelsDb[(cardNum, modelNum, quizNum)]),
                modelToPercentileDecay(modelsDb[(cardNum, modelNum, quizNum)], .8),
            ],
                          sep='/') for modelNum in range(len(initModels))
        ])
      for quizNum, t in enumerate(card.dts_hours):
        oldModels = initModels if quizNum == 0 else [
            modelsDb[(cardNum, modelNum, quizNum - 1)] for modelNum in range(len(initModels))
        ]
        ps.append([predictRecall(m, t) for m in oldModels])

      cumsumLls = np.cumsum(lls, axis=0)
      for indiv, cumulative, res, t, hl, p in zip(lls, cumsumLls, card.results, card.dts_hours, hls,
                                                  ps):
        print(
            f'  {res=}, {t:.1f}h, p={printableList(p, True)}, hl={printableList(hl)}, ll={printableList(indiv)}, cumulative={printableList(cumulative)}',
            file=outfile)


def analyzeModelsGrid(logLikDb, abVec, hlVec):
  # key: (card integer, model number, quiz number)
  sums = np.zeros((len(hlVec), len(abVec)))
  raveled = sums.ravel()
  for (cardNum, modelNum, quizNum), ll in logLikDb.items():
    raveled[modelNum] += ll
  return sums


def oneModelAllHalflives(modelsDb, numCards, p=0.5, modelNum=0):
  assert 0 < p < 1
  hls = []
  for cardNum in range(numCards):
    numQuizzes = next(filter(lambda q: (cardNum, modelNum, q) not in allModels, range(1000)))
    hls.append(modelToPercentileDecay(modelsDb[(cardNum, modelNum, numQuizzes - 1)], p))
  return hls


if __name__ == '__main__':
  FOCAL_GAMMA = 2

  import os
  from pathlib import Path
  ankiPath = Path(os.path.dirname(os.path.realpath(__file__))) / 'collection-no-fields.anki2'
  df = sqliteToDf(str(ankiPath), True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  cards = [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]
  # or
  cards = train

  initModels: list[tuple[float, float, float]] = [
      # (2.0, 2, 10), # leads to very large halflives
      (1.25, 1.25, 125),
      # (2, 2, 125),
      # (3, 3, 125),
      # # (2, 2, 10),
      # (3, 3, 10),
      # (5, 5, 10),
  ]

  GRID_MODE = False
  if GRID_MODE:
    abVec = list(np.arange(1.25, 3, .25))
    # abVec = list(range(2, 6))
    hlVec = list(range(10, 300, 25))
    initModels = [(ab, ab, hl) for hl in hlVec for ab in abVec]
  else:
    abVec, hlVec, GRID_MODE = [], [], False

  allModels = dict()  # key: (card integer, model number, quiz number)
  allLogliks = dict()
  for cardNum, card in tqdm(enumerate(cards), total=len(cards)):
    models = initModels

    for quizNum, (ankiResult, elapsedTime) in enumerate(zip(card.results, card.dts_hours)):
      resultArgs = convertAnkiResultToBinomial(ankiResult, 'approx')

      newModels = []
      for modelNum, m in enumerate(models):
        key = (cardNum, modelNum, quizNum)

        delta = elapsedTime / m[-1]
        l = np.log2(1 + delta)
        newModel = ebisu2.updateRecall(m, tnow=l * m[-1], rebalance=False, **resultArgs)
        newModels.append(newModel)
        allModels[key] = newModel

        if resultArgs['total'] == 1:
          z = resultArgs['successes'] >= 0.5
          q1 = max(resultArgs['successes'], 1 - resultArgs['successes'])
          q0 = resultArgs['q0'] if 'q0' in resultArgs else 1 - q1
          loglik = noisyLogProbabilityFocal(z, q1, q0, predictRecall(m, elapsedTime), FOCAL_GAMMA)
        else:
          loglik = binomialLogProbabilityFocal(resultArgs['successes'], resultArgs['total'],
                                               predictRecall(m, elapsedTime), FOCAL_GAMMA)
        allLogliks[key] = loglik
      models = newModels

  # SUMMARY
  summary = np.zeros((len(cards), len(initModels)))
  for (cardNum, modelNum, quizNum), ll in allLogliks.items():
    summary[cardNum, modelNum] += ll

  # DETAILS
  if len(initModels) < 10:
    printDetails(cards, models, allModels, allLogliks, outfile='beta-powerlaw-compare.txt')

    with open('beta-powerlaw-compare.json', 'w') as fid:
      json.dump(
          {
              str(p): oneModelAllHalflives(allModels, len(cards), p=p, modelNum=0)
              for p in [0.5, 0.8]
          }, fid)

      plt.figure()
      plt.plot(np.array(sorted(summary, key=lambda v: v[0])))
      plt.legend([f'{m}' for m in initModels])
      plt.ylim((-10, 1))
      plt.yticks(np.arange(-10, 0.1, 1))
      plt.xlabel('flashcard number')
      plt.ylabel('∑ focal loss')
      plt.title('Powerlaw-beta performance for training set')
      plt.savefig('beta-powerlaw-compare.png', dpi=300)
      plt.savefig('beta-powerlaw-compare.svg')

  if GRID_MODE:
    sums = analyzeModelsGrid(allLogliks, abVec, hlVec)

    def extents(f):
      delta = f[1] - f[0]
      return [f[0] - delta / 2, f[-1] + delta / 2]

    plt.figure()
    plt.imshow(
        sums,
        aspect='auto',
        interpolation='none',
        extent=extents(abVec) + extents(hlVec),
        origin='lower')
    plt.colorbar()
    plt.xlabel('initial α=β')
    plt.ylabel('initial halflife')
    plt.title('Focal loss, Beta-power-law\nall cards in training set (higher is better)')
    plt.grid(False)
    plt.savefig(f'focal-betapowerlaw.png', dpi=300)
    plt.savefig(f'focal-betapowerlaw.svg')
