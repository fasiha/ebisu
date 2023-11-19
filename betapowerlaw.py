from tqdm import tqdm  #type:ignore
from scipy.optimize import minimize_scalar  # type:ignore
import ebisu2beta as ebisu2
from scipy.stats import beta as betarv  # type:ignore
import numpy as np
import pylab as plt  # type:ignore
from scipy.special import beta as betafn  # type:ignore

plt.ion()

VIZ = False

alpha, beta = 2, 2
t = 10

bs = betarv.rvs(alpha, beta, size=100_000)

delta = 1.5
prior = bs**np.log2(1 + delta)

p = np.linspace(0, 1, 1001)[1:]
l = np.log2(1 + delta)
pdf = p**(alpha / l - 1) * (1 - p**(1 / l))**(beta - 1) / l / betafn(alpha, beta)
# pdf = (p**(1 / l))**(alpha + 1 - 1) * (1 - p**(1 / l))**(beta - 1) / p / l / betafn(alpha, beta)

if VIZ:
  plt.figure()
  hist = plt.hist(prior, bins=50, density=True, alpha=0.5)
  plt.plot(p, pdf)


def g(x, delta=3):
  return x**(np.log2(1 + delta))


def ginv(y, delta=3):
  return y**(1 / (np.log2(1 + delta)))


def predictRecall(model, elapsed):
  delta = elapsed / model[-1]
  l = np.log2(1 + delta)
  return ebisu2.predictRecall(model, l * model[-1], exact=True)


print([
    np.mean(prior),
    betafn(alpha + l, beta) / betafn(alpha, beta),
    predictRecall((alpha, beta, t), t * delta)
])


def updateRecall(model, result, elapsed):
  delta = elapsed / model[-1]
  l = np.log2(1 + delta)
  return ebisu2.updateRecall(model, result, 1, tnow=l * model[-1], rebalance=False)


def modelToPercentileDecay(model, percentile=0.5):
  logLeft, logRight = 0, 0
  counter = 0
  while predictRecall(model, 10**logLeft) <= 0.5:
    logLeft -= 1
    counter += 1
    if counter >= 20:
      raise Exception('unable to find left bound')

  counter = 0
  while predictRecall(model, 10**logRight) >= 0.5:
    logRight += 1
    counter += 1
    if counter >= 20:
      raise Exception('unable to find right bound')

  res = minimize_scalar(
      lambda h: abs(percentile - predictRecall(model, h)), bounds=[10**logLeft, 10**logRight])
  assert res.success
  return res.x


import typing
import utils
from scipy.special import betaln, logsumexp  # type:ignore
from math import log

ConvertAnkiMode = typing.Literal['approx', 'binary']
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def convertAnkiResultToBinomial(result: int, mode: ConvertAnkiMode) -> dict:
  if mode == 'approx':
    # Try to approximate hard to easy with binomial: this is tricky and ad hoc
    if result == 1:  # fail
      return dict(successes=0, total=1)
    elif result == 2:  # hard
      return dict(successes=1, total=1, q0=0.2)
    elif result == 3:  # good
      return dict(successes=1, total=1)
    elif result == 4:  # easy
      return dict(successes=2, total=2)
    else:
      raise Exception('unknown Anki result')
  elif mode == 'binary':
    # hard or better is pass
    return dict(successes=int(result > 1), total=1)


def binomln(n: int, k: int):
  "Log of scipy.special.binom calculated entirely in the log domain"
  return -betaln(1 + n - k, 1 + k) - log(n + 1)


def _binomialLogProbability(k: int, n: int, alpha: float, beta: float, delta: float) -> float:
  assert k <= n
  failures = n - k
  prefix = binomln(n, k) - betaln(alpha, beta)
  return prefix + logsumexp(
      [binomln(failures, i) + betaln(alpha + delta * (k + i), beta) for i in range(failures + 1)],
      b=[(-1)**i for i in range(failures + 1)])


def _noisyLogProbability(result: typing.Union[int, float], q1: float, q0: float, p: float) -> float:
  z = result >= 0.5
  return log(((q1 - q0) * p + q0) if z else (q0 - q1) * p + (1 - q0))


def printableList(v: list[int | float]) -> str:
  return ", ".join([f'{x:0.1f}' for x in v])


def printDetails(cards, initModels, modelsDb, logLikDb):
  for cardNum, card in enumerate(cards):
    sumLls = [
        sum([ll
             for k, ll in logLikDb.items()
             if k[0] == cardNum and k[1] == modelNum])
        for modelNum in range(len(initModels))
    ]
    print(f'{cardNum}, key={card.key}, lls={printableList(sumLls)}')
    numQuizzes = len([ll for k, ll in logLikDb.items() if k[0] == cardNum and k[1] == 0])
    lls = np.array([[ll
                     for k, ll in logLikDb.items()
                     if k[0] == cardNum and k[2] == quizNum]
                    for quizNum in range(numQuizzes)])
    hls = [[
        modelToPercentileDecay(m)
        for k, m in modelsDb.items()
        if k[0] == cardNum and k[2] == quizNum
    ]
           for quizNum in range(numQuizzes)]
    cumsumLls = np.cumsum(lls, axis=0)
    for indiv, cumulative, res, t, hl in zip(lls, cumsumLls, card.results, card.dts_hours, hls):
      print(
          f'  {res=}, {t:.1f}h, post-hl={printableList(hl)}, ll={printableList(indiv)}, cumulative={printableList(cumulative)}'
      )


def analyzeModelsGrid(cards, initModels, modelsDb, logLikDb, abVec, hlVec):
  # key: (card integer, model number, quiz number)
  sums = np.zeros((len(hlVec), len(abVec)))
  raveled = sums.ravel()
  for (cardNum, modelNum, quizNum), ll in logLikDb.items():
    raveled[modelNum] += ll
  return sums


if __name__ == '__main__':
  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  cards = [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]
  # or
  # cards = train

  initModels: list[tuple[float, float, float]] = [
      (2.0, 2, 10),
      (5, 5, 10),
      (2, 2, 100),  # seems to be the best for a wide range
      (5, 5, 100),
  ]

  abVec, hlVec, GRID_MODE = None, None, False
  GRID_MODE = True
  abVec = list(np.arange(2, 5, .5))
  # abVec = list(range(2, 6))
  hlVec = list(range(10, 1000, 50))
  initModels = [(ab, ab, hl) for hl in hlVec for ab in abVec]

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
          loglik = _noisyLogProbability(z, q1, q0, predictRecall(m, elapsedTime))
        else:
          loglik = _binomialLogProbability(resultArgs['successes'], resultArgs['total'], m[0], m[1],
                                           l)
        allLogliks[key] = loglik
      models = newModels

  # SUMMARY
  summary = np.zeros((len(cards), len(initModels)))
  for (cardNum, modelNum, quizNum), ll in allLogliks.items():
    summary[cardNum, modelNum] += ll

  # DETAILS
  DETAILS, VIZ = True, True
  DETAILS, VIZ = False, False
  if DETAILS:
    printDetails(cards, models, allModels, allLogliks)
  VIZ = True
  if VIZ:
    plt.figure()
    plt.plot(np.array(sorted(summary, key=lambda v: v[2])))
    plt.legend([f'{m}' for m in initModels])

  if GRID_MODE:
    sums = analyzeModelsGrid(cards, initModels, allModels, allLogliks, abVec, hlVec)

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
    plt.xlabel('')
