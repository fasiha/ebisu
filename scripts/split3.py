from scipy.optimize import minimize_scalar  # type:ignore
from typing import Optional, Tuple
import ebisu.ebisu2beta as ebisu2
import betapowerlaw

HOURS_PER_YEAR = 365 * 24
SubModel = Tuple[float, float, float, float]
Model = Tuple[SubModel, SubModel, SubModel]


def norm(v: list[float]) -> list[float]:
  s = sum(v)
  return [x / s for x in v]


def initModel(alphaBeta: float, hlHours: float, w1=0.65, w2=0.3, scale2=2) -> Model:
  ws = norm([w1, w2, 1 - w1 - w2])
  return (
      (ws[0], alphaBeta, alphaBeta, hlHours),
      (ws[1], alphaBeta, alphaBeta, hlHours * scale2),
      (ws[2], alphaBeta, alphaBeta, HOURS_PER_YEAR),
  )


def predictRecall(model: Model, elapsed: float, verbose=False) -> float:
  primary, strength, longterm = model
  if verbose:
    print([
        primary[0] * ebisu2.predictRecall(primary[1:], elapsed, exact=True),
        strength[0] * ebisu2.predictRecall(strength[1:], elapsed, exact=True),
        longterm[0] * betapowerlaw.predictRecall(longterm[1:], elapsed)
    ])
  return (primary[0] * ebisu2.predictRecall(primary[1:], elapsed, exact=True) +
          strength[0] * ebisu2.predictRecall(strength[1:], elapsed, exact=True) +
          longterm[0] * betapowerlaw.predictRecall(longterm[1:], elapsed))


def updateRecall(model: Model,
                 successes: float,
                 total: int,
                 elapsed: float,
                 q0: Optional[float] = None) -> Model:
  scale2 = model[1][-1] / model[0][-1]
  newPrimary = ebisu2.updateRecall(model[0][1:], successes, total, elapsed, q0=q0)
  strength = (*model[1][:-1], newPrimary[-1] * scale2)
  return ((model[0][0], *newPrimary), strength, model[2])


def modelToPercentileDecay(model: Model, percentile=0.5) -> float:
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


def printableSub(m: SubModel, i: int) -> str:
  return f'w={m[0]:g}, α=β={m[1]}, hl={m[-1]}' if i <= 1 else f'w={m[0]:g}, hl={m[-1]}'


def printableModel(m: Model) -> str:
  return '(' + "), (".join([printableSub(v, i) for i, v in enumerate(m)]) + ')'


if __name__ == "__main__":
  ab = 1.25
  initHl = 200
  ts = [initHl, initHl * 10, HOURS_PER_YEAR]

  m = initModel(ab, initHl)
  e = m[0][1:]
  print('split: ', [predictRecall(m, t) for t in ts])
  print('ebis2: ', [ebisu2.predictRecall(e, t, exact=True) for t in ts])
  print('hl: ', [modelToPercentileDecay(m), ebisu2.modelToPercentileDecay(e)])

  m = updateRecall(m, 1, 1, .2)
  e = ebisu2.updateRecall(e, 1, 1, .2)
  print('split: ', [predictRecall(m, t) for t in ts])
  print('ebis2: ', [ebisu2.predictRecall(e, t, exact=True) for t in ts])
  print('hl: ', [modelToPercentileDecay(m), ebisu2.modelToPercentileDecay(e)])

  #
  import pylab as plt  # type:ignore
  import os
  from pathlib import Path
  import numpy as np
  from tqdm import tqdm  #type:ignore
  from utils import binomialLogProbabilityFocal, convertAnkiResultToBinomial, noisyLogProbabilityFocal, printableList, sqliteToDf, traintest, clipclim
  import json

  plt.style.use('ggplot')
  plt.rcParams['svg.fonttype'] = 'none'
  plt.ion()

  FOCAL_GAMMA = 2

  ankiPath = Path(os.path.dirname(os.path.realpath(__file__))) / 'collection-no-fields.anki2'
  df = sqliteToDf(str(ankiPath), True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  cards = [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]
  # or
  cards = train

  initModels: list[Model] = [
      initModel(1.25, 24, w1=0.35, w2=0.35, scale2=5),
      initModel(1.25, 24, w1=0.35, w2=0.35),
      #
      initModel(1.25, 100, w1=0.35, w2=0.35, scale2=5),
      initModel(1.25, 100, w1=0.35, w2=0.35),
      #
      # initModel(1.25, 100, w1=0.6, w2=0.3),
      # initModel(1.25, 100, w1=0.9, w2=0.05),
  ]

  GRID_MODE = False
  # GRID_MODE = True
  if GRID_MODE:
    abVec = list(np.arange(1.25, 2.5, .25))
    hlVec = list(range(10, 200, 20))
    initModels = [initModel(ab, hl, w1=0.35, w2=0.35) for hl in hlVec for ab in abVec]
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

        newModel = updateRecall(m, elapsed=elapsedTime, **resultArgs)
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
    plt.figure()
    plt.plot(np.array(sorted(summary, key=lambda v: v[0])), alpha=0.5)
    plt.legend([printableModel(m) for m in initModels], fontsize="x-small")
    plt.ylim((-10, 1))
    plt.yticks(np.arange(-10, 0.1, 1))
    plt.xlabel('flashcard number')
    plt.ylabel('∑ focal loss')
    plt.title('Split-3-atom performance for training set')
    plt.savefig('split-compare.png', dpi=300)
    plt.savefig('split-compare.svg')

    printDetails(cards, models, allModels, allLogliks, outfile='split-compare.txt')

    with open('split-compare.json', 'w') as fid:
      json.dump(
          {
              str(p): oneModelAllHalflives(allModels, len(cards), p=p, modelNum=0)
              for p in [0.5, 0.8]
          }, fid)

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
    plt.title('Focal loss, Split\nall cards in training set (higher is better)')
    plt.grid(False)
    plt.savefig(f'focal-split.png', dpi=300)
    plt.savefig(f'focal-split.svg')
