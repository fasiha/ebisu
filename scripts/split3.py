from collections import namedtuple
from scipy.optimize import minimize_scalar  # type:ignore
from typing import Optional, Tuple
import ebisu2
import betapowerlaw

HOURS_PER_YEAR = 365 * 24
HOURS_PER_SECOND = 1 / 3600
SubModel = Tuple[float, float, float, float]
Model = Tuple[SubModel, SubModel, SubModel]
Ebisu2Model = Tuple[float, float, float]


def norm(v: list[float]) -> list[float]:
  s = sum(v)
  return [x / s for x in v]


def initModel(alphaBeta: float,
              hlHours: float,
              w1=0.65,
              w2=0.3,
              scale2=2,
              hl3=HOURS_PER_YEAR) -> Model:
  ws = norm([w1, w2, 1 - w1 - w2])
  return (
      (ws[0], alphaBeta, alphaBeta, hlHours),
      (ws[1], alphaBeta, alphaBeta, hlHours * scale2),
      (ws[2], alphaBeta, alphaBeta, hl3),
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


def modelToPercentileDecaySafe(m, *args, **kwargs):
  if type(m[0]) == tuple:
    return modelToPercentileDecay(m, *args, **kwargs)
  return ebisu2.modelToPercentileDecay(m, *args, **kwargs)


def predictRecallSafe(m, *args, **kwargs):
  if type(m[0]) == tuple:
    return predictRecall(m, *args, **kwargs)
  return ebisu2.predictRecall(m, *args, **kwargs, exact=True)


def printDetails(cards, initModels, modelsDb, logLikDb, outfile="out.txt"):
  # key: (card integer, model number, quiz number)
  if outfile:
    print(f'Writing details to {outfile}')
  with open(outfile, 'w') as outfile:
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
                modelToPercentileDecaySafe(modelsDb[(cardNum, modelNum, quizNum)]),
                modelToPercentileDecaySafe(modelsDb[(cardNum, modelNum, quizNum)], .8),
            ],
                          sep='/') for modelNum in range(len(initModels))
        ])
      for quizNum, t in enumerate(card.dts_hours):
        oldModels = initModels if quizNum == 0 else [
            modelsDb[(cardNum, modelNum, quizNum - 1)] for modelNum in range(len(initModels))
        ]
        ps.append([predictRecallSafe(m, t) for m in oldModels])

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


def printableModel(m: Model | Ebisu2Model) -> str:
  if type(m[0]) == tuple:
    return '(' + "), (".join([printableSub(v, i) for i, v in enumerate(m)]) + ')'
  return f'(α=β={m[0]:0.3f}, hl={m[2]:0.3f})'


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
  import time

  plt.style.use('ggplot')
  plt.rcParams['svg.fonttype'] = 'none'
  plt.ion()

  FOCAL_GAMMA = 2
  GRID_MODE = False
  GRID_MODE_EBISU2 = not True
  SAVE_RESULTS = False  # save card-by-card model-by-model results to text file
  PER_QUIZ_DETAILS = True or SAVE_RESULTS  # this will grow memory
  USE_FSRS_DATASET = False
  FSRS_MIN_CARDS = 2  # this many or more total reviews (including the first learn review)
  # FSRS_MIN_DELTA_T_SEC = 0  # this many seconds or more
  FSRS_CARD_PERCENT = 1
  FSRS_USER_PERCENT = .1
  FSRS_SEED = 102
  FSRS_LIMIT_CARDS = 10_000_000

  totalFsrsQuizzes = 186292444
  # `fsrsQuizCumulativeCounts[i]` is the number of cards with number of quizzes `<=i`
  fsrsQuizCumulativeCounts = [
      0, 16826431, 38800236, 60230753, 80187017, 98452344, 114624854, 128063183, 138994472,
      147540415, 153679407
  ]
  maxPossibleFsrsCards = (
      totalFsrsQuizzes -
      (fsrsQuizCumulativeCounts[FSRS_MIN_CARDS -
                                1] if FSRS_MIN_CARDS < len(fsrsQuizCumulativeCounts) else 0))

  aucThresholds = np.linspace(0, 1, 51)

  if USE_FSRS_DATASET:
    import fsrs_anki_20k_reader as fsrs_reader

    def gen():
      Mapped = namedtuple('Mapped', ['results', 'dts_hours', 'key'])

      cardNum = 0
      for card in fsrs_reader.allCards(
          os.path.join(os.getenv('FSRS_PATH', '.'), 'dataset'),
          min_reviews=FSRS_MIN_CARDS,
          # min_delta_t_sec=FSRS_MIN_DELTA_T_SEC,
          card_fraction=FSRS_CARD_PERCENT,
          user_fraction=FSRS_USER_PERCENT,
          seed=FSRS_SEED):
        innerList = [(review.rating, max(1, review.delta_t_sec) * HOURS_PER_SECOND,
                      f'{review.file}:{review.card_id}') for review in card]
        results, dts_hours, card_id = zip(*innerList)
        yield Mapped(results=results, dts_hours=dts_hours, key=card_id)
        cardNum += 1
        if cardNum >= FSRS_LIMIT_CARDS:
          break

    cards = gen()
    numTotalCards = min(FSRS_LIMIT_CARDS,
                        round(maxPossibleFsrsCards * min(FSRS_CARD_PERCENT, FSRS_USER_PERCENT)))
  else:
    ankiPath = Path(os.path.dirname(os.path.realpath(__file__))) / 'collection-no-fields.anki2'
    df = sqliteToDf(str(ankiPath), True)
    print(f'loaded SQL data, {len(df)} rows')

    train, TEST_TRAIN = traintest(df, noPerfectCardsInTraining=False)
    print(f'split flashcards into train/test, {len(train)} cards in train set')

    numTotalCards = len(train)
    cards = train

  initModels: list[Model | Ebisu2Model] = [
      initModel(2.25, 10, w1=0.1, w2=0.4, scale2=2, hl3=365 * 24 * 10),
      initModel(2.25, 20, w1=0.1, w2=0.4, scale2=2, hl3=365 * 24 * 10),
      initModel(2.25, 40, w1=0.1, w2=0.4, scale2=2, hl3=365 * 24 * 10),
      #
      initModel(1.25, 9, w1=0.35, w2=0.35, scale2=5, hl3=365 * 24 * 10),
      # initModel(1.25, 24, w1=0.35, w2=0.35, scale2=5),
      # initModel(1.25, 24, w1=0.35, w2=0.35),
      #
      # initModel(1.25, 100, w1=0.35, w2=0.35, scale2=5),
      # initModel(1.25, 100, w1=0.35, w2=0.35),
      #
      # initModel(1.25, 100, w1=0.6, w2=0.3),
      # initModel(1.25, 100, w1=0.9, w2=0.05),
      # ebisu2.defaultModel(24, 1.25),
      ebisu2.defaultModel(7, 1.5),
      # ebisu2.defaultModel(24, 2.5),
      ebisu2.defaultModel(24 * 7, 1.01),
  ]

  if GRID_MODE:
    if not GRID_MODE_EBISU2:
      abVec = list(np.arange(1.05, 2, .2))
      hlVec = list(np.arange(1, 30, 2.5))
      initModels = [
          initModel(ab, hl, w1=0.35, w2=0.35, scale2=5, hl3=365 * 24 * 10)
          for hl in hlVec
          for ab in abVec
      ]
    else:
      abVec = list(np.arange(1.05, 2.5, .25))
      hlVec = list(range(1, 20, 1))
      initModels = [ebisu2.defaultModel(hl, ab) for hl in hlVec for ab in abVec]
  else:
    abVec, hlVec, GRID_MODE = [], [], False

  allModels = dict()  # key: (card integer, model number, quiz number)
  allLogliks = dict()
  allPrecalls = dict()
  allCards = []

  ignoreAuc = False
  positivePopulation = 0
  negativePopulation = 0
  truePositives = np.zeros((len(aucThresholds), len(initModels)), dtype=int)
  falsePositives = np.zeros((len(aucThresholds), len(initModels)), dtype=int)

  logLossesPerCard: list[np.ndarray] = []  # this stops growing after a while (save memory)
  totalFocalLoss = np.zeros(len(initModels))

  for cardNum, card in tqdm(enumerate(cards), total=numTotalCards):
    models = initModels
    llsPerCard = np.zeros(len(initModels))
    for quizNum, (ankiResult, elapsedTime) in enumerate(zip(card.results, card.dts_hours)):
      resultArgs = convertAnkiResultToBinomial(ankiResult, 'binary')

      newModels = []
      pRecalls: list[float] = list()
      llsPerQuiz: list[float] = list()
      for modelNum, m in enumerate(models):
        key = (cardNum, modelNum, quizNum)

        try:
          newModel = (
              updateRecall(m, elapsed=elapsedTime, **resultArgs)
              if type(m[0]) == tuple else ebisu2.updateRecall(m, tnow=elapsedTime, **resultArgs))
        except Exception as e:
          print(f'ERROR {m=}, {elapsedTime=}, {resultArgs=}, {card=}')
          raise e
        newModels.append(newModel)

        pRecall = (
            predictRecall(m, elapsedTime) if type(m[0]) == tuple else ebisu2.predictRecall(
                m, elapsedTime, exact=True))
        if resultArgs['total'] == 1:
          z = resultArgs['successes'] >= 0.5
          q1 = max(resultArgs['successes'], 1 - resultArgs['successes'])
          q0 = resultArgs['q0'] if 'q0' in resultArgs else 1 - q1
          loglik = noisyLogProbabilityFocal(z, q1, q0, pRecall, FOCAL_GAMMA)
          if not ignoreAuc:
            pRecalls.append(pRecall)
        else:
          ignoreAuc = True
          loglik = binomialLogProbabilityFocal(resultArgs['successes'], resultArgs['total'],
                                               pRecall, FOCAL_GAMMA)

        llsPerQuiz.append(loglik)
        if PER_QUIZ_DETAILS:
          allPrecalls[key] = pRecall
          allLogliks[key] = loglik
          allModels[key] = newModel

      llsPerCard += np.array(llsPerQuiz)
      if not ignoreAuc:
        ps = np.array(pRecalls)
        truePositives += np.logical_and(np.atleast_2d(ps).T > aucThresholds, z).T
        falsePositives += np.logical_and(np.atleast_2d(ps).T > aucThresholds, not z).T
        positivePopulation += z
        negativePopulation += not z

      models = newModels
    if len(logLossesPerCard) < 10_000:
      logLossesPerCard.append(llsPerCard)
    totalFocalLoss += llsPerCard

    if PER_QUIZ_DETAILS:
      allCards.append(card)

    if cardNum % 50_000 == 49_999:
      print(
          f'\n{cardNum+1=}, auc',
          np.abs(
              np.trapz(
                  truePositives / positivePopulation,
                  falsePositives / negativePopulation,
                  axis=0,
              )))

  # SUMMARY
  print('completed cards analysis')
  numTotalCards = cardNum + 1  # update in case we got more or fewer
  logLosses = np.array(logLossesPerCard)

  if not ignoreAuc:
    truePositiveRate = truePositives / positivePopulation
    falsePositiveRate = falsePositives / negativePopulation
    aucs = np.abs(np.trapz(truePositiveRate, falsePositiveRate, axis=0))
    p20s = [
        np.interp(0.2, falsePositiveRate[:, i][::-1], truePositiveRate[:, i][::-1])
        for i in range(falsePositiveRate.shape[1])
    ]

  # DETAILS
  totalFocalLoss = np.sum(logLosses, axis=0)
  if len(initModels) < 10:
    runName = f'{time.time()}'
    with open(f'split3-{runName}.json', 'w') as fid:
      json.dump(
          dict(
              numTotalCards=numTotalCards,
              FOCAL_GAMMA=FOCAL_GAMMA,
              GRID_MODE=GRID_MODE,
              GRID_MODE_EBISU2=GRID_MODE_EBISU2,
              PER_QUIZ_DETAILS=PER_QUIZ_DETAILS,
              SAVE_RESULTS=SAVE_RESULTS,
              USE_FSRS_DATASET=USE_FSRS_DATASET,
              FSRS_MIN_CARDS=FSRS_MIN_CARDS,
              # FSRS_MIN_DELTA_T_SEC=FSRS_MIN_DELTA_T_SEC,
              FSRS_CARD_PERCENT=FSRS_CARD_PERCENT,
              FSRS_USER_PERCENT=FSRS_USER_PERCENT,
              FSRS_SEED=FSRS_SEED,
              FSRS_LIMIT_CARDS=FSRS_LIMIT_CARDS,
              aucThresholds=aucThresholds.tolist(),
              initModels=initModels,
              aucs=aucs.tolist(),
              totalFocalLoss=totalFocalLoss.tolist(),
              truePositiveRate=truePositiveRate.tolist(),
              falsePositiveRate=falsePositiveRate.tolist(),
          ),
          fid,
          indent=1)

    plt.figure()
    plt.plot(np.array(sorted(logLosses, key=lambda v: v[0])), alpha=0.5)
    plt.legend(
        [f'{printableModel(m)} (∑l {tot:0.3g})' for m, tot in zip(initModels, totalFocalLoss)],
        fontsize="x-small")
    plt.ylim((-10, 1))
    plt.yticks(np.arange(-10, 0.1, 1))
    plt.xlabel('flashcard number')
    plt.ylabel('∑ focal loss')
    plt.title('Split-3-atom performance for training set')
    plt.savefig(f'split-compare-{runName}.png', dpi=300)
    plt.savefig(f'split-compare-{runName}.svg')

    # ROC/AUC
    if not ignoreAuc:
      plt.figure()
      plt.plot(falsePositiveRate, truePositiveRate)
      plt.plot([0, 1], [0, 1], 'r--')
      plt.xlabel('false positive rate')
      plt.ylabel('true positive rate')
      plt.legend([f'{printableModel(m)} AUC={a:.3f}' for m, a in zip(initModels, aucs)],
                 fontsize="x-small")
      plt.title('AUC/ROC')
      plt.savefig(f'split-auc-{runName}.png', dpi=300)
      plt.savefig(f'split-auc-{runName}.svg')

    if PER_QUIZ_DETAILS:
      if SAVE_RESULTS:
        printDetails(allCards, models, allModels, allLogliks, outfile='split-compare.txt')
        with open('split-compare.json', 'w') as fid:
          json.dump(
              {
                  str(p): oneModelAllHalflives(allModels, numTotalCards, p=p, modelNum=0)
                  for p in [0.5, 0.8]
              }, fid)

      modelToPrecallRes: list[list[tuple[float, bool]]] = [[] for _ in initModels]
      for (cardNum, modelNum, quizNum), p in allPrecalls.items():
        modelToPrecallRes[modelNum].append((p, cards[cardNum].results[quizNum] > 1))
      modelToCenters = []
      modelToGalef = []
      for l in modelToPrecallRes:
        pX = np.array(l)
        pcounts, pbins = np.histogram(pX[:, 0])
        pcenters = np.diff(pbins) / 2 + pbins[:-1]
        pToBin = np.argmin(np.abs(pcenters - pX[:, 0][:, np.newaxis]), axis=1)
        pres = np.zeros_like(pcenters, dtype=int)
        for bin, res in zip(pToBin, pX[:, 1]):
          pres[bin] += res

        modelToCenters.append(pcenters)
        modelToGalef.append(pres / pcounts)
      plt.figure()
      plt.plot(np.array(modelToCenters).T, np.array(modelToGalef).T)

  if GRID_MODE:

    def extents(f):
      delta = f[1] - f[0]
      return [f[0] - delta / 2, f[-1] + delta / 2]

    plt.figure()
    plt.imshow(
        totalFocalLoss.reshape((len(hlVec), len(abVec))),
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

    plt.figure()
    plt.imshow(
        aucs.reshape((len(hlVec), len(abVec))),
        aspect='auto',
        interpolation='none',
        extent=extents(abVec) + extents(hlVec),
        origin='lower')
    plt.colorbar()
    plt.xlabel('initial α=β')
    plt.ylabel('initial halflife')
    plt.title('AUC')
    plt.grid(False)
    plt.savefig(f'auc-split.png', dpi=300)
    plt.savefig(f'auc-split.svg')

    plt.figure()
    plt.imshow(
        np.array(p20s).reshape((len(hlVec), len(abVec))),
        aspect='auto',
        interpolation='none',
        extent=extents(abVec) + extents(hlVec),
        origin='lower')
    plt.colorbar()
    plt.xlabel('initial α=β')
    plt.ylabel('initial halflife')
    plt.title('TPR @ FPR=0.2')
    plt.grid(False)
