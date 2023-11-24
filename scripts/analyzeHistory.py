import ebisu

import json
from tqdm import tqdm  #type:ignore
import numpy as np
import pylab as plt  # type:ignore
from utils import binomialLogProbabilityFocal, convertAnkiResultToBinomial, noisyLogProbabilityFocal, printableList, sqliteToDf, traintest, clipclim

plt.style.use('ggplot')
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()

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
            ebisu.modelToPercentileDecay(modelsDb[(cardNum, modelNum, quizNum)])
            for modelNum in range(len(initModels))
        ])
      for quizNum, t in enumerate(card.dts_hours):
        oldModels = initModels if quizNum == 0 else [
            modelsDb[(cardNum, modelNum, quizNum - 1)] for modelNum in range(len(initModels))
        ]
        ps.append([ebisu.predictRecall(m, t) for m in oldModels])

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
    hls.append(ebisu.modelToPercentileDecay(modelsDb[(cardNum, modelNum, numQuizzes - 1)], p))
  return hls


if __name__ == '__main__':
  FOCAL_GAMMA = 2

  df = sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  cards = [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]
  # or
  cards = train

  initModelParams = [
      dict(firstHalflife=50, firstWeight=0.5, initialAlphaBeta=1.25, lastHalflife=10e3),
      dict(firstHalflife=50, firstWeight=0.5, initialAlphaBeta=2, lastHalflife=10e3),
      dict(firstHalflife=100, firstWeight=0.5, lastHalflife=10e3),
  ]

  GRID_MODE = False
  if GRID_MODE:
    FIRST_WEIGHT = 0.5
    abVec = list(np.arange(1.25, 3, .25))
    hlVec = list(range(10, 300, 25))
    initModelParams = [
        dict(
            firstHalflife=hl,
            initialAlphaBeta=ab,
            lastHalflife=10e3,
            firstWeight=FIRST_WEIGHT,
        ) for hl in hlVec for ab in abVec
    ]
  else:
    abVec, hlVec, GRID_MODE, FIRST_WEIGHT = [], [], False, 0

  initModels = [ebisu.initModel(**args) for args in initModelParams]  #type:ignore

  allModels = dict()  # key: (card integer, model number, quiz number)
  allLogliks = dict()
  for cardNum, card in tqdm(enumerate(cards), total=len(cards)):
    models = initModels

    for quizNum, (ankiResult, elapsedTime) in enumerate(zip(card.results, card.dts_hours)):
      resultArgs = convertAnkiResultToBinomial(ankiResult, 'approx')

      newModels = []
      for modelNum, m in enumerate(models):
        key = (cardNum, modelNum, quizNum)

        newModel = ebisu.updateRecall(m, elapsedTime=elapsedTime, **resultArgs)
        newModels.append(newModel)
        allModels[key] = newModel

        if resultArgs['total'] == 1:
          z = resultArgs['successes'] >= 0.5
          q1 = max(resultArgs['successes'], 1 - resultArgs['successes'])
          q0 = resultArgs['q0'] if 'q0' in resultArgs else 1 - q1
          loglik = noisyLogProbabilityFocal(z, q1, q0, ebisu.predictRecall(m, elapsedTime),
                                            FOCAL_GAMMA)
        else:
          loglik = binomialLogProbabilityFocal(resultArgs['successes'], resultArgs['total'],
                                               ebisu.predictRecall(m, elapsedTime), FOCAL_GAMMA)
        allLogliks[key] = loglik
      models = newModels

  # SUMMARY
  summary = np.zeros((len(cards), len(initModels)))
  for (cardNum, modelNum, quizNum), ll in allLogliks.items():
    summary[cardNum, modelNum] += ll

  # DETAILS
  if len(initModelParams) < 10:
    printDetails(cards, models, allModels, allLogliks, outfile='ensemble-compare.txt')

    with open('ensemble-compare-halflives.json', 'w') as fid:
      json.dump(
          {
              str(p): oneModelAllHalflives(allModels, len(cards), p=p, modelNum=0)
              for p in [0.5, 0.8]
          }, fid)

      plt.figure()
      plt.plot(np.array(sorted(summary, key=lambda v: v[0])))
      plt.ylim((-10, 1))
      plt.yticks(np.arange(-10, 0.1, 1))
      plt.legend([f'{m}' for m in initModelParams])
      plt.xlabel('flashcard number')
      plt.ylabel('∑ focal loss')
      plt.title('Ensemble v3 performance for training set')
      plt.savefig('ensemble-compare.png', dpi=300)
      plt.savefig('ensemble-compare.svg')

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
    plt.title(f'Focal loss, v3-ensemble\nall cards in training set, w1={FIRST_WEIGHT}')
    plt.grid(False)
    plt.savefig(f'focal-ensemble-{FIRST_WEIGHT}.png', dpi=300)
    plt.savefig(f'focal-ensemble-{FIRST_WEIGHT}.svg')
