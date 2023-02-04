from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore
import numpy as np
import pylab as plt  # type:ignore
import typing

import ebisu
import ebisu3gammas
import utils

plt.ion()

ConvertAnkiMode = typing.Literal['approx', 'binary']
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec

wmaxs = np.linspace(0, 1, 101)
n = 10
hs = np.logspace(0, 4, n)


def convertAnkiResultToBinomial(result: int, mode: ConvertAnkiMode) -> typing.Tuple[int, int]:
  if mode == 'approx':
    # Try to approximate hard to easy with binomial: this is tricky and ad hoc
    if result == 1:  # fail
      return (0, 1)
    elif result == 2:  # hard
      return (1, 2)
    elif result == 3:  # good
      return (1, 1)
    elif result == 4:  # easy
      return (2, 2)
    else:  # easy
      raise Exception('unknown Anki result')
  elif mode == 'binary':
    # hard or better is pass
    return (int(result > 1), 1)


if __name__ == '__main__':
  origNow = ebisu.timeMs()

  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  card = None
  models = None
  modelsPerIter = None

  ePredictor, v3Predictor = [
      lambda model, elapsedTime: ebisu.predictRecall(
          model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
      lambda model, elapsedTime: ebisu3gammas.predictRecall(
          model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
  ]
  eUpdator, v3Updator = [
      lambda model, s, t, now: ebisu.updateRecall(model, successes=s, total=t, now=now),
      lambda model, s, t, now: ebisu3gammas.updateRecallHistory(
          ebisu3gammas.updateRecall(model, successes=s, total=t, now=now), size=1000),
  ]

  betasPredictor = lambda model, elapsedTime: ebisu.predictRecallBetas(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  betasUpdator = lambda model, s, t, now: ebisu.updateRecallBetas(
      model, successes=s, total=t, now=now)

  fracs = [0.8]
  # fracs = [1.0]
  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  # fracs = [0.8]
  for card in [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]:
    # for card in train:
    hlMeanStd = (24., 24 * .7)
    boostMeanStd = (3, 3 * .7)
    convertMode: ConvertAnkiMode = 'binary'
    now = origNow

    wmaxMean, initHlMean = 0.5, None
    wmaxMean, initHlMean = None, 12.0
    intermediate = not False

    models = [
        ebisu.initModel(wmaxMean=wmaxMean, initHlMean=initHlMean, now=now),
        ebisu3gammas.initModel(
            initHlMean=hlMeanStd[0],
            boostMean=boostMeanStd[0],
            initHlStd=hlMeanStd[1],
            boostStd=boostMeanStd[1],
            now=now),
        ebisu.initModel(wmaxMean=.02, now=now),
    ]
    modelsInit = models
    modelsPerIter = [modelsInit]

    predictors = [ePredictor, v3Predictor, betasPredictor]
    updators = [eUpdator, v3Updator, betasUpdator]

    logliks = []
    for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)

      pRecallForModels = [pred(model, elapsedTime) for model, pred in zip(models, predictors)]

      success = s * 2 > t
      ll = tuple(np.log(p) if success else np.log(1 - p) for p in pRecallForModels)
      logliks.append(ll)
      if intermediate:
        print(
            f'  {s}/{t}, {elapsedTime:.1f}: ps={[round(p,4) for p in pRecallForModels]}, ll={[round(l,3) for l in ll]}'
        )

      models = [update(model, s, t, now) for model, update in zip(models, updators)]
      modelsPerIter.append(models)

    loglikFinal = np.sum(np.array(logliks), axis=0).tolist()
    print(f'loglikFinal={[round(p,3) for p in loglikFinal]}, {card.key=}')
    np.set_printoptions(precision=2, suppress=True)
    print(
        np.array([[t + (w,)
                   for t, w in zip(v[-1].pred.betaModels, v[-1].pred.betaWeights)]
                  for v in modelsPerIter]))
"""
[[[     2.        2.        1.        1.  ]
  [     2.        2.        3.59      0.77]
  [     2.        2.       12.92      0.6 ]
  [     2.        2.       46.42      0.46]
  [     2.        2.      166.81      0.36]
  [     2.        2.      599.48      0.28]
  [     2.        2.     2154.43      0.22]
  [     2.        2.     7742.64      0.17]
  [     2.        2.    27825.59      0.13]
  [     2.        2.   100000.        0.1 ]]
  """