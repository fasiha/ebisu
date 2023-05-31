from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore
import numpy as np
import pylab as plt  # type:ignore
import typing

import ebisu
import ebisu3wmax
import ebisu3boost
import ebisu2beta
import ebisu3max
import ebisu.ebisu3weightedmean as ebisu3w
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

  e2pred = lambda model, elapsedTime: ebisu2beta.predictRecall(model, elapsedTime, True)
  # e2upd = lambda model, s, t, now: ebisu2beta.updateRecall(model, s, t, )
  ePredictor, v3Predictor = [
      lambda model, elapsedTime: ebisu3wmax.predictRecall(
          model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
      lambda model, elapsedTime: ebisu3boost.predictRecall(
          model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
  ]
  eUpdator, v3Updator = [
      lambda model, s, t, now: ebisu3wmax.updateRecall(model, successes=s, total=t, now=now),
      lambda model, s, t, now: ebisu3boost.updateRecallHistory(
          ebisu3boost.updateRecall(model, successes=s, total=t, now=now), size=1000),
  ]

  betasPredictor = lambda model, elapsedTime: ebisu3wmax.predictRecallBetas(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  betasUpdator = lambda model, s, t, now: ebisu3wmax.updateRecallBetas(
      model, successes=s, total=t, now=now)

  gammaPredictor = lambda model, elapsedTime: ebisu3wmax.predictRecallGammas(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  gammaUpdator = lambda model, s, t, now: ebisu3wmax.updateRecallGammas(
      model, successes=s, total=t, now=now)

  gamma3MaxPredictor = lambda model, elapsedTime: ebisu3max.predictRecall(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  gamma3MaxUpdator = lambda model, s, t, now: ebisu3max.updateRecall(
      model, successes=s, total=t, now=now)

  gamma3Predictor = lambda model, elapsedTime: ebisu.predictRecall(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  gamma3Updator = lambda model, s, t, now: ebisu.updateRecall(model, successes=s, total=t, now=now)
  gamma3Updator2 = lambda model, s, t, now: ebisu.updateRecall(model, successes=s, total=t, now=now)

  ebisu3wPredictor = lambda model, elapsedTime: ebisu3w.predictRecall(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  ebisu3wUpdator = lambda model, s, t, now: ebisu3w.updateRecall(
      model, successes=s, total=t, now=now, verbose=False)

  # np.seterr(all='raise')
  # np.seterr(under='warn')

  fracs = [0.8]
  # fracs = [1.0]
  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  #   fracs = [0.75]
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
        ebisu3wmax.initModel(wmaxMean=wmaxMean, initHlMean=initHlMean, now=now),
        ebisu3boost.initModel(
            initHlMean=hlMeanStd[0],
            boostMean=boostMeanStd[0],
            initHlStd=hlMeanStd[1],
            boostStd=boostMeanStd[1],
            now=now),
        ebisu3wmax.initModel(wmaxMean=.02, now=now),
        ebisu3wmax.initModel(wmaxMean=.02, now=now),
        ebisu3max.initModel(halflife=10, now=now),
        ebisu.initModel(halflife=10, now=now, power=14, n=4),  # 4 4 
        ebisu.initModel(halflife=10 * 10, now=now, power=20, n=4, firstHalflife=7.5),  # 4 4
        ebisu.initModel(halflife=10 * 10, now=now, power=20, n=4, firstHalflife=7.5,
                        stdScale=10),  # 4 4
    ]
    modelsInit = models
    modelsPerIter = [modelsInit]

    predictors = [
        ePredictor, v3Predictor, betasPredictor, gammaPredictor, gamma3MaxPredictor,
        gamma3Predictor, gamma3Predictor, gamma3Predictor
    ]
    updators = [
        eUpdator, v3Updator, betasUpdator, gammaUpdator, gamma3MaxUpdator, gamma3Updator,
        gamma3Updator, gamma3Updator
    ]

    logliks = []
    for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)

      pRecallForModels = [pred(model, elapsedTime) for model, pred in zip(models, predictors)]

      success = s * 2 > t
      ll = tuple(np.log(p) if success else np.log(1 - p) for p in pRecallForModels)
      logliks.append(ll)
      if intermediate:
        print(f'  {s}/{t}, {elapsedTime:.1f}h: ps={[round(p,4) for p in pRecallForModels]}')

      models = [update(model, s, t, now) for model, update in zip(models, updators)]
      #   print(f'    hl={[round(ebisu.hoursForRecallDecay(models[-1], p), 4) for p in [ .5, .8, .9]]}')
      modelsPerIter.append(models)

    loglikFinal = np.sum(np.array(logliks), axis=0).tolist()
    print(f'loglikFinal={[round(p,3) for p in loglikFinal]}, {card.key=}')
    np.set_printoptions(precision=2, suppress=True)

    norm = lambda log2s: np.exp2(log2s)
    weightsEvolution = np.array([[
        (t[0] / t[1],) + (w,)
        for t, w in zip(v[-1].pred.halflifeGammas, norm(v[-1].pred.log2weights))
    ]
                                 for v in modelsPerIter])
    # print(np.array2string(weightsEvolution, edgeitems=100000))
norm = lambda v: np.array(v) / np.sum(v)


def _powerMean(v: list[float] | np.ndarray, p: int | float) -> float:
  return float(np.mean(np.array(v)**p)**(1 / p))


def _powerMeanW(v: list[float] | np.ndarray, p: int | float, ws: list[float] | np.ndarray) -> float:
  ws = np.array(ws) / np.sum(ws)
  return float(sum(np.array(v)**p * ws))**(1 / p)


if True:
  ts = np.logspace(0, 5, 1001)
  plt.figure()
  for i, v in enumerate(modelsPerIter):
    m = v[-1]
    plt.loglog(
        ts, [
            ebisu.predictRecall(m, now=m.pred.lastEncounterMs + t * 3600e3, logDomain=False)
            for t in ts
        ],
        label=f'{i}')
  plt.legend()

  hls, ws = zip(*[[12.97, 0.59], [188.15, 0.37], [4228.49, 0.04], [100011.38, 0.]])
  weightsHalflifemusHls = [(np.exp2(v[-1].pred.log2weights),
                            [a / b for a, b in v[-1].pred.halflifeGammas],
                            ebisu.hoursForRecallDecay(v[-1])) for v in modelsPerIter]
  [_powerMeanW(2**(-np.array(hls) / mu), 14, ws) for ws, mu, hls in weightsHalflifemusHls]
