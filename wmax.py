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

  fracs = [0.8]
  # fracs = [1.0]
  # fracs = [0.8, 0.85, 0.9, 0.95, 1.]
  for card in [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]:
    hlMeanStd = (10., 10 * .7)
    boostMeanStd = (3, 3 * .7)
    convertMode: ConvertAnkiMode = 'binary'
    now = origNow

    models = [
        ebisu.initModel(
            wmax=.5,
            now=now,
            # format="rational",
        ),
        ebisu3gammas.initModel(
            initHlMean=hlMeanStd[0],
            boostMean=boostMeanStd[0],
            initHlStd=hlMeanStd[1],
            boostStd=boostMeanStd[1],
            now=now),
    ]
    predictors = [
        lambda model, elapsedTime: ebisu.predictRecall(
            model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
        lambda model, elapsedTime: ebisu3gammas.predictRecall(
            model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False),
    ]
    updators = [
        lambda model, s, t, now: ebisu.updateRecall(
            model, successes=s, total=t, now=now, wmaxPrior=[None, (3, 1.5)][0]),
        lambda model, s, t, now: ebisu3gammas.updateRecallHistory(
            ebisu3gammas.updateRecall(model, successes=s, total=t, now=now), size=1000),
    ]

    logliks = []
    for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)

      pRecallForModels = [pred(model, elapsedTime) for model, pred in zip(models, predictors)]

      success = s * 2 > t
      ll = tuple(np.log(p) if success else np.log(1 - p) for p in pRecallForModels)
      logliks.append(ll)
      print(
          f'  {s}/{t}, {elapsedTime:.1f}: ps={[round(p,4) for p in pRecallForModels]}, ll={[round(l,3) for l in ll]} (wmax={models[0].pred.wmax:0.3f}, hl={models[1].pred.currentHalflifeHours:0.3f})'
      )

      models = [update(model, s, t, now) for model, update in zip(models, updators)]

    loglikFinal = np.sum(np.array(logliks), axis=0)
    print(f'{loglikFinal=}, {card.key=}')

  assert models
  model: ebisu.Model = models[0]
  wmaxBetaPriors = (9, 3)
  wmaxBetaPriors = (3, 3)
  wmaxBetaPriors = (3, 1)
  ps = np.array(
      [ebisu.ebisuHelpers.posterior(model, wmax, wmaxBetaPriors, hs)[0] for wmax in wmaxs])

  plt.figure()
  parr = []
  for i in range(3, 1 + len(model.quiz.results[0])):
    thisModel = deepcopy(model)
    thisModel.quiz.results[0] = thisModel.quiz.results[0][:i]
    ab = wmaxBetaPriors
    v = [ebisu.ebisuHelpers.posterior(thisModel, wmax, ab, hs)[0] for wmax in wmaxs]
    parr.append(v)
    plt.plot(wmaxs, v, label=f'{i=}')
    imax = np.argmax(v)
    plt.plot(wmaxs[imax], v[imax], 'bo')

  plt.legend()
  assert card
  plt.title(f'card {card.key}, {n=}')

  ###

  n = hs.size
  tau = -(n - 1) / np.log(.1)
  ws = np.exp(-np.arange(n) / tau)

  ts = np.linspace(0, 1000, 501)
  arr = np.exp2(-ts[:, np.newaxis] * (1 / hs[np.newaxis, :])) * ws

  plt.figure()
  plt.plot(ts, arr)
  plt.plot(ts, np.max(arr, axis=1), linewidth=3, linestyle=':')

  arr2 = np.exp2(-ts[:, np.newaxis] * (1 / hs[np.newaxis, :])) * ws
