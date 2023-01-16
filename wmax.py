from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore
import numpy as np
import pylab as plt  # type:ignore
import typing

import ebisu
import utils

plt.ion()

ConvertAnkiMode = typing.Literal['approx', 'binary']
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec

wmaxs = np.linspace(0, 1, 101)
n = 5
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

  train, TEST_TRAIN = utils.traintest(df)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.8]
  # fracs = [0.9]
  # fracs = [0.8, 0.85, 0.9, 0.95]
  models = []
  model = ebisu.initModel(wmax=.1, now=origNow)
  for card in [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]:
    hlMeanStd = (10., 10 * .7)
    boostMeanStd = (3, 3 * .7)
    convertMode: ConvertAnkiMode = 'binary'

    now = origNow
    model = ebisu.initModel(wmax=.1, now=now)
    models = [model]

    USE_ANKI = not False

    ress = [1] * 10
    hours = [6.0] * len(ress)
    for ankiResult, elapsedTime in (zip(card.results, card.dts_hours) if USE_ANKI else zip(
        ress, hours)):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)
      model = ebisu.updateRecall(model, successes=s, total=t, now=now)
      models.append(model)

    logliks = []
    wmaxMaps = []
    for thisModel, thisQuiz in zip(models, model.quiz.results[0]):
      p1 = (
          ebisu.predictRecall(
              thisModel,
              thisModel.pred.lastEncounterMs + thisQuiz.hoursElapsed * 3600e3,
              logDomain=False))

      res = [
          ebisu.success(m)
          for m in (thisModel.quiz.results[-1] if len(thisModel.quiz.results) else [])
      ]
      successes = sum(res)
      ab = (max(2, successes), max(2, len(res) - successes))
      ps = [ebisu.ebisuHelpers.posterior(thisModel, wmax, ab, hs)[0] for wmax in wmaxs]
      wmaxMap = wmaxs[np.argmax(ps)]
      wmaxMaps.append(wmaxMap)

      n = hs.size
      # ws = 1 + (wmaxMap - 1) / (n - 1) * np.arange(n) # linear
      tau = -(n - 1) / np.log(wmaxMap)  # exp
      ws = np.exp(-np.arange(n) / tau)

      p2 = np.max(ws * np.exp2(-thisQuiz.hoursElapsed / hs))

      pRecallForModels = [p1, p2]
      ll = tuple(np.log(p) if ebisu.success(thisQuiz) else np.log(1 - p) for p in pRecallForModels)
      logliks.append(ll)
      print(
          f'{ebisu.success(thisQuiz)}/{thisQuiz.hoursElapsed:.1f}: ps={[round(p,4) for p in pRecallForModels]}, ll={[f"{l:.3f}" for l in ll]}'
      )

    loglikFinal = np.sum(np.array(logliks), axis=0)
    print(f'{loglikFinal=}, {card.key=}')

  wmaxBetaPriors = (9, 3)
  wmaxBetaPriors = (3, 3)
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
