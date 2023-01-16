from copy import deepcopy
from scipy.stats import gamma as gammarv  # type: ignore
import numpy as np
import pylab as plt  # type:ignore
import pandas as pd  # type:ignore
import json
import typing

from cmdstanpy import CmdStanModel  # type:ignore

import ebisu
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

  train, TEST_TRAIN = utils.traintest(df)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  fracs = [0.8]
  # fracs = [0.9]
  fracs = [0.8, 0.85, 0.9, 0.95]
  models = []
  for card in [next(t for t in train if t.fractionCorrect > frac) for frac in fracs]:
    hlMeanStd = (10., 10 * .7)
    boostMeanStd = (3, 3 * .7)
    convertMode: ConvertAnkiMode = 'binary'

    now = origNow
    model = ebisu.initModel(
        initHlMean=hlMeanStd[0],
        initHlStd=hlMeanStd[1],
        boostMean=boostMeanStd[0],
        boostStd=boostMeanStd[1],
        now=now,
    )
    models = [model]

    USE_ANKI = not False

    ress = [1] * 10
    hours = [6.0] * len(ress)
    for ankiResult, elapsedTime in (zip(card.results, card.dts_hours) if USE_ANKI else zip(
        ress, hours)):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      s, t = convertAnkiResultToBinomial(ankiResult, convertMode)
      model = ebisu.updateRecall(model, successes=s, total=t, now=now)
      models.append(ebisu.updateRecallHistory(model, size=1000))

    logliks = []
    wmaxMaps = []
    for thisModel, thisQuiz in zip(models, model.quiz.results[0]):
      p1 = (
          ebisu.predictRecall(
              thisModel,
              thisModel.pred.lastEncounterMs + thisQuiz.hoursElapsed * 3600e3,
              logDomain=False))

      res = [
          ebisu.ebisuHelpers.success(m)
          for m in (thisModel.quiz.results[-1] if len(thisModel.quiz.results) else [])
      ]
      successes = sum(res)
      ab = (max(2, successes), max(2, len(res) - successes))
      ps = [ebisu.ebisuHelpers.posterior2(thisModel, wmax, ab, hs) for wmax in wmaxs]
      wmaxMap = wmaxs[np.argmax(ps)]
      wmaxMaps.append(wmaxMap)

      n = hs.size
      # ws = 1 + (wmaxMap - 1) / (n - 1) * np.arange(n) # linear
      tau = -(n - 1) / np.log(wmaxMap)  # exp
      ws = np.exp(-np.arange(n) / tau)

      p2 = np.max(ws * np.exp2(-thisQuiz.hoursElapsed / hs))

      pRecallForModels = [p1, p2]
      ll = tuple(
          np.log(p) if ebisu.ebisuHelpers.success(thisQuiz) else np.log(1 - p)
          for p in pRecallForModels)
      logliks.append(ll)
      # print(
      #     f'{ebisu.ebisuHelpers.success(thisQuiz)}/{thisQuiz.hoursElapsed:.1f}: ps={[round(p,4) for p in pRecallForModels]}, ll={[f"{l:.3f}" for l in ll]} ({thisModel.pred.currentHalflifeHours:.1f})'
      # )

    loglikFinal = np.sum(np.array(logliks), axis=0)
    print(f'{loglikFinal=}, {card.key=}')

import pylab as plt

plt.ion()

wmaxBetaPriors = (9, 3)
wmaxBetaPriors = (3, 3)
ps = np.array([ebisu.ebisuHelpers.posterior2(model, wmax, wmaxBetaPriors, hs) for wmax in wmaxs])

plt.figure()
parr = []
for i in range(3, 1 + len(model.quiz.results[0])):
  thisModel = deepcopy(model)
  thisModel.quiz.results[0] = thisModel.quiz.results[0][:i]
  ab = wmaxBetaPriors
  v = [ebisu.ebisuHelpers.posterior2(thisModel, wmax, ab, hs) for wmax in wmaxs]
  parr.append(v)
  plt.plot(wmaxs, v, label=f'{i=}')
  imax = np.argmax(v)
  plt.plot(wmaxs[imax], v[imax], 'bo')

plt.legend()
plt.title(f'card {card.key}, {n=}')
