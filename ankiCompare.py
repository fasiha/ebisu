import numpy as np
import pylab as plt  # type:ignore
import typing

import ebisu
from ebisu.ebisu import resultToLogProbability
import ebisu3boost
import utils
import ebisu3beta

plt.ion()

ConvertAnkiMode = typing.Literal['approx', 'binary']
MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec

wmaxs = np.linspace(0, 1, 101)
n = 10
hs = np.logspace(0, 4, n)


def printableList(v: list[int | float]) -> str:
  return ", ".join([f'{x:0.1f}' for x in v])


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
    else:  # easy
      raise Exception('unknown Anki result')
  elif mode == 'binary':
    # hard or better is pass
    return dict(successes=int(result > 1), total=1)


def cardToLoglik(card: utils.Card,
                 model,
                 pred,
                 update,
                 now=0,
                 convertMode: ConvertAnkiMode = 'approx'):
  logliks = []
  models = [model]
  pRecalls = []
  for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
    now += elapsedTime * MILLISECONDS_PER_HOUR
    resultArgs = convertAnkiResultToBinomial(ankiResult, convertMode)

    pRecall = pred(model, elapsedTime)
    pRecalls.append(pRecall)

    model = update(model, now, **resultArgs)
    models.append(model)

    ll = resultToLogProbability(models[-1].results[-1], pRecall)
    logliks.append(ll)

  return sum(logliks)
  # return models, logliks, pRecalls


if __name__ == '__main__':
  origNow = ebisu.timeMs()

  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df, noPerfectCardsInTraining=False)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  card = None
  models = None
  modelsPerIter = None

  v3Predictor = lambda model, elapsedTime: ebisu3boost.predictRecall(
      model, model.pred.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  v3Updator = lambda model, now, **kwargs: ebisu3boost.updateRecallHistory(
      ebisu3boost.updateRecall(model, now=now, **kwargs), size=1000)

  gamma3Predictor = lambda model, elapsedTime: ebisu.predictRecall(
      model, model.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  gamma3Updator = lambda model, now, **kwargs: ebisu.updateRecall(model, now=now, **kwargs)

  beta3pred = lambda model, elapsedTime: ebisu3beta.predictRecall(
      model, model.lastEncounterMs + elapsedTime * 3600e3, logDomain=False)
  beta3up = lambda model, now, **kwargs: ebisu3beta.updateRecall(
      model, now=now, **kwargs, updateThreshold=.99, weightThreshold=0.49)

  # np.seterr(all='raise')
  # np.seterr(under='warn')

  fracs = [0.8]
  # fracs = [1.0]
  fracs = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
  # fracs = [0.75]
  # for card in train:

  listToStr = lambda v: ", ".join([f"{x:0.2f}" for x in v])
  w1s = []  #  np.arange(.4, .99, .05)
  allLiks = []
  for w1 in w1s:
    liks = []
    for card in [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]:
      liks.append(
          cardToLoglik(
              card,
              ebisu.initModel(halflife=100, now=0, power=1, n=5, stdScale=1, w1=w1),
              gamma3Predictor,
              lambda *args, **kwargs: gamma3Updator(
                  *args, **kwargs, updateThreshold=.99, weightThreshold=0.05),
          ))
    print(f'{w1=:0.2f}, liks=[{listToStr(liks)}]')
    allLiks.append(liks)

  for card in [next(t for t in train if t.fractionCorrect >= frac) for frac in fracs]:
    hlMeanStd = (24., 24 * .7)
    boostMeanStd = (3, 3 * .7)
    convertMode: ConvertAnkiMode = 'binary'
    convertMode: ConvertAnkiMode = 'approx'
    now = origNow

    wmaxMean, initHlMean = 0.5, None
    wmaxMean, initHlMean = None, 12.0
    intermediate = not False

    modelsPredictorsUpdators = [
        (
            ebisu3beta.initModel(100, 2.0, n=5, w1=.5, now=now),
            beta3pred,
            beta3up,
            ebisu3beta.hoursForRecallDecay,
        ),
        (
            ebisu3boost.initModel(
                initHlMean=hlMeanStd[0],
                boostMean=boostMeanStd[0],
                initHlStd=hlMeanStd[1],
                boostStd=boostMeanStd[1],
                now=now),
            v3Predictor,
            v3Updator,
            lambda model, *rest: model.pred.currentHalflifeHours,
        ),
        (
            ebisu.initModel(halflife=100, now=now, power=20, n=5, stdScale=1, w1=.9),
            gamma3Predictor,
            lambda *args, **kwargs: gamma3Updator(
                *args, **kwargs, updateThreshold=0.9, weightThreshold=10000),
            ebisu.hoursForRecallDecay,
        ),
        (
            ebisu.initModel(halflife=100, now=now, power=20, n=5, stdScale=1, w1=.9),
            gamma3Predictor,
            lambda *args, **kwargs: gamma3Updator(
                *args, **kwargs, updateThreshold=0.5, weightThreshold=10000),
            ebisu.hoursForRecallDecay,
        ),
        (
            ebisu.initModel(halflife=100, now=now, power=20, n=5, stdScale=1, w1=.9),
            gamma3Predictor,
            lambda *args, **kwargs: gamma3Updator(
                *args, **kwargs, updateThreshold=.99, weightThreshold=0.05),
            ebisu.hoursForRecallDecay,
        ),
        (
            ebisu.initModel(
                halflife=100,
                now=now,
                power=1,  ####################### !
                n=5,
                stdScale=1,
                w1=.5),
            gamma3Predictor,
            lambda *args, **kwargs: gamma3Updator(
                *args, **kwargs, updateThreshold=.99, weightThreshold=0.49),
            ebisu.hoursForRecallDecay,
        ),
    ]

    modelsPerIter = [[v[0] for v in modelsPredictorsUpdators]]

    logliks = []
    for ankiResult, elapsedTime in zip(card.results, card.dts_hours):
      now += elapsedTime * MILLISECONDS_PER_HOUR
      resultArgs = convertAnkiResultToBinomial(ankiResult, convertMode)

      pRecallForModels = [
          pred(model, elapsedTime) for model, pred, _, *r in modelsPredictorsUpdators
      ]

      models = [
          update(model, now, **resultArgs) for model, _, update, *r in modelsPredictorsUpdators
      ]
      for i, m in enumerate(models):
        _oldModel, p, u, *r = modelsPredictorsUpdators[i]
        modelsPredictorsUpdators[i] = (m, p, u, *r)

      ll = tuple(resultToLogProbability(models[-1].results[-1], p) for p in pRecallForModels)
      logliks.append(ll)

      if intermediate:
        hls = [
            f(model) if f else -1 for model, (_, _, _, f) in zip(models, modelsPredictorsUpdators)
        ]
        hls80 = [
            f(model, 0.8) if f else -1
            for model, (_, _, _, f) in zip(models, modelsPredictorsUpdators)
        ]
        ps = [round(p, 4) for p in pRecallForModels]
        modelToAtoms = lambda model: ", ".join([
            f'(w={2**l2w:0.2g}, h={round(a/b,2)}, a={a:0.2g}, b={b:0.2g})' for l2w,
            (a, b) in zip(model.log2weights, model.halflifeGammas)
        ])

        def betaModelToAtoms(model: ebisu3beta.BetaEnsemble) -> str:
          return ", ".join([
              f'(w={2**l2w:0.2g}, h={round(t,2)}, a={a:0.2g}, b={b:0.2g})'
              for l2w, (a, b, t) in zip(model.log2weights, model.models)
          ])

        printableResult = f'{resultArgs["successes"]}/{resultArgs["total"]}/{resultArgs.get("q0", 1)}'
        print(
            f'   {elapsedTime:.1f}h {printableResult}: {ps=}, hls=[{printableList(hls)}] h (80%=[{printableList(hls80)}])'
        )
        for i, m in enumerate(models):
          if type(m) == ebisu.models.Model:
            print(f'     ({i}) {modelToAtoms(m)}')
          elif type(m) == ebisu3beta.BetaEnsemble:
            print(f'     ({i}) {betaModelToAtoms(m)}')

        print('')
      modelsPerIter.append(models)

    loglikFinal = np.sum(np.array(logliks), axis=0).tolist()
    print(f'loglikFinal={[round(p,3) for p in loglikFinal]}, {card.key=}')
    np.set_printoptions(precision=2, suppress=True)
