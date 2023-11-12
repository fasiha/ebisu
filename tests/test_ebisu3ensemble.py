import numpy as np
from ebisu import initModel, modelToPercentileDecay, predictRecall, updateRecall
from ebisu.ebisu import BetaEnsemble
import ebisu.ebisu2beta as ebisu2


def weightedSum(weights: list[float], samples: list[float]) -> float:
  return sum(w * x for w, x in zip(weights, samples))


def testInit():
  init = initModel(10, 10e3)
  assert modelToPercentileDecay(init) > 10

  init = initModel(10, 10e3, numAtoms=10)
  assert len(init) == 10


def testPredict():
  init = initModel(10, 10e3)
  t = 10

  assert predictRecall(init, t) < 0
  pLinear = predictRecall(init, t, logDomain=False)
  assert 0 < pLinear < 1

  weights = [2**a.log2weight for a in init]
  pRecalls = [ebisu2.predictRecall((a.alpha, a.beta, a.time), t, exact=True) for a in init]
  expected = weightedSum(weights, pRecalls)

  assert np.allclose(pLinear, expected)


def testUpdate():
  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  summarizeModel(init)

  for t in [1, 10, 100, 1000]:
    for success in [True, False]:
      u = updateRecall(init, 1 if success else 0, 1, t)
      halflife = modelToPercentileDecay(u)
      if success:
        assert halflife > initHalflife
      else:
        summarizeModel(u)
        assert halflife < initHalflife, f'{t=}, {success=}'


def summarizeModel(model: BetaEnsemble):
  print([(2**a.log2weight, a.alpha, a.beta, a.time) for a in model])
