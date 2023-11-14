import numpy as np
from ebisu import initModel, modelToPercentileDecay, predictRecall, updateRecall, rescaleHalflife
from ebisu.ebisu import BetaEnsemble
import ebisu.ebisu2beta as ebisu2


def weightedSum(weights: list[float], samples: list[float]) -> float:
  return sum(w * x for w, x in zip(weights, samples))


def summarizeModel(model: BetaEnsemble):
  print([(2**a.log2weight, a.alpha, a.beta, a.time) for a in model])


def isMonotonicallyIncreasing(v: list[float]) -> bool:
  return bool(np.all(np.diff(v) > 0))


def testInit():
  init = initModel(10, 10e3)
  assert modelToPercentileDecay(init) > 10

  init = initModel(10, 10e3, numAtoms=10)
  assert len(init) == 10


def testPredict():
  init = initModel(10, 10e3)
  t = 10

  pLinear = predictRecall(init, t)
  assert 0 < pLinear < 1

  weights = [2**a.log2weight for a in init]
  pRecalls = [ebisu2.predictRecall((a.alpha, a.beta, a.time), t, exact=True) for a in init]
  expected = weightedSum(weights, pRecalls)

  assert np.allclose(pLinear, expected)


def testUpdateBinary():
  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    for success in [True, False]:
      u = updateRecall(init, 1 if success else 0, 1, t)
      halflife = modelToPercentileDecay(u)
      if success:
        assert halflife > initHalflife
      else:
        assert halflife < initHalflife, f'{t=}, {success=}'


def testUpdateBinomial():
  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    updateds = [updateRecall(init, success, 2, t) for success in [0, 1, 2]]
    halflives = [modelToPercentileDecay(m) for m in updateds]
    assert halflives[0] < halflives[1] < halflives[2], f'{halflives}'
    assert halflives[0] < initHalflife
    assert halflives[2] > initHalflife


def testUpdateFuzzy():
  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    q1s = [0.1, 0.4, 0.6, 0.9]
    updateds = [updateRecall(init, q1, 1, t) for q1 in q1s]
    halflives = [modelToPercentileDecay(m) for m in updateds]
    assert isMonotonicallyIncreasing(halflives)
    assert all(h < initHalflife if q1 < 0.5 else h > initHalflife for h, q1 in zip(halflives, q1s))


def testHalflife():
  init = initModel(10, 10e3)
  percentiles = [0.9, 0.7, 0.4, 0.2]
  times = [modelToPercentileDecay(init, p) for p in percentiles]
  assert isMonotonicallyIncreasing(times)


def testRescaling():
  init = initModel(10, 10e3)

  scales = [.1, .5, 1, 2, 10]
  updateds = [rescaleHalflife(init, s) for s in scales]
  halflives = [modelToPercentileDecay(u) for u in updateds]
  assert isMonotonicallyIncreasing(halflives)
