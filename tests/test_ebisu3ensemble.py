import numpy as np
from ebisu import initModel, modelToPercentileDecay, predictRecall, updateRecall, rescaleHalflife
from ebisu.ebisu import BetaEnsemble
import ebisu.ebisu2beta as ebisu2

testpoints = []


def weightedSum(weights: list[float], samples: list[float]) -> float:
  return sum(w * x for w, x in zip(weights, samples))


def summarizeModel(model: BetaEnsemble):
  print([(2**a.log2weight, a.alpha, a.beta, a.time) for a in model])


def isMonotonicallyIncreasing(v: list[float]) -> bool:
  return bool(np.all(np.diff(v) > 0))


def modelToJson(model: BetaEnsemble):
  return [m.to_dict() for m in model]  # type:ignore # ugh


def testInit():
  args = dict(firstHalflife=10, lastHalflife=10e3)
  init = initModel(**args)
  assert modelToPercentileDecay(init) > 10

  args2 = dict(firstHalflife=10, lastHalflife=10e3, numAtoms=10)
  init2 = initModel(**args2)
  assert len(init2) == 10

  global testpoints
  testpoints += [["init", args, modelToJson(init)]]
  testpoints += [["init", args2, modelToJson(init2)]]


def testPredict():
  global testpoints
  init = initModel(10, 10e3)
  for t in [1, 10, 100]:

    pLinear = predictRecall(init, t)
    assert 0 < pLinear < 1

    weights = [2**a.log2weight for a in init]
    pRecalls = [ebisu2.predictRecall((a.alpha, a.beta, a.time), t, exact=True) for a in init]
    expected = weightedSum(weights, pRecalls)

    assert np.allclose(pLinear, expected)
    testpoints += [["predict", [modelToJson(init), t], pLinear]]


def testUpdateBinary():
  global testpoints

  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    for success in [True, False]:
      arg = dict(model=init, successes=1 if success else 0, total=1, elapsedTime=t)
      u = updateRecall(**arg)
      halflife = modelToPercentileDecay(u)
      if success:
        assert halflife > initHalflife
      else:
        assert halflife < initHalflife, f'{t=}, {success=}'
      arg['model'] = modelToJson(arg['model'])
      testpoints += [["update", arg, modelToJson(u)]]


def testUpdateBinomial():
  global testpoints

  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    args = [dict(model=init, successes=success, total=2, elapsedTime=t) for success in [0, 1, 2]]
    updateds = [updateRecall(**arg) for arg in args]
    halflives = [modelToPercentileDecay(m) for m in updateds]
    assert halflives[0] < halflives[1] < halflives[2], f'{halflives}'
    assert halflives[0] < initHalflife
    assert halflives[2] > initHalflife
    for arg, u in zip(args, updateds):
      arg['model'] = modelToJson(arg['model'])
      testpoints += [["update", arg, modelToJson(u)]]


def testUpdateFuzzy():
  global testpoints

  init = initModel(10, 10e3)
  initHalflife = modelToPercentileDecay(init)

  for t in [1, 10, 100, 1000]:
    q1s = [0.1, 0.4, 0.6, 0.9]
    args = [dict(model=init, successes=q1, total=1, elapsedTime=t) for q1 in q1s]
    updateds = [updateRecall(**arg) for arg in args]
    halflives = [modelToPercentileDecay(m) for m in updateds]
    assert isMonotonicallyIncreasing(halflives)
    assert all(h < initHalflife if q1 < 0.5 else h > initHalflife for h, q1 in zip(halflives, q1s))
    for arg, u in zip(args, updateds):
      arg['model'] = modelToJson(arg['model'])
      testpoints += [["update", arg, modelToJson(u)]]


def testHalflife():
  init = initModel(10, 10e3)
  percentiles = [0.9, 0.7, 0.4, 0.2]
  times = [modelToPercentileDecay(init, p) for p in percentiles]
  assert isMonotonicallyIncreasing(times)

  global testpoints
  for p, t in zip(percentiles, times):
    testpoints += [["modelToPercentileDecay", [modelToJson(init), p], t]]


def testRescaling():
  init = initModel(10, 10e3)

  scales = [.1, .5, 1, 2, 10]
  updateds = [rescaleHalflife(init, s) for s in scales]
  halflives = [modelToPercentileDecay(u) for u in updateds]
  assert isMonotonicallyIncreasing(halflives)

  global testpoints
  for s, u in zip(scales, updateds):
    testpoints += [["rescale", [modelToJson(init), s], modelToJson(u)]]


if __name__ == '__main__':
  testInit()
  testPredict()
  testUpdateBinary()
  testUpdateBinomial()
  testUpdateFuzzy()
  testHalflife()
  testRescaling()

  with open("test3.json", "w") as out:
    import json
    out.write(json.dumps(testpoints))
