from copy import deepcopy
from math import fsum, log2
from attr import dataclass
from ebisu.ebisu import timeMs
from ebisu.ebisuHelpers import gammaUpdateBinomial, gammaUpdateNoisy
from ebisu.models import BinomialResult, NoisyBinaryResult, Result
import ebisu2beta as ebisu2
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from typing import Optional, Union

HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
LN2 = np.log(2)

BetaModel = tuple[float, float, float]


@dataclass
class BetaEnsemble:
  version: int
  results: list[Result]
  lastEncounterMs: float  # milliseconds since unix epoch
  log2weights: list[float]
  models: list[BetaModel]  # same length as log2weights


def initModel(
    halflife: Optional[float] = None,  # hours
    initAB: Optional[float] = None,
    finalHalflife=1e5,  # hours
    n: int = 10,
    w1: float = 0.9,
    # above: lazy inputs, below: exact inputs
    weightsModel: Optional[list[tuple[float, BetaModel]]] = None,
    now: Optional[float] = None,
) -> BetaEnsemble:
  now = now if now is not None else timeMs()
  if weightsModel is not None:
    n = len(weightsModel)
    weights: list[float] = []
    models: list[BetaModel] = []
    # could use `zip` here but then types get mixed
    for w, g in weightsModel:
      weights.append(w)
      models.append(g)
  else:
    assert halflife and initAB, "halflife & initAB needed if weightsModel not provided"
    dsol = minimize_scalar(lambda d: np.abs(1 - sum(np.hstack([w1, w1**(d * np.arange(1, n))]))),
                           [5, 20])
    weights = np.hstack([w1, w1**(dsol.x * np.arange(1, n))]).tolist()
    halflives = np.logspace(np.log10(halflife), np.log10(finalHalflife), n)
    models = [(initAB, initAB, t) for t in halflives]

  wsum = fsum(weights)
  weights = [w / wsum for w in weights]
  log2weights = np.log2(weights).tolist()

  return BetaEnsemble(
      version=1,
      results=[],
      lastEncounterMs=now,
      log2weights=log2weights,
      models=models,
  )


def predictRecall(
    model: BetaEnsemble,
    now: Optional[float] = None,
    logDomain=True,
) -> float:
  from ebisu.logsumexp import logsumexp

  now = now if now is not None else timeMs()
  elapsedHours = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  logps = [ebisu2.predictRecall(m, tnow=elapsedHours, exact=False) for m in model.models]
  logs = [LN2 * l2w + lp for l2w, lp in zip(model.log2weights, logps)]
  log2Expect = logsumexp(logs) / LN2

  # print(f'{np.exp2(log2Expect)=}, =?? {sum(np.exp(logps)*np.exp2(model.log2weights))}')

  assert np.isfinite(log2Expect) and log2Expect <= 0, f'{logps=}, {logs=}, {log2Expect=}'
  return log2Expect if logDomain else np.exp2(log2Expect)


def predictRecallApprox(
    model: BetaEnsemble,
    now: Optional[float] = None,
    logDomain=True,
) -> float:
  from ebisu.logsumexp import logsumexp

  now = now if now is not None else timeMs()
  elapsedHours = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  log2ps = [-elapsedHours / m[-1] for m in model.models]
  logs = [LN2 * (l2w + l2p) for l2w, l2p in zip(model.log2weights, log2ps)]
  log2Expect = logsumexp(logs) / LN2

  assert np.isfinite(log2Expect) and log2Expect <= 0, f'{log2ps=}, {logs=}, {log2Expect=}'
  return log2Expect if logDomain else np.exp2(log2Expect)


def hoursForRecallDecay(model: BetaEnsemble, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)

  res = minimize_scalar(
      lambda h: abs(lp - predictRecall(model, now=model.lastEncounterMs + 3600e3 * h)),
      bounds=[.01, 100e3])
  assert res.success
  return res.x


def resultToLogProbability(r: Result, p: float) -> float:
  if type(r) == NoisyBinaryResult:
    z = r.result >= 0.5
    return np.log(((r.q1 - r.q0) * p + r.q0) if z else (r.q0 - r.q1) * p + (1 - r.q0))
  elif type(r) == BinomialResult:
    return float(binom.logpmf(r.successes, r.total, p))
  raise Exception("unknown quiz type")


def updateRecall(
    model: BetaEnsemble,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    now: Optional[float] = None,
    updateThreshold=0.5,
    weightThreshold=0.1,
) -> BetaEnsemble:
  now = now or timeMs()
  t = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS

  updateds = [ebisu2.updateRecall(m, successes, total, tnow=t, q0=q0) for m in model.models]

  resultObj: Union[NoisyBinaryResult, BinomialResult]
  if total == 1:
    assert (0 <= successes <= 1), "`total=1` implies successes in [0, 1]"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t, rescale=1)
  else:
    assert successes == np.floor(successes), "float `successes` must have `total=1`"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    resultObj = BinomialResult(successes=int(successes), total=total, hoursElapsed=t, rescale=1)

  individualLogProbabilities = [
      resultToLogProbability(resultObj, ebisu2.predictRecall(m, t, exact=True))
      for m in model.models
  ]
  assert all(
      x < 0
      for x in individualLogProbabilities), f'{individualLogProbabilities=}, {model.models=}, {t=}'

  newModels: list[BetaModel] = []
  newLog2Weights: list[float] = []
  scales: list[float] = []
  for (
      m,
      updated,
      l2w,
      lp,
      exceededWeight,
  ) in zip(
      model.models,
      updateds,
      model.log2weights,
      individualLogProbabilities,
      _exceedsThresholdLeft([2**x for x in model.log2weights], weightThreshold),
  ):
    oldHl = m[-1]
    newHl = updated[-1]
    scal = newHl / oldHl
    scales.append(scal)

    newLog2Weights.append(l2w + lp / LN2)  # particle filter update
    if scal > updateThreshold or exceededWeight:
      newModels.append(updated)
    else:
      newModels.append(m)

  ret = deepcopy(model)  # clone
  ret.results.append(resultObj)
  ret.lastEncounterMs = now
  ret.models = newModels
  ret.log2weights = (newLog2Weights - logsumexp(np.array(newLog2Weights) * LN2) / LN2).tolist()
  return ret


def _exceedsThresholdLeft(v, threshold):
  ret = []
  last = False
  for x in v[::-1]:
    last = last or x > threshold
    ret.append(last)
  return ret[::-1]
