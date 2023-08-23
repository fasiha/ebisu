from math import fsum, log10, log2
from time import time_ns
import numpy as np
from copy import deepcopy
from typing import Union, Optional
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import binom
from ebisu.gammaDistribution import gammaToMean, meanVarToGamma

from .models import BinomialResult, HalflifeGamma, Model, NoisyBinaryResult, Result
from .ebisuHelpers import gammaPredictRecall, gammaUpdateBinomial, gammaUpdateNoisy

HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
LN2 = np.log(2)


def initModel(
    halflife: Optional[float] = None,  # hours
    finalHalflife=1e5,  # hours
    n: int = 10,
    w1: float = 0.9,
    # above: lazy inputs, below: exact inputs
    weightsHalflifeGammas: Optional[list[tuple[float, HalflifeGamma]]] = None,
    power: int = 1,
    stdScale: float = 0.5,
    now: Optional[float] = None,
) -> Model:
  """
  Create brand new Ebisu model

  Optional `now`: milliseconds since the Unix epoch when the fact was learned.
  """
  now = now if now is not None else timeMs()
  if weightsHalflifeGammas is not None:
    n = len(weightsHalflifeGammas)
    weights: list[float] = []
    halflifeGammas: list[tuple[float, float]] = []
    halflives: list[float] = []
    # could use `zip` here but then types get mixed
    for w, g in weightsHalflifeGammas:
      weights.append(w)
      halflifeGammas.append(g)
      halflives.append(gammaToMean(*g))
  else:
    assert halflife, "halflife or weightsHalflifeGammas needed"
    dsol = minimize_scalar(lambda d: np.abs(1 - sum(np.hstack([w1, w1**(d * np.arange(1, n))]))),
                           [5, 20])
    weights = np.hstack([w1, w1**(dsol.x * np.arange(1, n))]).tolist()
    halflives = np.logspace(np.log10(halflife), np.log10(finalHalflife), n)
    halflifeGammas = [meanVarToGamma(t, (t * stdScale)**2) for t in halflives]

  wsum = fsum(weights)
  weights = [w / wsum for w in weights]
  log2weights = np.log2(weights).tolist()

  return Model(
      version=1,
      startTimestampMs=now,
      results=[],
      lastEncounterMs=now,
      log2weights=log2weights,
      halflifeGammas=halflifeGammas,
      forSql=_genForSql(log2weights, halflives, power),
      power=power,
  )


def _genForSql(log2ws: list[float], hs: list[float], power: int) -> list[tuple[float, float]]:
  return [(lw, power / (h * 3600e3)) for lw, h in zip(log2ws, hs)]


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    now: Optional[float] = None,
    updateThreshold=0.5,
    weightThreshold=0.1,
    rescale: Union[float, int] = 1,
) -> Model:

  if rescale != 1:
    pRecall = predictRecallApprox(model, now, logDomain=False)
    model = rescaleHalflife(
        model, rescale, pRecall, updateThreshold=updateThreshold, weightThreshold=weightThreshold)

  now = now or timeMs()
  t = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if total == 1:
    assert (0 <= successes <= 1), "`total=1` implies successes in [0, 1]"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    z = successes >= 0.5
    updateds = [gammaUpdateNoisy(a, b, t, q1, q0, z) for a, b in model.halflifeGammas]
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t, rescale=rescale)
  else:
    assert successes == np.floor(successes), "float `successes` must have `total=1`"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    k = int(successes)
    n = total
    updateds = [gammaUpdateBinomial(a, b, t, k, n) for a, b in model.halflifeGammas]
    resultObj = BinomialResult(
        successes=int(successes), total=total, hoursElapsed=t, rescale=rescale)

  individualLogProbabilities = [
      resultToLogProbability(resultObj, 2**(-t / h))
      for h in map(lambda x: gammaToMean(*x), model.halflifeGammas)
  ]

  newModels: list[tuple[float, float]] = []
  newLog2Weights: list[float] = []
  scales: list[float] = []
  for (
      m,
      updated,
      l2w,
      lp,
      exceededWeight,
  ) in zip(
      model.halflifeGammas,
      updateds,
      model.log2weights,
      individualLogProbabilities,
      _exceedsThresholdLeft([2**x for x in model.log2weights], weightThreshold),
  ):
    oldHl = gammaToMean(*m)
    scal = updated.mean / oldHl
    scales.append(scal)

    newLog2Weights.append(l2w + lp / LN2)  # particle filter update
    if scal > updateThreshold or exceededWeight:
      newModels.append((updated.a, updated.b))
    else:
      newModels.append(m)

  ret = deepcopy(model)  # clone
  ret.results.append(resultObj)
  ret.lastEncounterMs = now
  ret.halflifeGammas = newModels
  ret.log2weights = (newLog2Weights - logsumexp(np.array(newLog2Weights) * LN2) / LN2).tolist()
  ret.forSql = _genForSql(ret.log2weights, [gammaToMean(*x) for x in ret.halflifeGammas],
                          model.power)
  return ret


def _exceedsThresholdLeft(v, threshold):
  ret = []
  last = False
  for x in v[::-1]:
    last = last or x > threshold
    ret.append(last)
  return ret[::-1]


def predictRecallApprox(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
) -> float:
  from .logsumexp import logsumexp

  now = now if now is not None else timeMs()
  elapsedHours = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  logs = [
      LN2 * (lw - model.power * elapsedHours * beta / alpha)
      for lw, (alpha, beta) in zip(model.log2weights, model.halflifeGammas)
  ]
  logExpect = logsumexp(logs) / (LN2 * model.power)

  assert np.isfinite(logExpect) and logExpect <= 0
  return logExpect if logDomain else np.exp2(logExpect)


def predictRecall(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
) -> float:
  from .logsumexp import logsumexp
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  l = [
      log2w * LN2 + gammaPredictRecall(alpha, beta, model.power * elapsedHours, True)
      for log2w, (alpha, beta) in zip(model.log2weights, model.halflifeGammas)
  ]

  logExpect = logsumexp(l) / model.power
  assert np.isfinite(logExpect) and logExpect <= 0
  return logExpect if logDomain else np.exp(logExpect)


def predictRecallMonteCarlo(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    size=10_000,
    extra: Optional[dict] = None,
) -> float:
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  from scipy.stats import gamma as grv
  gammas = [grv(a, scale=1 / b) for a, b in model.halflifeGammas]
  if model.power is not None:
    ws = np.exp2(model.log2weights)
    ws /= sum(ws)
    hmat = np.vstack([g.rvs(size=size) for g in gammas])
    p = np.sum(
        ws[:, np.newaxis] * np.exp2(-elapsedHours / hmat * model.power), axis=0)**(1 / model.power)
  else:
    logp = np.max((np.array(model.log2weights)[:, np.newaxis] -
                   elapsedHours / np.vstack([g.rvs(size=size) for g in gammas])),
                  axis=0)
    p = np.exp2(logp)
  expectation = np.mean(p)
  if extra is not None:
    extra['std'] = np.std(p)
  return np.log(expectation) if logDomain else expectation


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)

  res = minimize_scalar(
      lambda h: abs(lp - predictRecallApprox(model, now=model.lastEncounterMs + 3600e3 * h)),
      bounds=[.01, 100e3])
  assert res.success
  return res.x


def timeMs() -> float:
  return float(time_ns() // 1_000_000)


def resultToLogProbability(r: Result, p: float) -> float:
  if type(r) == NoisyBinaryResult:
    z = r.result >= 0.5
    return np.log(((r.q1 - r.q0) * p + r.q0) if z else (r.q0 - r.q1) * p + (1 - r.q0))
  elif type(r) == BinomialResult:
    return float(binom.logpmf(r.successes, r.total, p))
  raise Exception("unknown quiz type")


def rescaleHalflife(m: Model,
                    scale: float,
                    p: float,
                    updateThreshold=None,
                    weightThreshold=None) -> Model:
  if scale == 1:
    return deepcopy(m)

  result = 1 if scale > 0 else 0
  initDecay = hoursForRecallDecay(m, p)
  sol = minimize_scalar(lambda h: abs(initDecay * scale - hoursForRecallDecay(
      updateRecall(
          m,
          result,
          total=1,
          now=h * 3600e3 + m.lastEncounterMs,
          updateThreshold=updateThreshold,
          weightThreshold=weightThreshold), p)))
  newModel = updateRecall(
      m,
      result,
      total=1,
      now=sol.x * 3600e3 + m.lastEncounterMs,
      updateThreshold=updateThreshold,
      weightThreshold=weightThreshold)
  # undo the effect of this last quiz
  newModel.lastEncounterMs = m.lastEncounterMs
  newModel.results = newModel.results[:-1]

  return newModel
