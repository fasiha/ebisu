from math import fsum, log10, log2
from time import time_ns
import numpy as np
from copy import deepcopy
from typing import Union, Optional
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import binom
from ebisu.gammaDistribution import gammaToMean, meanVarToGamma

from .models import BinomialResult, HalflifeGamma, Model, NoisyBinaryResult, Predict, Quiz, Result
from .ebisuHelpers import gammaPredictRecall, gammaUpdateBinomial, gammaUpdateNoisy

HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
LN2 = np.log(2)


def initModel(
    halflife: Optional[float] = None,  # hours
    finalHalflife=1e5,  # hours
    n: int = 10,
    firstHalflife: Optional[float] = None,
    # above: lazy inputs, below: exact inputs
    weightsHalflifeGammas: Optional[list[tuple[float, HalflifeGamma]]] = None,
    power: int = 1,
    stdScale: float = 0.5,
    now: Optional[float] = None,
    newThing=False,
    w1: Optional[float] = None,
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
    if newThing and w1:
      dsol = minimize_scalar(lambda d: np.abs(1 - sum(np.hstack([w1, w1**(d * np.arange(1, n))]))),
                             [5, 20])
      weights = np.hstack([w1, w1**(dsol.x * np.arange(1, n))]).tolist()
      halflives = np.logspace(np.log10(halflife), np.log10(finalHalflife), n)
      halflifeGammas = [meanVarToGamma(t, (t * stdScale)**2) for t in halflives]
    else:
      halflives = np.logspace(log10(firstHalflife or (halflife * .1)), log10(finalHalflife),
                              n).tolist()
      # pick standard deviation to be half of the mean
      halflifeGammas = [meanVarToGamma(t, (t * stdScale)**2) for t in halflives]
      weights = _halflifeToFinalWeight(halflife, halflives, power)

  wsum = fsum(weights)
  weights = [w / wsum for w in weights]
  log2weights = np.log2(weights).tolist()

  return Model(
      version=1,
      quiz=Quiz(version=1, results=[[]], startTimestampMs=[now]),
      pred=Predict(
          version=1,
          lastEncounterMs=now,
          log2weights=log2weights,
          halflifeGammas=halflifeGammas,
          forSql=_genForSql(log2weights, halflives, power),
          power=power,
      ),
  )


def _genForSql(log2ws: list[float], hs: list[float], power: int) -> list[tuple[float, float]]:
  return [(lw, power / (h * 3600e3)) for lw, h in zip(log2ws, hs)]


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    now: Optional[float] = None,
    extra: Optional[dict] = None,
    updateThreshold=0.5,
    weightThreshold=0.1,
    verbose=False,
) -> Model:
  now = now or timeMs()
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if total == 1:
    assert (0 <= successes <= 1), "`total=1` implies successes in [0, 1]"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    z = successes >= 0.5
    updateds = [gammaUpdateNoisy(a, b, t, q1, q0, z) for a, b in model.pred.halflifeGammas]
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t)
  else:
    assert successes == np.floor(successes), "float `successes` must have `total=1`"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    k = int(successes)
    n = total
    updateds = [gammaUpdateBinomial(a, b, t, k, n) for a, b in model.pred.halflifeGammas]
    resultObj = BinomialResult(successes=int(successes), total=total, hoursElapsed=t)

  individualLogProbabilities = [
      resultToProbability(resultObj, 2**(-t / h))
      for h in map(lambda x: gammaToMean(*x), model.pred.halflifeGammas)
  ]

  ret = deepcopy(model)  # clone
  _appendQuizImpure(ret, resultObj)

  newModels: list[tuple[float, float]] = []
  newLog2Weights: list[float] = []
  scales: list[float] = []
  # ws = np.exp2()
  for idx, (
      m,
      updated,
      l2w,
      lp,
      exceededWeight,
  ) in enumerate(
      zip(model.pred.halflifeGammas, updateds, model.pred.log2weights, individualLogProbabilities,
          _exceedsThresholdLeft([2**x for x in model.pred.log2weights], weightThreshold))):
    oldHl = gammaToMean(*m)
    scal = updated.mean / oldHl
    scales.append(scal)

    newLog2Weights.append(l2w + lp / LN2)  # particle filter update
    # if verbose:
    #   print(f'_ {scal=}')
    if scal > updateThreshold or exceededWeight:
      newModels.append((updated.a, updated.b))
    else:
      newModels.append(m)

  if extra is not None:
    extra['scales'] = scales

  ret.pred.lastEncounterMs = now

  ret.pred.halflifeGammas = newModels
  ret.pred.log2weights = (newLog2Weights - logsumexp(np.array(newLog2Weights) * LN2) / LN2).tolist()
  ret.pred.forSql = _genForSql(ret.pred.log2weights,
                               [gammaToMean(*x) for x in ret.pred.halflifeGammas], model.pred.power)
  return ret


def _exceedsThresholdLeft(v, threshold):
  ret = []
  last = False
  for x in v[::-1]:
    last = last or x > threshold
    ret.append(last)
  return ret[::-1]


def _appendQuizImpure(model: Model, result: Result) -> None:
  if len(model.quiz.results) == 0:
    model.quiz.results = [[result]]
  else:
    model.quiz.results[-1].append(result)


def predictRecall(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
) -> float:
  from .logsumexp import logsumexp

  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  logs = [
      LN2 * (lw - model.pred.power * elapsedHours * beta / alpha)
      for lw, (alpha, beta) in zip(model.pred.log2weights, model.pred.halflifeGammas)
  ]
  logExpect = logsumexp(logs) / (LN2 * model.pred.power)

  assert np.isfinite(logExpect) and logExpect <= 0
  return logExpect if logDomain else np.exp2(logExpect)


def predictRecallSemiBayesian(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    innerPower=False,
) -> float:
  from .logsumexp import logsumexp

  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  if innerPower:
    l = [
        log2w * LN2 + gammaPredictRecall(alpha, beta, model.pred.power * elapsedHours, True)
        for log2w, (alpha, beta) in zip(model.pred.log2weights, model.pred.halflifeGammas)
    ]
  else:
    l = [
        log2w * LN2 + gammaPredictRecall(alpha, beta, elapsedHours, True) * model.pred.power
        for log2w, (alpha, beta) in zip(model.pred.log2weights, model.pred.halflifeGammas)
    ]

  logExpect = logsumexp(l) / model.pred.power
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
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  from scipy.stats import gamma as grv
  gammas = [grv(a, scale=1 / b) for a, b in model.pred.halflifeGammas]
  if model.pred.power is not None:
    ws = np.exp2(model.pred.log2weights)
    ws /= sum(ws)
    hmat = np.vstack([g.rvs(size=size) for g in gammas])
    p = np.sum(
        ws[:, np.newaxis] * np.exp2(-elapsedHours / hmat * model.pred.power),
        axis=0)**(1 / model.pred.power)
  else:
    logp = np.max((np.array(model.pred.log2weights)[:, np.newaxis] -
                   elapsedHours / np.vstack([g.rvs(size=size) for g in gammas])),
                  axis=0)
    p = np.exp2(logp)
  expectation = np.mean(p)
  if extra is not None:
    extra['std'] = np.std(p)
  return np.log(expectation) if logDomain else expectation


def _halflifeToFinalWeight(halflife: float, hs: list[float] | np.ndarray, pow: int) -> list[float]:
  """
  Complement to `hoursForRecallDecay`, which is $halflife(percentile, wmaxMean)$.

  This is $finalWeight(halflife, percentile=0.5)$.
  """
  n = len(hs)

  logv = (-halflife * np.log(2)) / np.array(hs)
  lp = np.log(0.5)
  n = len(hs)
  ivec = (np.arange(n) / (n - 1))

  def optlog(logwfinal):
    ws = np.exp(logwfinal)**ivec
    return abs(lp - _powerMeanLogW(logv, pow, ws))

  res = minimize_scalar(optlog, bounds=[np.log(1e-13), 0])
  assert res.success
  return (np.exp(res.x)**ivec).tolist()


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)

  res = minimize_scalar(
      lambda h: abs(lp - predictRecall(model, now=model.pred.lastEncounterMs + 3600e3 * h)),
      bounds=[.01, 100e3])
  assert res.success
  return res.x


def timeMs() -> float:
  return time_ns() / 1_000_000


def _powerMeanLogW(logv: list[float] | np.ndarray,
                   p: int | float,
                   ws: Optional[list[float] | np.ndarray] = None) -> float:
  "same as _powerMean but pass in log (base e) of `v`"
  logv = np.array(logv)
  ws = np.array(ws) / np.sum(ws) if ws is not None else [1 / logv.size] * logv.size
  res, sgn = logsumexp(p * logv, b=ws, return_sign=True)
  assert sgn > 0
  return float(res / p)


def resultToProbability(r: Result, p: float) -> float:
  if type(r) == NoisyBinaryResult:
    z = r.result >= 0.5
    return np.log(((r.q1 - r.q0) * p + r.q0) if z else (r.q0 - r.q1) * p + (1 - r.q0))
  elif type(r) == BinomialResult:
    return float(binom.logpmf(r.successes, r.total, p))
  raise Exception("unknown quiz type")
