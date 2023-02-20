from math import log10, log2
from time import time_ns
import numpy as np
from copy import deepcopy
from typing import Callable, Union, Optional
from itertools import takewhile
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .models import BinomialResult, HalflifeGamma, Model, NoisyBinaryResult, Predict, Quiz, Result
from .ebisuHelpers import gammaPredictRecall, gammaUpdateBinomial, gammaUpdateNoisy


def initModel(
    halflife: Optional[float] = None,  # hours
    finalHalflife=1e5,  # hours
    n: int = 10,
    # above: lazy inputs, below: exact inputs
    weightsHalflifeGammas: Optional[list[tuple[float, HalflifeGamma]]] = None,
    power: Optional[int] = None,
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
      halflives.append(_gammaToMean(*g))
  else:
    assert halflife, "halflife or weightsHalflifeGammas needed"
    halflives = np.logspace(log10(halflife * .1), log10(finalHalflife), n).tolist()
    # pick standard deviation to be half of the mean
    halflifeGammas = [_meanVarToGamma(t, (t * .5)**2) for t in halflives]
    weights = _halflifeToFinalWeight(halflife, halflives, power)

  log2weights = np.log2(weights).tolist()

  return Model(
      version=1,
      quiz=Quiz(version=1, results=[[]], startTimestampMs=[now]),
      pred=Predict(
          version=1,
          lastEncounterMs=now,
          log2weights=log2weights,
          halflifeGammas=halflifeGammas,
          weightsReached=[x == 0 for x in range(n)],
          forSql=_genForSql(log2weights, halflives),
          power=power,
      ),
  )


def _genForSql(log2ws: list[float], hs: list[float]) -> list[tuple[float, float]]:
  return [(lw, h * 3600e3) for lw, h in zip(log2ws, hs)]


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    now: Optional[float] = None,
    extra: Optional[dict] = None,
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

  ret = deepcopy(model)  # clone
  _appendQuizImpure(ret, resultObj)

  SCALE_THRESH = 0.95
  newModels: list[tuple[float, float]] = []
  newWeights: list[float] = []
  newReached: list[bool] = []
  scales: list[float] = []
  for m, updated, log2weight, reached in zip(model.pred.halflifeGammas, updateds,
                                             model.pred.log2weights, model.pred.weightsReached):
    weight = np.exp2(log2weight)
    scal = updated.mean / _gammaToMean(*m)
    scales.append(scal)

    if model.pred.power is not None:
      if reached:
        newModels.append((updated.a, updated.b))
        newWeights.append(1)
        newReached.append(True)
      else:
        if (scal < 1 and scal < SCALE_THRESH) or (scal > 1 and scal < 1.01):
          # early failure or success
          newModels.append(m)
          newWeights.append(weight)
          newReached.append(False)
        else:
          newModels.append((updated.a, updated.b))
          newWeights.append(1)
          newReached.append(True)

      continue

    if reached:
      newModels.append((updated.a, updated.b))
      newWeights.append(min(weight * scal if scal > 1 else weight, 1))
      newReached.append(True)
    else:
      if (scal < 1 and scal < SCALE_THRESH) or (scal > 1 and scal < 1.01):
        # early failure or success
        newModels.append(m)
        newWeights.append(weight)
        newReached.append(False)
      else:
        newModels.append((updated.a, updated.b))
        newWeights.append(min(_powerMean([weight, scal], 2), 1))
        newReached.append(True)

  if extra is not None:
    extra['scales'] = scales

  if model.pred.power is not None:
    leadingReached = sum(takewhile(lambda x: x, newReached))
    MAX = 4
    if leadingReached > MAX:
      toDrop = leadingReached - MAX
      newModels = newModels[toDrop:]
      newWeights = newWeights[toDrop:]
      newReached = newReached[toDrop:]

  ret.pred.lastEncounterMs = now

  ret.pred.halflifeGammas = newModels
  ret.pred.log2weights = np.log2(newWeights).tolist()
  ret.pred.weightsReached = newReached
  ret.pred.forSql = _genForSql(ret.pred.log2weights,
                               [_gammaToMean(*x) for x in ret.pred.halflifeGammas])

  return ret


def _appendQuizImpure(model: Model, result: Result) -> None:
  if len(model.quiz.results) == 0:
    model.quiz.results = [[result]]
  else:
    model.quiz.results[-1].append(result)


HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def predictRecall(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    extra: Optional[dict] = None,
) -> float:
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  if model.pred.power:
    l = [-elapsedHours / (alpha / beta) for (alpha, beta) in model.pred.halflifeGammas]
    log2Precall = (
        _powerMeanLogW(np.array(l) * np.log(2), model.pred.power, np.exp2(model.pred.log2weights)) /
        np.log(2))
  else:
    l = [
        log2w + -elapsedHours / (alpha / beta)
        for (alpha, beta), log2w in zip(model.pred.halflifeGammas, model.pred.log2weights)
    ]
    log2Precall = max(l)
  if extra is not None:
    extra['indiv'] = l

  assert np.isfinite(log2Precall) and log2Precall <= 0
  return log2Precall if logDomain else np.exp2(log2Precall)


LN2 = np.log(2)


def predictRecallSemiBayesian(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    extra: Optional[dict] = None,
) -> float:
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  if model.pred.power is not None:
    l = [
        gammaPredictRecall(alpha, beta, elapsedHours, True)
        for (alpha, beta) in model.pred.halflifeGammas
    ]
    logPrecall = _powerMeanLogW(l, model.pred.power, np.exp2(model.pred.log2weights))
  else:
    l = [
        log2w * LN2 + gammaPredictRecall(alpha, beta, elapsedHours, True)
        for (alpha, beta), log2w in zip(model.pred.halflifeGammas, model.pred.log2weights)
    ]
    logPrecall = max(l)
  if extra is not None:
    extra['indiv'] = l
  assert np.isfinite(logPrecall) and logPrecall <= 0
  return logPrecall if logDomain else np.exp(logPrecall)


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


def _halflifeToFinalWeight(halflife: float,
                           hs: list[float] | np.ndarray,
                           pow: Optional[int] = None) -> list[float]:
  """
  Complement to `hoursForRecallDecay`, which is $halflife(percentile, wmaxMean)$.

  This is $finalWeight(halflife, percentile=0.5)$.
  """
  n = len(hs)

  if pow is not None:
    logv = (-halflife * np.log(2)) / np.array(hs)
    lp = np.log(0.5)
    n = len(hs)
    ivec = (np.arange(n) / (n - 1))

    # ivec = np.arange(n + 1)[1:] / (n)

    def opt(wfinal):
      ws = wfinal**ivec
      return abs(lp - _powerMeanLogW(logv, pow, ws))

    res = minimize_scalar(opt, bounds=[1e-4, 1 - 1e-4])
    assert res.success
    return (res.x**ivec).tolist()

  if n == 1:
    return _makeWs(n, 1.0).tolist()
  i = np.arange(0, n)
  hs = np.array(hs)
  # Avoid divide-by-0 below by skipping first `i` and `hs`
  res = 2**(min((halflife / hs[1:] + log2(0.5)) * (n - 1) / i[1:]))
  return _makeWs(n, res).tolist()


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)

  if model.pred.power:
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda h: abs(lp - predictRecall(model, now=model.pred.lastEncounterMs + 3600e3 * h)),
        bounds=[.01, 1e3],
        tol=1e-12)
    assert res.success
    # print(res)
    return res.x

  return max((lw - lp) * _gammaToMean(*alphaBeta)
             for lw, alphaBeta in zip(model.pred.log2weights, model.pred.halflifeGammas))
  # max above will ALWAYS get at least one result given percentile ∈ (0, 1]


def _gammaToMean(alpha: float, beta: float) -> float:
  return alpha / beta


def _meanVarToGamma(mean, var) -> tuple[float, float]:
  a = mean**2 / var
  b = mean / var
  return a, b


def _makeWs(n: int, finalWeight: float) -> np.ndarray:
  if n == 1:
    return np.array([finalWeight])
  return finalWeight**(np.arange(n) / (n - 1))


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


def _powerMean(v: list[float] | np.ndarray, p: int | float) -> float:
  return float(np.mean(np.array(v)**p)**(1 / p))


def _powerMeanW(v: list[float] | np.ndarray, p: int | float, ws: list[float] | np.ndarray) -> float:
  ws = np.array(ws) / np.sum(ws)
  return float(sum(np.array(v)**p * ws))**(1 / p)
