from math import log10, log2
from time import time_ns
import numpy as np
from copy import deepcopy
from typing import Union, Optional

from .models import BinomialResult, HalflifeGamma, Model, NoisyBinaryResult, Predict, Quiz, Result
from .ebisuHelpers import gammaUpdateBinomial, gammaUpdateNoisy


def initModel(
    halflife: Optional[float] = None,
    finalWeight: Optional[float] = None,
    firstHalflife=1.0,
    finalHalflife=1e5,
    weights: Optional[list[float] | np.ndarray] = None,
    halflifeGammas: Optional[list[HalflifeGamma]] = None,
    n: int = 10,
    now: Optional[float] = None,
) -> Model:
  """
  Create brand new Ebisu model

  Optional `now`: milliseconds since the Unix epoch when the fact was learned.
  """
  now = now if now is not None else timeMs()
  if halflifeGammas is None:
    halflives = np.logspace(log10(firstHalflife), log10(finalHalflife), n).tolist()
    halflifeGammas = [_meanVarToGamma(t, (t * .5)**2) for t in halflives]
  else:
    halflives = [_gammaToMean(alpha, beta) for alpha, beta in halflifeGammas]

  if weights is None:
    if finalWeight is not None:
      weights = _makeWs(n, finalWeight)
    elif halflife is not None:
      weights = _makeWs(n, _halflifeToFinalWeight(halflife, halflives))
    else:
      raise Exception('need weights, finalWeight, or halflife')
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
  for m, updated, log2weight, reached in zip(model.pred.halflifeGammas, updateds,
                                             model.pred.log2weights, model.pred.weightsReached):
    weight = np.exp2(log2weight)
    scal = updated.mean / _gammaToMean(*m)
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
  l = [
      log2w + -elapsedHours / (alpha / beta)
      for (alpha, beta), log2w in zip(model.pred.halflifeGammas, model.pred.log2weights)
  ]
  if extra is not None:
    extra['indiv'] = l
  logPrecall = max(l)
  assert np.isfinite(logPrecall) and logPrecall <= 0
  return logPrecall if logDomain else np.exp2(logPrecall)


def _halflifeToFinalWeight(halflife: float, hs: list[float] | np.ndarray):
  """
  Complement to `hoursForRecallDecay`, which is $halflife(percentile, wmaxMean)$.

  This is $wmaxMean(halflife, percentile=0.5)$.
  """
  n = len(hs)
  i = np.arange(0, n)
  hs = np.array(hs)
  # Avoid divide-by-0 below by skipping first `i` and `hs`
  return 2**(min((halflife / hs[1:] + log2(0.5)) * (n - 1) / i[1:]))


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)
  return max((lw - lp) * _gammaToMean(*alphaBeta)
             for lw, alphaBeta in zip(model.pred.log2weights, model.pred.halflifeGammas))
  # max above will ALWAYS get at least one result given percentile ∈ (0, 1]


def _powerMean(v: list[float] | np.ndarray, p: int | float) -> float:
  assert p > 0
  return float(np.mean(np.array(v)**p)**(1 / p))


def _gammaToMean(alpha: float, beta: float) -> float:
  return alpha / beta


def _meanVarToGamma(mean, var) -> tuple[float, float]:
  a = mean**2 / var
  b = mean / var
  return a, b


def _makeWs(n: int, finalWeight: float) -> np.ndarray:
  return finalWeight**(np.arange(n) / (n - 1))


def timeMs() -> float:
  return time_ns() / 1_000_000
