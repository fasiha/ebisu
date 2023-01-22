from math import log2
import numpy as np
from copy import deepcopy
from typing import Union, Optional
from scipy.optimize import minimize_scalar  # type: ignore

from ebisu.ebisuHelpers import posterior, timeMs
from ebisu.models import BinomialResult, Model, NoisyBinaryResult, WeightsFormat, success, Predict, Quiz, Result


def initModel(
    wmaxMean: Optional[float] = None,
    hmin: float = 1,
    hmax: float = 1e5,  # 1e5 hours = 11+ years
    n: int = 10,
    initHlMean: Optional[float] = None,
    now: Optional[float] = None,
    format: WeightsFormat = "exp",
    m: Optional[float] = None,
) -> Model:
  """
  Create brand new Ebisu model

  Optional `now`: milliseconds since the Unix epoch when the fact was learned.
  """
  if wmaxMean is None:
    assert initHlMean, "must provide wmaxMean or initHlMean"
    wmaxMean = _halflifeToWmax(initHlMean, hmin=hmin, hmax=hmax, n=n)

  assert wmaxMean and 0 <= wmaxMean <= 1, 'wmaxMean should be between 0 and 1'
  wmaxMean = wmaxMean or np.spacing(1)
  now = now if now is not None else timeMs()

  hs = np.logspace(np.log10(hmin), np.log10(hmax), n).tolist()
  ws = _makeWs(n, wmaxMean, format, m)
  log2ws = np.log2(ws).tolist()
  return Model(
      version=1,
      quiz=Quiz(version=1, results=[[]], startTimestampMs=[now]),
      pred=Predict(
          version=1,
          lastEncounterMs=now,
          wmaxMean=wmaxMean,
          log2ws=log2ws,
          hs=hs,
          forSql=_genForSql(log2ws, hs),
          # custom
          format=format,
          m=m,
          initHlMean=initHlMean,
      ),
  )


def _genForSql(log2ws, hs) -> list[tuple[float, float]]:
  return [(lw, h * 3600e3) for lw, h in zip(log2ws, hs)]


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    wmaxPrior: Optional[tuple[float, float]] = None,
    now: Optional[float] = None,
) -> Model:
  now = now or timeMs()
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS

  if wmaxPrior is None:
    # don't even look at the result, this is supposed to be a prior
    nq = len(model.quiz.results[0])
    # Before the first quiz, just use this quiz time and the initHlMean if provided
    # AFTER that, use past quizzes' times and initHlMean if provided
    maxh = max([q.hoursElapsed for q in model.quiz.results[0]] +
               [t if nq == 0 else 0, model.pred.initHlMean or 0])
    wmaxPrior = _halflifeToWmaxPrior(maxh)
    # Note I don't use this or any past results, just past times. Given
    # http://www.stat.columbia.edu/~gelman/research/published/entropy-19-00555-v2.pdf
    # ("The Prior Can Often Only Be Understood in the Context of the Likelihood"
    # by Gelman, Simpson, Betancourt, 2017) I think this is ok

  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if total == 1:
    assert (0 <= successes <= 1), "`total=1` implies successes in [0, 1]"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t)

  else:
    assert successes == np.floor(successes), "float `successes` must have `total=1`"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    resultObj = BinomialResult(successes=int(successes), total=total, hoursElapsed=t)

  ret = deepcopy(model)  # clone
  _appendQuizImpure(ret, resultObj)

  hs = np.array(ret.pred.hs)
  res = minimize_scalar(lambda wmax: -(posterior(ret, wmax, wmaxPrior, hs)[0]), [0, 1], [0, 1])
  assert res.success
  wmaxMap = res.x

  ret.pred.lastEncounterMs = now
  ret.pred.wmaxMean = wmaxMap
  ret.pred.hs = hs.tolist()

  ws = _makeWs(len(hs), wmaxMap, model.pred.format, model.pred.m)
  ret.pred.log2ws = np.log2(ws).tolist()
  ret.pred.forSql = _genForSql(ret.pred.log2ws, ret.pred.hs)

  return ret


def _makeWs(n: int, wmax: float, format: WeightsFormat, m: Optional[float] = None) -> np.ndarray:
  if format == "exp":
    tau = -(n - 1) / (np.log(wmax) or np.spacing(1))
    return np.exp(-np.arange(n) / tau)
  elif format == "rational":
    assert m and (m > 0), "m must be positive"
    return 1 + (wmax - 1) * (np.arange(n) / (n - 1))**m
  raise Exception('unknown format')


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
) -> float:
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  logPrecall = max(log2w - elapsedHours / h for log2w, h in zip(model.pred.log2ws, model.pred.hs))
  return logPrecall if logDomain else np.exp2(logPrecall)


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)
  return max((lw - lp) * h for lw, h in zip(model.pred.log2ws, model.pred.hs))
  # max above will ALWAYS get at least one result given percentile ∈ (0, 1]


def _halflifeToWmax(h: float, hmin: float = 1.0, hmax: float = 1e5, n: int = 10):
  res = minimize_scalar(
      lambda w: abs(h - hoursForRecallDecay(initModel(w, hmin=hmin, hmax=hmax, n=n))), [0, 1],
      [0, 1])
  assert res.success, "wmax found s.t. h is a halflife"
  return res.x


def _halflifeToWmaxPrior(h: float,
                         hmin: float = 1.0,
                         hmax: float = 1e5,
                         n: int = 10) -> tuple[float, float]:
  wmaxMean = _halflifeToWmax(h, hmin=hmin, hmax=hmax, n=n)
  # find (α, β) such that Beta(α, β) is lowest variance AND α>=2, β>=2

  # following are solutions for μ = a/(a+b) such that a=2 and b=2, which will
  # be the maximum-var solution (proof by handwaving)
  b = -2 + 2 / wmaxMean  # beta = 2
  if b >= 2:
    return (2.0, min(b, 10 + np.log10(b)))
    # I mean sure β might be 300 (for e.g., `h=1` hour) but let's cap this for sanity

  a = -2 * wmaxMean / (wmaxMean - 1)  # alpha = 2
  if a >= 2:
    return (min(a, 10 + np.log10(a)), 2.0)
  raise Exception('unable to find prior')
