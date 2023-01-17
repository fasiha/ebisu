import numpy as np
from copy import deepcopy
from typing import Union, Optional

from ebisu.ebisuHelpers import posterior, timeMs
from ebisu.models import BinomialResult, Model, NoisyBinaryResult, success, Predict, Quiz, Result


def initModel(
    wmax: float,
    hmax: float = 1e5,  # 1e5 hours = 11+ years
    n: int = 10,  # only used for initial predictRecall. Can change in updateRecall
    now: Optional[float] = None,
) -> Model:
  """
  Create brand new Ebisu model

  Provide the mean and standard deviation for the initial halflife (units of
  hours) and boost (unitless).

  If you're really lazy and omit standard deviations, we'll pick them to be half
  their respective means (ensures a nice densityw ith non-zero mode and positive
  second-derivative at 0, though you'd get this for `std <= mean / sqrt(2)`).
  
  Optional `now`: milliseconds since the Unix epoch when the fact was learned.
  """
  assert 0 <= wmax <= 1, 'wmax should be between 0 and 1'
  wmax = wmax or np.spacing(1)
  now = now or timeMs()

  hs = np.logspace(0, np.log10(hmax), n)
  ws = _makeWs(n, wmax)
  log2ws = np.log2(ws)
  return Model(
      version=1,
      quiz=Quiz(version=1, results=[], startTimestampMs=[now]),
      pred=Predict(
          version=1,
          lastEncounterMs=now,
          wmax=wmax,
          hmax=hs[-1],
          log2ws=log2ws,
          hs=hs.tolist(),
      ),
  )


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    n: int = 10,
    wmaxPrior: Optional[tuple[float, float]] = None,
    now: Optional[float] = None,
) -> Model:
  now = now or timeMs()
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
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

  if wmaxPrior is None:
    res = [success(m) for m in (ret.quiz.results[-1] if len(ret.quiz.results) else [])]
    successes = sum(res)
    wmaxPrior = (max(2, successes), max(2, len(res) - successes))

  hs = np.logspace(0, np.log10(ret.pred.hmax), n)
  wmaxs = np.linspace(0, 1, 101)

  ps = [posterior(ret, wmax, wmaxPrior, hs)[0] for wmax in wmaxs]
  wmaxMap = wmaxs[np.argmax(ps)]

  ret.pred.lastEncounterMs = now
  ret.pred.wmax = wmaxMap
  ret.pred.hs = hs.tolist()

  ws = _makeWs(n, wmaxMap)
  ret.pred.log2ws = np.log2(ws)

  return ret


def _makeWs(n: int, wmax: float):
  tau = -(n - 1) / (np.log(wmax) or np.spacing(1))
  ws = np.exp(-np.arange(n) / tau)
  return ws


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
  now = now or timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  logPrecall = max([log2w - elapsedHours / h for log2w, h in zip(model.pred.log2ws, model.pred.hs)])
  return logPrecall if logDomain else np.exp2(logPrecall)
