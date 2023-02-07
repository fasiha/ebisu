from math import log10, log2
import numpy as np
from copy import deepcopy
from typing import Union, Optional
from scipy.optimize import minimize_scalar  # type: ignore

from .ebisuHelpers import _makeWs, posterior, timeMs
from .expectionMaxScaledPowBeta import expectationMaxScaledPowBeta
from .models import BinomialResult, Ebisu2Model, Model, NoisyBinaryResult, WeightsFormat, Predict, Quiz, Result

import ebisu2beta
from ebisu3boost.ebisuHelpers import gammaUpdateBinomial, gammaUpdateNoisy
from ebisu3boost.gammaDistribution import meanVarToGamma


def initModel(
    wmaxMean: Optional[float] = None,
    initHlMean: Optional[float] = None,
    hmin: float = 1,
    hmax: float = 1e5,  # 1e5 hours = 11+ years
    n: int = 10,
    initAlphaBeta=2.0,
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

  hs = np.logspace(log10(hmin), log10(hmax), n).tolist()
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
          # beta-on-recall models
          betaWeights=ws.tolist(),
          betaModels=[(initAlphaBeta, initAlphaBeta, t) for t in hs],
          betaWeightsReached=[x == 0 for x in range(n)],
          # gamma-on-hl models
          gammaWeights=ws.tolist(),
          gammaParams=[meanVarToGamma(t, (t * .5)**2) for t in hs],
          gammaWeightsReached=[x == 0 for x in range(n)],
      ),
  )


def _genForSql(log2ws, hs) -> list[tuple[float, float]]:
  return [(lw, h * 3600e3) for lw, h in zip(log2ws, hs)]


def updateRecallGammas(
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
    updateds = [gammaUpdateNoisy(a, b, t, q1, q0, z) for a, b in model.pred.gammaParams]
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t)
  else:
    assert successes == np.floor(successes), "float `successes` must have `total=1`"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    k = int(successes)
    n = total
    updateds = [gammaUpdateBinomial(a, b, t, k, n) for a, b in model.pred.gammaParams]
    resultObj = BinomialResult(successes=int(successes), total=total, hoursElapsed=t)

  ret = deepcopy(model)  # clone
  _appendQuizImpure(ret, resultObj)

  SCALE_THRESH = 0.95
  newModels: list[tuple[float, float]] = []
  newWeights: list[float] = []
  newReached: list[bool] = []
  for m, updated, weight, reached in zip(model.pred.gammaParams, updateds, model.pred.gammaWeights,
                                         model.pred.gammaWeightsReached):
    scal = (updated.a / updated.b) / (m[0] / m[1])
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
        newWeights.append(min(powerMean([weight, scal], 2), 1))
        newReached.append(True)

  ret.pred.lastEncounterMs = now

  ret.pred.gammaParams = newModels
  ret.pred.gammaWeights = newWeights
  ret.pred.gammaWeightsReached = newReached

  return ret


def updateRecallBetas(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    now: Optional[float] = None,
) -> Model:
  now = now or timeMs()
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  updatedModels = [
      ebisu2beta.updateRecall(prior, successes=successes, total=total, tnow=t, q0=q0)
      for prior in model.pred.betaModels
  ]
  updatedHls = [ebisu2beta.modelToPercentileDecay(m) for m in updatedModels]
  hls = [ebisu2beta.modelToPercentileDecay(m) for m in model.pred.betaModels]
  hlScales = [n / o for n, o in zip(updatedHls, hls)]
  SCALE_THRESH = 0.95
  newModels: list[Ebisu2Model] = []
  newWeights: list[float] = []
  newReached: list[bool] = []
  for m, updated, scal, weight, reached in zip(model.pred.betaModels, updatedModels, hlScales,
                                               model.pred.betaWeights,
                                               model.pred.betaWeightsReached):
    if reached:
      newModels.append(updated)
      newWeights.append(min(weight * scal if scal > 1 else weight, 1))
      newReached.append(True)
    else:
      if (scal < 1 and scal < SCALE_THRESH) or (scal > 1 and scal < 1.01):
        # early failure or success
        newModels.append(m)
        newWeights.append(weight)
        newReached.append(False)
      else:
        newModels.append(updated)
        newWeights.append(min(powerMean([weight, scal], 2), 1))
        newReached.append(True)

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

  ret.pred.lastEncounterMs = now

  ret.pred.betaModels = newModels
  ret.pred.betaWeights = newWeights
  ret.pred.betaWeightsReached = newReached

  return ret


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
  res = minimize_scalar(
      lambda wmax: -(posterior(ret, wmax, wmaxPrior, hs)[0]), bracket=[0, 1], bounds=[0, 1])
  assert res.success
  wmaxMap = res.x

  ret.pred.lastEncounterMs = now
  ret.pred.wmaxMean = wmaxMap
  ret.pred.hs = hs.tolist()

  ws = _makeWs(len(hs), wmaxMap, model.pred.format, model.pred.m)
  ret.pred.log2ws = np.log2(ws).tolist()
  ret.pred.forSql = _genForSql(ret.pred.log2ws, ret.pred.hs)

  return ret


def _appendQuizImpure(model: Model, result: Result) -> None:
  if len(model.quiz.results) == 0:
    model.quiz.results = [[result]]
  else:
    model.quiz.results[-1].append(result)


HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
LN2 = np.log(2)


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


def predictRecallGammas(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    extra: Optional[dict] = None,
) -> float:
  # To be more exact, you'd need to use https://github.com/spedygiorgio/mbbefd/blob/e8b1ef11a6289dbf23701860afebd54dc70c7b99/R/distr-GB1.R#L23-L35 and https://math.stackexchange.com/questions/3679707/expected-maximum-of-beta-random-variables
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  if logDomain:
    l = [
        np.log(w) + -elapsedHours / (alpha / beta) * LN2
        for (alpha, beta), w in zip(model.pred.gammaParams, model.pred.gammaWeights)
    ]
    if extra is not None:
      extra['indiv'] = l
    logPrecall = max(l)
    assert np.isfinite(logPrecall) and logPrecall <= 0
    return logPrecall
  else:
    l = [
        w * np.exp(-elapsedHours / (alpha / beta) * LN2)
        for (alpha, beta), w in zip(model.pred.gammaParams, model.pred.gammaWeights)
    ]
    if extra is not None:
      extra['indiv'] = l

    ret = max(l)
    assert np.isfinite(ret) and ret >= 0 and ret <= 1
    return ret


def predictRecallBetas(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
    extra: Optional[dict] = None,
) -> float:
  # To be more exact, you'd need to use https://github.com/spedygiorgio/mbbefd/blob/e8b1ef11a6289dbf23701860afebd54dc70c7b99/R/distr-GB1.R#L23-L35 and https://math.stackexchange.com/questions/3679707/expected-maximum-of-beta-random-variables
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"
  if logDomain:
    l = [
        np.log(w) + ebisu2beta.predictRecall(prior, elapsedHours, exact=not logDomain)
        for prior, w in zip(model.pred.betaModels, model.pred.betaWeights)
    ]
    if extra is not None:
      extra['indiv'] = l
    logPrecall = max(l)
    return logPrecall
  else:
    l = [
        w * ebisu2beta.predictRecall(prior, elapsedHours, exact=not logDomain)
        for prior, w in zip(model.pred.betaModels, model.pred.betaWeights)
    ]
    if extra is not None:
      extra['indiv'] = l

    return max(l)


def hoursForRecallDecayBetas(model: Model, percentile=0.5) -> float:
  logp = np.log(percentile)
  res = minimize_scalar(
      lambda h: abs(logp - predictRecallBetas(model, model.pred.lastEncounterMs + 3600e3 * h)),
      [0, 1e7], [0, 1e7])
  assert res.success
  return res.x


def hoursForRecallDecay(model: Model, percentile=0.5) -> float:
  "How many hours for this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  lp = log2(percentile)
  return max((lw - lp) * h for lw, h in zip(model.pred.log2ws, model.pred.hs))
  # max above will ALWAYS get at least one result given percentile ∈ (0, 1]


def _halflifeToWmax(halflife: float, hmin: float = 1.0, hmax: float = 1e5, n: int = 10):
  """
  Complement to `hoursForRecallDecay`, which is $halflife(percentile, wmaxMean)$.

  This is $wmaxMean(halflife, percentile=0.5)$.
  """
  hs = np.logspace(log10(hmin), log10(hmax), n)
  i = np.arange(0, len(hs))
  # Avoid divide-by-0 below by skipping first `i` and `hs`
  return 2**(min((halflife / hs[1:] + log2(0.5)) * (n - 1) / i[1:]))


def _halflifeToWmaxPrior(h: float,
                         hmin: float = 1.0,
                         hmax: float = 1e5,
                         n: int = 10) -> tuple[float, float]:
  wmaxMean = _halflifeToWmax(h, hmin=hmin, hmax=hmax, n=n)
  # find (α, β) such that Beta(α, β) is lowest variance AND α>=2, β>=2
  a, b = _fitMaxVarBetaGivenMeanAndABMins(wmaxMean, 2.0, 2.0)
  # I mean sure β might be 300 (for e.g., `h=1` hour) but let's cap this for
  # sanity. As a downside, this does mean the mean of the distribution
  # returned might not match what's passed in.
  return (min(a, 10 + np.log10(a)), min(b, 10 + np.log10(b)))


def _fitMaxVarBetaGivenMeanAndABMins(mean: float, amin: float, bmin: float):
  """Find the max-variance Beta distribution given mean and parameter minima

  For any given mean, an infinite number of Beta distributions parameterized by
  `(a, b)` can be found. This function finds the `(a, b)` parameters that have the
  max variance given an additional constraint: `a >= amin` and `b >= bmin`. This
  can be useful for example to find Beta distributions that are unimodal, when
  `amin = bmin = 2`.

  This is a fast arithmetic operation: we find the `b` for `a=amin` and the `a`
  for `b=bmin` and pick the one that yields positive parameters. This may seem
  too simple but it winds up (I think) giving the max variance because a Beta's
  variance increases as a and b both head to 0, so `amin` and `bmin` constraints
  make a L, a right angle box in the a-b plane. The a, b line for a fixed mean
  will hit that L-shaped "box" in only one place, which will be the highest
  variance because all higher-variance (a, b)s will be outside (left/below) the
  box.
  """
  assert 0 < mean < 1, 'mean ∈ (0, 1)'
  assert amin > 0 and bmin > 0, 'amax and bmax both > 0'
  # following are solutions for μ = a/(a+b) such that a=amin and b=bmin
  b = -amin + amin / mean  # a = amin and mean fully define b
  if b >= bmin:
    return (2.0, b)
  a = -bmin * mean / (mean - 1)  # b = bmin and mean fully define a
  if a >= amin:
    return (a, 2.0)
  # One of the above will happen but...
  raise Exception('unable to find prior')


def predictRecallBayesian(
    model: Model,
    wmaxVar: Optional[float] = None,
    wmaxMean: Optional[float] = None,
    now: Optional[float] = None,
) -> float:
  now = now if now is not None else timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  assert elapsedHours >= 0, "cannot go back in time"

  wmaxMean = wmaxMean if wmaxMean is not None else model.pred.wmaxMean
  if wmaxVar:
    assert wmaxVar > 0, 'variance must be positive'
    alpha, beta = _meanVarToBeta(wmaxMean, wmaxVar)
  else:
    alpha, beta = _fitMaxVarBetaGivenMeanAndABMins(wmaxMean, 2, 2)
    # we could cap alpha,beta to e.g., 10, like we do above, but this is just for prediction
  n = len(model.pred.hs)
  # print(f'{alpha=}, {beta=}')
  return expectationMaxScaledPowBeta(
      np.log(2) * -elapsedHours / np.array(model.pred.hs),
      np.arange(n) / (n - 1),
      alpha,
      beta,
      avecLog=True)
  # above equivalent to:
  # return expectationMaxScaledPowBeta(
  #     np.exp2(-elapsedHours / np.array(model.pred.hs)),
  #     np.arange(n) / (n - 1), alpha, beta)


def _meanVarToBeta(mean: float, var: float) -> tuple[float, float]:
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def powerMean(v: list[float] | np.ndarray, p: int | float) -> float:
  assert p > 0
  return float(np.mean(np.array(v)**p)**(1 / p))