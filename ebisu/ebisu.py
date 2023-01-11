from scipy.linalg import lstsq  #type:ignore
import numpy as np  # type:ignore
from scipy.stats import gamma as gammarv  # type: ignore
from scipy.stats import uniform as uniformrv  # type: ignore
from scipy.special import gammaln, logsumexp  #type: ignore
from typing import Any, Union, Optional, Callable
from copy import deepcopy

from ebisu.ebisuHelpers import (currentHalflifePrior, gammaUpdateBinomial, gammaUpdateNoisy,
                                _intGammaPdfExp, posterior, clampLerp, success, timeMs)
from ebisu.gammaDistribution import (gammaToMean, meanVarToGamma, _weightedGammaEstimate)
from ebisu.models import (BinomialResult, Model, NoisyBinaryResult, Predict, Probability, Quiz,
                          Result)

logsumexp: Callable = logsumexp


def initModel(initHlMean: float, initHlStd: float, boostMean: float, boostStd: float,
              now: Optional[float]) -> Model:
  """
  Create brand new Ebisu model

  Provide the mean and standard deviation for the initial halflife (units of
  hours) and boost (unitless).
  
  Optional: provide the timestamp in milliseconds since the Unix epoch.
  """
  hl0 = meanVarToGamma(initHlMean, initHlStd**2)
  b = meanVarToGamma(boostMean, boostStd**2)
  assert gammaToMean(*hl0) > 0, 'init halflife mean should be positive'
  assert gammaToMean(*b) >= 1.0, 'boost mean should be >= 1'
  now = now or timeMs()
  return Model(
      quiz=Quiz(results=[], startStrengths=[], startTimestampMs=[now]),
      prob=Probability(initHlPrior=hl0, boostPrior=b, initHl=hl0, boost=b),
      pred=Predict(lastEncounterMs=now, currentHalflifeHours=gammaToMean(*hl0), logStrength=0.0))


def resetHalflife(
    model: Model,
    initHlMean: float,
    initHlStd: float,
    now: Union[float, None] = None,
    strength: float = 1.0,
) -> Model:
  now = now or timeMs()

  ret = deepcopy(model)
  ret.quiz.results.append([])
  ret.quiz.startStrengths.append([])
  ret.quiz.startTimestampMs.append(now)

  ret.prob.initHlPrior = meanVarToGamma(initHlMean, initHlStd**2)
  ret.pred.currentHalflifeHours = initHlMean
  ret.pred.lastEncounterMs = now
  ret.pred.logStrength = np.log(strength)
  return ret


def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    now: Union[None, float] = None,
    q0: Union[None, float] = None,
    reinforcement: float = 1.0,
    left=0.3,
    right=1.0,
) -> Model:
  now = now or timeMs()
  (a, b), totalBoost = currentHalflifePrior(model)
  t = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  resultObj: Union[NoisyBinaryResult, BinomialResult]

  if (0 < successes < 1):
    assert total == 1, "float `successes` implies total==1"
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5
    z = successes >= 0.5
    updated = gammaUpdateNoisy(a, b, t, q1, q0, z)
    resultObj = NoisyBinaryResult(result=successes, q1=q1, q0=q0, hoursElapsed=t)

  else:  # int, or float outside (0, 1) band
    assert successes == np.floor(successes), "float `successes` must be between 0 and 1"
    assert successes >= 0, "negative `successes` meaningless"
    assert total > 0, "positive binomial trials"
    k = int(successes)
    n = total
    updated = gammaUpdateBinomial(a, b, t, k, n)
    resultObj = BinomialResult(successes=k, total=n, hoursElapsed=t)

  mean, newAlpha, newBeta = (updated.mean, updated.a, updated.b)

  ret = deepcopy(model)  # clone
  ret.prob.initHl = (newAlpha, newBeta * totalBoost)
  _appendQuizImpure(ret, resultObj, reinforcement)
  boostMean = gammaToMean(*ret.prob.boost)
  if success(resultObj):
    boostedHl = mean * clampLerp(left * model.pred.currentHalflifeHours,
                                 right * model.pred.currentHalflifeHours, 1, max(1.0, boostMean), t)
  else:
    boostedHl = mean

  if reinforcement > 0:
    ret.pred.currentHalflifeHours = boostedHl
    ret.pred.lastEncounterMs = now
    ret.pred.logStrength = np.log(reinforcement)
  else:
    ret.pred.currentHalflifeHours = boostedHl

  return ret


def _expand(thresh: float, minBoost: float, minHalflife: float, maxBoost: float, maxHalflife: float,
            size: int, lpVector):
  bvec = np.linspace(minBoost, maxBoost, int(np.sqrt(size)))
  hvec = np.linspace(minHalflife, maxHalflife, int(np.sqrt(size)))
  bs, hs = np.meshgrid(bvec, hvec)
  posteriors = lpVector(bs, hs)
  nz0, nz1 = np.nonzero(np.diff(np.sign(posteriors - np.max(posteriors) + abs(thresh)), axis=1))
  return bvec[np.max(nz1)] * 1.2, hvec[np.max(nz0)] * 1.2


def updateRecallHistory(
    model: Model,
    left=0.3,
    right=1.0,
    size=10_000,
    likelihoodFitWeight=0.9,
    likelihoodFitPower=2,
    likelihoodFitSize=600,
) -> Model:
  return updateRecallHistoryDebug(
      model=model,
      left=left,
      right=right,
      size=size,
      likelihoodFitWeight=likelihoodFitWeight,
      likelihoodFitPower=likelihoodFitPower,
      likelihoodFitSize=likelihoodFitSize,
  )[0]


# I'd like an automated way to ensure this function has the same args as above.
def updateRecallHistoryDebug(
    model: Model,
    left=0.3,
    right=1.0,
    size=10_000,
    likelihoodFitWeight=0.9,
    likelihoodFitPower=2,
    likelihoodFitSize=600,
) -> tuple[Model, dict]:
  ret = deepcopy(model)
  if len(ret.quiz.results[-1]) <= 2:
    # not enough quizzes to update boost
    return (ret, dict())

  assert likelihoodFitPower >= 0, "likelihoodFitPower non-negative"
  assert likelihoodFitSize > 0, "likelihoodFitSize positive"
  assert 0 < likelihoodFitWeight <= 1, "likelihoodFitWeight between 0 and 1"
  unifWeight = 1 - likelihoodFitWeight

  lpVector = np.vectorize(lambda b, h: posterior(b, h, ret, left, right), otypes=[float])

  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior
  # Look at the prior for rough min/max
  minBoost, maxBoost = gammarv.ppf([0.01, 0.9999], ab, scale=1 / bb)
  minHalflife, maxHalflife = gammarv.ppf([0.01, 0.9999], ah, scale=1 / bh)
  # and then use those to get the posterior's bulk of the posterior's support
  maxBoost, maxHalflife = _expand(10, minBoost / 3, minHalflife / 3, maxBoost * 3, maxHalflife * 3,
                                  likelihoodFitSize, lpVector)
  minBoost = 0
  minHalflife = 0

  def generatePosteriorSurface(bMinmax, hMinmax, f, size):
    bs, hs = np.random.rand(2, size)
    bs = bs * (bMinmax[1] - bMinmax[0]) + bMinmax[0]
    hs = hs * (hMinmax[1] - hMinmax[0]) + hMinmax[0]
    zs = f(bs, hs)
    return bs, hs, zs

  try:
    bs, hs, posteriors = generatePosteriorSurface([minBoost, maxBoost], [minHalflife, maxHalflife],
                                                  lpVector, likelihoodFitSize)
    fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=likelihoodFitPower)
  except AssertionError as e:
    if "positive gamma parameters" in e.args[0]:
      print('posterior fit failed but trying with more samples:', e)
      likelihoodFitSize *= 2

      bs, hs, posteriors = generatePosteriorSurface([minBoost, maxBoost],
                                                    [minHalflife, maxHalflife], lpVector,
                                                    likelihoodFitSize)
      fit = _fitJointToTwoGammas(bs, hs, posteriors, weightPower=likelihoodFitPower)
    else:
      raise e

  def mix(aWeight, aComponent, bComponent) -> dict[str, Callable]:

    def gen(size: int):
      numA = np.sum(np.random.rand(size) < aWeight)
      a = aComponent.rvs(size=numA)
      b = bComponent.rvs(size=size - numA)
      ret = np.hstack([a, b])
      np.random.shuffle(ret)
      return ret

    def logpdf(x):
      lpA = aComponent.logpdf(x)
      lpB = bComponent.logpdf(x)
      return logsumexp(np.vstack([lpA, lpB]), axis=0, b=np.array([[aWeight, 1 - aWeight]]).T)

    return dict(gen=gen, logpdf=logpdf)

  xmix = mix(unifWeight, uniformrv(loc=minBoost, scale=maxBoost - minBoost),
             gammarv(fit['alphax'], scale=1 / fit['betax']))  # type: ignore
  ymix = mix(unifWeight, uniformrv(loc=minHalflife, scale=maxHalflife - minHalflife),
             gammarv(fit['alphay'], scale=1 / fit['betay']))  # type: ignore

  betterFit = _monteCarloImprove(
      generateX=xmix['gen'],
      generateY=ymix['gen'],
      logpdfX=xmix['logpdf'],
      logpdfY=ymix['logpdf'],
      logposterior=lpVector,
      size=size,
  )
  # update prior(s)
  ret.prob.boost = (betterFit['alphax'], betterFit['betax'])
  ret.prob.initHl = (betterFit['alphay'], betterFit['betay'])
  # update SQL-friendly scalars
  bmean, hmean = [gammaToMean(*prior) for prior in [ret.prob.boost, ret.prob.initHl]]
  _, extra = posterior(bmean, hmean, ret, left, right, extra=True)  # type: ignore
  ret.pred.currentHalflifeHours = extra['currentHalflife']
  return ret, dict(
      origfit=fit,
      betterFit=betterFit,
      bs=bs,
      hs=hs,
      posteriors=posteriors,
      size=size,
      likelihoodFitSize=likelihoodFitSize,
  )


def _appendQuizImpure(model: Model, result: Result, startStrength: float) -> None:
  # IMPURE

  if len(model.quiz.results) == 0:
    model.quiz.results = [[result]]
  else:
    model.quiz.results[-1].append(result)

  if len(model.quiz.startStrengths) == 0:
    model.quiz.startStrengths = [[startStrength]]
  else:
    model.quiz.startStrengths[-1].append(startStrength)


LN2 = np.log(2)
HOURS_PER_MILLISECONDS = 1 / 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec


def _fitJointToTwoGammas(x: Union[list[float], np.ndarray],
                         y: Union[list[float], np.ndarray],
                         logPosterior: Union[list[float], np.ndarray],
                         weightPower=0.0) -> dict:
  """Fit two independent Gammas from samples and log-likelihood

  This is a fast linear system of equations that fits five parameters (two for
  each Gamma, and one offset).

  `weightPower` can be used to scale the system to give more importance to
  higher likelihoods. Zero, the default, will not scale it. 1 and 2 are
  reasonable alternatives.
  """
  # four-dimensional weighted least squares
  x = np.array(x)
  y = np.array(y)
  logPosterior = np.array(logPosterior)
  assert x.size == y.size
  assert x.size == logPosterior.size

  # If `N = x.size`, $A$ is N by 5.
  A = np.vstack([np.log(x), -x, np.log(y), -y, np.ones_like(x)]).T
  weights = np.exp((logPosterior - np.max(logPosterior)) * weightPower)
  sol = lstsq(A * weights[:, np.newaxis], weights * logPosterior)
  t = sol[0]
  alphax = t[0] + 1
  betax = t[1]
  alphay = t[2] + 1
  betay = t[3]

  assert all(x > 0 for x in [alphax, betax, alphay, betay]), 'positive gamma parameters'
  return dict(
      sol=sol,
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      meanX=gammaToMean(alphax, betax),
      meanY=gammaToMean(alphay, betay))


def _monteCarloImprove(generateX: Callable[[int], np.ndarray],
                       generateY: Callable[[int], np.ndarray],
                       logpdfX: Callable[[np.ndarray], np.ndarray],
                       logpdfY: Callable[[np.ndarray], np.ndarray],
                       logposterior: Callable,
                       size=10_000) -> dict[str, Any]:
  x = generateX(size)
  y = generateY(size)
  logp = logposterior(x, y)
  logw = logp - (logpdfX(x) + logpdfY(y))
  w = np.exp(logw)
  alphax, betax = _weightedGammaEstimate(x, w)
  alphay, betay = _weightedGammaEstimate(y, w)
  return dict(
      x=[alphax, betax],
      y=[alphay, betay],
      alphax=alphax,
      betax=betax,
      alphay=alphay,
      betay=betay,
      logw=logw,
      logp=logp,
      xs=x,
      ys=y)


def predictRecall(model: Model, now: Union[float, None] = None, logDomain=True) -> float:
  now = now or timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS
  logPrecall = -elapsedHours / model.pred.currentHalflifeHours * LN2 + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)


def _predictRecallBayesian(model: Model, now: Union[float, None] = None, logDomain=True) -> float:
  now = now or timeMs()
  elapsedHours = (now - model.pred.lastEncounterMs) * HOURS_PER_MILLISECONDS

  (a, b), _totalBoost = currentHalflifePrior(model)
  logPrecall = _intGammaPdfExp(
      a, b, elapsedHours * LN2,
      logDomain=True) + a * np.log(b) - gammaln(a) + model.pred.logStrength
  return logPrecall if logDomain else np.exp(logPrecall)
