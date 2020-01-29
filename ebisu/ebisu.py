# -*- coding: utf-8 -*-


def predictRecall(prior, tnow, exact=False):
  """Expected recall probability now, given a prior distribution on it. 🍏

  `prior` is a tuple representing the prior distribution on recall probability
  after a specific unit of time has elapsed since this fact's last review.
  Specifically,  it's a 3-tuple, `(alpha, beta, t)` where `alpha` and `beta`
  parameterize a Beta distribution that is the prior on recall probability at
  time `t`.

  `tnow` is the *actual* time elapsed since this fact's most recent review.

  Optional keyword parameter `exact` makes the return value a probability,
  specifically, the expected recall probability `tnow` after the last review: a
  number between 0 and 1. If `exact` is false (the default), some calculations
  are skipped and the return value won't be a probability, but can still be
  compared against other values returned by this function. That is, if
  
  > predictRecall(prior1, tnow1, exact=True) < predictRecall(prior2, tnow2, exact=True)

  then it is guaranteed that

  > predictRecall(prior1, tnow1, exact=False) < predictRecall(prior2, tnow2, exact=False)
  
  The default is set to false for computational efficiency.

  See README for derivation.
  """
  from math import exp
  a, b, t = prior
  dt = tnow / t
  ret = _betalnRatio(a + dt, a, b)
  return exp(ret) if exact else ret


from functools import lru_cache
from .gamma import gammaln


@lru_cache(maxsize=None)
def _gammalnCached(x):
  return gammaln(x)


def _betalnRatio(a1, a, b):
  return gammaln(a1) - gammaln(a1 + b) + _gammalnCached(a + b) - _gammalnCached(a)


def betaln(a, b):
  return _gammalnCached(a) + _gammalnCached(b) - _gammalnCached(a + b)


def updateRecall(prior, result, tnow, rebalance=True, tback=None):
  """Update a prior on recall probability with a quiz result and time. 🍌

  `prior` is same as in `ebisu.predictRecall`'s arguments: an object
  representing a prior distribution on recall probability at some specific time
  after a fact's most recent review.

  `result` is truthy for a successful quiz, falsy otherwise.

  `tnow` is the time elapsed between this fact's last review and the review
  being used to update.

  (The keyword arguments `rebalance` and `tback` are intended for internal use.)

  Returns a new object (like `prior`) describing the posterior distribution of
  recall probability at `tback` (which is an optional input, defaults to `tnow`).
  """
  from math import exp

  (alpha, beta, t) = prior
  if tback is None:
    tback = t
  dt = tnow / t
  et = tnow / tback

  if result:

    if tback == t:
      proposed = alpha + dt, beta, t
      return _rebalace(prior, result, tnow, proposed) if rebalance else proposed

    logmean = _betalnRatio(alpha + dt / et * (1 + et), alpha + dt, beta)
    logm2 = _betalnRatio(alpha + dt / et * (2 + et), alpha + dt, beta)
    mean = exp(logmean)
    var = _subexp(logm2, 2 * logmean)

  else:

    logDenominator = _logsubexp(betaln(alpha, beta), betaln(alpha + dt, beta))
    mean = _subexp(
        betaln(alpha + dt / et, beta) - logDenominator,
        betaln(alpha + dt / et * (et + 1), beta) - logDenominator)
    m2 = _subexp(
        betaln(alpha + 2 * dt / et, beta) - logDenominator,
        betaln(alpha + dt / et * (et + 2), beta) - logDenominator)
    assert m2 > 0
    var = m2 - mean**2

  assert mean > 0
  assert var > 0
  newAlpha, newBeta = _meanVarToBeta(mean, var)
  proposed = newAlpha, newBeta, tback
  return _rebalace(prior, result, tnow, proposed) if rebalance else proposed


def _rebalace(prior, result, tnow, proposed):
  newAlpha, newBeta, _ = proposed
  if (newAlpha > 2 * newBeta or newBeta > 2 * newAlpha):
    roughHalflife = modelToPercentileDecay(proposed, coarse=True)
    return updateRecall(prior, result, tnow, rebalance=False, tback=roughHalflife)
  return proposed


def _logsubexp(a, b):
  """Evaluate `log(exp(a) - exp(b))` preserving accuracy.
  
  Subtract log-domain numbers and return in the log-domain.
  Wraps `scipy.special.logsumexp`.
  """
  from .logsumexp import logsumexp
  return logsumexp([a, b], b=[1, -1])[0]


def _subexp(x, y):
  """Evaluates `exp(x) - exp(y)` a bit more accurately than that. ⚾️

  Subtract log-domain numbers and return in the *linear* domain.
  Similar to `scipy.special.logsumexp` except without the final `log`.
  """
  from math import exp
  maxval = max(x, y)
  return exp(maxval) * (exp(x - maxval) - exp(y - maxval))


def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def modelToPercentileDecay(model, percentile=0.5, coarse=False):
  """When will memory decay to a given percentile? 🏀
  
  Given a memory `model` of the kind consumed by `predictRecall`,
  etc., and optionally a `percentile` (defaults to 0.5, the
  half-life), find the time it takes for memory to decay to
  `percentile`. If `coarse`, the returned time (in the same units as
  `model`) is approximate.
  """
  # Use a root-finding routine in log-delta space to find the delta that
  # will cause the GB1 distribution to have a mean of the requested quantile.
  # Because we are using well-behaved normalized deltas instead of times, and
  # owing to the monotonicity of the expectation with respect to delta, we can
  # quickly scan for a rough estimate of the scale of delta, then do a finishing
  # optimization to get the right value.

  assert (percentile > 0 and percentile < 1)
  from .mingolden import mingolden
  from math import log, exp
  alpha, beta, t0 = model
  logBab = betaln(alpha, beta)
  logPercentile = log(percentile)

  def f(lndelta):
    logMean = betaln(alpha + exp(lndelta), beta) - logBab
    return logMean - logPercentile

  # Scan for a bracket.
  bracket_width = 1.0 if coarse else 6.0
  blow = -bracket_width / 2.0
  bhigh = bracket_width / 2.0
  flow = f(blow)
  fhigh = f(bhigh)
  while flow > 0 and fhigh > 0:
    # Move the bracket up.
    blow = bhigh
    flow = fhigh
    bhigh += bracket_width
    fhigh = f(bhigh)
  while flow < 0 and fhigh < 0:
    # Move the bracket down.
    bhigh = blow
    fhigh = flow
    blow -= bracket_width
    flow = f(blow)

  assert flow > 0 and fhigh < 0

  if coarse:
    return (exp(blow) + exp(bhigh)) / 2 * t0

  sol = mingolden(lambda x: abs(f(x)), blow, bhigh)
  if not sol['converged']:
    raise ValueError('minimization failed to converge')
  t1 = exp(sol['argmin']) * t0
  return t1


def defaultModel(t, alpha=3.0, beta=None):
  """Convert recall probability prior's raw parameters into a model object. 🍗

  `t` is your guess as to the half-life of any given fact, in units that you
  must be consistent with throughout your use of Ebisu.

  `alpha` and `beta` are the parameters of the Beta distribution that describe
  your beliefs about the recall probability of a fact `t` time units after that
  fact has been studied/reviewed/quizzed. If they are the same, `t` is a true
  half-life, and this is a recommended way to create a default model for all
  newly-learned facts. If `beta` is omitted, it is taken to be the same as
  `alpha`.
  """
  return (alpha, beta or alpha, t)
