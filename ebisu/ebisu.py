# -*- coding: utf-8 -*-

from scipy.special import betaln, logsumexp
import numpy as np


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
  from numpy import exp
  a, b, t = prior
  dt = tnow / t
  ret = betaln(a + dt, b) - _cachedBetaln(a, b)
  return exp(ret) if exact else ret


_BETALNCACHE = {}


def _cachedBetaln(a, b):
  "Caches `betaln(a, b)` calls in the `_BETALNCACHE` dictionary."
  if (a, b) in _BETALNCACHE:
    return _BETALNCACHE[(a, b)]
  x = betaln(a, b)
  _BETALNCACHE[(a, b)] = x
  return x
def binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def updateRecall(prior, successes, total, tnow, rebalance=True, tback=None):
  """Update a prior on recall probability with a quiz result and time. 🍌

  `prior` is same as in `ebisu.predictRecall`'s arguments: an object
  representing a prior distribution on recall probability at some specific time
  after a fact's most recent review.

  `successes` is the number of times the user *successfully* exercised this
  memory during this review session, out of `n` attempts. Therefore, `0 <=
  successes <= total` and `1 <= total`.

  If the user was shown this flashcard only once during this review session,
  then `total=1`. If the quiz was a success, then `successes=1`, else
  `successes=0`.
  
  If the user was shown this flashcard *multiple* times during the review
  session (e.g., Duolingo-style), then `total` can be greater than 1.

  `tnow` is the time elapsed between this fact's last review and the review
  being used to update.

  (The keyword arguments `rebalance` and `tback` are intended for internal use.)

  Returns a new object (like `prior`) describing the posterior distribution of
  recall probability at `tback` (which is an optional input, defaults to
  `tnow`).

  N.B. This function is tested for numerical stability for small `total < 5`. It
  may be unstable for much larger `total`.

  N.B.2. This function may throw an assertion error upon numerical instability.
  This can happen if the algorithm is *extremely* surprised by a result; for
  example, if `successes=0` and `total=5` (complete failure) when `tnow` is very
  small compared to the halflife encoded in `prior`. Calling functions are asked
  to call this inside a try-except block and to handle any possible
  `AssertionError`s in a manner consistent with user expectations, for example,
  by faking a more reasonable `tnow`. Please open an issue if you encounter such
  exceptions for cases that you think are reasonable.
  """
  assert (0 <= successes and successes <= total and 1 <= total)

  (alpha, beta, t) = prior
  if tback is None:
    tback = t
  dt = tnow / t
  et = tback / tnow

  binomlns = [binomln(total - successes, i) for i in range(total - successes + 1)]
  logDenominator, logMeanNum, logM2Num = [
      logsumexp([
          binomlns[i] + betaln(beta, alpha + dt * (successes + i) + m * dt * et)
          for i in range(total - successes + 1)
      ],
                b=[(-1)**i
                   for i in range(total - successes + 1)])
      for m in range(3)
  ]
  mean = np.exp(logMeanNum - logDenominator)
  m2 = np.exp(logM2Num - logDenominator)

  message = dict(
      prior=prior, successes=successes, total=total, tnow=tnow, rebalance=rebalance, tback=tback)
  assert mean > 0, message
  assert m2 > 0, message

  meanSq = np.exp(2 * (logMeanNum - logDenominator))
  var = m2 - meanSq
  assert var > 0, message
  newAlpha, newBeta = _meanVarToBeta(mean, var)
  proposed = newAlpha, newBeta, tback
  return _rebalace(prior, successes, total, tnow, proposed) if rebalance else proposed


def _rebalace(prior, k, n, tnow, proposed):
  newAlpha, newBeta, _ = proposed
  if (newAlpha > 2 * newBeta or newBeta > 2 * newAlpha):
    roughHalflife = modelToPercentileDecay(proposed, coarse=True)
    return updateRecall(prior, k, n, tnow, rebalance=False, tback=roughHalflife)
  return proposed


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
  from scipy.special import betaln
  from scipy.optimize import root_scalar
  alpha, beta, t0 = model
  logBab = betaln(alpha, beta)
  logPercentile = np.log(percentile)

  def f(lndelta):
    logMean = betaln(alpha + np.exp(lndelta), beta) - logBab
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
    return (np.exp(blow) + np.exp(bhigh)) / 2 * t0

  sol = root_scalar(f, bracket=[blow, bhigh])
  t1 = np.exp(sol.root) * t0
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
