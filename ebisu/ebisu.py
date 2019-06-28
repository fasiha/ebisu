# -*- coding: utf-8 -*-

def predictRecall(prior, tnow, exact=False, independent=None):
  """Expected recall probability now, given a prior distribution on it. 🍏

  `prior` is a tuple representing the prior distribution on recall probability
  after a specific unit of time has elapsed since this fact's last review.
  Specifically,  it's a 3-tuple, `(alpha, beta, t)` where `alpha` and `beta`
  parameterize a Beta distribution that is the prior on recall probability at
  time `t`.

  `tnow` is the *actual* time elapsed since this fact's most recent review.

  Optional keyword paramter `exact` makes the return value a probability, specifically, the expected recall probability `tnow` after the last review: a number between 0 and 1. If `exact` is false (the default), some calculations are skipped and the return value won't be a probability, but can still be compared against other values returned by this function. That is, if `predictRecall(prior1, tnow1, exact=True) < predictRecall(prior2, tnow2, exact=True)`, then it is guaranteed that `predictRecall(prior1, tnow1, exact=False) < predictRecall(prior2, tnow2, exact=False)`. The default is set to false for computational reasons.

  Optional keyword parameter `independent` is a precalculated number that is only dependent on `prior` and independent of `tnow`, allowing some computational speedup if cached ahead of time. It can be obtained with `ebisu.cacheIndependent`.

  See README for derivation.
  """
  from scipy.special import betaln
  from numpy import exp
  a, b, t = prior
  if not independent:
    independent = cacheIndependent(prior)
  dt = tnow / t
  ret = betaln(a + dt, b) - independent
  return exp(ret) if exact else ret


def cacheIndependent(prior):
  """Precompute a value to speed up `predictRecall`. 🥥

  Send the output of this function to `predictRecall`'s `independent` keyword argument.
  """
  from scipy.special import betaln
  alpha, beta, t = prior
  return betaln(alpha, beta)
def updateRecall(prior, result, tnow, tback=None):
  """Update a prior on recall probability with a quiz result and time. 🍌

  `prior` is same as for `ebisu.predictRecall` and `predictRecallVar`: an object
  representing a prior distribution on recall probability at some specific time
  after a fact's most recent review.

  `result` is truthy for a successful quiz, false-ish otherwise.

  `tnow` is the time elapsed between this fact's last review and the review
  being used to update.

  Returns a new object (like `prior`) describing the posterior distribution of
  recall probability at `tback` (which is an optional input, defaults to `tnow`).
  """
  from scipy.special import betaln
  from numpy import exp

  (alpha, beta, t) = prior
  if tback is None:
    tback = t
  dt = tnow / t
  et = tnow / tback

  if result:

    if tback == t:
      return alpha + dt, beta, t

    denominator = betaln(alpha, beta)
    m1numerator = betaln(alpha + dt / et * (1 + et), beta)
    m2numerator = betaln(alpha + dt / et * (2 + et), beta)
    logmean = m1numerator - denominator
    logvar = _sub(m2numerator - denominator, 2 * logmean)

  else:

    denominator = _sub(betaln(alpha, beta), betaln(alpha + dt, beta))
    logmean = _sub(
        betaln(alpha + dt / et, beta) - denominator,
        betaln(alpha + dt / et * (et + 1), beta) - denominator)
    m2 = _sub(
        betaln(alpha + 2 * dt / et, beta) - denominator,
        betaln(alpha + dt / et * (et + 2), beta) - denominator)
    logvar = _sub(m2, 2 * logmean)

  newAlpha, newBeta = _meanVarToBeta(exp(logmean), exp(logvar))
  return newAlpha, newBeta, tback


def _sub(a, b):
  """Subtract two numbers in the log-domain, returning in log-domain
  """
  from scipy.special import logsumexp
  return logsumexp([a, b], b=[1, -1])


def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance. 🏈"""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta
def modelToPercentileDecay(model, percentile=0.5):
  """When will memory decay to a given percentile? 🏀
  
  Use a root-finding routine in log-delta space to find the delta that
  will cause the GB1 distribution to have a mean of the requested quantile.
  Because we are using well-behaved normalized deltas instead of times, and
  owing to the monotonicity of the expectation with respect to delta, we can
  quickly scan for a rough estimate of the scale of delta, then do a finishing
  optimization to get the right value.
  """
  from scipy.special import betaln
  from scipy.optimize import root_scalar
  import numpy as np
  alpha, beta, t0 = model
  logBab = betaln(alpha, beta)
  logPercentile = np.log(percentile)

  def f(lndelta):
    logMean = betaln(alpha + np.exp(lndelta), beta) - logBab
    return logMean - logPercentile

  # Scan for a bracket.
  bracket_width = 6.0
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
