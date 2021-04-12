# -*- coding: utf-8 -*-

from scipy.special import betaln, logsumexp
from scipy.special import beta as betafn
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


def updateRecallFuzzy(prior, result, tnow, rebalance=True, tback=None, q0=None):
  (alpha, beta, t) = prior

  z = result > 0.5
  q1 = result if z else 1 - result  # alternatively, max(result, 1-result)
  if q0 is None:
    q0 = 1 - q1

  dt = tnow / t

  if z == False:
    c, d = (q0 - q1, 1 - q0)
  else:
    c, d = (q1 - q0, q0)

  den = c * betafn(alpha + dt, beta) + d * (betafn(alpha, beta) if d else 0)

  def moment(N, et):
    num = 0
    if c != 0:
      num += c * betafn(alpha + dt + N * dt * et, beta)
    if d != 0:
      num += d * betafn(alpha + N * dt * et, beta)
    return num / den

  if rebalance:
    from scipy.optimize import newton
    et = newton(lambda et: moment(1, et) - 0.5, 1 / dt)
    tback = et * tnow
  elif tback:
    et = tback / tnow
  else:
    tback = t
    et = tback / tnow

  mean = moment(1, et)  # could be just a bit away from 0.5 after rebal, so reevaluate
  secondMoment = moment(2, et)

  var = secondMoment - mean * mean
  newAlpha, newBeta = _meanVarToBeta(mean, var)
  return (newAlpha, newBeta, tback)


# via https://stats.stackexchange.com/q/419197/31187
def up2(prior, result, tnow, tback=None):
  (alpha, beta, t) = prior
  if tback is None:
    tback = t
  dt = tnow / t
  et = tback / tnow

  z = result > 0.5
  q1 = result if z else 1 - result  # alternatively, max(result, 1-result)
  q0 = 1 - q1

  c = (alpha * q1 + beta * q0) / (alpha + beta)
  if z:
    r = q1 / c
    s = q0 / c
  else:
    r = (1 - q1) / (1 - c)
    s = (1 - q0) / (1 - c)

  moment = lambda N: logsumexp([
      betaln(alpha + dt + N * dt * et, beta),
      betaln(alpha + N * dt * et, beta),
  ],
                               b=[r - s, s])

  logDen = moment(0)
  logmean = moment(1) - logDen
  mean = np.exp(logmean)
  logm2 = moment(2) - logDen
  var = np.exp(logm2) - np.exp(2 * logmean)

  # logm3 = moment(3) - logDen
  # skewness = (np.exp(logm3) - 3 * mean * var - mean**3) / var**(3 / 2)
  # print(dict(et=et, skewness=skewness))

  newAlpha, newBeta = _meanVarToBeta(mean, var)
  proposed = newAlpha, newBeta, tback
  return proposed


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
  dt = tnow / t
  failures = total - successes
  binomlns = [binomln(failures, i) for i in range(failures + 1)]

  def unnormalizedLogMoment(m, et):
    return logsumexp([
        binomlns[i] + betaln(alpha + dt * (successes + i) + m * dt * et, beta)
        for i in range(failures + 1)
    ],
                     b=[(-1)**i for i in range(failures + 1)])

  logDenominator = unnormalizedLogMoment(0, et=0)  # et doesn't matter for 0th moment
  message = dict(
      prior=prior, successes=successes, total=total, tnow=tnow, rebalance=rebalance, tback=tback)

  if rebalance:
    from scipy.optimize import newton
    et = newton(
        lambda et: np.exp(unnormalizedLogMoment(1, et) - logDenominator) - 0.5,
        1 / dt,
        maxiter=5000)
    tback = et * tnow

    m2 = np.exp(unnormalizedLogMoment(2, et) - logDenominator)
    assert m2 > 0, message

    newAlphaBeta = 1 / (8 * m2 - 2) - 0.5
    return (newAlphaBeta, newAlphaBeta, tback)

  if tback:
    et = tback / tnow
  else:
    tback = t
    et = tback / tnow

  logMean = unnormalizedLogMoment(1, et) - logDenominator
  mean = np.exp(logMean)
  m2 = np.exp(unnormalizedLogMoment(2, et) - logDenominator)

  assert mean > 0, message
  assert m2 > 0, message

  meanSq = np.exp(2 * logMean)
  var = m2 - meanSq
  assert var > 0, message
  newAlpha, newBeta = _meanVarToBeta(mean, var)
  return (newAlpha, newBeta, tback)


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


def rescaleHalflife(prior, scale=1.):
  """Given any model, return a new model with the original's halflife scaled.
  Use this function to adjust the halflife of a model.
  Perhaps you want to see this flashcard far less, because you *really* know it.
  `newModel = rescaleHalflife(model, 5)` to shift its memory model out to five
  times the old halflife.
  Or if there's a flashcard that suddenly you want to review more frequently,
  perhaps because you've recently learned a confuser flashcard that interferes
  with your memory of the first, `newModel = rescaleHalflife(model, 0.1)` will
  reduce its halflife by a factor of one-tenth.
  Useful tip: the returned model will have matching α = β, where `alpha, beta,
  newHalflife = newModel`. This happens because we first find the old model's
  halflife, then we time-shift its probability density to that halflife. That's
  the distribution this function returns, except at the *scaled* halflife.
  """
  (alpha, beta, t) = prior
  oldHalflife = modelToPercentileDecay(prior)
  dt = oldHalflife / t

  logDenominator = betaln(alpha, beta)
  logm2 = betaln(alpha + 2 * dt, beta) - logDenominator
  m2 = np.exp(logm2)
  newAlphaBeta = 1 / (8 * m2 - 2) - 0.5
  return (newAlphaBeta, newAlphaBeta, oldHalflife * scale)


for tnow in [1., 40.]:
  for z in [1., 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]:
    pre = (3., 4., 10.)
    post = updateRecallFuzzy(pre, z, tnow)
    print('fuzzy', dict(z=z, tnow=tnow, hl=modelToPercentileDecay(post), new=post))
    post = updateRecallFuzzy(pre, z, tnow, q0=0)
    print('q0=0', dict(z=z, tnow=tnow, hl=modelToPercentileDecay(post), new=post))
    post = up2(pre, z, tnow, tback=pre[-1])
    print('2020', dict(z=z, tnow=tnow, hl=modelToPercentileDecay(post), new=post))
    if z == 1. or z == 0:
      post = updateRecall(pre, z > 0.5, 1, tnow)
      print('binom', dict(z=z, tnow=tnow, hl=modelToPercentileDecay(post), new=post))

print('BINOMIAL')
for n in [1, 2, 3, 5]:
  for tnow in [40., 1.]:
    for k in range(n + 1):
      try:
        post = updateRecall(pre, k, n, tnow)
        print(dict(n=n, tnow=tnow, k=k, hl=modelToPercentileDecay(post), post=post))
      except Exception as e:
        print(dict(n=n, tnow=tnow, k=k))
        raise e

pre = (3., 4., 10.)
tnow = 19.
post = rescaleHalflife(pre, 2.)
print(dict(oldhl=modelToPercentileDecay(pre), rescaled=post, newhl=modelToPercentileDecay(post)))
