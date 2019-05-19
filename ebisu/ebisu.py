# -*- coding: utf-8 -*-

def predictRecall(prior, tnow):
  """Expected recall probability now, given a prior distribution on it. 🍏

  `prior` is a tuple representing the prior distribution on recall probability
  after a specific unit of time has elapsed since this fact's last review.
  Specifically,  it's a 3-tuple, `(alpha, beta, t)` where `alpha` and `beta`
  parameterize a Beta distribution that is the prior on recall probability at
  time `t`.

  `tnow` is the *actual* time elapsed since this fact's most recent review.

  Returns the expectation of the recall probability `tnow` after last review, a
  float between 0 and 1.

  See documentation for derivation.
  """
  from scipy.special import gammaln
  from numpy import exp
  alpha, beta, t = prior
  dt = tnow / t
  return exp(
      gammaln(alpha + dt) - gammaln(alpha + beta + dt) -
      (gammaln(alpha) - gammaln(alpha + beta)))


def _subtractexp(x, y):
  """Evaluates exp(x) - exp(y) a bit more accurately than that. ⚾️

  This can avoid cancellation in case `x` and `y` are both large and close,
  similar to scipy.misc.logsumexp except without the last log.
  """
  from numpy import exp, maximum
  maxval = maximum(x, y)
  return exp(maxval) * (exp(x - maxval) - exp(y - maxval))


def predictRecallVar(prior, tnow):
  """Variance of recall probability now. 🍋

  This function returns the variance of the distribution whose mean is given by
  `ebisu.predictRecall`. See it's documentation for details.

  Returns a float.
  """
  from scipy.special import gammaln
  alpha, beta, t = prior
  dt = tnow / t
  s = [
      gammaln(alpha + n * dt) - gammaln(alpha + beta + n * dt) for n in range(3)
  ]
  md = 2 * (s[1] - s[0])
  md2 = s[2] - s[0]

  return _subtractexp(md2, md)
def updateRecall(prior, result, tnow):
  """Update a prior on recall probability with a quiz result and time. 🍌

  `prior` is same as for `ebisu.predictRecall` and `predictRecallVar`: an object
  representing a prior distribution on recall probability at some specific time
  after a fact's most recent review.

  `result` is truthy for a successful quiz, false-ish otherwise.

  `tnow` is the time elapsed between this fact's last review and the review
  being used to update.

  Returns a new object (like `prior`) describing the posterior distribution of
  recall probability at `tnow`.
  """
  from scipy.special import gammaln
  from numpy import exp
  alpha, beta, t = prior
  dt = tnow / t
  if result:
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p, {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*p, {p,0,1}]`
    # variance: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*(p - m)^2, {p,0,1}]`
    # Simplify all three to get the following:
    same = gammaln(alpha + beta + dt) - gammaln(alpha + dt)
    muln = gammaln(alpha + 2 * dt) - gammaln(alpha + beta + 2 * dt) + same
    mu = exp(muln)
    var = _subtractexp(
        same + gammaln(alpha + 3 * dt) - gammaln(alpha + beta + 3 * dt),
        2 * muln)
  else:
    # Mathematica code is same as above, but replace one `p` with `(1-p)`
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p), {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*p, {p,0,1}]`
    # var: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*(p - m)^2, {p,0,1}]`
    # Then simplify and combine

    from scipy.misc import logsumexp
    from numpy import expm1

    s = [
        gammaln(alpha + n * dt) - gammaln(alpha + beta + n * dt)
        for n in range(4)
    ]

    mu = expm1(s[2] - s[1]) / -expm1(s[0] - s[1])

    def lse(a, b):
      return list(logsumexp(a, b=b, return_sign=True))

    n1 = lse([s[1], s[0]], [1, -1])
    n1[0] += s[3]
    n2 = lse([s[0], s[1], s[2]], [1, 1, -1])
    n2[0] += s[2]
    n3 = [s[1] * 2, 1.]
    d = lse([s[1], s[0]], [1, -1])
    d[0] *= 2
    n = lse([n1[0], n2[0], n3[0]], [n1[1], n2[1], -n3[1]])
    var = exp(n[0] - d[0])

  newAlpha, newBeta = _meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow
def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance. 🏈"""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def priorToHalflife(prior, percentile=0.5, maxt=100, mint=1e-3):
  """Find the half-life corresponding to a time-based prior on recall. 🏀"""
  from scipy.optimize import brentq
  return brentq(lambda now: predictRecall(prior, now) - percentile, mint, maxt)


def defaultModel(t, alpha=4.0, beta=None):
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
