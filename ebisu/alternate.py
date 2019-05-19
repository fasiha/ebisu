# -*- coding: utf-8 -*-

from .ebisu import _meanVarToBeta
import numpy as np


def predictRecallMode(prior, tnow):
  """Mode of the immediate recall probability.

  Same arguments as `ebisu.predictRecall`, see that docstring for details. A
  returned value of 0 or 1 may indicate divergence.
  """
  # [1] Mathematica: `Solve[ D[p**((a-t)/t) * (1-p**(1/t))**(b-1), p] == 0, p]`
  alpha, beta, t = prior
  dt = tnow / t
  pr = lambda p: p**((alpha - dt) / dt) * (1 - p**(1 / dt))**(beta - 1)

  # See [1]. The actual mode is `modeBase ** dt`, but since `modeBase` might
  # be negative or otherwise invalid, check it.
  modeBase = (alpha - dt) / (alpha + beta - dt - 1)
  if modeBase >= 0 and modeBase <= 1:
    # Still need to confirm this is not a minimum (anti-mode). Do this with a
    # coarse check of other points likely to be the mode.
    mode = modeBase**dt
    modePr = pr(mode)

    eps = 1e-3
    others = [
        eps, mode - eps if mode > eps else mode / 2,
        mode + eps if mode < 1 - eps else (1 + mode) / 2, 1 - eps
    ]
    otherPr = map(pr, others)
    if max(otherPr) <= modePr:
      return mode
  # If anti-mode detected, that means one of the edges is the mode, likely
  # caused by a very large or very small `dt`. Just use `dt` to guess which
  # extreme it was pushed to. If `dt` == 1.0, and we get to this point, likely
  # we have malformed alpha/beta (i.e., <1)
  return 0.5 if dt == 1. else (0. if dt > 1 else 1.)


def predictRecallMedian(prior, tnow, percentile=0.5):
  """Median (or percentile) of the immediate recall probability.

  Same arguments as `ebisu.predictRecall`, see that docstring for details.

  An extra keyword argument, `percentile`, is a float between 0 and 1, and
  specifies the percentile rather than 50% (median).
  """
  # [1] `Integrate[p**((a-t)/t) * (1-p**(1/t))**(b-1) / t / Beta[a,b], p]`
  # and see "Alternate form assuming a, b, p, and t are positive".
  from scipy.optimize import brentq
  from scipy.special import betainc
  alpha, beta, t = prior
  dt = tnow / t

  # See [1]. If the mode doesn't exist (or can't be found), find the median (or
  # `percentile`) using a root-finder and the cumulative distribution function.
  cdfPercentile = lambda p: betainc(alpha, beta, p**(1 / dt)) - percentile
  return brentq(cdfPercentile, 0, 1)


def predictRecallMonteCarlo(prior, tnow, N=1000 * 1000):
  """Monte Carlo simulation of the immediate recall probability.

  Same arguments as `ebisu.predictRecall`, see that docstring for details. An
  extra keyword argument, `N`, specifies the number of samples to draw.

  This function returns a dict containing the mean, variance, median, and mode
  of the current recall probability.
  """
  import scipy.stats as stats
  alpha, beta, t = prior
  tPrior = stats.beta.rvs(alpha, beta, size=N)
  tnowPrior = tPrior**(tnow / t)
  freqs, bins = np.histogram(tnowPrior, 'auto')
  bincenters = bins[:-1] + np.diff(bins) / 2
  return dict(
      mean=np.mean(tnowPrior),
      median=np.median(tnowPrior),
      mode=bincenters[freqs.argmax()],
      var=np.var(tnowPrior))
def updateRecallQuad(prior, result, tnow, analyticMarginal=True):
  """Update recall probability with quiz result via quadrature integration.

  Same arguments as `ebisu.updateRecall`, see that docstring for details.

  An extra keyword argument: `analyticMarginal` if false will compute the
  marginal (the denominator in Bayes rule) using quadrature as well. If true, an
  analytical expression will be used.
  """
  from scipy.integrate import quad
  alpha, beta, t = prior
  dt = tnow / t

  if result == 1:
    marginalInt = lambda p: p**((alpha - dt) / dt) * (1 - p**(1 / dt))**(beta -
                                                                         1) * p
  else:
    # difference from above: -------------------------------------------^vvvv
    marginalInt = lambda p: p**((alpha - dt) / dt) * (1 - p**(1 / dt))**(
        beta - 1) * (1 - p)

  if analyticMarginal:
    from scipy.special import beta as fbeta
    if result == 1:
      marginal = dt * fbeta(alpha + dt, beta)
    else:
      marginal = dt * (fbeta(alpha, beta) - fbeta(alpha + dt, beta))
  else:
    marginalEst = quad(marginalInt, 0, 1)
    if marginalEst[0] < marginalEst[1] * 10.:
      raise OverflowError(
          'Marginal integral error too high: value={}, error={}'.format(
              marginalEst[0], marginalEst[1]))
    marginal = marginalEst[0]

  muInt = lambda p: marginalInt(p) * p
  muEst = quad(muInt, 0, 1)
  if muEst[0] < muEst[1] * 10.:
    raise OverflowError(
        'Mean integral error too high: value={}, error={}'.format(
            muEst[0], muEst[1]))
  mu = muEst[0] / marginal

  varInt = lambda p: marginalInt(p) * (p - mu)**2
  varEst = quad(varInt, 0, 1)
  if varEst[0] < varEst[1] * 10.:
    raise OverflowError(
        'Variance integral error too high: value={}, error={}'.format(
            varEst[0], varEst[1]))
  var = varEst[0] / marginal

  newAlpha, newBeta = _meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow


def updateRecallMonteCarlo(prior, result, tnow, N=10 * 1000):
  """Update recall probability with quiz result via Monte Carlo simulation.

  Same arguments as `ebisu.updateRecall`, see that docstring for details.

  An extra keyword argument `N` specifies the number of samples to draw.
  """
  # [bernoulliLikelihood] https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=769806318#Properties_of_the_Bernoulli_Distribution, third equation
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  # [weightedVar] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  import scipy.stats as stats

  alpha, beta, t = prior

  tPrior = stats.beta.rvs(alpha, beta, size=N)
  tnowPrior = tPrior**(tnow / t)

  # This is the Bernoulli likelihood [bernoulliLikelihood]
  weights = (tnowPrior)**result * ((1 - tnowPrior)**(1 - result))

  # See [weightedMean]
  weightedMean = np.sum(weights * tnowPrior) / np.sum(weights)
  # See [weightedVar]
  weightedVar = np.sum(weights *
                       (tnowPrior - weightedMean)**2) / np.sum(weights)

  newAlpha, newBeta = _meanVarToBeta(weightedMean, weightedVar)

  return newAlpha, newBeta, tnow
