"""
run as 
$ python -m unittest --failfast
"""

from itertools import product
from functools import cache
import ebisu
import unittest
from scipy.stats import gamma as gammarv, binom as binomrv, bernoulli  # type: ignore
from scipy.special import logsumexp  # type: ignore
import numpy as np
from typing import Optional, Union, Any, Callable
import math
from copy import deepcopy
import pickle
from datetime import datetime
import mpmath as mp  # type:ignore
import csv
import io

from ebisu.ebisuHelpers import GammaUpdate, gammaUpdateBinomial, gammaUpdateNoisy
from ebisu.gammaDistribution import gammaToMean, meanVarToGamma, gammaToStats, gammaToStd

results: dict[Union[str, tuple], Any] = dict()
testStartTime = datetime.utcnow().isoformat()

seed = np.random.randint(1, 1_000_000_000)
print(f'{seed=}')

MILLISECONDS_PER_HOUR = 3600e3  # 60 min/hour * 60 sec/min * 1e3 ms/sec
logsumexp: Callable = logsumexp


def weightedMeanVarLogw(logw: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  # [weightedVar] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  logsumexpw, sgn = logsumexp(logw, return_sign=True)
  assert sgn > 0, 'positive'
  mean = np.exp(logsumexp(logw, b=x) - logsumexpw)
  m2 = np.exp(logsumexp(logw, b=x**2) - logsumexpw)
  var = m2 - mean**2
  return (mean, var, m2, np.sqrt(m2))


def fourQuiz(fraction: float, result: int, lastNoisy: bool):
  initHlMean = 10.0
  now = ebisu.timeMs()
  init = ebisu.initModel(halflife=initHlMean, now=now)

  upd = deepcopy(init)
  elapsedHours = fraction * initHlMean
  thisNow = now + elapsedHours * MILLISECONDS_PER_HOUR
  upd = ebisu.updateRecall(upd, result, now=thisNow)

  for nextResult, nextElapsed, nextTotal in zip(
      [1, 1, 1 if not lastNoisy else (0.2)],
      [elapsedHours * 3, elapsedHours * 5, elapsedHours * 7],
      [1, 2, 2 if not lastNoisy else 1],
  ):
    thisNow += nextElapsed * MILLISECONDS_PER_HOUR
    upd = ebisu.updateRecall(upd, nextResult, total=nextTotal, q0=0.05, now=thisNow)
  return init, upd


def _gammaUpdateBinomialMonteCarlo(
    a: float,
    b: float,
    t: float,
    k: int,
    n: int,
    size=1_000_000,
) -> GammaUpdate:
  # Scipy Gamma random variable is inverted: it needs (a, scale=1/b) for the usual (a, b) parameterization
  halflife = gammarv.rvs(a, scale=1 / b, size=size)
  pRecall = np.exp2(-t / halflife)

  logweight = binomrv.logpmf(k, n, pRecall)  # this is the likelihood of observing the data
  weight = np.exp(logweight)
  # use logpmf because macOS Scipy `pmf` overflows for pRecall around 2.16337e-319?

  wsum = math.fsum(weight)
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  postMean = math.fsum(weight * halflife) / wsum
  # See https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  postVar = math.fsum(weight * (halflife - postMean)**2) / wsum

  if False:
    # This is a fancy mixed type log-moment estimator for fitting Gamma rvs from (weighted) samples.
    # It's (much) closer to the maximum likelihood fit than the method-of-moments fit we use.
    # See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1066334959#Closed-form_estimators
    # However, in Ebisu, we just have central moments of the posterior, not samples, and there doesn't seem to be
    # an easy way to get a closed form "mixed type log-moment estimator" from moments.
    h = halflife
    w = weight
    that2 = np.sum(w * h * np.log(h)) / wsum - np.sum(w * h) / wsum * np.sum(w * np.log(h)) / wsum
    khat2 = np.sum(w * h) / wsum / that2
    fit = (khat2, 1 / that2)

  newA, newB = meanVarToGamma(postMean, postVar)
  return GammaUpdate(newA, newB, postMean)


def _gammaUpdateNoisyMonteCarlo(
    a: float,
    b: float,
    t: float,
    q1: float,
    q0: float,
    z: bool,
    size=1_000_000,
) -> GammaUpdate:
  halflife = gammarv.rvs(a, scale=1 / b, size=size)
  pRecall = np.exp2(-t / halflife)

  # this weight is `P(z | pRecall)` and derived and checked via Stan in
  # https://github.com/fasiha/ebisu/issues/52
  # Notably, this expression is NOT used by ebisu, so it's a great independent check
  weight = bernoulli.pmf(z, q1) * pRecall + bernoulli.pmf(z, q0) * (1 - pRecall)

  wsum = math.fsum(weight)
  # for references to formulas, see `_gammaUpdateBinomialMonteCarlo`
  postMean = math.fsum(weight * halflife) / wsum
  postVar = math.fsum(weight * (halflife - postMean)**2) / wsum

  newA, newB = meanVarToGamma(postMean, postVar)
  return GammaUpdate(newA, newB, postMean)


def relativeError(actual, expected):
  e, a = np.array(expected), np.array(actual)
  return np.abs(a - e) / np.abs(e)


class TestEbisu(unittest.TestCase):

  def setUp(self):
    np.random.seed(seed=seed)  # for sanity when testing with Monte Carlo

  def test_gamma_update_noisy(self):
    """Test _gammaUpdateNoisy for various q0 and against Monte Carlo

    These are the Ebisu v2-style updates, in that there's no boost, just a prior
    on halflife and either quiz type. These have to be correct for the boost
    mechanism to work.
    """
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    MAX_RELERR_AB = .02
    MAX_RELERR_MEAN = .01
    for fraction in [0.1, 0.5, 1., 2., 10.]:
      t = initHlMean * fraction
      for q0 in [.15, 0, None]:
        prev: Optional[GammaUpdate] = None
        for noisy in [0.1, 0.3, 0.7, 0.9]:
          z = noisy >= 0.5
          q1 = noisy if z else 1 - noisy
          q0 = 1 - q1 if q0 is None else q0
          updated = gammaUpdateNoisy(a, b, t, q1, q0, z)

          for size in [100_000, 500_000, 1_000_000]:
            u2 = _gammaUpdateNoisyMonteCarlo(a, b, t, q1, q0, z, size=size)
            if (relativeError(updated.a, u2.a) < MAX_RELERR_AB and
                relativeError(updated.b, u2.b) < MAX_RELERR_AB and
                relativeError(updated.mean, u2.mean) < MAX_RELERR_MEAN):
              # found a size that should match the actual tests below
              break

          self.assertLess(relativeError(updated.a, u2.a), MAX_RELERR_AB)  #type:ignore
          self.assertLess(relativeError(updated.b, u2.b), MAX_RELERR_AB)  #type:ignore
          self.assertLess(relativeError(updated.mean, u2.mean), MAX_RELERR_MEAN)  #type:ignore

          msg = f'q0={q0}, z={z}, noisy={noisy}'
          if z:
            self.assertGreaterEqual(updated.mean, initHlMean, msg)
          else:
            self.assertLessEqual(updated.mean, initHlMean, msg)

          if prev:
            # Noisy updates should be monotonic in `z` (the noisy result)
            lt = prev.mean <= updated.mean
            approx = relativeError(prev.mean, updated.mean) < (np.spacing(updated.mean) * 1e3)
            self.assertTrue(
                lt or approx,
                f'{msg}, prev.mean={prev.mean}, updated.mean={updated.mean}, lt={lt}, approx={approx}'
            )
          # Means WILL NOT be monotonic in `t`: for `q0 > 0`,
          # means rise with `t`, then peak, then drop: see
          # https://github.com/fasiha/ebisu/issues/52

          prev = updated

  def test_gamma_update_binom(self):
    """Test BASIC _gammaUpdateBinomial"""
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)

    a, b = initHlPrior

    maxN = 4
    ts = [fraction * initHlMean for fraction in [0.1, 0.5, 1., 2., 10.]]
    us: dict[tuple[int, int, int], GammaUpdate] = dict()
    for tidx, t in enumerate(ts):
      for n in range(1, maxN + 1):
        for result in range(n + 1):
          updated = gammaUpdateBinomial(a, b, t, result, n)
          self.assertTrue(np.all(np.isfinite([updated.a, updated.b, updated.mean])))
          if result == n:
            self.assertGreaterEqual(updated.mean, initHlMean, (t, result, n))
          elif result == 0:
            self.assertLessEqual(updated.mean, initHlMean, (t, result, n))
          us[(tidx, result, n)] = updated

    for tidx, k, n in us:
      curr = us[(tidx, k, n)]

      # Binomial updated means should be monotonic in `t`
      prev = us.get((tidx - 1, k, n))
      if prev:
        self.assertLess(prev.mean, curr.mean, (tidx, k, n))

      # Means should be monotonic in `k`/`result` for fixed `n`
      prev = us.get((tidx, k - 1, n))
      if prev:
        self.assertLess(prev.mean, curr.mean, (tidx, k, n))

      # And should be monotonic in `n` for fixed `k`/`result`
      prev = us.get((tidx, k, n - 1))
      if prev:
        self.assertLess(curr.mean, prev.mean, (tidx, k, n))

  def test_gamma_update_vs_montecarlo(self):
    "Test Gamma-only updates via Monte Carlo"
    initHlMean = 10  # hours
    initHlBeta = 0.1
    initHlPrior = (initHlBeta * initHlMean, initHlBeta)
    a, b = initHlPrior

    # These thresholds on relative error between the analytical and Monte Carlo updates
    # should be enough for several trials of this unit test (see `trial` below). Nonetheless
    # I set the seed to avoid test surprises.
    MAX_RELERR_AB = .05
    MAX_RELERR_MEAN = .01
    for trial in range(1):
      for fraction in [0.1, 1., 10.]:
        t = initHlMean * fraction
        for n in [1, 2, 3, 4]:  # total number of binomial attempts
          for result in range(n + 1):  # number of binomial successes
            updated = gammaUpdateBinomial(a, b, t, result, n)
            self.assertTrue(
                np.all(np.isfinite([updated.a, updated.b, updated.mean])), f'k={result}, n={n}')

            # in order to avoid egregiously long tests, scale up the number of Monte Carlo samples
            # to meet the thresholds above.
            for size in [100_000, 500_000, 2_000_000, 5_000_000]:
              u2 = _gammaUpdateBinomialMonteCarlo(a, b, t, result, n, size=size)

              if (relativeError(updated.a, u2.a) < MAX_RELERR_AB and
                  relativeError(updated.b, u2.b) < MAX_RELERR_AB and
                  relativeError(updated.mean, u2.mean) < MAX_RELERR_MEAN):
                # found a size that should match the actual tests below
                break

            msg = f'{(trial, t, result, n, size)}'  #type:ignore
            self.assertLess(relativeError(updated.a, u2.a), MAX_RELERR_AB, msg)  #type:ignore
            self.assertLess(relativeError(updated.b, u2.b), MAX_RELERR_AB, msg)  #type:ignore

            #type:ignore
            self.assertLess(
                relativeError(updated.mean, u2.mean),  #type:ignore
                MAX_RELERR_MEAN,  #type:ignore
                msg)

  def test_1_then_0s(self, verbose=False):
    initHlMean = 10  # hours
    now = ebisu.timeMs()
    base = ebisu.initModel(initHlMean, now=now)

    ts = [20., 10., 5., 4., 3., 2., 1.]  # hours elapsed for each quiz
    correct_ts = [ts[0]]  # just one success

    fulls: list[ebisu.Model] = [base]
    for t in ts:
      now += t * MILLISECONDS_PER_HOUR
      fulls.append(ebisu.updateRecall(fulls[-1], int(t in correct_ts), now=now))

    # require monotonically decreasing halflife since we have an initial success followed by a string of failures
    for left, right in zip(fulls[1:], fulls[2:]):
      self.assertGreater(ebisu.hoursForRecallDecay(left), ebisu.hoursForRecallDecay(right))


if __name__ == '__main__':
  import os
  # get just this file's module name: no `.py` and no path
  name = os.path.basename(__file__).replace(".py", "")
  unittest.TextTestRunner(failfast=True).run(unittest.TestLoader().loadTestsFromName(name))
