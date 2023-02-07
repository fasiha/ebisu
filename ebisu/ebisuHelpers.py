from scipy.special import kv, kve, gammaln, gamma, betaln, logsumexp  #type: ignore
from functools import cache
from math import log, exp
from typing import Callable
import numpy as np
from time import time_ns
from .gammaDistribution import logmeanlogVarToGamma
from .models import GammaUpdate

LN2 = log(2)
logsumexp: Callable = logsumexp


@cache
def _binomln(n, k):
  "Log of scipy.special.binom calculated entirely in the log domain"
  return -betaln(1 + n - k, 1 + k) - log(n + 1)


def gammaUpdateBinomial(a: float, b: float, t: float, k: int, n: int) -> GammaUpdate:
  """Core Ebisu v2-style Bayesian update on binomial quizzes

  Assuming a halflife $h ~ Gamma(a, b)$ and a Binomial quiz at time $t$
  resulting in $k ~ Binomial(n, 2^(t/h))$ successes out of a total $n$
  trials, this function computes moments of the true posterior $h|k$,
  which is a nonstandard distribution, approximates it to a new
  $Gamma(newA, newB)$ and returns that approximate posterior.

  Note that all probabilistic parameters are assumed to be known ($a,
  b$), as are data parameters $t, n$, and experimental result $k$.

  See also `gammaUpdateNoisy`.
  """

  def logmoment(nth) -> float:
    loglik = []
    scales = []
    for i in range(0, n - k + 1):
      loglik.append(_binomln(n - k, i) + _intGammaPdfExp(a + nth, b, t * (k + i), logDomain=True))
      scales.append((-1)**i)
    return logsumexp(loglik, b=scales)

  logm0 = logmoment(0)
  logmean = logmoment(1) - logm0
  logm2 = logmoment(2) - logm0
  logvar = logsumexp([logm2, 2 * logmean], b=[1, -1])
  newAlpha, newBeta = logmeanlogVarToGamma(logmean, logvar)
  assert np.isfinite(newAlpha)
  assert np.isfinite(newBeta)
  return GammaUpdate(a=newAlpha, b=newBeta, mean=exp(logmean))


def _intGammaPdf(a: float, b: float, logDomain: bool):
  # \int_0^∞ h^(a-1) \exp(-b h) dh$, via Wolfram Alpha, etc.
  if not logDomain:
    return b**(-a) * gamma(a)
  return -a * log(b) + gammaln(a)


def _intGammaPdfExp(a: float, b: float, c: float, logDomain: bool):
  # $s(a,b,c) = \int_0^∞ h^(a-1) \exp(-b h - c / h) dh$, via sympy
  if c == 0:
    return _intGammaPdf(a, b, logDomain=logDomain)

  z = 2 * np.sqrt(b * c)  # arg to kv
  if not logDomain:
    return 2 * (c / b)**(a * 0.5) * kv(a, z)

  # `kve = kv * exp(z)` -> `log(kve) = log(kv) + z` -> `log(kv) = log(kve) - z`
  besselK = kve(a, z)
  if np.isfinite(besselK):
    return LN2 + log(c / b) * (a * 0.5) + log(besselK) - z

  # Use large-order approximation https://dlmf.nist.gov/10.41 -> 10.41.2
  logBesselK = log(np.e * z / (2 * a)) * -a + 0.5 * log(np.pi / (2 * a))
  return LN2 + log(c / b) * (a * 0.5) + logBesselK


def gammaUpdateNoisy(a: float, b: float, t: float, q1: float, q0: float, z: bool) -> GammaUpdate:
  """Core Ebisu v2-style Bayesian update on noisy binary quizzes

  Assuming a halflife $h ~ Gamma(a, b)$, a hidden quiz result $x ~
  Bernoulli(2^(t/h))$ (for $t$ time elapsed since review), and an
  observed *noisy* quiz report $z|x ~ Bernoulli(q0, q1)$, this function
  computes moments of the true posterior $h|z$, which is a nonstandard
  distribution, approximates it to a new $Gamma(newA, newB)$ and returns
  that approximate posterior.

  Note that all probabilistic parameters are assumed to be known: $a, b,
  q0, q1$, as are data $t, z$. Only $x$, the true quiz result is
  unknown, as well as of course the true halflife.

  See also `gammaUpdateBinomial`.
  """
  qz = (q0, q1) if z else (1 - q0, 1 - q1)

  def logmoment(n):
    an = a + n
    res, sgn = logsumexp([
        _intGammaPdfExp(an, b, t, logDomain=True),
        log(qz[0] or np.spacing(1)) + gammaln(an) - an * log(b)
    ],
                         b=[qz[1] - qz[0], 1],
                         return_sign=True)
    assert sgn > 0
    return res

  logm0 = logmoment(0)
  logmean = logmoment(1) - logm0
  logm2 = logmoment(2) - logm0
  logvar = logsumexp([logm2, 2 * logmean], b=[1, -1])
  newAlpha, newBeta = logmeanlogVarToGamma(logmean, logvar)
  assert np.isfinite(newAlpha)
  assert np.isfinite(newBeta)
  return GammaUpdate(a=newAlpha, b=newBeta, mean=exp(logmean))
