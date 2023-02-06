from scipy.special import kv, kve, gammaln, gamma, betaln, logsumexp  #type: ignore
from functools import cache
from math import fsum, exp, log, expm1
from typing import Callable, Union
import numpy as np
from time import time_ns
from .gammaDistribution import gammaToMean, logmeanlogVarToGamma, meanVarToGamma, _weightedMeanVarLogw

from .models import BinomialResult, GammaUpdate, Model, NoisyBinaryResult, Result

LN2 = np.log(2)
logsumexp: Callable = logsumexp


def timeMs() -> float:
  return time_ns() / 1_000_000


def _noisyBinaryToLogPmfs(quiz: NoisyBinaryResult) -> tuple[float, float]:
  z = quiz.result > 0.5
  return _noisyHelper(z, quiz.q1, quiz.q0)


@cache
def _noisyHelper(z: bool, q1: float, q0: float) -> tuple[float, float]:
  return (_logBernPmf(z, q1), _logBernPmf(z, q0))


def _safeLog(x):
  return (x == 0 and -np.inf) or (x == 1 and 0) or log(x)


def _logBernPmf(z: Union[int, bool], p: float) -> float:
  return _safeLog(p) if z else _safeLog(1 - p)


@cache
def _logFactorial(n: int) -> float:
  return gammaln(n + 1)


def _logComb(n: int, k: int) -> float:
  return _logFactorial(n) - _logFactorial(k) - _logFactorial(n - k)


def _logBinomPmfLogp(n: int, k: int, logp: float) -> float:
  assert (n >= k >= 0)
  logcomb = _logComb(n, k)
  if n - k > 0:
    logq = log(-expm1(logp))
    return logcomb + k * logp + (n - k) * logq
  return logcomb + k * logp


def posterior(b: float, h: float, ret: Model, left: float, right: float, extra=False):
  "log posterior up to a constant offset"
  ab, bb = ret.prob.boostPrior
  ah, bh = ret.prob.initHlPrior

  logb = log(b)
  logh = log(h)
  logprior = -bb * b - bh * h + (ab - 1) * logb + (ah - 1) * logh

  loglik = []
  currHalflife = h
  halflives = [currHalflife]
  for res in ret.quiz.results[-1]:
    logPrecall = -res.hoursElapsed / currHalflife * LN2
    if isinstance(res, NoisyBinaryResult):
      # noisy binary/Bernoulli
      q1LogPmf, q0LogPmf = _noisyBinaryToLogPmfs(res)
      logPfail = log(-expm1(logPrecall))
      # Stan has this nice function, log_mix, which is perfect for this...
      # Scipy logsumexp here is much slower??
      loglik.append(_logaddexp(logPrecall + q1LogPmf, logPfail + q0LogPmf))
    else:
      # binomial
      loglik.append(_logBinomPmfLogp(res.total, res.successes, logPrecall))
    if success(res):
      currHalflife *= clampLerp(left * currHalflife, right * currHalflife, 1, max(1, b),
                                res.hoursElapsed)
    halflives.append(currHalflife)
  logposterior = fsum(loglik + [logprior])
  if extra:
    return logposterior, dict(currentHalflife=currHalflife, loglik=loglik, halflives=halflives)
  return logposterior


def success(res: Result) -> bool:
  if isinstance(res, NoisyBinaryResult):
    return res.result > 0.5
  elif isinstance(res, BinomialResult):
    return res.successes * 2 > res.total
  else:
    raise Exception("unknown result type")


def clampLerp(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  mu = (x - x1) / (x2 - x1)
  y = (y1 * (1 - mu) + y2 * mu)
  return min(y2, max(y1, y))


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

  return GammaUpdate(a=newAlpha, b=newBeta, mean=np.exp(logmean))


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
  return LN2 + log(c / b) * (a * 0.5) + log(kve(a, z)) - z


def currentHalflifePrior(model: Model) -> tuple[tuple[float, float], float]:
  # if X ~ Gamma(a, b), (c*X) ~ Gamma(a, b/c)
  a0, b0 = model.prob.initHl
  boosted = model.pred.currentHalflifeHours / gammaToMean(a0, b0)
  return (a0, b0 / boosted), boosted


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
    # return _intGammaPdfExp(an, b, t, logDomain=False) * (qz[1] - qz[0]) + qz[0] * gamma(an) / b**an
    res, sgn = logsumexp([
        _intGammaPdfExp(an, b, t, logDomain=True),
        np.log(qz[0] or np.spacing(1)) + gammaln(an) - an * np.log(b)
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
  return GammaUpdate(a=newAlpha, b=newBeta, mean=np.exp(logmean))


def _enrichDebug(fullDebug):
  logw = fullDebug['betterFit']['logw']
  fullDebug['kish'] = _kishLog(logw)
  fullDebug['bEbisuSamplesStats'] = _weightedMeanVarLogw(logw, fullDebug['betterFit']['xs'])
  fullDebug['hEbisuSamplesStats'] = _weightedMeanVarLogw(logw, fullDebug['betterFit']['ys'])
  return fullDebug


def _kishLog(logweights) -> float:
  "kish effective sample fraction, given log-weights"
  return np.exp(2 * logsumexp(logweights) - logsumexp(2 * logweights)) / logweights.size


def _logaddexp(x, y):
  "This is ~50x faster than Scipy logsumexp and 2x faster than Numpy logaddexp"
  a_max = max(x, y)
  s = abs(exp(y - a_max) + exp(x - a_max))
  return log(s) + a_max
