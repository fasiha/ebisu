from scipy.special import gammaln  #type: ignore
from functools import cache
from math import fsum, exp, log, expm1
from typing import Union
import numpy as np
from time import time_ns

from .models import BinomialResult, Model, NoisyBinaryResult


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


def posterior(
    ret: Model,
    wmax: float,
    wmaxBetaPriors: tuple[float, float],
    hs: np.ndarray,
):
  from scipy.stats import beta as betarv  #type: ignore
  logprior = betarv.logpdf(wmax, *wmaxBetaPriors)

  n = hs.size
  tau = -(n - 1) / np.log(wmax + 1e-12)  # exp
  ws = np.exp(-np.arange(n) / tau)

  loglik = []
  for res in ret.quiz.results[-1] if len(ret.quiz.results) else []:
    logPrecall = np.log(np.max(ws * np.exp2(-res.hoursElapsed / hs)))

    if isinstance(res, NoisyBinaryResult):
      # noisy binary/Bernoulli
      q1LogPmf, q0LogPmf = _noisyBinaryToLogPmfs(res)
      logPfail = log(-expm1(logPrecall))
      # Stan has this nice function, log_mix, which is perfect for this...
      # Scipy logsumexp here is much slower??
      loglik.append(_logaddexp(logPrecall + q1LogPmf, logPfail + q0LogPmf))
    elif isinstance(res, BinomialResult):
      # binomial
      loglik.append(_logBinomPmfLogp(res.total, res.successes, logPrecall))
    else:
      raise Exception('unknown quiz type')
  logposterior = fsum(loglik + [logprior])
  return logposterior, dict(loglik=loglik, logprior=logprior)


def clampLerp(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
  mu = (x - x1) / (x2 - x1)
  y = (y1 * (1 - mu) + y2 * mu)
  return min(y2, max(y1, y))


def _logaddexp(x, y):
  "This is ~50x faster than Scipy logsumexp and 2x faster than Numpy logaddexp"
  a_max = max(x, y)
  s = abs(exp(y - a_max) + exp(x - a_max))
  return log(s) + a_max
