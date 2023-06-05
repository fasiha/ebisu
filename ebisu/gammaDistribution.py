from typing import Callable
from scipy.optimize import minimize  #type:ignore
from scipy.special import gammaln, logsumexp  #type: ignore
import numpy as np
from math import fsum

logsumexp: Callable = logsumexp


def gammaToMode(a, b):
  return (a - b) / b if a >= 1 else 0


def gammaToMean(alpha: float, beta: float) -> float:
  return alpha / beta


def meanVarToGamma(mean, var) -> tuple[float, float]:
  a = mean**2 / var
  b = mean / var
  return a, b


def logmeanlogVarToGamma(logmean, logvar) -> tuple[float, float]:
  loga = 2 * logmean - logvar
  logb = logmean - logvar
  return np.exp(loga), np.exp(logb)


def _weightedGammaEstimate(h, w, biasCorrected=True):
  """
  See https://en.wikipedia.org/w/index.php?title=Gamma_distribution&oldid=1067698046#Closed-form_estimators
  """
  wsum = fsum(w)
  whsum = fsum(w * h)
  wlh = w * np.log(h)
  t = fsum(h * wlh) / wsum - whsum / wsum * fsum(wlh) / wsum
  k = whsum / wsum / t

  if biasCorrected:
    # this is the bias-corrected form on Wikipedia
    n = len(h)
    t2 = n / (n - 1) * t
    k2 = k - (3 * k - 2 / 3 * (k / (1 + k)) - 0.8 * k / (1 + k)**2) / n
    fit2 = (k2, 1 / t2)
    return fit2

  return (k, 1 / t)


def _weightedGammaEstimateMaxLik(x, w):
  "Maximum likelihood Gamma fit given weighted samples"
  est = _weightedGammaEstimate(x, w)
  wsum = fsum(w)
  meanLnX = fsum(np.log(x) * w) / wsum
  meanX = fsum(w * x) / wsum
  n = len(x)

  def opt(input):
    k = input[0]
    lik = (k - 1) * n * meanLnX - n * k - n * k * np.log(meanX / k) - n * gammaln(k)
    return -lik

  res = minimize(opt, [est[0]], bounds=[[.01, np.inf]])
  k = res.x[0]
  b = k / meanX  # b = 1/theta, theta = meanX / k
  return (k, b)


def gammaToStd(a, b):
  return np.sqrt(a) / b


def gammaToVar(a, b):
  return a / (b)**2


def gammaToMeanStd(a, b):
  return (gammaToMean(a, b), gammaToStd(a, b))


def gammaToMeanVar(a, b):
  return (gammaToMean(a, b), gammaToVar(a, b))


def gammaToStats(a: float, b: float):
  mean = gammaToMean(a, b)
  var = gammaToVar(a, b)
  k = a
  t = 1 / b
  m2 = t**2 * k * (k + 1)  # second non-central moment
  return (mean, var, m2, np.sqrt(m2))
