from ebisu import *
import matplotlib.pylab as plt


def calcSkew(prior, successes, total, tnow, tback, N=3):
  (alpha, beta, t) = prior
  dt = tnow / t
  et = tback / tnow

  binomlns = [binomln(total - successes, i) for i in range(total - successes + 1)]
  ncLogMoment = lambda m: logsumexp([
      binomlns[i] + betaln(beta, alpha + dt * (successes + i) + m * dt * et)
      for i in range(total - successes + 1)
  ],
                                    b=[(-1)**i for i in range(total - successes + 1)])
  logDenominator = ncLogMoment(0)
  logMean = ncLogMoment(1) - logDenominator
  logStdev = ncLogMoment(2) - logDenominator
  logSkew = ncLogMoment(3) - logDenominator
  print([logMean, logStdev, logSkew])
  # https://en.wikipedia.org/wiki/Skewness#Pearson's_moment_coefficient_of_skewness
  skew = (np.exp(logSkew) - 3 * np.exp(logMean) * np.exp(2 * logStdev) -
          np.exp(3 * logMean)) / np.exp(3 * logStdev)
  return skew


if False:
  calcSkew((3.3, 3.3, 1.), successes=1, total=1, tnow=2., tback=1.4802435605039237)
  tbacks = np.logspace(-2, 2, 500)

  skewness = np.array(
      list([calcSkew((3.3, 3.3, 1.), successes=0, total=10, tnow=.1 / 4, tback=t) for t in tbacks]))
  plt.figure()
  plt.semilogx(tbacks, (skewness), '-')
  plt.grid()
  plt.ylim([-4, 1])


def mkPosterior(prior, successes, total, tnow, tback):
  (alpha, beta, t) = prior
  dt = tnow / t
  et = tback / tnow
  binomlns = [binomln(total - successes, i) for i in range(total - successes + 1)]

  signs = [(-1)**i for i in range(total - successes + 1)]
  logDenominator = logsumexp([
      binomlns[i] + betaln(beta, alpha + dt * (successes + i)) for i in range(total - successes + 1)
  ],
                             b=signs) + np.log(dt * et)
  logPdf = lambda logp: logsumexp([
      binomlns[i] + logp * ((alpha + dt * (successes + i)) / (dt * et) - 1) +
      (beta - 1) * np.log(1 - np.exp(logp / (et * dt))) for i in range(total - successes + 1)
  ],
                                  b=signs) - logDenominator
  return logPdf


from scipy.stats import beta as betarv
from scipy.integrate import trapz


def compare(prior, successes, total, tnow, tback, viz=True, ps=np.linspace(0, 1, 5000)):
  p1 = np.vectorize(mkPosterior(prior, successes, total, tnow, tback=tback))
  pdf = np.exp(p1(np.log(ps)))

  try:
    m1 = updateRecall(prior, successes, total, tnow, tback=tback, rebalance=False)
  except:
    m1 = None

  if m1:
    betapdf = betarv.pdf(ps, m1[0], m1[1])
    err = trapz(np.abs(pdf - betapdf), ps)
    integrals = [trapz(pdf, ps), trapz(betapdf, ps)]
  else:
    err = -1
    integrals = ''

  if viz:
    plt.figure()
    plt.subplot(211)
    plt.semilogx(ps, pdf)
    if m1:
      plt.semilogx(ps, betapdf)
      plt.subplot(212)
      plt.semilogx(ps, pdf - betapdf)
  else:
    integrals = 'skipped'

  ret = dict(err=err, m1=m1, p1=p1, integrals=integrals)
  return ret


pre1 = (3.3, 4.4, 1.)
k = 3
n = 3
tnow = 100
tbackOrig = updateRecall(pre1, k, n, tnow)[2]
hl = modelToPercentileDecay(updateRecall(pre1, k, n, tnow))

# resOrig = compare(pre1, k, n, tnow, tbackOrig)
# resHl = compare(pre1, k, n, tnow, hl)
# print([resOrig, resHl])

pre3 = (3.3, 4.4, 1.)
tback3 = 1.
tnow = 1 / 50.
f = np.vectorize(mkPosterior(pre3, 1, 10, tnow, tback3))
ps = np.logspace(-5, 0, 5000)
pdf = np.exp(f(np.log(ps)))
mom1 = trapz((ps * pdf)[np.isfinite(pdf)], ps[np.isfinite(pdf)])
mom2 = trapz((ps**2 * pdf)[np.isfinite(pdf)], ps[np.isfinite(pdf)])
var = mom2 - mom1**2

from scipy.optimize import minimize
res3 = minimize(
    lambda x: np.sum(
        np.abs(betarv.pdf(ps[np.isfinite(pdf)], x[0], x[1]) - pdf[np.isfinite(pdf)])**2), [1.5, 20])


def _meanVarToBeta(mean, var):
  """Fit a Beta distribution to a mean and variance."""
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


model3 = list(_meanVarToBeta(mom1, var)) + [tback3]
plt.figure()
plt.semilogx(ps, pdf)

# tbacks = np.logspace(-2, 2)
# errs1 = np.array(list([compare(pre1, 3, 3, 50, t, viz=False)['err'] for t in tbacks]))
# errs2 = np.array(list([compare(pre1, 0, 3, 1 / 50., t, viz=False)['err'] for t in tbacks]))
# plt.figure()
# plt.loglog(tbacks, errs1, tbacks, errs2)

from scipy.special import betaln, logsumexp


def cancel(prior, successes, total, tnow, tback):
  assert (0 <= successes and successes <= total and 1 <= total)
  (alpha, beta, t) = prior
  dt = tnow / t
  et = tback / tnow
  # binomln = lambda n,k: -betaln(1 + n - k, 1 + k) - np.log(n + 1)
  binomlns = [binomln(total - successes, i) for i in range(total - successes + 1)]
  logDenominator, logMeanNum, logM2Num = [([
      binomlns[i] + betaln(beta, alpha + dt * (successes + i) + m * dt * et)
      for i in range(total - successes + 1)
  ], [(-1)**i
      for i in range(total - successes + 1)])
                                          for m in range(3)]
  l = lambda a, b: logsumexp(a, b=b)
  den = l(*logDenominator)
  mean = np.exp(l(*logMeanNum) - den)
  m2 = np.exp(l(*logM2Num) - den)
  meanSq = np.exp(2 * (l(*logMeanNum) - den))
  var = m2 - meanSq

  return logDenominator, logMeanNum, logM2Num, mean, m2, var, den


c3 = cancel(pre3, 1, 10, tnow, tback3)

from shewchuck import msum
msum(np.array(c3[0][0]) * np.array(c3[0][1]))

lden, logm1n, logm2n = [msum(np.exp(np.array(a)) * np.array(b)) for a, b in c3[:3]]

import mpmath as mp
mp.mp.dps = 30


def cancel2(prior, successes, total, tnow, tback, dps=None):
  if dps:
    mp.mp.dps = dps
  assert (0 <= successes and successes <= total and 1 <= total)
  (alpha, beta, t) = prior
  dt = tnow / t
  et = tback / tnow
  print(dict(dt=dt, et=et, dtet=dt * et))

  def ncMoment(N):
    num = []
    for i in range(successes + total + 1):
      numA = alpha + dt * (successes + i) + N * dt * et
      # numB = beta
      denA = n - k + 1 - i
      denB = 1 + i
      num.append((-1)**i * mp.gammaprod([numA, denA + denB], [numA + beta, denA, denB]))
    den = []
    for i in range(successes + total + 1):
      numA = alpha + dt * (successes + i)
      # numB = beta
      denA = n - k + 1 - i
      denB = 1 + i
      den.append((-1)**i * mp.gammaprod([numA, denA + denB], [numA + beta, denA, denB]))
    return sum(num) / sum(den), num, den

  mean, mean1, mean2 = ncMoment(1)
  m2, m21, m22 = ncMoment(2)
  var = m2 - mean**2

  return mean, m2, var, [mean1, mean2], [m21, m22]


k = 1
n = 9
tnow = 1 / 50.
# above disagree, but this they agree?
# n=8
# tnow=2.
foo = cancel(pre3, k, n, tnow, tback3)[-4:-1]
print(list(_meanVarToBeta(foo[0], foo[2])) + [tback3])

mpres = cancel2(pre3, k, n, tnow, tback3)
model4 = list(_meanVarToBeta(mpres[0], mpres[2])) + [tback3]
print(model4)
