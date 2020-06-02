from ebisu import *


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


def compare(prior, successes, total, tnow, tback, viz=True):
  m1 = updateRecall(prior, successes, total, tnow, tback=tback, rebalance=False)

  p1 = np.vectorize(mkPosterior(prior, successes, total, tnow, tback=tback))
  from scipy.stats import beta as betarv
  from scipy.integrate import trapz

  ps = np.linspace(0, 1, 5000)
  pdf = np.exp(p1(np.log(ps)))

  betapdf = betarv.pdf(ps, m1[0], m1[1])
  err = trapz(np.abs(pdf - betapdf), ps)

  if viz:
    integrals = [trapz(pdf, ps), trapz(betapdf, ps)]

    plt.figure()
    plt.subplot(211)
    plt.plot(ps, pdf)
    plt.plot(ps, betapdf)
    plt.subplot(212)
    plt.plot(ps, pdf - betapdf)
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

resOrig = compare(pre1, k, n, tnow, tbackOrig)
resHl = compare(pre1, k, n, tnow, hl)
print([resOrig, resHl])

tbacks = np.logspace(-2, 2)
errs1 = np.array(list([compare(pre1, 3, 3, 50, t, viz=False)['err'] for t in tbacks]))
errs2 = np.array(list([compare(pre1, 0, 3, 1 / 50., t, viz=False)['err'] for t in tbacks]))
plt.figure()
plt.loglog(tbacks, errs1, tbacks, errs2)
