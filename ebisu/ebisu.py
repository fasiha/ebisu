def predictRecall(prior, tnow):
  # `Integrate[p^((a - t)/t) * (1 - p^(1/t))^(b - 1) * (p)/t/(Gamma[a]*Gamma[b]/Gamma[a+b]), {p, 0, 1}]`
  from scipy.special import gammaln
  from numpy import exp
  alpha, beta, t = prior
  dt = tnow / t
  return exp(
      gammaln(alpha + dt) - gammaln(alpha + beta + dt) - (
          gammaln(alpha) - gammaln(alpha + beta)))


def predictRecallVar(prior, tnow):
  from numpy import exp
  from scipy.special import gammaln
  alpha, beta, t = prior
  # `Assuming[a>0 && b>0 && t>0, {Integrate[p^((a - t)/t) * (1 - p^(1/t))^(b - 1) * (p-m)^2/t/(Gamma[a]*Gamma[b]/Gamma[a+b]), {p, 0, 1}]}]``
  # And then plug in mean for `m` & simplify to get:
  dt = tnow / t
  same0 = gammaln(alpha) - gammaln(alpha + beta)
  same1 = gammaln(alpha + dt) - gammaln(alpha + beta + dt)
  same2 = gammaln(alpha + 2 * dt) - gammaln(alpha + beta + 2 * dt)
  md = same1 - same0
  md2 = same2 - same0
  return exp(md2) - exp(2 * md)


def updateRecall(prior, result, tnow):
  from scipy.special import gammaln, gamma
  from numpy import exp
  alpha, beta, t = prior
  dt = tnow / t
  if result == 1:
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p, {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*p, {p,0,1}]`
    # variance: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*(p - m)^2, {p,0,1}]`
    # Simplify all three to get the following:
    same = gammaln(alpha + beta + dt) - gammaln(alpha + dt)
    muln = gammaln(alpha + 2 * dt) - gammaln(alpha + beta + 2 * dt) + same
    mu = exp(muln)
    var = exp(same + gammaln(alpha + 3 * dt) -
              gammaln(alpha + beta + 3 * dt)) - exp(2 * muln)
  else:
    # Mathematica code is same as above, but replace one `p` with `(1-p)`
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p), {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*p, {p,0,1}]`
    # var: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*(p - m)^2, {p,0,1}]`
    # Then simplify and combine

    # Caveat TODO: if alpha+beta > 1e-14 and alpha+beta+3*dt < 17, we could just evaluate `gamma` directly, rather than gammaln+expm1+logsumexp. Is it worth it?

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

  newAlpha, newBeta = meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow


def meanVarToBeta(mean, var):
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def priorToHalflife(prior, percentile=0.5, maxt=100, mint=1e-3):
  from math import sqrt
  from scipy.optimize import brentq
  h = brentq(lambda now: predictRecall(prior, now) - percentile, mint, maxt)
  # `h` is the expected half-life, i.e., the time at which recall probability drops to 0.5.
  # To get the variance about this half-life, we have to convert probability variance (around 0.5) to a time variance. This is a really dumb way to do that.
  # This 'variance' number should not be taken seriously, but it can be used for notional plotting.
  v = predictRecallVar(prior, h)

  from scipy.stats import beta as fbeta
  lo, hi = fbeta.interval(.68, *meanVarToBeta(percentile, v))

  h2 = brentq(lambda now: predictRecall(prior, now) - lo, mint, maxt)
  h3 = brentq(lambda now: predictRecall(prior, now) - hi, mint, maxt)

  return h, ((abs(h2 - h) + abs(h3 - h)) / 2)**2
