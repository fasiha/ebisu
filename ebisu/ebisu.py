
def recallProbabilityMean(alpha, beta, t, tnow):
  # `Integrate[p^((a - t)/t) * (1 - p^(1/t))^(b - 1) * (p)/t/(Gamma[a]*Gamma[b]/Gamma[a+b]), {p, 0, 1}]`
  from scipy.special import gamma
  dt = tnow / t
  same0 = gamma(alpha) / gamma(alpha+beta)
  same1 = gamma(alpha+dt) / gamma(alpha+beta+dt)
  return same1 / same0


def recallProbabilityVar(alpha, beta, t, tnow):
  from scipy.special import gamma
  # `Assuming[a>0 && b>0 && t>0, {Integrate[p^((a - t)/t) * (1 - p^(1/t))^(b - 1) * (p-m)^2/t/(Gamma[a]*Gamma[b]/Gamma[a+b]), {p, 0, 1}]}]``
  # And then plug in mean for `m` & simplify to get:
  dt = tnow / t
  same0 = gamma(alpha) / gamma(alpha+beta)
  same1 = gamma(alpha+dt) / gamma(alpha+beta+dt)
  same2 = gamma(alpha+2*dt) / gamma(alpha+beta+2*dt)
  md = same1 / same0
  md2 = same2 / same0
  return md2 - md**2


def posteriorAnalytic(alpha, beta, t, result, tnow):
  from scipy.special import gammaln, gamma
  from math import exp
  dt = tnow / t
  if result == 1:
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p, {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*p, {p,0,1}]`
    # variance: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*(p - m)^2, {p,0,1}]`
    # Simplify all three to get the following:
    same = gammaln(alpha+beta+dt) - gammaln(alpha+dt)
    mu = exp(gammaln(alpha + 2*dt)
           - gammaln(alpha + beta + 2*dt)
           + same)
    var = exp(same + gammaln(alpha + 3*dt) - gammaln(alpha + beta + 3*dt)) - mu**2
  else:
    # Mathematica code is same as above, but replace one `p` with `(1-p)`
    # marginal: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p), {p,0,1}]`
    # mean: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*p, {p,0,1}]`
    # var: `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*(p - m)^2, {p,0,1}]`
    # Then simplify and combine
    same0 = gamma(alpha) / gamma(alpha+beta)
    same1 = gamma(alpha+dt) / gamma(alpha+beta+dt)
    same2 = gamma(alpha+2*dt) / gamma(alpha+beta+2*dt)
    same3 = gamma(alpha+3*dt) / gamma(alpha+beta+3*dt)
    mu = (same1 - same2) / (same0 - same1)
    var = (same3 * (same1 - same0) + same2 * (same0 + same1 - same2) - same1**2) / (same1 - same0) ** 2
  newAlpha, newBeta = meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow


def meanVarToBeta(mean, var):
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters
  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


