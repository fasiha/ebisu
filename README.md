# Ebisu

Super-simple library and engine for spaced-repetition systems in the vein of Anki and Memrise.

It just needs an append-only table. Each row has:

- user ID doing the addressing
- fact ID addressed
- what’s happening:
  - quiz?, with result boolean
  - restart?
  - upsert mem?, with new mem string
  - stop quizzing?
- timestamp

Given a table of these, we have a few functions we need:

- for a given user, and an optional timestamp, give next fact to review
- after that fact is reviewed, update it’s spacing.

## Algorithm

This is, as far as I can tell, correct implementations of the above, using Bayesian statistics. Forthcoming:

- complete details including Python code
- JavaScript (potentially TypeScript/PureScript) implementation

### TL;DR

```py
import numpy as np

from scipy.special import hyp2f1, beta


def recallProbability(alpha, beta, t, tnow, percentile=0.5):
  """Returns the mode (or median) of the immediate (pseudo-Beta) prior"""
  # [peak] [WolframAlpha result](https://www.wolframalpha.com/input/?i=Solve%5B+D%5Bp%5E((a-t)%2Ft)+*+(1-p%5E(1%2Ft))%5E(b-1),+p%5D+%3D%3D+0,+p%5D) for `Solve[ D[p**((a-t)/t) * (1-p**(1/t))**(b-1), p] == 0, p]`
  # [cdf] [WolframAlpha result](https://www.wolframalpha.com/input/?i=Integrate%5Bp%5E((a-t)%2Ft)+*+(1-p%5E(1%2Ft))%5E(b-1)+%2F+t+%2F+Beta%5Ba,b%5D,+p%5D) for `Integrate[p**((a-t)/t) * (1-p**(1/t))**(b-1) / t / Beta[a,b], p]`

  dt = tnow / t

  # See [peak]. This is the mode but can be nonsense for PDFs that blow up
  tentativePeak = ((alpha - dt) / (alpha + beta - dt - 1)) ** dt
  if tentativePeak.imag == 0 and tentativePeak > 0. and tentativePeak < 1.:
    return tentativePeak

  from scipy.optimize import brentq
  from scipy.special import beta as fbeta

  # See [cdf]. If the mode doesn't exist (or can't be found), find the median (or `percentile`) using a root-finder and the cumulative distribution function.
  # N.B. I prefer to try to find the mode (above) because it will be much faster than this.
  cdfPercentile = lambda p: (p**(alpha/dt) *
                             hyp2f1(alpha, 1 - beta, 1 + alpha, p**(1/dt)) /
                             alpha /
                             fbeta(alpha,beta)) - percentile
  return brentq(cdfPercentile, 0, 1)

def recallProbabilityMedian(alpha, beta, t, tnow, percentile=0.5):
  from scipy.optimize import brentq
  from scipy.special import beta as fbeta
  dt = tnow / t

  # See [cdf]. If the mode doesn't exist (or can't be found), find the median (or `percentile`) using a root-finder and the cumulative distribution function.
  # N.B. I prefer to try to find the mode (above) because it will be much faster than this.
  cdfPercentile = lambda p: (p**(alpha/dt) *
                             hyp2f1(alpha, 1 - beta, 1 + alpha, p**(1/dt)) /
                             alpha /
                             fbeta(alpha,beta)) - percentile
  return brentq(cdfPercentile, 0, 1)

# Looks like the median-finding `brentq` approach is faster than the mode approach, and it works all the time. Consider switching to median-only.
%timeit(lambda : recallProbability(betaa,betab,t0, 1*t0))
%timeit(lambda : recallProbabilityMedian(betaa,betab,t0, 1*t0))

%timeit(lambda : recallProbability(betaa,betab,t0, 10*t0))
%timeit(lambda : recallProbabilityMedian(betaa,betab,t0, 10*t0))

# So we have several ways of evaluating the posterior mean/var:
# - Monte Carlo
# - Quadrature integration
# - Analytic expression, with several hyp2f1
# - Simplified analytic expression with fewer hyp2f1 (recurrence relations)

def posteriorAnalytic(alpha, beta, t, result, tnow):
  from scipy.special import beta as fbeta

  dt = tnow / t
  if result == 1:
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p, {p,0,1}]`
    marginal = dt * fbeta(alpha+dt, beta)
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*p, {p,0,1}]`
    mu = dt * fbeta(alpha + 2*dt, beta) / marginal
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*p*(p - m)^2, {p,0,1}]`
    var = dt * (mu**2 * fbeta(alpha + dt, beta)
              - 2 * mu * fbeta(alpha+ 2*dt, beta)
              + fbeta(alpha+ 3*dt, beta)) / marginal
  else:
    # Mathematica code is same as above, but replace one `p` with `(1-p)`
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p), {p,0,1}]`
    marginal = dt * (fbeta(alpha, beta) - fbeta(alpha+dt, beta))
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*p, {p,0,1}]`
    mu = dt * (fbeta(alpha + dt, beta) - fbeta(alpha + 2*dt, beta)) / marginal
    # `Integrate[p^((a - t)/t)*(1 - p^(1/t))^(b - 1)*(1-p)*(p - m)^2, {p,0,1}]`
    var = dt * (mu**2 * fbeta(alpha, beta)
              - mu * (2 + mu) * fbeta(alpha+dt, beta)
              + (1 + 2 * mu) * fbeta(alpha+2*dt, beta)
              - fbeta(alpha+3*dt, beta)) / marginal
  print(mu, var, marginal)

  newAlpha, newBeta = meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow


def posteriorQuad(alpha, beta, t, result, tnow, analyticMarginal=True, maxiter=100):
  """Update a time-dependent Beta distribution with a new data sample"""
  from scipy.integrate import quadrature

  dt = tnow / t

  if result == 1:
    marginalInt = lambda p: p**((alpha-dt)/dt) * (1-p**(1/dt))**(beta-1)*p
  else:
    # difference from above: -------------------------------------------^vvvv
    marginalInt = lambda p: p**((alpha-dt)/dt) * (1-p**(1/dt))**(beta-1)*(1-p)

  if analyticMarginal:
    from scipy.special import beta as fbeta
    if result == 1:
      marginal = dt * fbeta(alpha+dt, beta)
    else:
      marginal = dt * (fbeta(alpha, beta) - fbeta(alpha+dt, beta))
  else:
    marginalEst = quadrature(marginalInt, 0, 1, maxiter=maxiter)
    marginal = marginalEst[0]

  muInt = lambda p: marginalInt(p) * p
  muEst = quadrature(muInt, 0, 1, maxiter=maxiter)
  mu = muEst[0] / marginal

  varInt = lambda p: marginalInt(p) * (p - mu)**2
  varEst = quadrature(varInt, 0, 1, maxiter=maxiter)
  var = varEst[0] / marginal
  print(mu, var, marginal)

  newAlpha, newBeta = meanVarToBeta(mu, var)
  return newAlpha, newBeta, tnow


def meanVarToBeta(mean, var):
  # [betaFit] https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Two_unknown_parameters

  tmp = mean * (1 - mean) / var - 1
  alpha = mean * tmp
  beta = (1 - mean) * tmp
  return alpha, beta


def posteriorMonteCarlo(alpha, beta, t, result, tnow, N=10000):
  """Update a time-dependent Beta distribution with a new data sample"""
  # [bernoulliLikelihood] https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=769806318#Properties_of_the_Bernoulli_Distribution, third (last) equation
  # [weightedMean] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Mathematical_definition
  # [weightedVar] https://en.wikipedia.org/w/index.php?title=Weighted_arithmetic_mean&oldid=770608018#Weighted_sample_variance
  import scipy.stats as stats

  tPrior = stats.beta.rvs(alpha, beta, size=N)

  # To see where this comes from, read the rest of this document!
  tnowPrior = tPrior ** (tnow / t)

  # This is the Bernoulli likelihood [bernoulliLikelihood]
  weights = (tnowPrior)**result * ((1 - tnowPrior)**(1 - result))

  # See [weightedMean]
  weightedMean = np.sum(weights * tnowPrior) / np.sum(weights)
  # See [weightedVar]
  weightedVar = np.sum(weights * (tnowPrior - weightedMean)**2) / np.sum(weights)
  print(weightedMean, weightedVar)

  newAlpha, newBeta = meanVarToBeta(weightedMean, weightedVar)

  return newAlpha, newBeta, tnow

```

```py
betaa = 4.
betab = 4.
t0 = 7.
dt = 3.3
[posteriorMonteCarlo(betaa, betab, t0, 1., t0 * dt, N=100000), posteriorQuad(betaa, betab, t0, 1., t0 * dt, analyticMarginal=True), posteriorQuad(betaa, betab, t0, 1., t0 * dt, analyticMarginal=False), posteriorAnalytic(betaa, betab, t0, 1., t0 * dt)]
[posteriorMonteCarlo(betaa, betab, t0, 0., t0 * dt, N=100000), posteriorQuad(betaa, betab, t0, 0., t0 * dt, analyticMarginal=True), posteriorQuad(betaa, betab, t0, 0., t0 * dt, analyticMarginal=False), posteriorAnalytic(betaa, betab, t0, 0., t0 * dt)]

def priorToHalflife(a, b, t, N=10000):
  p = stats.beta.rvs(a, b, size=N)
  return pis2gammaest(p, t)
```

### Details (😱😪)
```py
import numpy as np
import scipy.stats as stats
import pylab

from pymc3 import Model, Gamma, Bernoulli
from pymc3 import NUTS, sample
from pymc3 import traceplot



t = 3.
xobs = 1.

def unscented(a,b,dt):
  # https://www.mitre.org/sites/default/files/pdf/vanzandt_unscented.pdf
  mean = a/(a+b)
  var = a*b/((a+b)**2*(a+b+1))
  xset = np.array([-1., 1.]) * np.sqrt(var) + mean
  yset = xset**dt
  mean2 = np.mean(yset)
  var2 = np.var(yset)
  a2 = mean2 * (mean2*(1-mean2)/var2 - 1)
  b2 = (1-mean2) * (mean2*(1-mean2)/var2 - 1)
  return a2, b2


def up(pis,t0,t,xobs):
  pis2 = pis ** (t / t0)
  b2, a2, pis3 = piposterior(pis2, xobs)
  return pis2gammaest(pis3, t)

t0 = 7.
betaa = 12.
betab = 12.
betaa = 4.
betab = 4.
pis = stats.beta.rvs(betaa, betab, size=50000)
pis2 = pis ** (t / t0)
_, _, pis3 = piposterior(pis2, xobs)

dt = t/t0
ps = np.linspace(0 + np.spacing(1), 1 - np.spacing(1))

newprior = lambda dt: mkpdf(ps, stats.beta.pdf(ps**(1/dt), betaa-dt+1, betab))
# via wolfram alpha: zeros of derivative of "new prior"
peakpi = lambda dt: ((betaa - dt)/(betaa + betab - dt - 1))**dt
peakpi(3/7.)



[np.mean(pis3), np.var(pis3)]
w = (pis2)**xobs*(1-pis2)**(1-xobs)
foo = [np.sum(w*pis2) / np.sum(w), np.sum(w * (pis2 - foo[0])**2) / np.sum(w)]
foo


stats.beta.fit(w, floc=0,fscale=1)
pylab.close('all')
pylab.figure()
# pylab.hist(pis,20,normed=True,alpha=.25,label='prior');
# pylab.hist(pis2,20,normed=True,alpha=.25,label='new prior');
pylab.hist(pis3,20,range=[0,1.],normed=True,alpha=.25, label='posterior');
pylab.hist(pis2,20,weights=w,range=[0,1.], normed=True,alpha=.25, label='weighted');
pylab.legend(loc=0);
# pylab.hist(w,20,normed=True,alpha=.5);

##
pispost = stats.beta.rvs(betaa+xobs, betab+1-xobs, size=50000)
foo = stats.beta.pdf(pis, betaa+xobs, betab+1-xobs) / stats.beta.pdf(pis, betaa, betab)
pylab.figure()
pylab.hist(pispost,20,normed=True,alpha=.5);
pylab.hist(pis,20,weights=foo,normed=True,alpha=.5);

##


up(pis, t0, t, xobs)
up(pis, t0, 40, 1)
up(pis, t0, 40, 0)
up(pis, t0, .1, 1)

stats.beta.fit(pis, floc=0,fscale=1)
stats.beta.fit(stats.beta.rvs(betaa, betab, size=50000)**(t/t0), floc=0,fscale=1)
stats.beta.fit(stats.beta.rvs(betaa, betab, size=500)**(t/t0), floc=0,fscale=1)
unscented(betaa, betab, t/t0)

pylab.close('all')
pylab.hist(pis2, 50, alpha=0.5,normed=True, range=[0,1.]);
pylab.hist(pis3, 50, alpha=0.33, normed=True, range=[0,1.]);

stats.beta.fit(pis3, floc=0,fscale=1)

# true PDF of posterior, likely not Beta
dt = t/t0
ps = np.linspace(0 + np.spacing(1), 1 - np.spacing(1))

pylab.plot(ps, stats.beta.pdf(ps**(1/dt), betaa, betab) * ps**(1/dt-1)/dt);

from scipy.special import beta
def mkpdf(x,f):
  return f / f.sum() / (x[1]-x[0])

pylab.hist(pis2, 50, alpha=0.5,normed=True, range=[0,1.]);
pylab.hist(pis3, 50, alpha=0.33, normed=True, range=[0,1.]);
# this is the "new" prior
pylab.plot(ps, mkpdf(ps, stats.beta.pdf(ps**(1/dt), betaa, betab) * ps**(1/dt-1)/dt), '.-');
pylab.plot(ps, mkpdf(ps, stats.beta.pdf(ps**(1/dt), betaa-dt+1, betab)),'--');

# this is the posterior, after Bernoulli likelihood
pylab.plot(ps, mkpdf(ps,ps**(xobs + (betaa-dt)/dt) * (1-ps)**(1-xobs) * (1-ps**(1/dt))**(betab-1) / t / beta(betaa, betab)));

post = mkpdf(ps,ps**(xobs + (betaa-dt)/dt) * (1-ps)**(1-xobs) * (1-ps**(1/dt))**(betab-1) / t / beta(betaa, betab))
postbare = ps**(xobs + (betaa-dt)/dt) * (1-ps)**(1-xobs) * (1-ps**(1/dt))**(betab-1)
from numpy.linalg import lstsq

import scipy.linalg as sla
Amat = np.vstack([np.log(ps), np.log(1-ps)]).T
apm1, bpm1 = lstsq(Amat, np.log(post))[0]
ap, bp = apm1 + 1, bpm1 + 1

import scipy.optimize as opt
apo, bpo = opt.leastsq(lambda v: stats.beta.pdf(ps, v[0], v[1]) - post, np.array([10., 3.5]))
apo, bpo
stats.beta.fit(pis3, floc=0,fscale=1)


##

pylab.close('all')
pylab.hist(pis3, 50, alpha=0.33, normed=True, range=[0,1.], label='post beta');
pylab.plot(ps, mkpdf(ps,ps**(xobs + (betaa-dt)/dt) * (1-ps)**(1-xobs) * (1-ps**(1/dt))**(betab-1)), label='true post');
pylab.plot(ps, stats.beta.pdf(ps, ap, bp), label='fit post')
pylab.legend(loc=0)

ts = np.arange(1,40)
success = np.array(list(map(lambda t: up(pis,t0,t,1.), ts)))
fail = np.array(list(map(lambda t: up(pis,t0,t,0.), ts)))
pylab.close('all')
pylab.errorbar(ts, success[:,0], yerr=success[:,1]/3,fmt='x-');
pylab.errorbar(ts, fail[:,0], yerr=fail[:,1]/3,fmt='o-');
pylab.grid(True)

pylab.grid(True)


def anki(i, d=0, m=1., m4=1.3, f=1.3):
  i2 = max(i + 1, (i + d/4) * 1.2 * m)
  i3 = max(i2 + 1, (i + d/2) * (f / 1000) * m)
  i4 = max(i3 + 1, (i + d) * (f / 1000) * m * m4)
  return (i2, i3, i4)

anki(7)
anki(7,-3)
anki(7,3)

def piposterior(priorpis, x):
  betaa, betab, _, _ = stats.beta.fit(priorpis, floc=0, fscale=1)
  betaa2 = betaa + x
  betab2 = betab + 1 - x
  postsamples = stats.beta.rvs(betaa2, betab2, size=hs.size)
  return betaa2, betab2, postsamples


def pis2gammaest(pis, t):
  h2s = -t / np.log2(pis)
  alpha2, _, beta2inv = stats.gamma.fit(h2s, floc=0)
  beta2 = 1 / beta2inv

  uh2, vh2 = (alpha2/beta2, alpha2 / beta2**2)
  return (uh2, np.sqrt(vh2))


pis2gammaest(pis, t0)
pis2gammaest(pis2, t)
pylab.close('all')
pylab.hist(pis,40)
pylab.hist(pis2,40)
pylab.show()


uh = 7. # days (mean)
vh = 4. # days**2 (variance)

beta = uh / vh
alpha = vh * beta**2

assert(np.isclose(uh, alpha / beta))
assert(np.isclose(vh, alpha / beta**2))

hs = stats.gamma.rvs(alpha, scale=1/beta, size=50000)
hs.mean()
hs.var()

def newMeanVar(t):
  π = 2.**(-t/hs)

  betaa, betab, _, _ = stats.beta.fit(π,floc=0,fscale=1)

  betaa2 = betaa + xobs
  betab2 = betab + 1 - xobs

  beta2s = stats.beta.rvs(betaa2, betab2, size=hs.size)
  h2s = -t / np.log2(beta2s)
  alpha2, _, beta2inv = stats.gamma.fit(h2s, floc=0)
  beta2 = 1 / beta2inv

  uh2, vh2 = (alpha2/beta2, alpha2 / beta2**2)
  return (uh2, vh2)

ts = np.arange(1,20)
news = np.array(list(map(newMeanVar, ts)))
pylab.plot(ts,news,'.-')
pylab.grid(True)

pylab.plot(ts,news,'.-')
pylab.grid(True)

betas = stats.beta.rvs(betaa, betab, size=10000)
pylab.close('all')
pylab.hist(π,40)
pylab.hist(betas,40)
pylab.hist(beta2s,40)
pylab.show()


basic_model = Model()
with basic_model:
  h = Gamma('h', alpha, beta)
  pi = 2 ** (-t / h)
  x = Bernoulli('x', pi, observed=xobs)

with basic_model:
  trace = sample(5000)
[trace['h'].mean(), trace['h'].var()]

# [6.6263918353576061, 2.0113573783697993] with vh = 2 & x=0, t=3
# [6.4593888708603382, 3.512451646364334] with vh = 4 & x = 0, t=3
# [7.1920570690327379, 3.7075010758265678] with v=4 & x = 1, t=3
# [7.6331225997896226, 4.0075803038947049] with v=4 & x = 1 & t=12
# [6.7015398215486259, 3.5228654317357759] v=4, x=0, t=12

pylab.ion()
traceplot(trace);

pylab.close('all')
pylab.figure()
pylab.hist(posteriors, bins=100)
pylab.show()

```
## Implementation thoughts

Lua, Erlang, Elixir, Red, F#, OCaml, Reason, PureScript, JS, TypeScript, Rust, …

Postgres (w/ or w/o GraphQL), SQLite, LevelDB, Redis, Lovefield, …
