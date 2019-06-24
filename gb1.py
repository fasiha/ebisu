# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import ebisu
from typing import Dict, Tuple, Sequence


def failureMoments(model: Tuple[float, float, float],
                   result: bool,
                   tnow: float,
                   num: int = 4,
                   returnLog: bool = True):
  """Moments of the posterior on recall at time tnow upon quiz failure"""
  a, b, t0 = model
  t = tnow / t0
  from scipy.special import gammaln, logsumexp
  from numpy import exp
  s = [gammaln(a + n * t) - gammaln(a + b + n * t) for n in range(num + 2)]
  marginal = logsumexp([s[0], s[1]], b=[1, -1])
  ret = [(logsumexp([s[n], s[n + 1]], b=[1, -1]) - marginal) for n in range(1, num + 1)]
  return ret if returnLog else [exp(x) for x in ret]


def gb1Moments(a: float, b: float, p: float, q: float, num: int = 2, returnLog: bool = True):
  """Raw moments of GB1 via Wikipedia"""
  from scipy.special import betaln
  bpq = betaln(p, q)
  logb = np.log(b)
  ret = [(h * logb + betaln(p + h / a, q) - bpq) for h in np.arange(1.0, num + 1)]
  return ret if returnLog else [np.exp(x) for x in ret]


def gb1ToBeta(gb1: Tuple[float, float, float, float, float]):
  """Convert a GB1 model (four GB1 parameters, and time) to a Beta model"""
  return (gb1[2], gb1[3], gb1[4] * gb1[0])


def updateGb1(prior: Tuple[float, float, float], result: bool, tnow: float):
  """GB1-based replacement to ebisu.updateRecall"""
  (alpha, beta, t) = prior
  dt = tnow / t
  if result:
    gb1 = (1.0 / dt, 1.0, alpha + dt, beta, tnow)
  else:
    mom = np.array(failureMoments(prior, result, tnow, num=2))

    def f(bd):
      b, d = bd
      this = np.array(gb1Moments(1 / d, 1., alpha, b, num=2))
      return this - mom

    from scipy.optimize import least_squares
    res = least_squares(f, [beta, dt], bounds=((1.01, 0), (np.inf, np.inf)))
    # print("optimization cost", res)
    newBeta, newDelta = res.x
    gb1 = (1 / newDelta, 1, alpha, newBeta, tnow)

  return gb1ToBeta(gb1)


def updateTest(prior: Tuple[float, float, float], result: bool, tnow: float):
  """Alternate to ebisu.updateRecall that returns several posterior Beta models"""
  (alpha, beta, t) = prior
  delta = tnow / t
  if result:
    gb1 = (1.0 / delta, 1.0, alpha + delta, beta, tnow)
    mom = gb1Moments(*gb1[:-1])
  else:
    mom = np.array(failureMoments(prior, result, tnow, num=2))

    def f(bd):
      b, d = bd
      this = np.array(gb1Moments(1 / d, 1., alpha, b, num=2))
      return this - mom

    from scipy.optimize import least_squares
    res = least_squares(f, [beta, delta], bounds=((1.01, 0), (np.inf, np.inf)))
    # print("optimization cost", res)
    newBeta, newDelta = res.x
    gb1 = (1 / newDelta, 1, alpha, newBeta, tnow)

  moved2 = moveBeta(gb1ToBeta(gb1))
  return dict(
      simple=gb1ToBeta(gb1),
      moved=gb1ToBeta(moveGb1(gb1, prior[2])),
      moved2=moved2,
      moved3=moveBeta2(gb1ToBeta(gb1)),
      moveBetasOnly=moveBeta(ebisu.updateRecall(prior, result, tnow)),
      postToBeta=posteriorMomToBeta(mom, moved2, tnow),
  )


def betaMoments(a, b, num=2):
  mom = [a / (a + b)]
  for k in range(2, num + 1):
    mom.append(mom[-1] * (a + k - 1) / (a + b + k - 1))
  return mom


def posteriorMomToBeta(mom, approxModel, tnow):

  def f(ah):
    newAlpha, newH = ah
    this = np.array(gb1Moments(newH / tnow, 1., newAlpha, newAlpha))
    return this - mom

  from scipy.optimize import least_squares
  res = least_squares(f, (approxModel[0], approxModel[2]))
  return [res.x[0], res.x[0], res.x[1]]


def moveGb1(gb1: Tuple[float, float, float, float, float], priorT):
  """Given a GB1 model (4 parameters and time), find the closest GB1 distribution where alpha=beta
  
  This will be at the halflife. Can produce terrible results when stressed. Why?"""
  mom = np.array(gb1Moments(*gb1[:-1], num=4))

  def f(aa):
    newA, newAlpha = np.exp(aa[0]), np.exp(aa[1])
    this = np.array(gb1Moments(newA, 1., newAlpha, newAlpha, num=4))
    return this - mom

  from scipy.optimize import least_squares
  res = least_squares(f, [np.log(gb1[0]), np.log((gb1[2] + gb1[3]) / 2)])
  # print("optimization cost", res.cost)
  print('movegb1res', np.exp(res.x), 'cost', res.cost)
  expx = np.exp(res.x)
  return [expx[0], gb1[1], expx[1], expx[1], gb1[4]]


def estimate_half_life(model, quantile=0.5):
  """Robust half-life (or quantile-life) estimation.
    
    Use a root-finding routine in log-delta space to find the delta that
    will cause the GB1 distribution to have a mean of the requested quantile.
    Because we are using well-behaved normalized deltas instead of times, and
    owing to the monotonicity of the expectation with respect to delta, we can
    quickly scan for a rough estimate of the scale of delta, then do a finishing
    optimization to get the right value.
    """
  alpha, beta, t0 = model
  logBab = special.betaln(alpha, beta)
  logQuantile = np.log(quantile)

  def f(lndelta):
    logMean = special.betaln(alpha + np.exp(lndelta), beta) - logBab
    return logMean - logQuantile

  # Scan for a bracket.
  bracket_width = 6.0
  blow = -bracket_width / 2.0
  bhigh = bracket_width / 2.0
  flow = f(blow)
  fhigh = f(bhigh)
  while flow > 0 and fhigh > 0:
    # Move the bracket up.
    blow = bhigh
    flow = fhigh
    bhigh += bracket_width
    fhigh = f(bhigh)
  while flow < 0 and fhigh < 0:
    # Move the bracket down.
    bhigh = blow
    fhigh = flow
    blow -= bracket_width
    flow = f(blow)

  assert flow > 0 and fhigh < 0

  sol = optimize.root_scalar(f, bracket=[blow, bhigh])
  t1 = np.exp(sol.root) * t0
  return t1


def moveBeta2(model: Tuple[float, float, float]):
  th = estimate_half_life(model, 0.5)
  a, b, t = model
  d = th / t
  m1, m2 = gb1Moments(1 / d, 1., a, b)
  var = m2 - m1**2
  alpha1, beta1 = ebisu.alternate._meanVarToBeta(m1, var)
  return (alpha1, beta1, th)


def moveBeta(model: Tuple[float, float, float]):
  """Given a Beta model (2 parameters and time), find the closest Beta model at the halflife
  
  Works bmuch better than `moveGb1`"""
  th = estimate_half_life(model, 0.5)
  m = ebisu.predictRecall(model, th)
  v = ebisu.predictRecallVar(model, th)
  alpha1, beta1 = ebisu.alternate._meanVarToBeta(m, v)
  return (alpha1, beta1, th)


def meanVarToGB1(mu, var):
  m2 = var + mu**2
  mom = np.array([mu, m2])

  def f(bd):
    b, d = bd
    this = np.array(gb1Moments(1 / d, 1., alpha, b, num=2))
    return this - mom

  from scipy.optimize import least_squares
  res = least_squares(f, [beta, dt], bounds=((1.01, 0), (np.inf, np.inf)))
  newBeta, newDelta = res.x
  return (1 / newDelta, 1, alpha, newBeta, tnow)


if __name__ == '__main__':
  import pylab as plt
  plt.style.use('ggplot')
  model = (4., 30., 1.)
  model = (40., 6., 1.)
  model = (4., 6., 1.)

  def updateRecallMonteCarlo(prior, result, tnow, N=10 * 1000):
    import scipy.stats as stats
    alpha, beta, t = prior
    tPrior = stats.beta.rvs(alpha, beta, size=N)
    tnowPrior = tPrior**(tnow / t)
    # This is the Bernoulli likelihood [bernoulliLikelihood]
    weights = (tnowPrior)**result * ((1 - tnowPrior)**(1 - result))
    # See [weightedMean]
    weightedMean = np.sum(weights * tnowPrior) / np.sum(weights)
    # See [weightedVar]
    weightedVar = np.sum(weights * (tnowPrior - weightedMean)**2) / np.sum(weights)
    newAlpha, newBeta = _meanVarToBeta(weightedMean, weightedVar)
    return newAlpha, newBeta, tnow

  def simulation(model, result, tnow):
    gold = ebisu.updateRecall(model, result, tnow)

    newModel = updateTest(model, result, tnow)
    print(newModel)
    t = np.linspace(.1, 100 * 1, 50 * 1)

    def trace(model):
      return np.vectorize(lambda t: ebisu.predictRecall(model, t))(t)

    def yerr(model):
      return np.vstack([
          np.vectorize(lambda t: ebisu.alternate.predictRecallMedian(model, t, 0.25))(t),
          np.vectorize(lambda t: ebisu.alternate.predictRecallMedian(model, t, 0.75))(t),
      ])

    def both(model):
      y = trace(model)
      return dict(y=y, yerr=np.abs(yerr(model) - y))

    plt.ion()
    plt.figure()
    plt.semilogy(t, trace(model), linewidth=6, label='prior')
    # plt.semilogy(t, trace(gold), linewidth=5, label='orig')
    plt.semilogy(t, trace(newModel['simple']), '--', linewidth=4, label='via GB1')
    # plt.semilogy(t, trace(newModel['moved']), linewidth=3, label='via GB1@HL')
    plt.semilogy(t, trace(newModel['moved2']), '--', linewidth=2, label='Beta@HL')
    plt.semilogy(t, trace(newModel['postToBeta']), linewidth=1, label='post2beta')
    # plt.semilogy(t, trace(newModel['moveBetasOnly']), '-', linewidth=2, label='orig@HL')
    plt.legend()
    plt.title('Model=({},{},{}), Quiz={} @ Tnow={}'.format(*model, result, tnow))

    result = False
    import scipy.stats as stats
    alpha, beta, tau = model
    N = 10 * 1000 * 1000
    priorTau = stats.beta.rvs(alpha, beta, size=N)
    tQuiz = 3.
    priorTQuiz = priorTau**(tQuiz / tau)
    posteriorWeights = (priorTQuiz)**result * ((1 - priorTQuiz)**(1 - result))
    posteriorModel = updateGb1(model, result, tQuiz)
    posteriorNewTau = (priorTQuiz)**(posteriorModel[2] / tQuiz)
    weightedMean = np.sum(posteriorWeights * posteriorNewTau) / np.sum(posteriorWeights)
    weightedVar = np.sum(posteriorWeights *
                         (posteriorNewTau - weightedMean)**2) / np.sum(posteriorWeights)
    print('expected', summarizeBeta(posteriorModel[0], posteriorModel[1]))
    print('actual', [weightedMean, weightedVar])

    def summarizeBeta(a, b):
      return dict(mean=a / (a + b), var=a * b / (a + b)**2 / (a + b + 1))

  simulation(model, True, 100.)
  simulation(model, False, 100.)

  simulation(model, True, 30.)
  simulation(model, False, 30.)

  simulation(model, True, 3.)
  simulation(model, False, 3.)

  simulation(model, True, .01)
  simulation(model, False, .01)

  def predictAnalysis():
    models = []
    hlmodels = []
    for i in range(1000):
      a = np.random.rand() * 10 + 2
      b = np.random.rand() * 10 + 2
      t = np.random.rand() * 20
      model = [a, b, t]
      hlmodel = moveBeta(model)
      models.append(model)
      hlmodels.append(hlmodel)
    t = np.linspace(.1, 500, 1000)
    t0 = t[50]
    ps = [ebisu.predictRecall(model, t0) for model in hlmodels]
    arg = np.argsort(ps)
    pviat = [model[2] / t0 for model in hlmodels]
    arg2 = np.argsort(pviat)
    plt.figure()
    plt.scatter(ps, pviat, np.array([m[0] for m in hlmodels]))

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ps, pviat, np.array([m[0] for m in hlmodels]))
    ax.set_xlabel('Precall')
    ax.set_ylabel('t/tnow')
    ax.set_zlabel('alpha')

    deltav = np.logspace(-2, 2)
    deltav = np.linspace(.01, 10)
    alphav = np.linspace(1.2, 40)
    dmesh, amesh = np.meshgrid(deltav, alphav)
    from scipy.special import gammaln
    pmesh = gammaln(
        2 * amesh) - gammaln(amesh) + gammaln(amesh + dmesh) - gammaln(2 * amesh + dmesh)

    def myimshow(x, y, data):

      def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

      ax = plt.imshow(
          data, aspect='auto', interpolation='none', extent=extents(x) + extents(y), origin='lower')

    plt.figure()
    myimshow(deltav, alphav, pmesh)
    plt.xlabel('delta')
    plt.ylabel('alpha')
    plt.colorbar()


"""
For `model = (4., 6., 1.)`, 
- the "via GB1" model, which doesn't move the prior halflife (much) and
- the Beta@HL model, which telescopes the "via GB1" model to the halflife,
both agree very well for severe over/under-review, pass/fail quizzes.

Similarly for `model = (40, 6, 1)` and `(4., 30., 1.)`.

Well, I don't like how the telescoping behaves for `(4., 30., 1.)`. True
and False at T=0.01 both are not great far past the new halflife.

Similarly for (40, 6, 1), True and false at T=0.01 don't behave as expected. 
"""