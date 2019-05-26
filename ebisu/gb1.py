# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import ebisu
from typing import Dict, Tuple, Sequence


def failureMoments(model: Tuple[float, float, float], result: bool, tnow: float, num: int = 4):
  """Moments of the posterior on recall at time tnow upon quiz failure"""
  a, b, t0 = model
  t = tnow / t0
  from scipy.special import gammaln, logsumexp
  from numpy import log, exp
  s = [gammaln(a + n * t) - gammaln(a + b + n * t) for n in range(num + 2)]
  logt = log(t)
  marginal = logsumexp([s[0], s[1]], b=[1, -1])
  return [exp(logsumexp([s[n], s[n + 1]], b=[1, -1]) - marginal) for n in range(1, num + 1)]


def gb1Moments(a: float, b: float, p: float, q: float, num: int = 2):
  """Raw moments of GB1 via Wikipedia"""
  from scipy.special import betaln
  bpq = betaln(p, q)
  logb = np.log(b)
  return [np.exp(h * logb + betaln(p + h / a, q) - bpq) for h in np.arange(1.0, num + 1)]


def gb1ToBeta(gb1: Tuple[float, float, float, float, float]):
  """Convert a GB1 model (four GB1 parameters, and time) to a Beta model"""
  return (gb1[2], gb1[3], gb1[4] * gb1[0])


def updateViaGb1(prior: Tuple[float, float, float], result: bool, tnow: float):
  """Alternate to ebisu.updateRecall that returns several posterior Beta models"""
  (alpha, beta, t) = prior
  delta = tnow / t
  if result:
    gb1 = (1.0 / delta, 1.0, alpha + delta, beta, tnow)
  else:
    mom = np.array(failureMoments(prior, result, tnow, num=2))

    def f(bd):
      b, d = bd
      this = np.array(gb1Moments(1 / d, 1., alpha, b, num=2))
      return this - mom

    from scipy.optimize import least_squares
    res = least_squares(f, [beta, delta], bounds=((1.01, 0), (np.inf, np.inf)))
    # print("optimization cost", res.cost)
    newBeta, newDelta = res.x
    gb1 = (1 / newDelta, 1, alpha, newBeta, tnow)
  return dict(
      simple=gb1ToBeta(gb1),
      moved=gb1ToBeta(moveGb1(gb1, prior[2])),
      moved2=moveBeta(gb1ToBeta(gb1)))


def moveGb1(gb1: Tuple[float, float, float, float, float], priorT):
  """Given a GB1 model (4 parameters and time), find the closest GB1 distribution where alpha=beta
  
  This will be at the halflife. Can produce terrible results when stressed. Why?"""
  mom1, mom2 = gb1Moments(*gb1[:-1])

  def f(aa):
    newA, newAlpha = aa
    this1, this2 = gb1Moments(newA, 1., newAlpha, newAlpha)
    return np.array([this1 - mom1, this2 - mom2])

  from scipy.optimize import least_squares
  res = least_squares(f, [gb1[0], (gb1[2] + gb1[3]) / 2], bounds=((0, 1.01), (np.inf, np.inf)))
  # print("optimization cost", res.cost)
  print('movegb1res', res.x, 'cost', res.cost)
  return [res.x[0], gb1[1], res.x[1], res.x[1], gb1[4]]


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
  Bab = special.beta(alpha, beta)

  def f(lndelta):
    mean = special.beta(alpha + np.exp(lndelta), beta) / Bab
    return mean - quantile

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


def moveBeta(model: Tuple[float, float, float]):
  """Given a Beta model (2 parameters and time), find the closest Beta model at the halflife
  
  Works bmuch better than `moveGb1`"""
  th = estimate_half_life(model, 0.5)
  m = ebisu.predictRecall(model, th)
  v = ebisu.predictRecallVar(model, th)
  alpha1, beta1 = ebisu._meanVarToBeta(m, v)
  return (alpha1, beta1, th)


if __name__ == '__main__':
  model = (4., 30., 1.)
  model = (40., 6., 1.)
  model = (4., 6., 1.)
  result = True
  result = False
  tnow = 30.

  def simulation(model, result, tnow):
    gold = ebisu.updateRecall(model, result, tnow)

    newModel = updateViaGb1(model, result, tnow)
    print(newModel)
    t = np.linspace(.1, 25, 200)

    def trace(model):
      return np.vectorize(lambda t: ebisu.predictRecall(model, t))(t)

    import pylab as plt
    plt.style.use('ggplot')
    plt.ion()
    plt.figure()
    plt.semilogy(t, trace(model), linewidth=5, label='prior')
    plt.semilogy(t, trace(gold), linewidth=4, label='post')
    plt.semilogy(t, trace(newModel['simple']), '--', linewidth=3, label='via GB1')
    plt.semilogy(t, trace(newModel['moved']), linewidth=2, label='via GB1@HL')
    plt.semilogy(t, trace(newModel['moved2']), '--', linewidth=1, label='Beta@HL')
    plt.legend()
    plt.title('Quiz={}, T={}'.format(result, tnow))

  simulation(model, True, 100.)
  simulation(model, False, 100.)

  simulation(model, True, 30.)
  simulation(model, False, 30.)

  simulation(model, True, 3.)
  simulation(model, False, 3.)

  simulation(model, True, .01)
  simulation(model, False, .01)
"""
For `model = (4., 6., 1.)`, 
- the "via GB1" model, which doesn't move the prior halflife (much) and
- the Beta@HL model, which telescopes the "via GB1" model to the halflife,
both agree very well for severe over/under-review, pass/fail quizzes.

Similarly for `model = (40, 6, 1)` and `(4., 30., 1.)`.
"""