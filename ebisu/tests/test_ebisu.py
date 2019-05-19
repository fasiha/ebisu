# -*- coding: utf-8 -*-

from ebisu import *
from ebisu.alternate import *
import unittest


def relerr(dirt, gold):
  return abs(dirt - gold) / abs(gold)


def maxrelerr(dirts, golds):
  return max(map(relerr, dirts, golds))


def klDivBeta(a, b, a2, b2):
  """Kullback-Leibler divergence between two Beta distributions in nats"""
  # Via http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
  from scipy.special import gammaln, psi
  import numpy as np
  left = np.array([a, b])
  right = np.array([a2, b2])
  return gammaln(sum(left)) - gammaln(sum(right)) - sum(gammaln(left)) + sum(
      gammaln(right)) + np.dot(left - right, psi(left) - psi(sum(left)))


def kl(v, w):
  return (klDivBeta(v[0], v[1], w[0], w[1]) + klDivBeta(w[0], w[1], v[0], v[1])
         ) / 2.


testpoints = []


class TestEbisu(unittest.TestCase):

  def test_predictRecallMedian(self):
    model0 = (4.0, 4.0, 1.0)
    model1 = updateRecall(model0, False, 1.0)
    model2 = updateRecall(model1, True, 0.01)
    ts = np.linspace(0.01, 4.0, 81.0)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    for t in ts:
      for q in qs:
        self.assertGreater(predictRecallMedian(model2, t, q), 0)

  def test_kl(self):
    # See https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Quantities_of_information_.28entropy.29 for these numbers
    self.assertAlmostEqual(klDivBeta(1., 1., 3., 3.), 0.598803, places=5)
    self.assertAlmostEqual(klDivBeta(3., 3., 1., 1.), 0.267864, places=5)

  def test_prior(self):

    def inner(a, b, t0):
      global testpoints
      for t in map(lambda dt: dt * t0, [0.1, .99, 1., 1.01, 5.5]):
        mc = predictRecallMonteCarlo((a, b, t0), t, N=100 * 1000)
        mean = predictRecall((a, b, t0), t)
        var = predictRecallVar((a, b, t0), t)
        self.assertLess(relerr(mean, mc['mean']), 5e-2)
        self.assertLess(relerr(var, mc['var']), 5e-2)
        testpoints += [['predict', [a, b, t0], [t], dict(mean=mean, var=var)]]

    inner(3.3, 4.4, 1.)
    inner(34.4, 34.4, 1.)

  def test_posterior(self):

    def inner(a, b, t0, dts):
      global testpoints
      for t in map(lambda dt: dt * t0, dts):
        for x in [False, True]:
          msg = 'a={},b={},t0={},x={},t={}'.format(a, b, t0, x, t)
          mc = updateRecallMonteCarlo((a, b, t0), x, t, N=1 * 100 * 1000)
          an = updateRecall((a, b, t0), x, t)
          self.assertLess(
              kl(an, mc), 1e-3, msg=msg + ' an={}, mc={}'.format(an, mc))

          try:
            quad1 = updateRecallQuad((a, b, t0), x, t, analyticMarginal=True)
          except OverflowError:
            quad1 = None
          if quad1 is not None:
            self.assertLess(kl(quad1, mc), 1e-3, msg=msg)

          try:
            quad2 = updateRecallQuad((a, b, t0), x, t, analyticMarginal=False)
          except OverflowError:
            quad2 = None
          if quad2 is not None:
            self.assertLess(kl(quad2, mc), 1e-3, msg=msg)

          testpoints += [['update', [a, b, t0], [x, t], dict(post=an)]]

    inner(3.3, 4.4, 1., [0.1, 1., 9.5])
    inner(341.4, 3.4, 1., [0.1, 1., 5.5, 50.])


if __name__ == '__main__':
  unittest.TextTestRunner().run(
      unittest.TestLoader().loadTestsFromModule(TestEbisu()))

  with open("test.json", "w") as out:
    import json
    out.write(json.dumps(testpoints))
