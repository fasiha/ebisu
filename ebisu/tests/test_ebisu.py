# -*- coding: utf-8 -*-

from ebisu import *
from ebisu.alternate import *
import unittest
import numpy as np


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
      gammaln(right)) + np.dot(left - right,
                               psi(left) - psi(sum(left)))


def kl(v, w):
  return (klDivBeta(v[0], v[1], w[0], w[1]) + klDivBeta(w[0], w[1], v[0], v[1])) / 2.


testpoints = []


class TestEbisu(unittest.TestCase):

  def test_predictRecallMedian(self):
    model0 = (4.0, 4.0, 1.0)
    model1 = updateRecall(model0, False, 1.0)
    model2 = updateRecall(model1, True, 0.01)
    ts = np.linspace(0.01, 4.0, 81)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    for t in ts:
      for q in qs:
        self.assertGreater(predictRecallMedian(model2, t, q), 0)

  def test_kl(self):
    # See https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=774237683#Quantities_of_information_.28entropy.29 for these numbers
    self.assertAlmostEqual(klDivBeta(1., 1., 3., 3.), 0.598803, places=5)
    self.assertAlmostEqual(klDivBeta(3., 3., 1., 1.), 0.267864, places=5)

  def test_prior(self):
    "test predictRecall vs predictRecallMonteCarlo"

    def inner(a, b, t0):
      global testpoints
      for t in map(lambda dt: dt * t0, [0.1, .99, 1., 1.01, 5.5]):
        mc = predictRecallMonteCarlo((a, b, t0), t, N=100 * 1000)
        mean = predictRecall((a, b, t0), t, exact=True)
        self.assertLess(relerr(mean, mc['mean']), 5e-2)
        testpoints += [['predict', [a, b, t0], [t], dict(mean=mean)]]

    inner(3.3, 4.4, 1.)
    inner(34.4, 34.4, 1.)

  def test_posterior(self):
    "Test updateRecall via updateRecallMonteCarlo"

    def inner(a, b, t0, dts):
      global testpoints
      for t in map(lambda dt: dt * t0, dts):
        for x in [False, True]:
          msg = 'a={},b={},t0={},x={},t={}'.format(a, b, t0, x, t)
          an = updateRecall((a, b, t0), x, t)
          mc = updateRecallMonteCarlo((a, b, t0), x, t, an[2], N=100 * 1000)
          self.assertLess(kl(an, mc), 5e-3, msg=msg + ' an={}, mc={}'.format(an, mc))

          testpoints += [['update', [a, b, t0], [x, t], dict(post=an)]]

    inner(3.3, 4.4, 1., [0.1, 1., 9.5])
    inner(34.4, 3.4, 1., [0.1, 1., 5.5, 50.])

  def test_update_then_predict(self):
    "Ensure #1 is fixed: prediction after update is monotonic"
    future = np.linspace(.01, 1000, 101)

    def inner(a, b, t0, dts):
      for t in map(lambda dt: dt * t0, dts):
        for x in [False, True]:
          msg = 'a={},b={},t0={},x={},t={}'.format(a, b, t0, x, t)
          newModel = updateRecall((a, b, t0), x, t)
          predicted = np.vectorize(lambda tnow: predictRecall(newModel, tnow))(future)
          self.assertTrue(
              np.all(np.diff(predicted) < 0), msg=msg + ' predicted={}'.format(predicted))

    inner(3.3, 4.4, 1., [0.1, 1., 9.5])
    inner(34.4, 3.4, 1., [0.1, 1., 5.5, 50.])

  def test_halflife(self):
    "Exercise modelToPercentileDecay"
    percentiles = np.linspace(.01, .99, 101)

    def inner(a, b, t0, dts):
      for t in map(lambda dt: dt * t0, dts):
        msg = 'a={},b={},t0={},t={}'.format(a, b, t0, t)
        ts = np.vectorize(lambda p: modelToPercentileDecay((a, b, t), p))(percentiles)
        self.assertTrue(monotonicDecreasing(ts), msg=msg + ' ts={}'.format(ts))

    inner(3.3, 4.4, 1., [0.1, 1., 9.5])
    inner(34.4, 3.4, 1., [0.1, 1., 5.5, 50.])

  def test_asymptotic(self):
    """Failing quizzes in far future shouldn't modify model when updating.
    Passing quizzes right away shouldn't modify model when updating.
    """

    def inner(a, b):
      prior = (a, b, 1.0)
      hl = modelToPercentileDecay(prior)
      ts = np.linspace(.001, 1000, 101)
      passhl = np.vectorize(lambda tnow: modelToPercentileDecay(
          updateRecall(prior, True, tnow, 1.0)))(
              ts)
      failhl = np.vectorize(lambda tnow: modelToPercentileDecay(
          updateRecall(prior, False, tnow, 1.0)))(
              ts)
      self.assertTrue(monotonicIncreasing(passhl))
      self.assertTrue(monotonicIncreasing(failhl))
      # Passing should only increase halflife
      self.assertTrue(np.all(passhl >= hl * .999))
      # Failing should only decrease halflife
      self.assertTrue(np.all(failhl <= hl * 1.001))

    for a in [2., 20, 200]:
      for b in [2., 20, 200]:
        inner(a, b)


def monotonicIncreasing(v):
  return np.all(np.diff(v) >= -np.spacing(1.) * 1e8)


def monotonicDecreasing(v):
  return np.all(np.diff(v) <= np.spacing(1.) * 1e8)


if __name__ == '__main__':
  unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromModule(TestEbisu()))

  with open("test.json", "w") as out:
    import json
    out.write(json.dumps(testpoints))
