from random import random
from math import sqrt, log
from typing import Callable  # just for static type checks


def exponential(scale: float) -> float:
  """Replacement for numpy.random.exponential"""
  return -log(random()) * scale


def intensity(t: float, timeToRecallProb: Callable[[float], float], q: float) -> float:
  """Evaluate the review intensity u(t)

  Inputs:
  - `t` time units in the future
  - `timeToRecallProb` function that returns recall probability at any given time
  - `q` tunable parameter balancing forgetting versus high review rates

  Returns a Poisson process' intensity.
  """
  p = timeToRecallProb(t)
  assert p >= 0 and p <= 1
  return 1.0 / sqrt(q) * (1 - p)


def sampler(timeToRecallProb: Callable[[float], float], q: float, T: float):
  """Random draw from the thinned Poisson point process

  Inputs:
  - `timeToRecallProb` function that returns recall probability at any given time
  - `q` tunable parameter balancing forgetting versus high review rates
  - `T` maximum time horizon

  Returns a time in the future to schedule this item *or* `None` if the item should not be
  scheduled within `T` time units.
  """
  t = 0
  max_int = 1.0 / sqrt(q)
  max_int_inv = 1.0 / max_int
  while (True):
    t_ = exponential(max_int_inv)
    if t_ + t > T:
      return None
    t = t + t_
    proposed_int = intensity(t, timeToRecallProb, q)
    if random() < (proposed_int * max_int_inv):
      return t


if __name__ == '__main__':
  model = [3., 3., 5.5]
  # 0 probability of recall -> sqrt(q) reviews per unit time
  q = 1e-2
  T = 1000.

  mean = lambda v: sum(v) / len(v)
  import ebisu
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 0.0, True), q, T) for i in range(1000)]
  print('mean', mean(res))
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 1.0, True), q, T) for i in range(1000)]
  print('mean', mean(res))
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 22.0, True), q, T) for i in range(1000)]
  print('mean', mean(res))
