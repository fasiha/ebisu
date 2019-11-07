from typing import Callable
import numpy as np


def intensity(t: float, timeToRecallProb: Callable[[float], float], q: float) -> float:
  p = timeToRecallProb(t)
  assert p >= 0 and p <= 1
  return 1.0 / np.sqrt(q) * (1 - p)


def sampler(timeToRecallProb: Callable[[float], float], q: float, T: float):
  t = 0
  max_int = 1.0 / np.sqrt(q)
  max_int_inv = 1.0 / max_int
  while (True):
    t_ = np.random.exponential(max_int_inv)
    if t_ + t > T:
      return None
    t = t + t_
    proposed_int = intensity(t, timeToRecallProb, q)
    if np.random.uniform(0, 1, 1)[0] < (proposed_int * max_int_inv):
      return t


if __name__ == '__main__':
  import ebisu
  model = [3., 3., 5.5]
  # 0 probability of recall -> sqrt(q) reviews per unit time
  q = 1e-2
  T = 1000.
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 0.0, True), q, T) for i in range(1000)]
  print('mean', np.mean(res))
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 1.0, True), q, T) for i in range(1000)]
  print('mean', np.mean(res))
  res = [sampler(lambda t: ebisu.predictRecall(model, t + 22.0, True), q, T) for i in range(1000)]
  print('mean', np.mean(res))
