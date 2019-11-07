import numpy as np
import ebisu


def intensity(t: float, model: [float, float, float], q: float,
              time_since_last_quiz: float = 0) -> float:
  p = ebisu.predictRecall(model, t + time_since_last_quiz, exact=True)
  # print('p=', p)
  return 1.0 / np.sqrt(q) * (1 - p)


def sampler(model, q, T, time_since_last_quiz):
  t = 0
  max_int = 1.0 / np.sqrt(q)
  max_int_inv = 1.0 / max_int
  while (True):
    t_ = np.random.exponential(max_int_inv)
    if t_ + t > T:
      return None
    t = t + t_
    proposed_int = intensity(t, model, q, time_since_last_quiz)
    if np.random.uniform(0, 1, 1)[0] < (proposed_int * max_int_inv):
      return t


if __name__ == '__main__':
  model = [3., 3., 5.5]
  # 0 probability of recall -> sqrt(q) reviews per unit time
  q = 1e-2
  T = 1000.
  res = [sampler(model, q, T, 0.) for i in range(1000)]
  print('mean', np.mean(res))
  res = [sampler(model, q, T, 1.) for i in range(1000)]
  print('mean', np.mean(res))
  res = [sampler(model, q, T, 22.) for i in range(1000)]
  print('mean', np.mean(res))
  # for i in range(5):
  #   print(sampler(model, q, T, 0.0))
