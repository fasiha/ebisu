from math import exp, log


def logsumexp(a, b):
  a_max = max(a)
  s = 0
  for i in range(len(a) - 1, -1, -1):
    s += b[i] * exp(a[i] - a_max)
  sgn = 1 if s >= 0 else 0
  s *= sgn
  out = log(s) + a_max
  return [out, sgn]
