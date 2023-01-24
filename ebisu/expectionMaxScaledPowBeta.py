from scipy.special import betainc, beta as betafn  #type:ignore
import numpy as np
from itertools import pairwise
from functools import cache


# see https://stats.stackexchange.com/q/602835 and my answer there
def expectationMaxScaledPowBeta(avec: list[float] | np.ndarray, bvec: list[float] | np.ndarray,
                                alpha: float, beta: float) -> float:
  crossings = [0.0] + [
      (a2 / a1)**(1 / (b1 - b2)) for ((a1, b1), (a2, b2)) in pairwise(zip(avec, bvec))
  ] + [1.0]

  betaincLoHi = lambda a, b, c1, c2: (cachedBetainc(a, b, c2) - cachedBetainc(a, b, c1)
                                     ) * cachedBeta(a, b)
  bab = cachedBeta(alpha, beta)
  correct = sum(a * betaincLoHi(alpha + b, beta, lo, hi)
                for (a, b, (lo, hi)) in zip(avec, bvec, pairwise(crossings))) / bab
  return correct


@cache
def cachedBetainc(*args, **kwargs):
  return betainc(*args, **kwargs)


@cache
def cachedBeta(*args, **kwargs):
  return betafn(*args, **kwargs)
