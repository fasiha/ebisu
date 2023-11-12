from typing import Optional
import numpy as np
from math import fsum, exp, log
from itertools import repeat


def helper(
    av: list[float] | np.ndarray,
    bv: Optional[list[float] | np.ndarray] = None,
) -> tuple[float, float]:
  amax = max(av)
  s = fsum(exp(a - amax) * b for a, b in zip(av, bv if bv is not None else repeat(1)))
  assert s >= 0, 'positive'
  return s, amax


def logsumexp(
    av: list[float] | np.ndarray,
    bv: Optional[list[float] | np.ndarray] = None,
) -> float:
  s, amax = helper(av, bv)
  return log(s) + amax


def sumexp(
    av: list[float] | np.ndarray,
    bv: Optional[list[float] | np.ndarray] = None,
) -> float:
  s, amax = helper(av, bv)
  return s * exp(amax)
