from itertools import combinations
from math import prod
from typing import Any


def expand(lst: list[Any]):
  """
  Expands `prod(1 - x for x in lst)` into a sum of monomials

  Returns a list which summed will yield the product of binomials above.
  """
  n = len(lst)
  coefficients = [1]
  sign = 1
  for k in range(1, n + 1):
    sign *= -1
    for combo in combinations(lst, k):
      coefficients.append(sign * prod(combo))
  return coefficients


def expandLog(lst: list[Any]):
  """
  Expand a product of binomials in terms of their log
  
  If the input is taken to mean
  ```
  prod(1 - exp(x) for x in lst)
  ```
  then the following operation on the returned list will 
  equal this to floating precision:
  ```
  coefs, signs = expandLog(lst)
  sum(sign * exp(x) for x, sign in zip(coefs, signs))
  ```
  """
  n = len(lst)
  coefficients = [0]
  signs = [1]

  sign = 1
  for k in range(1, n + 1):
    sign *= -1
    for combo in combinations(lst, k):
      coefficients.append(sum(combo))
      signs.append(sign)

  return coefficients, signs


if __name__ == '__main__':
  import sympy as sp
  for n in range(1, 5):
    l = sp.symbols(f'v:{n}')
    assert 0 == sp.simplify(sum(expand(l)) - sp.prod(1 - x for x in l))

    expected = sp.expand(sp.prod(1 - sp.exp(x) for x in l))
    coefs, signs = expandLog(l)
    actual = sum(sign * sp.exp(x) for x, sign in zip(coefs, signs))
    assert 0 == sp.simplify(actual - expected)

  import numpy as np
  relerr = lambda act, exp: np.abs((act - exp) / exp)
  EPS = np.spacing(1) * 1e5

  for n in range(1, 5):
    for trial in range(10):
      l2 = -np.random.rand(n)
      e = relerr(sum(expand(l2)), np.prod(1 - l2))
      assert e < EPS, f'{e=}, {n=}, {trial=}'

      expected = np.prod(1 - np.exp(l2))
      coefs, signs = expandLog(l2)
      actual = sum(signs * np.exp(coefs))
      e = relerr(actual, expected)
      assert e < EPS, f'{e=}, {n=}, {trial=}'
