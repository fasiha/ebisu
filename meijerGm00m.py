import sympy as s
from sympy import S

isInt = lambda x: x == int(x)
isZeroNegInt = lambda x: x <= 0 and isInt(x)


def meijerGm00m(bs, z, numSeries=20, verbose=False):
  """
  An approach to implementing the MeijerG^{m,0}_{0,m} function.

  This Meijer G function defines a contour integral with m Γ functions (no
  denominator). Based on https://math.stackexchange.com/a/4700880/81266

  `bs` are the coefficients of the Meijer G function. A list of numbers (ideally
  Sympy symbolic numbers rather than floats?)

  `z` is the scalar argument of the Meijer G function.

  `numSeries` is how many terms of the infinite series to use. TODO: we can use
  relative tolerances to exit sooner too.

  `verbose` will compare this approach's estimates of the residues with Sympy's
  symbolic residues. Use this with `numSeries < 5` because this can get really
  slow.
  """
  resSum2 = S(0)
  seenPoles = set()

  if verbose:
    ds = s.symbols('s')
    f = s.prod([s.gamma(x - ds) for x in meijerList]) * z**ds

  for i in range(numSeries):
    for b in bs:
      thisPole = b + i  # we know that f(thisPole) -> infinity
      if thisPole in seenPoles:
        continue

      # both these lists are arguments to Gamma
      poleBs = []  # these args to f to blow up
      nonpoleBs = []  # these dont
      for b2 in bs:
        other = b2 - thisPole
        if isZeroNegInt(other):
          poleBs.append(b2)
        else:
          nonpoleBs.append(b2)
      if len(poleBs) == 1:
        # simple pole!
        negN = -(poleBs[0] - thisPole)
        laurentMinus1 = (-1)**negN / s.gamma(negN + 1)
        rest = s.prod([s.gamma(nonpole - thisPole) for nonpole in nonpoleBs]) * z**thisPole
        res = laurentMinus1 * rest
        if verbose:
          print('simple pole calculated', res.evalf())
          print('simple pole residue', s.simplify(s.residue(f, ds, thisPole)).evalf())
        # res should be `s.simplify(s.residue(f, ds, thisPole))`
      else:
        # See https://math.stackexchange.com/a/4700880/
        l1 = s.prod(s.gamma(b - thisPole) for b in nonpoleBs) * z**thisPole * s.prod(
            (-1)**(1 + b - thisPole) / s.gamma(-(b - thisPole) + 1) for b in poleBs)
        l2 = s.log(z) - sum(s.polygamma(0, b - thisPole) for b in nonpoleBs) - sum(
            s.polygamma(0, thisPole - b + 1) for b in poleBs)
        res = -l1 * l2
        # res should be `s.simplify(s.residue(f, ds, thisPole).subs({dz: zsym}))`
        if verbose:
          print(f'MULTI={len(poleBs)} pole calculated', res.evalf())
          print(f'MULTI pole residue', s.simplify(s.residue(f, ds, thisPole)).evalf())
      resSum2 += float(res.evalf())
      # print(f'{i=}, {resSum2=}=?{expected}, {thisPole=}, {poleBs=}')
      seenPoles.add(thisPole)
  # print('FINAL meijerG residue sum:', resSum2, expected)
  return resSum2


if __name__ == "__main__":
  import mpmath as mp
  z = S(1) / 80

  meijerList = [S(i) / 3 for i in range(15)]
  expected = mp.meijerg([[], []], [meijerList, []], z)
  actual = meijerGm00m(meijerList, z)
  print(actual, expected, 'error', (actual - expected) / actual)

  n = 1
  p, q = S(1), S(3)
  meijerList = [(x - (p / q) - n) / p for x in range(1, p)] + [-1 / q - n / p
                                                              ] + [x / q for x in range(int(q))]

  expected = mp.meijerg([[], []], [meijerList, []], z)
  actual = meijerGm00m(meijerList, z)
  print(actual, expected, 'error', (actual - expected) / actual)

  # meijerList = [S(i) for i in [1, 2, 3, 4]]
  # expected = mp.meijerg([[], []], [meijerList, []], z)
  # actual = meijerGm00m(meijerList, z, 20)
  # print(actual, expected, 'error', (actual - expected) / actual)
