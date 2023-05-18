import pdb
import itertools as it
import mpmath as mp
import sympy as s
from sympy import S


def makeIntegrand2(n, k, l, t):
  return lambda x: x**n * (x / l)**(k - 1) * s.exp(-t * x - (x / l)**k) * k / l


def makeIntegrand3(n, k, l, t):
  return lambda x: x**n * (x / l)**(k - 1) * s.exp(-t / x - (x / l)**k) * k / l


k, l, x, t, n, v = s.symbols('k l x t n nu', real=True, positive=True)
f = s.exp(-t / x) * x**(k - 1) * s.exp(-l * (x)**k)

extraSub = {n: 0, l: 1, t: 10}

p, q = S(2), S(3)

for pp in [2]:
  p = S(pp)
  print(f'# p/q={p}/{q}')
  if False:
    # This is the Wikipedia/Sagias integral
    print('## E[X^n exp(-t X)]')
    f2 = s.simplify(x**n * (x / l)**(k - 1) * s.exp(-t * x - (x / l)**k) * k / l)
    res2 = s.integrate(f2.subs({k: p / q}), (x, 0, s.oo)).simplify()
    print(s.latex(f2))
    s.pprint(res2)
    meijerList2 = ([(x - (p / q) - n) / p for x in range(1, p + 1)], [x / q for x in range(int(q))])
    prefix = (p / t)**(p / q + n) * s.sqrt(p / q) / l**(p / q) / s.sqrt(2 * s.pi)**(p + q - 2)
    arg = (p / l / t)**p / q**q
    print(meijerList2)
    s.pprint(prefix)
    s.pprint(arg)
    # continue

  print('## E[X^n exp(-t / X)]')
  f3 = s.simplify(x**n * (x / l)**(k - 1) * s.exp(-t / x - (x / l)**k) * k / l)
  res3 = s.integrate(f3.subs({k: p / q}), (x, 0, s.oo)).simplify()
  s.pprint(res3)

  meijerList = [(x - (p / q) - n) / p for x in range(1, p)] + [-1 / q - n / p
                                                              ] + [x / q for x in range(int(q))]
  prefix = (t / p)**(p / q + n) * s.sqrt(p / q) / l**(p / q) / s.sqrt(2 * s.pi)**(p + q - 2)
  arg = (t / l / p)**p / q**q

  recon = prefix * s.meijerg(((), ()), (meijerList, ()), arg)
  s.pprint(recon)

  qres3 = mp.quad(
      makeIntegrand3(n=extraSub[n], k=float(pp / q), l=extraSub[l], t=extraSub[t]), [0, mp.inf])
  mpdirect = float(prefix.subs(extraSub)) * mp.meijerg(
      [[], []], [[float(x.subs(extraSub)) for x in meijerList], []], float(arg.subs(extraSub)))

  print('comparison', [qres3, res3.subs(extraSub).evalf(), recon.subs(extraSub).evalf(), mpdirect])

makeMeijerList = lambda p, q, n: [(S(x) - (p / q) - n) / p for x in range(1, p)
                                 ] + [-S(1) / q - n / p] + [S(x) / q for x in range(q)]

# verbose
checkMeijer = mp.meijerg([[], []], [[(x.subs(extraSub)) for x in meijerList], []],
                         (arg.subs(extraSub)),
                         verbose=True)

import pylab as plt
import numpy as np

plt.ion()
l = [float(x.subs(extraSub)) for x in meijerList]
z = float(arg.subs(extraSub))

zl = mp.log(z)


def f(s):
  try:
    return mp.fsum(mp.loggamma(b - s) for b in l) + s * zl
  except ValueError:
    return 20


xaxis = np.linspace(-10, 10, 1001)
yaxis = np.array([float(f(x).real) for x in xaxis])

plt.figure()
plt.plot(xaxis, yaxis)
plt.grid()

expected = mp.meijerg([[], []], [[float(x.subs(extraSub)) for x in meijerList], []],
                      float(arg.subs(extraSub)))
zsym = arg.subs(extraSub)
z = float(zsym)
ds, dz, dp, dq = s.symbols('s z p q')
f = s.prod([s.gamma(x.subs(extraSub) - ds) for x in meijerList]) * dz**ds

if pp == 1:
  # bj are evenly spaced, e.g., [-1/3, 0, 1/3, 2/3] or [-1/4, 0, 1/4, 1/2, 3/4]
  residues = []
  for i in range(15):
    # slow. works for p=1 for now
    print(i)
    pole = -S(1) / q + S(i) / q
    residues.append(s.residue(f, ds, pole))

    floatRes = [float(r.subs({dz: z})) for r in residues]
    print('expected', expected, 'actual', sum(floatRes), floatRes)

isInt = lambda x: x == int(x)
isZeroNegInt = lambda x: x <= 0 and isInt(x)
bs = [x.subs(extraSub) for x in meijerList]

resSum2 = S(0)
seenPoles = set()
for i in range(0, 6):
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
      rest = s.prod([s.gamma(nonpole - thisPole) for nonpole in nonpoleBs]) * zsym**thisPole
      res = laurentMinus1 * rest
      # print('simple pole calculated', res.evalf())
      # print('simple pole residue', s.simplify(s.residue(f, ds, thisPole)).subs({dz: zsym}).evalf())
      # res should be `s.simplify(s.residue(f, ds, thisPole))`
    else:
      # See https://math.stackexchange.com/a/4700880/
      # pdb.set_trace()
      assert len(poleBs) == 2, "unimplemented: >2 coincident poles"
      l1 = s.prod(s.gamma(b - thisPole) for b in nonpoleBs) * zsym**thisPole * s.prod(
          (-1)**(1 + b - thisPole) / s.gamma(-(b - thisPole) + 1) for b in poleBs)
      l2 = s.log(zsym) - sum(s.polygamma(0, b - thisPole) for b in nonpoleBs) - sum(
          s.polygamma(0, thisPole - b + 1) for b in poleBs)
      res = -l1 * l2
      # res should be `s.simplify(s.residue(f, ds, thisPole).subs({dz: zsym}))`
      # print('DOUBLE pole calculated', res.evalf())
      # print('DOUBLE pole residue', s.simplify(s.residue(f, ds, thisPole)).subs({dz: zsym}).evalf())
    resSum2 += float(res)
    print(f'{i=}, {resSum2=}=?{expected}, {thisPole=}, {poleBs=}')
    seenPoles.add(thisPole)
print('FINAL meijerG residue sum:', resSum2, expected)
