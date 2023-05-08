import sympy as s
from sympy import S


def flatten(l):
  return [item for sublist in l for item in sublist]


def rotate(l, n):
  return l[n:] + l[:n]


def relerr(actual, expected):
  return abs(actual - expected) / expected


k, l, x, t, n, v = s.symbols('k l x t n nu', real=True, positive=True)
f = s.exp(-t / x) * x**(k - 1) * s.exp(-l * (x)**k)

f2 = x**(n - 1) * s.exp(-x - l * x**k)
s.pprint(s.integrate(f.subs({k: S(1) / S(3)}), (x, 0, s.oo)).simplify())

den = s.sympify(3)
z = 100

for num in range(1, den):
  ffin = f.subs({k: num / den})
  res = s.integrate(ffin, (x, 0, s.oo)).simplify()
  s.pprint(s.simplify(ffin))
  s.pprint(res)
  nums = s.sympify(num)
  meijerList = [n / den for n in range(int(den))] + [n / nums for n in range(1, 1 + num)]
  print('Meiger G terms', meijerList)

  zHyper = (-1)**len(meijerList) / z
  expansion = []
  for rot in range(len(meijerList)):
    hyperArg = [-meijerList[0] + i + 1 for i in meijerList[1:]]
    if 0 in hyperArg:
      print('! SKIPPING ', hyperArg)
      continue
    print(f'∝ 0F{len(meijerList)-1}(; {hyperArg})')
    terms = [[
        n * s.log(zHyper),
        -s.loggamma(n + 1),
        *[-(s.loggamma(b + n) - s.loggamma(b)) if b else 1 for b in hyperArg],
    ] for n in range(40)]
    summedTerms = [sum(term).evalf() for term in terms]
    actual = sum(map(s.exp, summedTerms)).evalf()
    expected = (float(s.hyper([], hyperArg, zHyper)))
    print([dict(actual=actual, expected=expected, err=relerr(actual, expected))])
    meijerList = rotate(meijerList, 1)

    leading = meijerList[0]
    newExpansion = (
        expected / s.prod(map(s.gamma, hyperArg)) * s.pi**len(hyperArg) * z**(leading - 1) *
        s.prod(s.csc(s.pi * (x - 1)) for x in hyperArg))
    expansion.append(newExpansion)

  actual = sum(expansion).evalf()
  expected = s.meijerg((tuple(meijerList), ()), ((), ()), z)
  print('actual vs expected', actual, expected.evalf())
  break
# res2 = s.integrate(f.subs({k: s.sympify(7) / 10}), (x, 0, s.oo)).simplify()
# s.pprint(res2)

Pi = s.pi
Csc = s.csc
Gamma = s.gamma
HypergeometricPFQ = s.hyper


def foo(a, b, c, z):
  return [
      (Pi**2 * z**(-1 + b) * Csc((-a + b) * Pi) * Csc(
          (b - c) * Pi) * HypergeometricPFQ([], [1 + a - b, 1 - b + c], -z**(-1))) /
      (Gamma(1 + a - b) * Gamma(1 - b + c)),
      (Pi**2 * z**(-1 + a) * Csc(a * Pi - b * Pi) * Csc(a * Pi - c * Pi) *
       HypergeometricPFQ([], [1 - a + b, 1 - a + c], -z**(-1))) /
      (Gamma(1 - a + b) * Gamma(1 - a + c)),
      (Pi**2 * z**(-1 + c) * Csc((a - c) * Pi) * Csc(
          (b - c) * Pi) * HypergeometricPFQ([], [1 + a - c, 1 + b - c], -z**(-1))) /
      (Gamma(1 + a - c) * Gamma(1 + b - c)),
  ]


"""
HypergeometricPFQ[{}, {1 + a - b, 1 - b + c, 1 - b + d, 1 - b + e}, -z^(-1)])
HypergeometricPFQ[{}, {1 - a + b, 1 - a + c, 1 - a + d, 1 - a + e}, -z^(-1)])
HypergeometricPFQ[{}, {1 + a - c, 1 + b - c, 1 - c + d, 1 - c + e}, -z^(-1)])
HypergeometricPFQ[{}, {1 + a - d, 1 + b - d, 1 + c - d, 1 - d + e}, -z^(-1)])
HypergeometricPFQ[{}, {1 + a - e, 1 + b - e, 1 + c - e, 1 + d - e}, -z^(-1)])
"""
"""
def integral(x, y, a, g1, g2=1):
  "
  Returns ∫_0^∞ t^(a-1) exp(-y t^(g1)) exp(-x / t^(g2)) dt
  See https://math.stackexchange.com/a/4482122
  "
  iters = 20
  logx = np.log(x)
  logy = np.log(y)
  from math import log
  logSums = []
  weights = []
  for n in range(iters):
    weights.append((-1)**n)
    weights.append((-1)**n)

    logSums.append(([
        -gammaln(n + 1),
        log(g1 / g2),
        gammaln((n * g1 + a) / -g2),
        (n * g1 + a) / g2 * (log(x) + g2 / g1 * log(y)),
    ]))
    logSums.append(([
        -gammaln(n + 1),
        (a - n * g2) / g1,
        gammaln((a - n * g2) / g1),
        n * (log(x) + g2 / g1 * log(y)),
    ]))
  pprint(logSums)
  # return logsumexp(sums, b=weights)


# def testIntegral():
import mpmath as mp


def integrand(t, x, y, a, g1):
  return t**(a - 1) * mp.exp(-y * t**g1 - x / t)


expected = mp.quad(lambda t: integrand(t, a=k, y=1 / l**k, g1=k, x=10), (0, mp.inf))
actual = integral(a=k, y=1 / l**k, g1=k, x=10)
"""

import mpmath as mp

kVal, lVal = mp.mpf(1) / int(den), 1
tVal = 10
qres = mp.quad(lambda x: mp.exp(-tVal / x) * (x / lVal)**(kVal - 1) * mp.exp(-(x / lVal)**kVal),
               [0, mp.inf])
qres2 = res.subs({t: tVal, l: lVal}).evalf()
print([qres, qres2])
mp.meijerg