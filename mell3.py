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

n = 0

p, q = S(2), S(4)

for pp in range(1, q):
  p = S(pp)
  print(f'# p/q={p}/{q}')
  print('## E[X^n exp(-t X)]')
  f2 = s.simplify(x**n * (x / l)**(k - 1) * s.exp(-t * x - (x / l)**k) * k / l)
  res2 = s.integrate(f2.subs({k: p / q}), (x, 0, s.oo)).simplify()
  s.pprint(res2)

  print('## E[X^n exp(-t / X)]')
  f3 = s.simplify(x**n * (x / l)**(k - 1) * s.exp(-t / x - (x / l)**k) * k / l)
  res3 = s.integrate(f3.subs({k: p / q}), (x, 0, s.oo)).simplify()
  s.pprint(res3)

  meijerList = [(n - k).subs({k: p / q}) / p - (0 if n < p else 1) for n in range(1, 1 + p)
               ] + [n / q for n in range(int(q))]
  print(meijerList)

  qres2 = mp.quad(makeIntegrand2(0, float(pp / q), 1.0, 10), [0, mp.inf])
  qres3 = mp.quad(makeIntegrand3(0, float(pp / q), 1.0, 10), [0, mp.inf])
  print([qres2, res2.subs(extraSub).evalf()])
  print([qres3, res3.subs(extraSub).evalf()])

  # expected = s.meijerg(
  #     ((), ()),
  #     ((S(11) / 20, S(3) / 10, S(1) / 20, -S(1) / 5, 0, S(1) / 5, S(2) / 5, S(3) / 5, S(4) / 5),
  #      ()),
  #     S(1) / 80).evalf()
  # actual = s.meijerg(((), ()), (tuple(meijerList), ()), S(1) / 80).evalf()
  # s.pprint([expected, actual])
