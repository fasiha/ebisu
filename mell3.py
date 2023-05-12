import mpmath as mp
import sympy as s
from sympy import S


def makeIntegrand2(n, k, l, t):
  return lambda x: x**n * (x / l)**(k - 1) * s.exp(-t * x - (x / l)**k) * k / l


def makeIntegrand3(n, k, l, t):
  return lambda x: x**n * (x / l)**(k - 1) * s.exp(-t / x - (x / l)**k) * k / l


k, l, x, t, n, v = s.symbols('k l x t n nu', real=True, positive=True)
f = s.exp(-t / x) * x**(k - 1) * s.exp(-l * (x)**k)

# n = 2

extraSub = {n: 2, l: 1, t: 10}

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

  meijerList = [(x - (p / q) - n) / p for x in range(1, p)] + [-1 / q - n / p
                                                              ] + [x / q for x in range(int(q))]
  recon = 1 / l**(p / q) * (t / p)**(p / q + n) * s.sqrt(p / q) / s.sqrt(
      2 * s.pi)**(p + q - 2) * s.meijerg(((), ()), (meijerList, ()), 1 / p**p / q**q / l**p * t**p)
  s.pprint(recon)

  # qres2 = mp.quad(
  #     makeIntegrand2(n=extraSub[n], k=float(pp / q), l=extraSub[l], t=extraSub[t]), [0, mp.inf])
  # print([qres2, res2.subs(extraSub).evalf()])
  qres3 = mp.quad(
      makeIntegrand3(n=extraSub[n], k=float(pp / q), l=extraSub[l], t=extraSub[t]), [0, mp.inf])
  print([qres3, res3.subs(extraSub).evalf(), recon.subs(extraSub).evalf()])
