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


xaxis = np.linspace(-10, 10, 10001)
yaxis = np.array([float(f(x).real) for x in xaxis])

plt.figure()
plt.plot(xaxis, yaxis)
plt.grid()

expected = mp.meijerg([[], []], [[float(x.subs(extraSub)) for x in meijerList], []],
                      float(arg.subs(extraSub)))
z = float(arg.subs(extraSub))
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

residues = []
resSum = S(0)
seenPoles = set()
for i in range(0, -5, -1):
  for b in [x.subs(extraSub) for x in meijerList]:
    pole = S(b) - S(i)
    if pole in seenPoles:
      continue
    seenPoles.add(pole)
    res = s.residue(f, ds, pole).subs({dz: z})
    resSum += res.evalf()
    residues.append(res)
    print(f'{expected=}')
    print(dict(b=b, pole=pole, res=res.evalf(), actual=resSum))

# fsimple = s.gamma(-ds) * s.gamma(-ds - 1)
