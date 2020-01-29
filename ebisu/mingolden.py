from math import sqrt, isnan
PHI_RATIO = 2 / (1 + sqrt(5))


def mingolden(f, xL, xU, tol=1e-8, maxIterations=100):
  iteration = 1
  x1 = xU - PHI_RATIO * (xU - xL)
  x2 = xL + PHI_RATIO * (xU - xL)
  f1 = f(x1)
  f2 = f(x2)
  f10 = f(xL)
  f20 = f(xU)
  xL0 = xL
  xU0 = xU
  while (iteration < maxIterations and abs(xU - xL) > tol):
    if (f2 > f1):
      xU = x2
      x2 = x1
      f2 = f1
      x1 = xU - PHI_RATIO * (xU - xL)
      f1 = f(x1)
    else:
      xL = x1
      x1 = x2
      f1 = f2
      x2 = xL + PHI_RATIO * (xU - xL)
      f2 = f(x2)
  iteration += 1

  xF = 0.5 * (xU + xL)
  fF = 0.5 * (f1 + f2)

  if (f10 < fF):
    argmin = xL0
  elif (f20 < fF):
    argmin = xU0
  else:
    argmin = xF

  return dict(
      iterations=iteration,
      argmin=argmin,
      minimum=fF,
      converged=not (isnan(f2) or isnan(f1) or iteration == maxIterations))
