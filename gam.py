import numpy as np
from scipy.stats import gamma as gammarv
from scipy.special import gamma
from scipy.integrate import trapz
from scipy.special import kv

x = np.linspace(0, 10, 10_000)
a = 2.
b = 1 / 2. * 3
t = 13.
post1 = np.exp(-t / x) * x**(a - 1) * np.exp(-b * x)
numericIntegral = trapz(post1, x)
numericMean = trapz(x * post1, x) / numericIntegral
numericM2 = trapz(x**2 * post1, x) / numericIntegral

post0 = (1 - np.exp(-t / x)) * x**(a - 1) * np.exp(-b * x)
numericIntegral0 = trapz(post0, x)
numericMean0 = trapz(x * post0, x)
numericM20 = trapz(x**2 * post0, x)

analyticalIntegralN = lambda N=0: 2 * t**(
    (a + N) / 2) * b**(-(a + N) / 2) * kv(-(a + N), 2 * np.sqrt(t * b))
analyticalIntegral = analyticalIntegralN(0)

postMean = analyticalIntegralN(1) / analyticalIntegral
postM2 = analyticalIntegralN(2) / analyticalIntegral
postVar = postM2 - postMean**2

meanVarToGamma = lambda mean, var: (mean**2 / var, mean / var)

newA, newB = meanVarToGamma(postMean, postVar)
print([[a, b, a / b], [newA, newB, newA / newB]])

priorMean = a / b
priorVar = a / b**2
priorM2 = priorVar + priorMean**2

analyticalIntegral0 = gamma(a) / b**a - analyticalIntegral

c0 = gamma(a) * b**(-a) - analyticalIntegral
c0Numeric = trapz((1 - np.exp(-t / x)) * x**(a - 1) * np.exp(-b * x), x)
post0MeanNum = trapz(x * (1 - np.exp(-t / x)) * x**(a - 1) * np.exp(-b * x), x) / c0

post0Mean = (priorMean * gamma(a) * b**(-a) - analyticalIntegralN(1)) / c0
post0M2 = (priorM2 * gamma(a) * b**(-a) - analyticalIntegralN(2)) / c0
post0Var = post0M2 - post0Mean**2

new0A, new0B = meanVarToGamma(post0Mean, post0Var)
print([[a, b, a / b], [new0A, new0B, new0A / new0B]])

# import matplotlib.pylab as plt
# plt.ion()
# plt.figure()
# plt.plot(x, post1)
