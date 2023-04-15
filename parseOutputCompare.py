import numpy as np
import pylab as plt
import re
import json

plt.ion()

r = re.compile(r"loglik.*=(\[.*\]),")

with open("output-entropy-pow14-exact-quadratic-ent.txt", "r") as fid:
  lines = fid.readlines()

loglik = np.array([json.loads(r.match(l).groups()[0]) for l in lines if 'loglik' in l])

import pandas as pd

df = pd.DataFrame(loglik)
arr = df.sort_values(by=6).values

fig, ax = plt.subplots(2, 1)
ax[0].plot(arr)
ax[1].plot(arr - arr[:, -1][:, np.newaxis])
ax[0].legend('wmax,boost,wmaxB,wmaxG,maxmax,ent/p=14/n=4,poly/p=14/n=14'.split(','))
