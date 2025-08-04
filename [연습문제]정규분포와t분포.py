import pandas as pd
import numpy as np
from scipy.stats import norm

# 1번
from scipy.stats import t
t(9).cdf(1.812)

# 2번
t(14).ppf(0.95)

# 3번
1-t(11).cdf(2.18)

# 4번
X = norm(loc=12,scale=1.5)
1-X.cdf(10)

# 5번
X = norm(loc=8,scale=0.8)
1 - X.cdf(9)

# 6-1
x = np.array([84.3, 85.7, 83.9, 86.1, 84.5, 85.2, 85.8, 86.3, 84.7, 85.5])
xbar_mean = x.mean()
xbar_var = X.var()
xbar_std = X.std()
Z = norm(loc=0,scale=1)
xbar_mean - (Z.ppf(0.975) * (xbar_std/np.sqrt(10))) # 하한
xbar_mean + (Z.ppf(0.975) * (xbar_std/np.sqrt(10))) # 상한
