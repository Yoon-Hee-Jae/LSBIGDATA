import pandas as pd
from scipy.stats import poisson

1-poisson.cdf(14,10)

from scipy.stats import expon
expon.cdf(5,scale=1/0.5) - expon.cdf(2,scale=1/0.5)

1-poisson.cdf(14,10)

from scipy.stats import binom
binom.pmf(2,20,0.05)
binom.cdf(2,20,0.05)
from scipy.stats import norm
1-norm.cdf(1.25)

0.2+2+2.7-2.1**2
0.49*4
0.3+1.6+1.8-1.7**2
16*0.3
7-2.4**2

p_break = 0.08*0.06 + 0.15*0.03 + 0.49*0.02 + 0.28*0.04
likelihood = 0.08*0.06
likelihood/p_break

5*0.1 + 10*0.15 + 15*0.2 + 20*0.3 + 25*0.15 + 30*0.05 + 35*0.05
5**2*0.1 + 10**2*0.15 + 15**2*0.2 + 20**2*0.3 + 25**2*0.15 + 30**2*0.05 + 35**2*0.05
382.5-18**2
import numpy as np
np.sqrt(58.5)

# 3
from scipy.stats import binom
binom.pmf(1,3,0.2) # np.float64(0.38400000000000006)
# 4
1-binom.cdf(4,7,0.5) # np.float64(0.2265625)
# 5
binom(6,0.3).mean() # np.float64(1.7999999999999998)
binom(6,0.3).var() # np.float64(1.2599999999999998)
6 * 0.3
6 * 0.3 * 0.7
# 6
from scipy.stats import poisson
poisson.pmf(3,2) # np.float64(0.18044704431548356)
# 7
1-poisson.cdf(1,4) # np.float64(0.9084218055563291)
# 8
poisson(5).mean()
poisson(5).var()
# 9
binom.pmf(4,10,0.6) # np.float64(0.11147673600000009)
# 10
poisson.cdf(2,3.5) # np.float64(0.32084719886213414)
# 13
poisson.cdf(3,1.5)-poisson.cdf(0,1.5) # np.float64(0.7112273854731201)
# 15
visits = [0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2]


#2
1-expon.cdf(1,scale=1/2)
#3
expon(scale=1/3).mean()
expon(scale=1/3).var()
#4
from scipy.stats import uniform
uniform(loc=2,scale=5-2).cdf(4) - uniform(loc=2,scale=5-2).cdf(3)