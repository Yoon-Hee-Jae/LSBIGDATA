import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


# 베르누이 분포
X = bernoulli(p=0.3)
X.mean()
x_values = np.array([0,1])
likelihood = np.array([0.7,0.3])
exp_X = np.sum(x_values*likelihood)
var_x = np.sum((x_values-exp_X)**2*likelihood)
exp_X # 베르누이 분포는 p 값(1이 나올 확률)이 기대값과 같다
var_x # 베르누이 분포 분산은 pq 로 계산 가능하다 0.3 * 0.7

# 이항분포
from scipy.stats import binom

X = binom(n=5,p=0.3)
X.mean()
X.pmf(0)
X.pmf(1)
X.pmf(2)
X.pmf(3)
X.pmf(4)
X.pmf(5)

# 이항분포 파라미터
n = 5
p = 0.3

# x값 (가능한 성공 횟수)
x = np.arange(0, n+1)

# 확률질량함수 값
pmf = binom.pmf(x, n, p)

# 시각화
plt.figure(figsize=(8, 5))
plt.bar(x, pmf, color='skyblue', edgecolor='black')
plt.title(f'Binomial PMF (n={n}, p={p})', fontsize=14)
plt.xlabel('Number of Successes (x)')
plt.ylabel('Probability P(X = x)')
plt.xticks(x)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 이항분포 계산법
from scipy.special import comb
comb(5,3)
P(X=3)
comb(5,3) * (0.3**3) * (0.7**2)

# 앞면이 나올 확률이 0.6인 동전을
# 10번 던져서 앞면이 5번 나올 확률은?
comb(10,5) * (0.6**5) * (0.4**5)
X.mean() # 이항분포 기대값 = np
X.var() # 이항분포 분산 = np(1-p)

Y = bernoulli(p=0.3)
sum(Y.rvs(size=5)) # B(5,0.3)

import math

n = 5
result = math.factorial(11)
result / (24*24*2)

10*9*8*7