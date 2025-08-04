import pandas as pd

# 문제 1
from scipy.stats import expon
expon.cdf(2,scale=2)

# 문제 2
1-expon.cdf(1,scale=0.5)

# 문제 3
expon(scale=1/3).mean() # 0.333333
expon(scale=1/3).var() # 0.11111111

# 문제 4
from scipy.stats import uniform
uniform(loc=2,scale=3).cdf(4)- uniform(loc=2,scale=3).cdf(3)

# 문제 5
uniform(loc=0,scale=8).mean() # 4
uniform(loc=0,scale=8).var() # 5.333333

# 문제 6
1-expon.cdf(5,scale=10)

# 문제 7
lamda = 1/6
expon.cdf(3,scale=6)

# 문제 8
lamda_8 = 1/12
1-expon.cdf(10,scale=1/lamda_8)

# 문제 9
lamda_9 = 5
expon.cdf(2,1/lamda_9)

# 문제 10
expon.cdf(3,scale=1/0.5)

# 문제 11
from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt

# 지수분포 파라미터
lam = 2
scale = 1 / lam  # scale = 0.5

# x 범위
x = np.linspace(0, 3, 500)
cdf = expon.cdf(x, scale=scale)

# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, cdf, label='CDF of Exponential(λ=2)', color='green')
plt.title('Exponential Distribution CDF (λ=2)')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend()
plt.show()

# 문제 12
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

# Uniform(2, 6) → loc=2, scale=4
dist = uniform(loc=2, scale=4)

# x축 범위 지정 (약간 넓게 그려보자)
x = np.linspace(0, 8, 1000)
pdf = dist.pdf(x)

# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, pdf, label='PDF of Uniform(2, 6)', color='blue')
plt.fill_between(x, pdf, alpha=0.2)
plt.title('Uniform Distribution PDF (a=2, b=6)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True)
plt.legend()
plt.show()

# 문제 13
X = uniform(loc=0,scale=10)
1-X.cdf(7)

# 문제 14
1-expon.cdf(6,scale=4)



