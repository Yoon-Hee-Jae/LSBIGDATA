
# 문제 1번
0.6 0.4
# 문제 2번
0.8
0.8*0.2
# 문제 3번
(0.2 * 0.8**2)*(3*2*1)/2

# 문제 4번
from scipy.stats import binom
X = binom(n=7,p=0.5)
1 - X.cdf(4)

# 문제 5번
X = binom(n=6,p=0.3)
X.mean() # np
X.var() # npq

# 문제 6번
from scipy.stats import poisson
poisson.pmf(3,2)

# 문제 7번
1-poisson.cdf(2,4)

# 문제 8번
poisson(mu=5).mean()
poisson(mu=5).var() # 포아송분포는 람다와 기대값 그리고 분산이 모두 같다

# 문제 9번
X = binom(n=10,p=0.6)
binom.pmf(4,10,0.6)

# 문제 10번
poisson.cdf(2,3.5)

# 문제 11번
from scipy.stats import bernoulli
X = bernoulli(0.7)

import matplotlib.pyplot as plt

x = [0, 1]
pmf = X.pmf(x)

plt.bar(x, pmf, color='skyblue', edgecolor='black')
plt.xticks([0, 1])
plt.title('Bernoulli PMF (p=0.7)')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 문제 12번
0 1 2 3 4
X = binom(n=4,p=0.5)
a = X.pmf(0)
b = X.pmf(1)
c = X.pmf(2)
d = X.pmf(3)
e = X.pmf(4)

# 문제 13번
poisson.cdf(3,1.5) - poisson.cdf(1,1.5)

# 문제 14번
X = binom(n=5,p=0.2)
X.pmf(0) + X.pmf(1)

# 문제 15번
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from collections import Counter

# 방문 횟수 데이터
visits = [0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2]

# 평균 확인
mu = np.mean(visits)  # 약 1.85
print("데이터 평균 λ:", mu)

# 방문 횟수 세기
count = Counter(visits)
x_vals = sorted(count.keys())
obs_freq = [count[x] / len(visits) for x in x_vals]  # 상대도수

# 포아송 이론 확률
poisson_pmf = poisson.pmf(x_vals, mu)

# 시각화
plt.figure(figsize=(8, 5))
bar_width = 0.35

# 실제 데이터
plt.bar(np.array(x_vals) - bar_width/2, obs_freq, width=bar_width, label='Observed', color='skyblue', edgecolor='black')

# 포아송 분포
plt.bar(np.array(x_vals) + bar_width/2, poisson_pmf, width=bar_width, label='Poisson(λ=1.85)', color='orange', edgecolor='black')

# 그래프 설정
plt.title('Observed Data vs Poisson Distribution (λ=1.85)')
plt.xlabel('Number of Visits')
plt.ylabel('Relative Frequency / Probability')
plt.xticks(x_vals)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 하루동안 5번 이상 확률은?
1-X.cdf(4)







import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# 다양한 λ 값
lambdas = [1, 3, 5]
x = np.arange(0, 15)

plt.figure(figsize=(10, 6))

for lam in lambdas:
    pmf = poisson.pmf(x, mu=lam)
    plt.plot(x, pmf, marker='o', label=f'λ = {lam}')

plt.title('Poisson Distribution (Different λ)', fontsize=14)
plt.xlabel('Number of Events (x)')
plt.ylabel('P(X = x)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


