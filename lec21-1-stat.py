import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# 비교할 자유도 리스트
dfs = [2, 5, 10]

# x 범위 설정
x = np.linspace(0, 25, 500)

plt.figure(figsize=(8, 5))

# 각 자유도별 PDF 계산 및 그래프 그리기
for df in dfs:
    pdf = chi2.pdf(x, df)
    plt.plot(x, pdf, label=f'df={df}')

plt.title('Chi-square Distribution PDF Comparison')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import chi2
from scipy.stats import norm

X = chi2(df=3)
1-X.cdf(8)

Y = norm(loc=3,scale=2)
data_set = Y.rvs(7500).reshape(500,-1)
data_set.shape
s_2 = data_set.std(ddof=1,axis=1)
s_2.shape
statistics = s_2 * (15-1) / 2**2


# 히스토그램
plt.figure(figsize=(8,5))
plt.hist(statistics, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Statistics')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

X = np.array([10.67, 9.92, 9.62, 9.53, 9.14, 9.74, 8.45, 12.65, 11.47, 8.62])

1-chi2.cdf((10-1) *X.var(ddof=1) / 1.3,df=9)

# 비흡 = 14 /28
# 흡 = 14/28
# 운 = 18/28
# 일 = 10/28