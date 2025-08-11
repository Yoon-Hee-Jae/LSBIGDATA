from scipy.stats import poisson
import pandas as pd
from scipy.stats import expon
from scipy.stats import binom
from scipy.stats import norm
import numpy as np
from scipy.stats import uniform
df = pd.read_csv('./data/datasetSalaries.csv')
df

# 1번
8*7*6
# 2번
(12*11*10*9*8)/ (2*3*4*5)
12*66
# 3번
1-(1/12 + 1/6 + 1/2)
# 4번
0.5*0.8 + 0.3*0.5 + 0.2*0.9
# 5번
ex = 4
ey = 2
varx = 5
vary = 3
3*ex - 2*ey + 7 # 15
9*varx + 4*vary # 57
# 6번
(0.01*0.95) / ((0.01*0.95)+(0.99*0.05))
0.0095/(0.0095+0.0495)
# 10번
0.75**2 - 0.25**2
# 11번
5/16
1.5**2/4 - 1/4
# 13번
4.9 - 2.1**2
0.3*3
# 14번
1 - 0.5**4
15/16
# 15번 하나의 상자가
error_1 = 1 - binom.cdf(1,20,0.02)
good = binom.cdf(1,20,0.02)
error_1 * good * good * 3
# 16번
a = 0
b = 4
# scipy는 loc=a, scale=b-a 형태로 파라미터 설정
uniform(loc=a, scale=b - a).cdf(3)-uniform(loc=a, scale=b - a).cdf(1)
uniform(loc=a, scale=b - a).var()
# 18번
norm.cdf(4.95,loc=5,scale=0.05)
# 19번
norm.cdf(6,loc=8,scale=2)
# 20번
norm.ppf(0.9,loc=32,scale=6)

# 22번
a = 0.85 - 0.14*(2/3) - 0.64*(7/8)
0.22* x = a
x = a/0.22
x
# 23번
data = np.array([21, 12, 24, 18, 25, 28, 22, 22, 29, 14, 20, 45, 16, 18, 15, 17, 23, 55, 19, 26])
sorted_data = np.sort(data) 
median = np.median(sorted_data)
lower_half = sorted_data[sorted_data < median]
upper_half = sorted_data[sorted_data > median]
q1 = np.median(lower_half)
q3 = np.median(upper_half)
iqr = q3 - q1
q1-1.5*iqr # 5.5
q3 + 1.5*iqr # 37.5
sorted_data

# 24번
norm.ppf(0.75,loc=3,scale=2) - norm.ppf(0.25,loc=3,scale=2)

# 25번
df.info()
df['salary'].mean()
df
from scipy.stats import ttest_1samp
t_statistic, p_value = ttest_1samp(df['salary'], popmean=50221, alternative='two-sided') # 양측 검정
print("t-statistic:", t_statistic, "p-value:", p_value)

# 28
import scipy.stats as sp
data_a = np.array([2011, 2005, 1998, 2003, 2008, 2001, 2006])
data_b = np.array([1985, 1991, 1988, 1992, 1986, 1990, 1987])
data_c = np.array([2020, 2024, 2019, 2026, 2023, 2025, 2022])
w, p_value = sp.shapiro(data_a)
print("W:", w, "p-value:", p_value) # 정규분포
w, p_value = sp.shapiro(data_b)
print("W:", w, "p-value:", p_value) # 정규분포
w, p_value = sp.shapiro(data_c)
print("W:", w, "p-value:", p_value) # 정규분포

# 29
drug_a = [142.9, 140.6, 144.7, 144.0, 142.4, 146.0, 149.1, 150.4]
drug_b = [139.1, 136.4, 147.3, 139.4, 143.0, 142.2, 142.2, 147.9]
# 대응표본 t검정
from scipy.stats import ttest_rel
# 단측 검정 (큰 쪽)
t_statistic, p_value = ttest_rel(drug_b, drug_a, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value) # 귀무가설 기각

# 30
data = [1.2, 0.9, 1.5, 2.1, 0.7, 0.8, 1.8, 2.2, 1.0, 1.3, 2.5, 2.0, 1.1, 1.6, 0.6]
lambda_x = 1/np.mean(data)
from scipy.stats import anderson
result = anderson(data, dist='expon')
print('검정통계량',result[0], '임계값:',result[1], '유의수준:',result[2])