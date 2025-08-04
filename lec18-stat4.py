from scipy.stats import norm

# 평균(mu)=0, 표준편차(sigma)=1인 정규분포에서 10개 샘플 생성
X = norm(loc=5, scale=3)
sample = X.rvs(size=10000000)
sample.mean()
X.var()

import math
6.22/math.sqrt(20)
9.96/math.sqrt(20) 
X = norm(loc=14,scale=2.227)
X.ppf(0.025)
X.ppf(0.975)

import numpy as np
sample = np.array([14,17,12,14,13,14,16,10,14,15,13,17,12,12,16])
sam_mean = sample.mean()
all_sig = 3
sam_sig = 3/math.sqrt(15)
X = norm(loc=sam_mean, scale=sam_sig)
X.ppf(0.05) # 12
X.ppf(0.95) # 15
X.cdf(15.20)

# 1. Xi ~ 모분포는 균일분포 (3,7)
# 2. i=1,2,...,20
# 3. Xi들에서 표본을 하나씩 뽑은 후 표본 평균을 계산하고,
#  95% 신뢰구간을 계산 ( 모분산값 1번 정보로 사용 )
# 3번의 과정을 1000번 실행해서 ( 1000번의 신뢰구간 발생 ) 각 신뢰구간이 모평균을 포함하고 있는지 체크
from scipy.stats import uniform
X_var = (7-3)**2/12
X = uniform(loc=3, scale=4)
X.std()
ans = []
for i in range(1000):
    samples = X.rvs(20)
    sam_mean = samples.mean()
    sam_std = np.sqrt(X_var)/np.sqrt(20)
    sam_X = norm(loc=sam_mean,scale=sam_std)
    sam_1 = sam_X.ppf(0.025)
    sam_2 = sam_X.ppf(0.975)
    if (sam_1 <= 5)&(sam_2>=5)==True:
        ans.append(True)
    else:
        ans.append(False)
len(ans)
np.sum(ans)

# 시각화
# 이론적인 표본 평균의 분포의 pdf를 그리고, 모평균을 빨간색 막대기로 표현
# 3번에서 뽑은 표본들을 x축에 녹색 점들로 표시하고
# 95%의 신뢰구간을 녹색 막대기 2개로 표현
# 표본이 바뀔 때마다 녹색 막대기안에 빨간 막대기가 있는지 확인
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
import matplotlib.font_manager as fm

# 한글 폰트 설정 (예: Windows에서는 'Malgun Gothic', macOS는 'AppleGothic')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지


# 모분포 정의
X = uniform(loc=3, scale=4)
X_var = (7 - 3) ** 2 / 12
mu = 5  # 모평균
n = 20
sigma = np.sqrt(X_var)
se = sigma / np.sqrt(n)  # 표본평균의 표준오차

# 표본추출 및 신뢰구간 계산 (1회만)
samples = X.rvs(n)
sample_mean = samples.mean()

# 신뢰구간 계산
z = norm.ppf(0.975)
ci_lower = sample_mean - z * se
ci_upper = sample_mean + z * se

# x축 범위 및 표본평균 분포 정의
x = np.linspace(3.5, 6.5, 1000)
pdf = norm(loc=mu, scale=se).pdf(x)

# 시각화
plt.figure(figsize=(10, 6))

# 1. 이론적인 표본평균 분포의 PDF
plt.plot(x, pdf, label='표본평균의 분포', color='blue')

# 2. 모평균 (빨간색 세로선)
plt.axvline(x=mu, color='red', linestyle='--', label='모평균 (5)')

# 3. 표본 평균 (녹색 점)
plt.plot(sample_mean, 0, 'go', markersize=8, label='표본평균')

# 4. 신뢰구간 양 끝 (초록 세로 막대기)
plt.vlines([ci_lower, ci_upper], ymin=0, ymax=norm(loc=mu, scale=se).pdf(sample_mean), 
           color='green', linestyle='-', label='95% 신뢰구간')

plt.title('표본평균 분포 및 신뢰구간 시각화')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.show()

#########################
# 모분포 정의
X = uniform(loc=3, scale=4)
X_var = (7 - 3) ** 2 / 12
mu = 5  # 모평균
n = 20
sigma = np.sqrt(X_var)
se = sigma / np.sqrt(n)  # 표본평균의 표준오차

# 표본추출 및 신뢰구간 계산 (1회만)
samples = X.rvs(n)
sample_mean = samples.mean()

# 신뢰구간 계산
Z = norm(loc=sample_mean,scale=se)
ci_lower = Z.ppf(0.025)
ci_upper = Z.ppf(0.975)

# x축 범위 및 표본평균 분포 정의
x = np.linspace(3.5, 6.5, 1000)
pdf = norm(loc=mu, scale=se).pdf(x)

# 시각화
plt.figure(figsize=(10, 6))

# 1. 이론적인 표본평균 분포의 PDF
plt.plot(x, pdf, label='표본평균의 분포', color='blue')

# 2. 모평균 (빨간색 세로선)
plt.axvline(x=mu, color='red', linestyle='--', label='모평균 (5)')

# 3. 표본 평균 (녹색 점)
plt.plot(sample_mean, 0, 'go', markersize=8, label='표본평균')

# 4. 신뢰구간 양 끝 (초록 세로 막대기)
plt.vlines([ci_lower, ci_upper], ymin=0, ymax=norm(loc=mu, scale=se).pdf(sample_mean), 
           color='green', linestyle='-', label='95% 신뢰구간')

plt.title('표본평균 분포 및 신뢰구간 시각화')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend()
plt.grid(True)
plt.show()


x = np.array([14,17,12,14,13,14,16,10,14,15,13,17,12,12,16])
n = len(X)
sd = 3 / np.sqrt(n)

X = norm(loc=x.mean(),scale=sd)
z_05 = norm.ppf(0.05,loc=0,scale=1)
x.mean() + z_05 * 3/np.sqrt(n)
x.mean() - z_05 * 3/np.sqrt(n)