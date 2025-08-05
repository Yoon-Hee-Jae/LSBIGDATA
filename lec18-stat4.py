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

# t 분포와 표준 정규분포 비교
# t 분포는 자유도에 따라서 그래프 모양이 변함
# t 값이 작을수록 평균에 값이 몰리기보다 다른 사이드 값이 나오는 빈도가 높아짐
# 자유도는 계속 커질 수 있지만 표준 정규분포보다 높게 올라가지 않고 일치하게 된다.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# x 축 범위 설정
x = np.linspace(-5, 5, 500)

# 표준 정규분포 PDF
pdf_norm = norm.pdf(x)

# 자유도 5인 t-분포 PDF
df = 100
pdf_t = t.pdf(x, df=df)

# 시각화
plt.plot(x, pdf_norm, label='표준 정규분포', color='red')
plt.plot(x, pdf_t, label=f't-분포 (자유도={df})', linestyle='--', color='blue')

plt.title('표준 정규분포 vs t-분포 PDF')
plt.xlabel('x')
plt.ylabel('확률밀도 (PDF)')
plt.legend()
plt.grid(True)
plt.show()

# 신뢰구간
data = [4.3,4.1,5.2,4.9,5.0,4.5,4.7,4.8,5.2,4.6]
from scipy.stats import t
import numpy as np
mean = np.mean(data)
n = len(data)
se= np.std(data,ddof=1)/np.sqrt(n)
mean - t.ppf(0.975,loc=mean,scale=se,df=n-1)
t.interval(0.975,loc=mean,scale=se,df=n-1)
9.89/np.sqrt(20)

from scipy.stats import norm
X = norm(loc=17,scale=2.21)
X.cdf(14)

X = norm(loc=0,scale=1)
X.cdf(-1.357)

# 문제 1
# # 본포 = 정규분포
se = 50/np.sqrt(100)
X_bar = norm(loc=500,scale=se)
1-X_bar.cdf(510)  

# 문제 2
from scipy.stats import binom
binom.pmf(2,20,0.05)
binom.cdf(2,20,0.05)
1-binom.cdf(2,20,0.05)

(20*19)/2 * 0.05**2 * (1-0.05)**18

# 문제 3
X=norm(loc=75,scale=8)
1- X.cdf(85)
X.sf(85)

X.sf(70)-X.sf(80)

X.ppf(0.9)

# 문제 4
# 귀무가설 : 한잔의 평균 온도가 75도이다
# 대립가설 : 한잔의 평균 온도가 75도가 아니다
from scipy.stats import t
data = [72.4,74.1,73.7,76.5,75.3,74.8,75.9,73.4,74.6,75.1]
len(data)
sam_std = np.std(data,ddof=1)
sam_mean = np.mean(data)
T_STATS = (sam_mean-75)/(sam_std/np.sqrt(10))
t.cdf(T_STATS,df=9)*2

X_bar = norm(loc=53-50,scale=8/np.sqrt(40))

z_stats = (53-50)/(8/np.sqrt(40))
(1 - norm(loc=0,scale=1).cdf(abs(z_stats))) * 2
