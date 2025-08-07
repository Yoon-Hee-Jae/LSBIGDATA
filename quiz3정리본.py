# 베르누이 분포

# 이항 분포
from scipy.stats import binom
binom.pmf(1,3,0.2) # k, n , p (p: 1이 일어날 확률)
binom(6,0.3).mean() # np // 분산 = npq

# 포아송분포 (확률질량함수) > 누적분포함수 구할 때 P(2)이면 1-poisson.cdf(1, 람다) 이렇게 해줘여 함
from scipy.stats import poisson
poisson.pmf(3,2) # k, mu ( 포아송분포의 평균은 람다 분산도 람다 )
# PPF(Q,람다)

# 지수 분포 (확률밀도함수)
from scipy.stats import expon
lambda_val = 0.5 # 람다의 역수 즉 scale에는 평균이 들어감!
x=2
expon.cdf(x,scale = 1/lambda_val) 
expon(scale=1/3).mean()
expon(scale=1/3).var()
# 지수분포의 평균은 1/람다, 분산은 1/람다^2
# 어떤 콜센터에 전화가 평균 6분 간격으로 걸려온다. 전화가 3분 이내에 걸려올 확률을 구하시오.
expon.cdf(3,scale=6) # 조심!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 평균 고장 시간 (mean)
mean_time = 10
# λ는 평균의 역수
lambda_val = 1 / mean_time
scale_param = 1 / lambda_val 
# P(X > 5)
x = 5
1 - expon.cdf(5, scale = scale_param)

# 균일분포
from scipy.stats import uniform
a = 2
b = 6
# scipy는 loc=a, scale=b-a 형태로 파라미터 설정
rv = uniform(loc=a, scale=b - a)
# 균일평균 a+b/2 분산 = (b-a)**2/12

# 신뢰구간
import numpy as np
from scipy.stats import norm, t

# 예시 데이터
sample = np.array([52, 55, 53, 57, 54, 56, 58, 59, 60, 51])  # n = 10
n = len(sample)
mean = np.mean(sample)
std = np.std(sample, ddof=1)  # 표본표준편차
confidence = 0.95
alpha = 1 - confidence

# z분포
# 모표준편차가 주어졌다고 가정
sigma = 4

z_crit = norm.ppf(1 - alpha/2)  # z 값
margin_of_error = z_crit * sigma / np.sqrt(n)

ci_z = (mean - margin_of_error, mean + margin_of_error)
print("Z-분포 기반 신뢰구간:", ci_z)

t_crit = t.ppf(1 - alpha/2, df=n-1)  # 자유도 = n - 1
margin_of_error = t_crit * std / np.sqrt(n)

ci_t = (mean - margin_of_error, mean + margin_of_error)
print("T-분포 기반 신뢰구간:", ci_t)

# 단일표본t검정
from scipy.stats import ttest_1samp
t_statistic, p_value = ttest_1samp(sample, popmean=10, alternative='two-sided') # 양측 검정
print("t-statistic:", t_statistic, "p-value:", p_value)

# 독립2표본 t검정
from scipy.stats import ttest_ind
male = my_tab2[my_tab2['gender'] == 'Male']
female = my_tab2[my_tab2['gender'] == 'Female']
t_statistic, p_value = ttest_ind(male['score'], female['score'], # 단측 검정 (큰 쪽)
equal_var=True, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)
# alternative 옵션 설정 주의
#alternative 옵션을 설정 할 때, 대립가설의 형태를 정확히 살펴 넣어야 합니다.
#alternative='greater'의 의미는 대립가설이 첫번째 그룹의 평균이 두번째 그룹의 평균보다
# 높다고 설정되어 있다는 의미입니다.


# 대응표본 t검정
from scipy.stats import ttest_rel

# 단측 검정 (큰 쪽)
t_statistic, p_value = ttest_rel(after, before, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)

# ks 검정 정규분포 확인
from scipy.stats import kstest, norm
import numpy as np
sample_data = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
4 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

# 표본 평균과 표준편차로 정규분포 생성
loc = np.mean(sample_data)
scale = np.std(sample_data, ddof=1)

# 정규분포를 기준으로 K-S 검정 수행
result = kstest(sample_data, 'norm', args=(loc, scale))
print("검정통계량:", result.statistic)



