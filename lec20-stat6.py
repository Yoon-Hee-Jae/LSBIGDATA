import numpy as np
data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
sorted_data = np.sort(data) # 데이터 정렬
minimum = np.min(sorted_data) # 최소값과 최대값
maximum = np.max(sorted_data)
median = np.median(sorted_data) # 중앙값
lower_half = sorted_data[sorted_data < median] # 중앙값보다 크거나, 작은 데이터들 필터
upper_half = sorted_data[sorted_data > median]
q1 = np.median(lower_half) # 1사분위수와 3사분위수
q3 = np.median(upper_half)
print("최소값:", minimum, "제 1사분위수:", q1, "중앙값:", median, "제 3사분위수:", q3, "최대값:", maximum)


from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna().reset_index()
penguins['species'].unique()
# 팔머펭귄 각 종별 부리길이의 사분위수를 계산하세요
adelie = penguins[penguins['species']=='Adelie']
chinstrap = penguins[penguins['species']=='Chinstrap']
gentoo = penguins[penguins['species']=='Gentoo']

adelie.sort_values('bill_length_mm',inplace=True)
chinstrap.sort_values('bill_length_mm',inplace=True)
gentoo.sort_values('bill_length_mm',inplace=True)

def QR_result(x):
    median_1 = x['bill_length_mm'].median()
    lower_half = x[x['bill_length_mm']<median_1]['bill_length_mm'].median()
    upper_half = x[x['bill_length_mm']>median_1]['bill_length_mm'].median()
    print('중앙값 =',np.round(median_1,2), '1QR =',lower_half, '2QR =',upper_half)

QR_result(adelie)
QR_result(chinstrap)
QR_result(gentoo)

# 코드로 쉽게 사분위수 구하기 값은 살짝 다를 수 있음
np.quantile(adelie['bill_length_mm'],0.25)
np.quantile(adelie['bill_length_mm'],0.5)
np.quantile(adelie['bill_length_mm'],0.75)

# 이상치 판별 방법
# 1QR에서 -1.5*IQR
# 3QR에서 +1.5*IQR 
# 해당 구간에서 벗어나는 데이터는 이상치로 판단
import numpy as np
import matplotlib.pyplot as plt

# 데이터 정의
data = np.array([155, 126, 82, 115, 140, 73, 92, 110, 134,5])

# 박스플롯 그리기
plt.figure(figsize=(6, 4))
plt.boxplot(data, vert=True, patch_artist=True)  # patch_artist=True는 색칠된 박스를 그림
plt.title('Boxplot of Data')
plt.ylabel('Value')
plt.grid(True)
plt.show()
# 상자 윗부분 = 3Q
# 상자 아랫부분 = 1Q
# 상자 안 가로선 = 2Q
# 점으로 찍힌 것은 이상치
# 상자 밖 가로선 = 이상치가 아닌 데이터 최대,최소값

import numpy as np
import matplotlib.pyplot as plt

# 재현 가능성 확보
np.random.seed(42)

# ▶ 1. 정규분포를 따르는 3개 그룹
group1 = np.random.normal(loc=50, scale=10, size=100)
group2 = np.random.normal(loc=60, scale=15, size=100)
group3 = np.random.normal(loc=55, scale=5, size=100)

# ▶ 2. 이상치 포함 데이터
group_with_outliers = np.concatenate([
    np.random.normal(100, 10, 95),
    np.array([200, 210, 215, 220, 300])  # 이상치
])

# ▶ 3. 다양한 분포 비교 (균등, 정규, 지수)
uniform_data = np.random.uniform(20, 80, 100)
normal_data = np.random.normal(50, 10, 100)
exponential_data = np.random.exponential(30, 100)

# ─────────────────────────────────────
# 박스플롯 3개 그리기
plt.figure(figsize=(15, 4))

# 첫 번째 그래프: 정규분포 그룹 3개
plt.subplot(1, 3, 1)
plt.boxplot([group1, group2, group3], patch_artist=True, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Normal Distribution Groups')
plt.ylabel('Value')
plt.grid(True)

# 두 번째 그래프: 이상치 포함
plt.subplot(1, 3, 2)
plt.boxplot(group_with_outliers, patch_artist=True)
plt.title('Data with Outliers')
plt.grid(True)

# 세 번째 그래프: 다양한 분포 비교
plt.subplot(1, 3, 3)
plt.boxplot([uniform_data, normal_data, exponential_data], patch_artist=True, labels=['Uniform', 'Normal', 'Exponential'])
plt.title('Different Distributions')
plt.grid(True)

plt.tight_layout()
plt.show()

scores = np.array([88,92,95,91,87,89,94,90,92,100,43])
sco_2Q = np.median(scores)
sco_higher = scores[scores>sco_2Q]
sco_lower = scores[scores<sco_2Q]
sco_1Q = np.median(sco_lower)
sco_3Q = np.median(sco_higher)
sco_IQR = sco_3Q-sco_1Q
print('중앙값 =',np.round(sco_2Q,2), '1QR =',sco_1Q, '3QR =',sco_3Q, 'IQR =', sco_IQR)

# 박스플롯 그리기
plt.figure(figsize=(6, 4))
plt.boxplot(scores, patch_artist=True)

# 계산된 사분위수 선 추가
plt.axhline(sco_1Q, color='blue', linestyle='--', label=f'Q1 = {sco_1Q}')
plt.axhline(sco_2Q, color='green', linestyle='--', label=f'Q2 (Median) = {sco_2Q}')
plt.axhline(sco_3Q, color='red', linestyle='--', label=f'Q3 = {sco_3Q}')

# 그래프 설정
plt.title('Boxplot with Custom Quartiles')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()

out_low = scores[scores < sco_1Q - 1.5*sco_IQR]
out_high = scores[scores > sco_3Q + 1.5*sco_IQR]
print(out_low)
print(out_high)

filtered_scores = scores[(scores >= out_low)]
##
data = np.array([155, 126, 27, 82, 115, 140, 73, 92, 110, 134])
sorted_data = np.sort(data)
n = len(data)

np.quantile(data,[0.25,0.5,0.75])
# 넘파이르 0.01~0.99 R까지
np.arange(0.01,1,0.01)
np.quantile(data,np.arange(0.01,1,0.01)) # 백분위수
np.percentile(data,[25,50,75])
from scipy.stats import norm
data.mean()
data.std(ddof=1)
X = norm(loc=data.mean(), scale=data.std(ddof=1))
X.ppf(0.5)
X.ppf(0.25)
# 실제로 정규분포를 따른다면 4분위수와 정규분포의 ppf로 구한 4분위수와 비슷해야함

norm_q = X.ppf(np.arange(0.01,1,0.01))
norm_q
data_q = np.quantile(data,np.arange(0.01,1,0.01)) # 백분위수

plt.figure(figsize=(6, 6))
plt.scatter(data_q, norm_q)
plt.plot([min(norm_q), max(norm_q)], [min(norm_q), max(norm_q)], 'r--', label='y = x')  # 기준선

plt.title('Q-Q Plot: Normal vs Empirical Quantiles')
plt.xlabel('Theoretical Quantiles (Normal Distribution)')
plt.ylabel('Empirical Quantiles (Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# qq플랏 쉽게 그리기
import scipy.stats as sp
sp.probplot(data,dist='norm',plot=plt)
plt.show()





