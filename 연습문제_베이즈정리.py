import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# 문제 1 – 회사 직원의 건강 문제와 흡연 확률
# 한 회사에서 무작위로 선택된 직원이 건강 문제가 있을 확률은 0.25입니다. 건강 문제가 있는 직원은 건강 문제가 없는 직원보다 흡연자일 확률이 두 배 높습니다.
# 직원이 흡연자라는 사실을 알았을 때, 그가 건강 문제를 가지고 있을 확률을 계산하십시오.
prior = np.array([0.25,0.75])
likelihood = np.array([2,1])
p_break = np.sum(prior*likelihood)
p_break

posterior = (prior*likelihood) / p_break
posterior

# 문제 2
prior = np.array([0.16,0.18,0.20]) / 0.54 # 0.54에서 3가지 중 하나이니까
likelihood = np.array([0.05,0.02,0.03])
p_break = np.sum(prior*likelihood)
posterior = (prior*likelihood) / p_break
posterior
# 정답
ans = round(posterior[0],2)
print(ans)

# 문제 3









# 동전 두개 던질 때 확률분포표 시각화

x_array = np.array([0,1,2])
likelihood = np.array([0.25,0.5,0.25])

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(x_array, likelihood, width=0.4, color='skyblue', edgecolor='black')

# 라벨과 타이틀
plt.xlabel('x 값')
plt.ylabel('확률 (P(x))')
plt.title('확률분포표 시각화')
plt.xticks(x_array)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 확률 값 표시
for x, p in zip(x_array, likelihood):
    plt.text(x, p + 0.02, f'{p:.2f}', ha='center')

plt.tight_layout()
plt.show()

# 동전 두번 던져서 나온 앞면 수
# 이때 앞면이 나올 확률 = 0.4
x_array = np.array([0,1,2])
likelihood = np.array([0.6*0.6, 0.6*0.4 + 0.4*0.6, 0.4*0.4])

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(x_array, likelihood, width=0.4, color='skyblue', edgecolor='black')

# 라벨과 타이틀
plt.xlabel('x 값')
plt.ylabel('확률 (P(x))')
plt.title('확률분포표 시각화')
plt.xticks(x_array)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 확률 값 표시
for x, p in zip(x_array, likelihood):
    plt.text(x, p + 0.02, f'{p:.2f}', ha='center')

# GPT 코드

# 확률변수 X의 값과 각각의 확률
x_values = np.array([0, 1,2])
probabilities = np.array([0.36, 0.48,0.16])

# 무작위로 확률변수 X에서 1000개 샘플 생성
samples = np.random.choice(x_values, size=333, p=probabilities)
samples
samples.mean() # 표본평균

# 샘플에서 값별 비율 확인 (확률 근사)
unique, counts = np.unique(samples, return_counts=True)
sample_distribution = dict(zip(unique, counts / len(samples)))

print("샘플 분포:", sample_distribution)

# 확률분포의 무게중심
exp_X = np.sum(x_values*probabilities)
exp_X # 기대값


# 펭귄 데이터 적용
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins.dropna(inplace=True)
penguins['bill_length_mm'].mean()


# 예제
x_values = np.array([1,2,3,4])
probabilities = np.array([0.1,0.3,0.2,0.4])

samples = np.random.choice(x_values, size=200,p=probabilities)
samples.mean()

exp_X = np.sum(x_values*probabilities)
exp_X

# 시각화 : 히스토그렘

# 확률변수 X의 값과 각 확률
x_values = np.array([1, 2, 3, 4])
probabilities = np.array([0.1, 0.3, 0.2, 0.4])

# 표본 추출
samples = np.random.choice(x_values, size=200, p=probabilities)

# 상대도수 계산 (히스토그램용)
counts, _ = np.histogram(samples, bins=np.arange(0.5, 5.5, 1))
relative_freq = counts / counts.sum()

# 히스토그램 그리기
plt.figure(figsize=(8, 5))
plt.bar(x_values, relative_freq, width=0.6, color='skyblue', edgecolor='black', label='표본 상대도수')

# 모집단 확률을 수직선으로 표시
for x, p in zip(x_values, probabilities):
    plt.vlines(x=x, ymin=0, ymax=p, colors='red', linestyles='--', linewidth=2, label='모집단 확률' if x == 1 else "")

# 그래프 꾸미기
plt.xlabel('X의 값')
plt.ylabel('비율')
plt.title('표본 상대도수 vs 모집단 확률')
plt.xticks(x_values)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(max(relative_freq), max(probabilities)) + 0.1)

plt.show()

# 분산
# 확률변수 X의 값과 각 확률
x_values = np.array([1, 2, 3, 4])
probabilities = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(x_values*probabilities)
samples = np.random.choice(x_values,size=300,p=probabilities)
(samples-E_X)**2 # 이건 확률 변수

# 확률변수가 갖는 값과 확률은?
(x_values - E_X)**2
probabilities
# ddof =1 >> 
samples.var(ddof=1)
# ddof=1 계산방식
# np.sum((samples - samples.mean())**2) / (300-1)
# 300-1 로 나눈 이유는 n-1로 나눈 값이 더 크고 실제로 여러 실험을 진행한 결과 실제 분산과 더 가깝다

# 확률 변수 분산
np.sum((x_values - E_X)**2 * probabilities)


# n-1 이 더 정확한 이유 실제로 확인

x_values = np.array([1, 2, 3, 4])
probabilities = np.array([0.1, 0.3, 0.2, 0.4])
E_X = np.sum(x_values*probabilities)
E_X

# 이론분산
true_variance = np.sum((x_values-E_X)**2 * probabilities)

samples = np.random.choice(x_values,size=500,p=probabilities)
samples.mean()

var_500 = []
var_499 = []

for i in range(1000):
    samples = np.random.choice(x_values,size=500,p=probabilities)
    var_500.append(np.sum((samples-samples.mean())**2) / 500)
    var_499.append(np.sum((samples-samples.mean())**2) / 499)

len(var_500)
var_499

# 표본분산 (나누기 500)
plt.figure(figsize=(10, 4))
plt.hist(var_500, bins=30, color='skyblue', edgecolor='black')
plt.axvline(np.mean(var_500), color='green', linestyle='--', linewidth=2, label=f'표본분산 평균 = {np.mean(var_500):.3f}')
plt.axvline(true_variance, color='red', linestyle='-', linewidth=2, label='이론 분산 = 1.09')
plt.title('표본분산 (나누기 500)')
plt.xlabel('분산 값')
plt.ylabel('빈도수')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# 불편분산 (나누기 499)
plt.figure(figsize=(10, 4))
plt.hist(var_499, bins=30, color='salmon', edgecolor='black')
plt.axvline(np.mean(var_499), color='green', linestyle='--', linewidth=2, label=f'불편분산 평균 = {np.mean(var_499):.3f}')
plt.axvline(true_variance, color='red', linestyle='-', linewidth=2, label='이론 분산 = 1.09')
plt.title('불편분산 (나누기 499)')
plt.xlabel('분산 값')
plt.ylabel('빈도수')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()


# 균일분포 확률변수
from scipy.stats import uniform

# X~균일분포 U(2,4)
a = 2
b = 4
X = uniform(loc=a,scale=b-a)
X.mean() # (b-a) / 2
X.var() # (b-a)^2/12
X.rvs(size=100)
X.cdf(3.5) # P(X<=3.5)

X.cdf(3.2) - X.cdf(2.1)
# y축 값
X.pdf(2.5)