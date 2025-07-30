import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

#지수분포
# SCALE = 지수분포의 평균, 기댓값
# X1: θ = 0.5, X2: θ = 3
X1 = expon(scale=0.5)
X2 = expon(scale=3)

# rvs 샘플 추출
x1 = X1.rvs(size=100)
x1
x2 = X2.rvs(size=100)
x2

sum(x1 <= 2)
sum(x2 <= 2)
# X1이 스케일이 0.5이기 때문에 2이하의 값이 더 많이 잡힘

# x축 범위: -1 ~ 10
x = np.linspace(-1, 10, 500)

# 각각의 PDF 계산
pdf1 = X1.pdf(x)
pdf2 = X2.pdf(x)

# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, pdf1, label='Exp(θ=0.5)', color='blue')
plt.plot(x, pdf2, label='Exp(θ=3)', color='orange')

# 그래프 꾸미기
plt.title('지수분포 확률밀도함수 (PDF)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# p(2<=x<=6)
a = X1.cdf(6)-X1.cdf(2)
b = X2.cdf(6)-X2.cdf(2)
b-a

# 평균 구하는 방법
X1.mean()
X2.mean()
# 분포도 확인을 위한 분산
X1.var()
X2.var()
# ppf() 특정 부분의 넓이를 알 때 x값을 구해주는 함수
X1.ppf(0.2)




# 정규분포
# (LOC=0,SCALE=1) 확률변수를 만들어보세요
# 평균
from scipy.stats import norm
X1 = norm(loc=0,scale=1)
x = np.linspace(-4, 4, 500)
X1.mean()
# 분산
X1.var()
# PDF 그리기
pdf = X1.pdf(x)
# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, pdf, color='blue', label='PDF of N(0,1)')
plt.title('표준정규분포 PDF (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.show()



# 정규분포 loc=2,scale=3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 정규분포 객체 (평균=0, 표준편차=1)
X2 = norm(loc=2, scale=3)

# 1. 300개 표본 생성
samples = X2.rvs(size=300)

# 2. 상위 10% 경계값 계산
threshold = X2.ppf(0.9) 

# 3. 시각화
plt.figure(figsize=(10, 5))

# (1) 히스토그램
plt.hist(samples, bins=30, alpha=0.6, density=True, color='skyblue', edgecolor='gray', label='Histogram (samples)')

# (2) 이론 정규분포 PDF 곡선
x = np.linspace(-3, 7, 500)
pdf = X2.pdf(x)
plt.plot(x, pdf, 'r-', lw=2, label='PDF of N(0,1)')

# (3) 상위 10% 경계선 및 음영 처리
plt.axvline(threshold, color='orange', linestyle='--', linewidth=2,
            label=f'상위 10% 경계 (x ≈ {threshold:.2f})')
x_fill = np.linspace(threshold, 7, 300)
plt.fill_between(x_fill, X2.pdf(x_fill), color='orange', alpha=0.3, label='상위 10% 영역')

# 4. 타이틀 및 라벨
plt.title('정규분포 히스토그램과 상위 10% 영역 표시')
plt.xlabel('x')
plt.ylabel('밀도')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# 5. 상위 10% 확률 계산 (이론값)
prob_upper_10 = 1 - X2.cdf(threshold)
print(f"상위 10% 확률값: {prob_upper_10:.2f}")  # 약 0.10

# 6. 표본 중 상위 10% 개수 (실제)
count_upper_10 = np.sum(samples >= threshold)
print(f"300개 중 상위 10% 이상 표본 개수: {count_upper_10}")


# 표준 정규뷴포
X1 = norm(loc=0,scale=1)
x = np.linspace(-4, 4, 500)
X1.mean()
# 분산
X1.var()
# PDF 그리기
pdf = X1.pdf(x)
# 시각화
plt.figure(figsize=(8, 4))
plt.plot(x, pdf, color='blue', label='PDF of N(0,1)')
plt.title('표준정규분포 PDF (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.show()

# X~N(0,1)에서-1에서 1사이 값이 나올 확률
X1.cdf(1) - X1.cdf(-1)

# X~N(2,3^2)에서 -1~5가 나올 확률
X2.cdf(5) - X1.cdf(-1)
# 두 값이 같은 이유는 평균과 분산은 다르지만
# 평균 + 시그마, 평균 - 시그마(표준편차) 이기 때문에 값이 동일하다.
# 마찬가지로 2표준편차일 경우
X1.cdf(2) - X1.cdf(-2)
X2.cdf(8) - X2.cdf(-4)





