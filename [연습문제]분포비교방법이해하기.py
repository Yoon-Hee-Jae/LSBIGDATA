import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 문제 1번
filename = "problem5_32.csv"
dat = pd.read_csv(filename)
dat.info()

male = dat.loc[dat['Gender']=='Male']
female = dat.loc[dat['Gender']=='Female']
male.info()
female.info()

# QQ 플롯 그리기
import scipy.stats as stats
# male
stats.probplot(male['Salary'], dist="norm", plot=plt)
plt.title('QQ Plot of male Salary')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()
# female
stats.probplot(female['Salary'], dist="norm", plot=plt)
plt.title('QQ Plot of female Salary')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

# 정규성 검정 > shapiro wiki
import scipy.stats as sp
w, p_value = sp.shapiro(male['Salary'])
print("W:", w, "p-value:", p_value)
# p-value가 0.92 이므로 정규분포이다 라는 귀무가설을 기각할 수 없다
w, p_value = sp.shapiro(female['Salary'])
print("W:", w, "p-value:", p_value)
# 여자도 귀무가설 기각 불가

# 문제 2번
filename = "heart_disease.csv"
dat = pd.read_csv(filename)
dat
health = dat.loc[dat['target']=='yes',:][['target','chol']]
weak = dat.loc[dat['target']=='no'][['target','chol']]
health.info()
weak.info() # 결측치 존재
weak.dropna(inplace=True)
weak.info()
# QQ 플랏
# 심장질환 O
stats.probplot(weak['chol'], dist="norm", plot=plt)
plt.title('QQ Plot of weak')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()
# 심장질환 x
stats.probplot(health['chol'], dist="norm", plot=plt)
plt.title('QQ Plot of health')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

# 정규성 검정 > shapiro wiki
import scipy.stats as sp
w, p_value = sp.shapiro(weak['chol'])
print("W:", w, "p-value:", p_value)
# p-value가 0.39 이므로 정규분포이다 라는 귀무가설을 기각할 수 없다
w, p_value = sp.shapiro(health['chol'])
print("W:", w, "p-value:", p_value)
# p-value가 작으므로 귀무가설을 기각하고 건강한 사람의 콜레스트롤 분포는 정규분포를 따르지 않음

# 문제 3
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()

BMI = dat.loc[:,['Outcome','BMI']]
BMI.info()
sweet = BMI[BMI['Outcome']==1]
nope = BMI[BMI['Outcome']==0]
# QQ 플랏
# 당뇨병 O
stats.probplot(sweet['BMI'], dist="norm", plot=plt)
plt.title('QQ Plot of sweet')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()
# 당뇨병 x
stats.probplot(nope['BMI'], dist="norm", plot=plt)
plt.title('QQ Plot of nope')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True)
plt.show()

# 정규성 검정 > shapiro wiki
import scipy.stats as sp
w, p_value = sp.shapiro(sweet['BMI'])
print("W:", w, "p-value:", p_value)
# p-value가 0.05보다 작으므로 정규분포이다 라는 귀무가설을 기각하고 즉 정규분포를 따르지않음
w, p_value = sp.shapiro(nope['BMI'])
print("W:", w, "p-value:", p_value)
# 마찬가지로 귀무가설 기각 정규분포를 따르지 않음

# 문제 4번
# 데이터 불러오기
filename = "problem5_44.csv"
df = pd.read_csv('./data/' + filename)
data = df.iloc[:, 0].values
data
from scipy.stats import expon
from statsmodels.distributions.empirical_distribution import ECDF
data_mean = np.mean(data)
data_lambda = 1/data_mean
# ecdf
data_ecdf = ECDF(data)
# 이론적 cdf 계산
# 시각화를 위한 x 값 범위
x = np.linspace(min(data), max(data), 100)
cdf_theoretical = expon.cdf(x, scale=1/data_lambda)

# 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, cdf_theoretical, label='Theoretical CDF (Exponential)', color='red')
plt.step(data_ecdf.x, data_ecdf.y, where='post', label='Empirical CDF (ECDF)', color='blue')
plt.xlabel('x')
plt.ylabel('CDF')
plt.title('Empirical CDF vs Theoretical Exponential CDF')
plt.legend()
plt.grid(True)
plt.show()

# 앤더슨 달링 검정
from scipy.stats import anderson
# 앤더슨-달링 검정 (지수분포)
result = anderson(data, dist='expon')
# 결과 출력
print("A-D 검정 통계량:", result.statistic)
print("임계값들:", result.critical_values)
print("유의수준들:", result.significance_level)
# 검종 통계량보다 임계값이 작으므로 유의수준 0.05%에서 귀무가설을 기각할 수 있다 즉 지수분포를 따르지 않는다




