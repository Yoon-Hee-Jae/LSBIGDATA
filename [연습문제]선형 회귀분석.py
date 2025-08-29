# 연습문제 1
import numpy as np
from scipy.stats import pearsonr
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 13])
corr_coeff, p_value = stats.pearsonr(x, y)
print(f"피어슨 상관계수 (r): {corr_coeff:.4f}")
print(f"p-value: {p_value:.4f}")

# 연습문제 2
x = np.array([1, 2, 3, 4, 10, 11, 12])
y = np.array([2, 4, 6, 8, 100, 200, -100])
# 이상치 포함
corr_coeff, p_value = stats.pearsonr(x, y)
print(f"피어슨 상관계수 (r): {corr_coeff:.4f}")
print(f"p-value: {p_value:.4f}") # 상관관계 없음
# 이상치 제외
# x
Q1 = np.percentile(x, 25)
Q3 = np.percentile(x, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
x = x[(x >= lower_bound) & (x <= upper_bound)]
# y
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
y = y[(y >= lower_bound) & (y <= upper_bound)]
x = x[:5]
corr_coeff, p_value = stats.pearsonr(x, y)
print(f"피어슨 상관계수 (r): {corr_coeff:.4f}")
print(f"p-value: {p_value:.4f}") # 상관계수 존재

# 연습문제 3
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])
# 1. 평균 계산
x_mean = np.mean(x)
y_mean = np.mean(y)

# 2. 회귀계수 계산
beta_1 = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
beta_0 = y_mean - beta_1 * x_mean
print(f"기울기 (beta_1): {beta_1}")
print(f"절편 (beta_0): {beta_0}")
print(f"회귀직선: y = {beta_0:.2f} + {beta_1:.2f}*x")

# 연습문제 4
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
# 평균과 표준편차
x_mean = np.mean(x)
y_mean = np.mean(y)
s_x = np.std(x, ddof=1)  # 표본 표준편차
s_y = np.std(y, ddof=1)

# 상관계수
r = np.corrcoef(x, y)[0,1]

# 기울기와 절편
beta_1 = r * (s_y / s_x)
beta_0 = y_mean - beta_1 * x_mean

print(f"상관계수 r: {r:.4f}")
print(f"기울기 beta_1: {beta_1:.4f}")
print(f"절편 beta_0: {beta_0:.4f}")
print(f"회귀직선: y = {beta_0:.2f} + {beta_1:.2f}*x")

# 연습문제 5
import pandas as pd
from sklearn.datasets import fetch_california_housing
cal = fetch_california_housing(as_frame=True)
df = cal.frame
model = smf.ols("MedHouseVal ~ AveRooms+AveOccup", data = df).fit()
model.summary()

# 연습문제 6
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
model.summary()

# 연습문제 7
df['IncomeLevel'] = pd.qcut(df['MedInc'], q=3, labels=['Low', 'Mid', 'High'])
model = smf.ols('MedHouseVal ~ AveRooms + AveOccup + C(IncomeLevel)', data=df).fit()
model.summary()

# 연습문제 8 
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model.resid)
dw_stat

# 연습문제 9 
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, model.model.exog)


# 연습문제 11
from sklearn.datasets import load_diabetes
# 데이터 불러오기 및 DataFrame 변환
diabetes = load_diabetes(as_frame=True)
df2 = diabetes.frame
model = smf.ols('target ~ bmi + bp + s1', data=df2).fit()
model.rsquared_adj

# 연습문제 12
model.summary()
# 650.247	911.247

# 연습문제 13
# bp

# 연습문제 14
model2 = smf.ols("target ~ bmi + bp + s1 + s2", data=df2).fit()
model2.summary() # 96 > 73 작아짐

# 연습문제 15
model3 = smf.ols("target ~ bmi + bp + s1 + s2 + s3", data=df2).fit()
model3.summary() # 모델3가 더 값이 작아서 좋은 모델

# 연습문제 16
import seaborn as sns
penguins = sns.load_dataset("penguins").dropna()
model0 = smf.ols("body_mass_g ~ bill_length_mm + flipper_length_mm", data=penguins).fit()
model.summary() # 0.397

# 연습문제 17
model0 = smf.ols("body_mass_g ~ bill_length_mm + flipper_length_mm + C(species)", data=penguins).fit()
model0.params # 친스트랩

# 연습문제 18
model0 = smf.ols("body_mass_g ~ bill_length_mm + flipper_length_mm + C(species) + sex", data=penguins).fit()
model0.summary()

# 연습문제 19

# 연습문제 20
import matplotlib.pyplot as plt
residuals = model0.resid
fitted_values = model0.fittedvalues
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)
plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt)
plt.show()

# 연습문제 21
# 예제 데이터 생성
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(n_samples)
df = pd.DataFrame(X, columns=['var1', 'var2', 'var3', 'var4', 'var5'])
df['target'] = y
# 데이터 확인
print(df.head())
x1 = df['var1']
corr_coeff1, p_value = stats.pearsonr(x1, y)
print(corr_coeff)

x2 = df['var2']
corr_coeff2, p_value = stats.pearsonr(x2, y)
print(corr_coeff2)

x3 = df['var3']
corr_coeff3, p_value = stats.pearsonr(x3, y)
print(corr_coeff3)

x4 = df['var4']
corr_coeff4, p_value = stats.pearsonr(x4, y)
print(corr_coeff4)

x5 = df['var5']
corr_coeff5, p_value = stats.pearsonr(x5, y)
print(corr_coeff5) # x1

model99 = smf.ols("y ~ x1 + x2 + x3 + x4 + x5", data=df).fit()
model99.summary() # 0.935

# 연습문제 22
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import statsmodels.api as sm
# 예제 데이터 생성
X, y = make_regression(n_samples=100, n_features=3, 
                       noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'var{i}' for i in range(3)])
df['target'] = y
# 데이터 확인
print(df.head())
model99 = smf.ols("y ~ var0 + var1 + var2", data=df).fit()
model99.summary() # var 1 이 가장 작음
# 결정계수 1
new_data = pd.DataFrame({
    "var0": [0.5],
    "var1": [1.2],
    "var2": [0.3]
})

y_pred = model99.predict(new_data)
print(y_pred)



