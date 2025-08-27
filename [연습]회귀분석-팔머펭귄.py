import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)

np.random.seed(2022)
train_index=np.random.choice(penguins.shape[0],200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.reset_index(drop=True,inplace=True)
train_data.info()
# 1번 펭귄 데이터의 bill_length_mm를 종속 변수로 두고
# 부리 깊이를 독립변수로 설정하여 회귀직선을 구하시오.
# 그리고 산점도를 그린 후, 회귀직선을 시각화해주세요
model = smf.ols("bill_length_mm	 ~ bill_depth_mm", data = train_data).fit()
print(model.summary())
intercept, slope = model.params

plt.figure(figsize=(8,6))
sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", hue="species", data=train_data)

# 회귀 직선 그리기
x_vals = np.linspace(train_data["bill_depth_mm"].min(), train_data["bill_depth_mm"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="회귀직선")

plt.xlabel("부리깊이")
plt.ylabel("부리길이")
plt.title("단순회귀: 부리길이 ~ 부리깊이")
plt.legend()
plt.show()

# 2번 독립변수와 종속변수의 관계를 직선 계수를 사용해서 해석해보세요
model.params
# 부리깊이가 1단위만큼 커질수록 부리길이는 -0.7만큼 줄어든다

# 3번 계수 유의성을 통해 해석 가능성을 이야기해보세요
model.pvalues
model.tvalues
# 부리깊이 계수의 t검정통계량 값은 약 -3.6이고 p-vale값은 0.05이하이다.
# 따라서 부리깊이가 부리길이에 영향을 주지 않는다는 귀무가설을 기각한다.
# t검정은 단일표본t검정을 사용한다.
# 그 이유는 해당 계수가 0일 경우 영향이 없는 것이기에
# 해당 계수가 0인지 아닌지를 체크하는 단일표본 t검정을 사용하고, 따라서 양측검정이다.
# p-value(해당 t 값보다 절대값이 더 큰 확률) * 2

# 넘파이를 이용해서, 직선과 주어진 점들의 수직거리를 한 변으로 하는 사각형들의 넓이 합을 계산하세요

x = train_data["bill_depth_mm"].values
y = train_data["bill_length_mm"].values

# 각 점의 수직거리 기반 넓이 합
square_sum = np.sum((y - (intercept + slope * x))**2)

# 시각화
plt.figure(figsize=(8,6))
sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", hue="species", data=train_data)

# 회귀 직선 그리기
x_vals = np.linspace(train_data["bill_depth_mm"].min(), train_data["bill_depth_mm"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="회귀직선")

# square_sum 값을 그래프에 텍스트로 표시
plt.text(
    x=train_data["bill_depth_mm"].min() + 0.5,   # x 위치
    y=train_data["bill_length_mm"].max() - 2,   # y 위치
    s=f"사각형 넓이 합 = {square_sum:.2f}",      # 텍스트 내용
    fontsize=12,
    color="blue",
    bbox=dict(facecolor="white", alpha=0.7)     # 박스 배경
)

plt.xlabel("부리깊이")
plt.ylabel("부리길이")
plt.title("단순회귀: 부리길이 ~ 부리깊이")
plt.legend()
plt.show()

X = np.array([0,1])
Y = np.array([2,4,6])
X_likelihood = np.array([0.5,0.5])
Y_likelihood = np.array([0.3,0.3,0.4])
E_X = np.sum(X * X_likelihood)
E_Y = np.sum(Y * Y_likelihood)
VAR_X = np.sum(X**2*X_likelihood)-E_X**2
VAR_Y = np.sum(Y**2*Y_likelihood)-E_Y**2

x = np.array([0,0,0,1,1,1])
y = np.array([2,4,6,2,4,6])
(x-E_X)*(y-E_Y)
px = np.array([0.2,0,0.3,0.1,0.3,0.1])
COV_XY = np.sum((x-0.5)*(y-4.2)*px) # 공분산

p_XY = COV_XY/(np.sqrt(VAR_X)*np.sqrt(VAR_Y))

# 연습 2 상관계수 구하기
X = np.array([0,1])
Y = np.array([2,4,6])
X_likelihood = np.array([0.5,0.5])
Y_likelihood = np.array([0.3,0.3,0.4])
E_X = np.sum(X * X_likelihood)
E_Y = np.sum(Y * Y_likelihood)
VAR_X = np.sum(X**2*X_likelihood)-E_X**2
VAR_Y = np.sum(Y**2*Y_likelihood)-E_Y**2

x = np.array([0,0,0,1,1,1])
y = np.array([2,4,6,2,4,6])
px = np.array([0.15,0.15,0.2,0.15,0.15,0.2])
COV_XY = np.sum((x-E_X)*(y-E_Y)*px)
p_XY = COV_XY/(np.sqrt(VAR_X)*np.sqrt(VAR_Y))

# 샘플 뽑기
N = 10000
idx = np.random.choice(len(x),size=N,p=px)
X = np.array([x[i] for i in idx])
Y = np.array([y[i] for i in idx])
upper = np.sum((X - X.mean())*(Y - Y.mean())) 
lower_left = np.sqrt(np.sum((X - X.mean())**2))
lower_right = np.sqrt(np.sum((Y - Y.mean())**2))
upper / (lower_left*lower_right)

import scipy.stats as stats
corr_coeff, p_value = stats.pearsonr(X, Y)
print(f"피어슨 상관계수 (r): {corr_coeff:.4f}")


# 상관계수 
x = np.array([2,5,3,7])
noise = np.random.normal(loc=0,scale=2,size=4)
y = 2*x + 3 +noise

plt.scatter(x,y,color='black',label='points')

x_vals = np.linspace(x.min()-1, x.max()+1, 100)
y_vals = 3 + 2 * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="y=2x+3")
plt.grid(True)
plt.legend()
plt.show()

#
x = np.array([10,20,30,40,50])
y = np.array([5,15,25,35,48])
x = x.reshape(-1,1)
X = np.hstack([np.ones((x.shape[0],1)),x])

beta = np.array([2.0,1.0]).reshape(-1,1)

def ssr(beta_vec):
    return (y - X @ beta_vec).transpose() @ (y-X @ beta_vec)


ssr(beta)

from scipy.optimize import minimize

def ssr(beta):
    beta = beta.reshape(-1,1)
    r = y - X @ beta
    return float((r.T @ r))

res = minimize(ssr,x0=np.zeros(2))
print("최적해 beta =", res.x)
print("최소 ssr =", res.fun)

a = np.linalg.inv(X.transpose()@X)
