import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train 셋
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data_for_learning = pd.DataFrame({'x': x, 'y': y})

# 산점도
plt.figure(figsize=(8, 5))
plt.scatter(data_for_learning["x"], data_for_learning["y"], 
            color="blue", alpha=0.7, label="Training data")
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(data_for_learning[["x"]], 
       data_for_learning["y"])
lr.coef_
lr.intercept_


data_for_learning["x^2"] = data_for_learning["x"]**2
data_for_learning["x^3"] = data_for_learning["x"]**3
data_for_learning.head()

tr_X = data_for_learning.drop(columns="y")
tr_y = data_for_learning["y"]
lr.fit(tr_X, tr_y)
lr.coef_
lr.intercept_

# 회귀직선식은 어떻게 되나요?
# y_hat = 0.929 - 2.03 * X1 + 0.352 * X2
# y_hat = 0.929 - 2.03 * X + 0.352 * X^2

# 회귀직선식을 시각화 해보면?
y_hat = lr.predict(tr_X)

# 산점도
plt.figure(figsize=(8, 5))
plt.scatter(data_for_learning["x"], 
            data_for_learning["y"], 
            color="blue", alpha=0.7, label="Training data")

# 두 번째 산점도 겹치기
plt.scatter(tr_X["x"], y_hat, 
            color="red", alpha=0.7, label="regression line")

plt.show()


# train 셋 나누기 -> train, valid
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)
# print(train.shape)
# print(valid.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

i=2   # i = 1에서 변동시키면서 MSE 체크 할 것
k = np.linspace(0, 1, 100)
sin_k = np.sin(2 * np.pi * k)
poly1 = PolynomialFeatures(degree=i, include_bias=True)
train_X = poly1.fit_transform(train[['x']])
model1 = LinearRegression().fit(train_X, train['y'])
model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

# 예측값 계산
train_y_pred = model1.predict(poly1.transform(train[['x']]))
valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

# MSE 계산
mse_train = mean_squared_error(train['y'], train_y_pred)
mse_valid = mean_squared_error(valid['y'], valid_y_pred)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1행 2열 서브플롯

# 왼쪽: 학습 데이터와 모델 피팅 결과
axes[0].scatter(train['x'], train['y'], color='black', label='Train Observed')
axes[0].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[0].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')
axes[0].set_ylim((-2.0, 2.0))
axes[0].legend()
axes[0].grid(True)
# 오른쪽: 검증 데이터
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')
axes[1].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')
axes[1].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # MSE 추가
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')
axes[1].set_ylim((-2.0, 2.0))
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.show()

# 집값 데이터에서 위 모델을 어떻게 적용할 것인가?
X1, X2, ... , X20, ... , X38



