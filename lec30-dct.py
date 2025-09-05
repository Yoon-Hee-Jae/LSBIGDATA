from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna()
penguins_a=penguins.loc[penguins["species"] == "Adelie",]


x = penguins_a[["bill_length_mm"]]
y = penguins_a["bill_depth_mm"]

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.6, color="blue", edgecolor="k")
plt.grid(True)
plt.show()

from sklearn.tree import DecisionTreeRegressor

dct = DecisionTreeRegressor()
dct.get_params()

import numpy as np
dct_params = {'max_depth' : np.arange(1, 7),
              'ccp_alpha': np.linspace(0, 1, 10)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                          param_grid=dct_params, 
                          cv = cv, 
                          scoring='neg_mean_squared_error')
dct_search.fit(x, y)
# dct.fit(x, y)
dct_search.best_params_
pred_y=dct_search.predict(x)

plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.6, color="blue", edgecolor="k")
plt.scatter(x, pred_y, alpha=0.6, color="red", edgecolor="k")
plt.grid(True)
plt.show()

# 평균의 모델의 MSE
import numpy as np
mse_1=np.mean((y - y.mean())**2)

num=30
def cal_benefit(num):
    # 41.6을 기준으로 나눈 모델의 MSE
    idx_416 = x["bill_length_mm"] < num
    y[idx_416].mean()  # 왼쪽 평행선 높이
    y[~idx_416].mean() # 오른쪽 평행선 높이
    # 왼쪽 그룹 mse
    n1=len(y[idx_416])
    n2=len(y[~idx_416])
    l_mse=np.mean((y[idx_416] - y[idx_416].mean())**2)
    # 오른쪽 그룹 mse
    r_mse=np.mean((y[~idx_416] - y[~idx_416].mean())**2)
    # 41.6을 기준으로 나눈 모델의 최종 MSE
    mse_2=(n1*l_mse + n2*r_mse) / (n1+n2)
    # 41.6을 기준으로 나눈 모델의 얻게된 효용
    return (mse_1 - mse_2)

cal_benefit(41.6)
cal_benefit(38)
cal_benefit(45)
cal_benefit(46)



