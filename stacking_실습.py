import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/house_prediction/train.csv')
test_df = pd.read_csv('./data/house_prediction/test.csv')
# 이상치 제거와 같은 EDA가 제일 중요
# 이상치 확인할 때는 X축에는 독립변수 Y축에는 종속변수를 설정

train_df.shape
train_df.info()
# train_df = train_df.select_dtypes(include=['number'])

num_columns = train_df.select_dtypes(include=['number']).columns
cat_columns = train_df.select_dtypes(include=['object']).columns

# 상관계수 계산 (Pearson 기본)
corr_matrix = train_df[num_columns].corr()
corr_matrix = np.abs(corr_matrix['SalePrice'])
corr_matrix.sort_values('')
# 수치형변수 히스토그렘
# 히스토그렘 확인
for col in num_columns:
    sns.histplot(train_df[col], bins=5, kde=True)  # kde=True 하면 밀도 곡선도 함께 표시
    plt.title(f"{col} Histogram")
    plt.show()


# 결측치 채우기 (간단히 처리)
# train_df = train_df.dropna()
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
# freq_impute.statistics_
# mean_impute.statistics_

# 다중공선성 확인
X_const = add_constant(train_df[num_columns])

# 3️⃣ VIF 계산
vif = pd.DataFrame()
vif["변수"] = X_const.columns
vif["VIF"] = [variance_inflation_factor(X_const.values, i) 
              for i in range(X_const.shape[1])]

print(vif)


# 이상치 확인



from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])
# train_df[num_columns].mean(axis=0)
# train_df[num_columns].std(axis=0, ddof=1)
# train_df_num.mean(axis=0)
# std_scaler.mean_

train_df_all = pd.concat([train_df_cat,
                          train_df_num], axis = 1)

# 독립변수(X)와 종속변수(y) 분리
# np.log(y_train).hist()
X_train = train_df_all
y_train = np.log1p(train_df['SalePrice'])

from sklearn.linear_model import ElasticNet
import numpy as np

elastic = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}
# 파라미터 확인 
ElasticNet().get_params()

# 의사결정나무 모델 
from sklearn.tree import DecisionTreeRegressor
dct = DecisionTreeRegressor()

dct_params = {'max_depth' : np.arange(1, 7),
              'ccp_alpha': np.linspace(0, 1, 10)}

# knn 회귀모델
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()
# knn.get_params()

knn_params = {"n_neighbors": np.arange(1, 6)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, 
           shuffle=True, 
           random_state=2025)

# 그리드서치 - ElasticNet
elastic_search = GridSearchCV(estimator=elastic, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')
elastic_search.fit(X_train, y_train)

# 그리드서치 - DCT
dct_search = GridSearchCV(estimator=dct, 
                          param_grid=dct_params, 
                          cv = cv, 
                          scoring='neg_mean_squared_error')
dct_search.fit(X_train, y_train)


# 그리드서치 - KNN 회귀
knn_search = GridSearchCV(estimator=knn, 
                          param_grid=knn_params, 
                          cv = cv, 
                          scoring='neg_mean_squared_error')
knn_search.fit(X_train, y_train)

# pd.DataFrame(elastic_search.cv_results_)

# best prameter
print(elastic_search.best_params_)
print(dct_search.best_params_)
print(knn_search.best_params_)

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])

test_df_cat = onehot.transform(test_df[cat_columns])
test_df_num = std_scaler.transform(test_df[num_columns])

test_df_all = pd.concat([test_df_cat,
                         test_df_num], axis = 1)

# 예측
# y_pred = elastic.predict(test_df)
y_pred1 = elastic_search.predict(test_df_all)
y_pred2 = dct_search.predict(test_df_all)
y_pred3 = knn_search.predict(test_df_all)

# 블렌딩: 단순하게 각 모델 출력값을 합침 (평균)
y_pred = (y_pred1 + y_pred2 + y_pred3) / 3

# 스택킹: 새로운 모델(메타모델)을 사용해서 합침(선형회귀)
from sklearn.linear_model import LinearRegression

lr=LinearRegression(fit_intercept=False)
y_pred_el = elastic_search.predict(train_df_all)
y_pred_dct = dct_search.predict(train_df_all)
y_pred_knn = knn_search.predict(train_df_all)
lr.get_params()
X_meta = pd.DataFrame({
            "elastic" : y_pred_el,
            "dct" : y_pred_dct,
            "knn" : y_pred_knn
        })

lr.fit(X_meta,y_train)
lr.coef_
lr.intercept_

X_meta_test = pd.DataFrame({
            "elastic" : y_pred1,
            "dct" : y_pred2,
            "knn" : y_pred3
        })
y_pred = lr.predict(X_meta_test)



submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=np.expm1(y_pred)

# CSV로 저장
submit.to_csv('./data/houseprice/elasticnet_grid.csv', index=False)
