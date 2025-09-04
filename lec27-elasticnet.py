import pandas as pd
import numpy as np

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/house_prediction/train.csv')
test_df = pd.read_csv('./data/house_prediction/test.csv')

# 독립 변수 선별

train_df.hist()
train_df.columns
train_df['MiscVal'].hist()
train_df['MiscVal'].value_counts()
# train_df = train_df.select_dtypes(include=['number'])

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns.drop('SalePrice')
cat_columns = train_df.select_dtypes(include=['object']).columns
test_num_columns = test_df.select_dtypes(include=['number']).columns
test_cat_columns = test_df.select_dtypes(include=['object']).columns

# 결측치 대체
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')
train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
test_df[test_cat_columns] = freq_impute.transform(test_df[test_cat_columns])
test_df[test_num_columns] = mean_impute.transform(test_df[test_num_columns])

# 범주형 변수 인코딩
from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder(handle_unknown='ignore', 
                       sparse_output=False).set_output(transform='pandas')

train_df_cat = onehot.fit_transform(train_df[cat_columns])
test_df_cat = onehot.transform(test_df[test_cat_columns])

# 수치형 변수 표준화
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler().set_output(transform='pandas')

train_df_num = std_scaler.fit_transform(train_df[num_columns])
test_df_num = std_scaler.transform(test_df[test_num_columns])

train_df_all = pd.concat([train_df_num, train_df_cat], axis = 1)
test_df_all = pd.concat([test_df_num, test_df_cat], axis = 1)

train_df_all.shape
test_df_all.shape

# 결측치 제거 (간단히 처리)
# train_df = train_df.dropna()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_all
y_train = train_df['SalePrice']
y_train = np.log1p(y_train)#log 0 은 마이너스 무한대니까  +1 을 해주는 log1p

###################################################################
# 모델 학습 1 elastic

# elasticnet 최적 파라미터 계산
from sklearn.linear_model import ElasticNet

elastic = ElasticNet()

# 고려할 파라미터 경우의 수
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

# 파라미터 확인 
ElasticNet().get_params()

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
elastic_search = GridSearchCV(estimator=elastic, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))

# best prameter
print(elastic_search.best_params_)

# 교차검증 best score 
print(-elastic_search.best_score_)

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
# test_df = test_df.select_dtypes(include=['number'])
# test_df = test_df.fillna(train_df.mean())
# 예측
y_pred_elastic = elastic_search.predict(test_df_all)

######################################################
# 모델 학습 2 knn

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()

# 파라미터 확인 
KNeighborsRegressor().get_params()

# 고려할 파라미터 경우의 수
knn_params = {'n_neighbors' : np.arange(1, 11, 1)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
knn_search = GridSearchCV(estimator=knn, 
                              param_grid=knn_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

knn_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(knn_search.cv_results_))

# best prameter
print(knn_search.best_params_)

# 교차검증 best score 
print(-knn_search.best_score_)

# 예측
y_pred_knn = knn_search.predict(test_df_all)

# 제출 데이터 평균 내서 생성
submit = pd.read_csv('./data/house_prediction/sample_submission.csv')
submit["SalePrice"] = (np.expm1(y_pred_elastic) + np.expm1(y_pred_knn)) / 2

# CSV로 저장
submit.to_csv('./data/house_prediction/elasticnet_log+knn.csv', index=False)

##################################################################
