import pandas as pd
import numpy as np
#pd.set_option('display.float_format','{:.4f}'.format)

train_df = pd.read_csv('./data/car_prediction/train.csv')
train_df.head()
test_df = pd.read_csv('./data/car_prediction/test.csv')
train_df["engine"].head(50)
# engine

# 문자열을 공백 기준으로 분리
df_split = train_df["engine"].str.split(" ", expand=True)
train_df['engine']
# 열 매핑
# Horsepower 추출 (숫자+HP)
train_df["Horsepower"] = train_df["engine"].str.extract(r'(\d+\.?\d*)HP').astype(float)

# 배기량(L) 추출
train_df["Displacement(L)"] = train_df["engine"].str.extract(r'(\d+\.?\d*)L').astype(float)

# "4 Cylinder" 또는 "V6" 형태 둘 다 잡기
cylinders = train_df["engine"].str.extract(r'(\d+)\sCylinder|V(\d+)')
train_df["Cylinders"] = cylinders.bfill(axis=1).iloc[:, 0].astype(float)

# 연료 종류 추출
train_df["Fuel_Type"] = train_df["engine"].str.extract(r'(Gasoline|Diesel|Electric|Hybrid|LPG)')

# 엔진 타입(예: "Cylinder Engine")
train_df["Engine_Type"] = train_df["engine"].str.extract(r'(\d+\sCylinder Engine)')

# test 데이터
# 열 매핑
# Horsepower 추출 (숫자+HP)
test_df["Horsepower"] = test_df["engine"].str.extract(r'(\d+\.?\d*)HP').astype(float)

# 배기량(L) 추출
test_df["Displacement(L)"] = test_df["engine"].str.extract(r'(\d+\.?\d*)L').astype(float)

# "4 Cylinder" 또는 "V6" 형태 둘 다 잡기
cylinders = test_df["engine"].str.extract(r'(\d+)\sCylinder|V(\d+)')
test_df["Cylinders"] = cylinders.bfill(axis=1).iloc[:, 0].astype(float)

# 연료 종류 추출
test_df["Fuel_Type"] = test_df["engine"].str.extract(r'(Gasoline|Diesel|Electric|Hybrid|LPG)')

# 엔진 타입(예: "Cylinder Engine")
test_df["Engine_Type"] = test_df["engine"].str.extract(r'(\d+\sCylinder Engine)')

# accident 열 추가
## accident칼럼
mapping = {
    'None reported': 0,
    'At least 1 accident or damage reported': 1
}

train_df['accident_numeric'] = train_df['accident'].map(mapping)
test_df['accident_numeric'] = train_df['accident'].map(mapping)
# 예시 데이터프레임 생성 (사용자의 train_df와 유사하게)
data = {
    'accident': ['None reported', 'At least 1 accident or damage reported', 'None reported', np.nan, 'None reported', 'At least 1 accident or damage reported']
}

# 칼럼 선택
num_columns = train_df.select_dtypes(include=['number']).columns.drop('price')
cat_columns = train_df.select_dtypes(include=['object']).columns

# 결측치 대체 범주형
from sklearn.impute import SimpleImputer
freq_impute = SimpleImputer(strategy='most_frequent')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
test_df[cat_columns] = freq_impute.transform(test_df[cat_columns])

# 결측치 대체 범주형
mean_impute = SimpleImputer(strategy='mean')
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])
test_df[num_columns] = mean_impute.transform(test_df[num_columns])

train_df.info()
test_df.info()

# 범주형 변수 인코딩

import category_encoders as ce

# OrdinalEncoder 생성 (범주형 컬럼만)
encoder = ce.OrdinalEncoder(cols=cat_columns)

# train: fit + transform
train_df[cat_columns] = encoder.fit_transform(train_df[cat_columns])

# test: train 기준으로 transform
test_df[cat_columns] = encoder.transform(test_df[cat_columns])

# 수치형 변수 표준화
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler().set_output(transform='pandas')

train_df[num_columns] = std_scaler.fit_transform(train_df[num_columns])
test_df[num_columns] = std_scaler.transform(test_df[num_columns])

######################
# 데이터프레임 정리
train_df_all = train_df
test_df_all = test_df

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df_all
y_train = np.log1p(train_df['price'])
y_train.describe()
y_train.hist(bins=100)
np.log1p(y_train).hist(bins=100)
X_train.columns


###############################################################
#모델학습
####################################################
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
y_pred_elastic = elastic_search.predict(test_df)
X_train.columns
test_df.columns
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
submit = pd.read_csv('./data/car_prediction/sample_submission.csv')
submit["SalePrice"] = (np.expm1(y_pred_elastic) + np.expm1(y_pred_knn)) / 2

# CSV로 저장
submit.to_csv('./data/car_prediction/elasticnet_log+knn_1700.csv', index=False)

