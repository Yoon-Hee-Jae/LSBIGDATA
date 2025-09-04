import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV

# 데이터 로드
train_df = pd.read_csv('./data/car_prediction/train.csv').drop('id',axis=1)
test_df = pd.read_csv('./data/car_prediction/test.csv').drop('id',axis=1)

# ====================================================================
# 데이터 전처리 함수 정의 (반복 코드 제거)
def preprocess_engine_data(df):
    df["Horsepower"] = df["engine"].str.extract(r'(\d+\.?\d*)HP').astype(float)
    df["Displacement(L)"] = df["engine"].str.extract(r'(\d+\.?\d*)L').astype(float)
    cylinders = df["engine"].str.extract(r'(\d+)\sCylinder|V(\d+)')
    df["Cylinders"] = cylinders.bfill(axis=1).iloc[:, 0].astype(float)
    df["Fuel_Type"] = df["engine"].str.extract(r'(Gasoline|Diesel|Electric|Hybrid|LPG)')
    df["Engine_Type"] = df["engine"].str.extract(r'(\d+\sCylinder Engine)')
    return df

train_df = preprocess_engine_data(train_df)
test_df = preprocess_engine_data(test_df)

# ====================================================================
# accident 열 처리 (데이터 누수 수정)
mapping = {
    'None reported': 0,
    'At least 1 accident or damage reported': 1
}
train_df['accident_numeric'] = train_df['accident'].map(mapping)
test_df['accident_numeric'] = test_df['accident'].map(mapping)

# ====================================================================
# 독립변수(X)와 종속변수(y) 분리
y_train = np.log1p(train_df['price'])
X_train = train_df.drop('price', axis=1) # 'price' 컬럼을 여기서 먼저 제거
X_test = test_df.copy()

# ====================================================================
# 결측치 처리
num_columns = X_train.select_dtypes(include=['number']).columns
cat_columns = X_train.select_dtypes(include=['object']).columns

freq_impute = SimpleImputer(strategy='most_frequent')
X_train[cat_columns] = freq_impute.fit_transform(X_train[cat_columns])
X_test[cat_columns] = freq_impute.transform(X_test[cat_columns])

mean_impute = SimpleImputer(strategy='mean')
X_train[num_columns] = mean_impute.fit_transform(X_train[num_columns])
X_test[num_columns] = mean_impute.transform(X_test[num_columns])

# ====================================================================
# 범주형 변수 인코딩 (CatBoostEncoder 사용)
import category_encoders as ce

encoder = ce.CatBoostEncoder(cols=cat_columns)

# y_train을 명시적으로 전달하여 fit_transform을 수행
X_train_encoded = encoder.fit_transform(X_train[cat_columns], y_train)

# test 데이터에 transform만 수행
X_test_encoded = encoder.transform(X_test[cat_columns])

# 인코딩된 범주형 변수를 원래 데이터프레임에 병합
X_train = pd.concat([X_train.drop(cat_columns, axis=1), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(cat_columns, axis=1), X_test_encoded], axis=1)

# ====================================================================
# 수치형 변수 표준화
std_scaler = StandardScaler().set_output(transform='pandas')

# 수치형 변수 컬럼 재선택
num_columns_after_encoding = X_train.select_dtypes(include=['number']).columns

X_train[num_columns_after_encoding] = std_scaler.fit_transform(X_train[num_columns_after_encoding])
X_test[num_columns_after_encoding] = std_scaler.transform(X_test[num_columns_after_encoding])


# ====================================================================
# 모델 학습 및 예측
elastic = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1), 'l1_ratio': np.linspace(0, 1, 5)}
cv = KFold(n_splits=5, shuffle=True, random_state=0)

elastic_search = GridSearchCV(estimator=elastic,
                              param_grid=elastic_params,
                              cv=cv,
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train, y_train)

# 결과 확인
print("Best Parameters:", elastic_search.best_params_)
print("Best Score (MSE):", -elastic_search.best_score_)

# 예측
y_pred_elastic = elastic_search.predict(X_test)
y_pred_final = np.expm1(y_pred_elastic)

print("\nFinal Predicted Prices (Sample):")
print(y_pred_final[:5])

# 제출 파일 생성
submit = pd.read_csv('./data/car_prediction/sample_submission.csv')
submit["price"] = y_pred_final
submit.to_csv('./data/car_prediction/elasticnet_log+knn_1703.csv', index=False)
print("Submission file shape:", submit.shape)

# 모델 학습 2: KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, GridSearchCV

knn = KNeighborsRegressor()

# 고려할 파라미터 경우의 수
knn_params = {'n_neighbors': np.arange(1, 11, 1)}

# 교차검증 설정
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
knn_search = GridSearchCV(estimator=knn,
                          param_grid=knn_params,
                          cv=cv,
                          scoring='neg_mean_squared_error')

# 모델 학습
knn_search.fit(X_train, y_train)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(knn_search.cv_results_))

# 최적 파라미터 및 점수
print("Best Parameters (KNN):", knn_search.best_params_)
print("Best Score (MSE, KNN):", -knn_search.best_score_)

# 예측 (수정된 부분: X_test 사용)
y_pred_knn = knn_search.predict(X_test)

# 제출 데이터 생성
submit = pd.read_csv('./data/car_prediction/sample_submission.csv')

# 예측값 역변환 후 평균
submit["SalePrice"] = (np.expm1(y_pred_elastic) + np.expm1(y_pred_knn)) / 2

# CSV 파일로 저장
submit.to_csv('./data/car_prediction/elasticnet_log+knn_1700.csv', index=False)

print("\nSubmission file 'elasticnet_log+knn_1700.csv' has been created successfully.")