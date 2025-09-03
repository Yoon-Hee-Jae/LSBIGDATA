import pandas as pd
import numpy as np

train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')
train_data.info()
test_data.info()

# 예시로 평균값 채우기
submit_data['SalePrice'] = train_data['SalePrice'].mean()
#submit_data.to_csv('./data/house_prediction/first_submit.csv',index=False,
#                  encoding='utf-8')

# train_data 수치형 변수만 선택해서 train_data 업데이트
# 이후 모든 변수 사용해서 선형회귀분석
# test_data의 정보를 사용해서 집값 예측
# 예측한 값을 사용해서 submit_csv의 집값 채우기
numeric_df = train_data.select_dtypes(include=['int64','float64'])
from sklearn.linear_model import LinearRegression
model = LinearRegression()
numeric_df.info()
numeric_df.dropna(inplace=True)
train_X = numeric_df.drop('SalePrice',axis=1)
train_y = numeric_df['SalePrice']
model.fit(train_X, train_y)
model.coef_
model.intercept_
# test 데이터도 train 데이터와 같은 형태로 만들어줌
numeric_test = test_data.select_dtypes(include=['int64','float64'])
numeric_test.info()
# 결측치를 dropna 시키면 안됨
numeric_test.fillna(0,inplace=True)
submit_data['SalePrice'] = model.predict(numeric_test)
submit_data.to_csv('./data/house_prediction/second_submit.csv',index=False,
                 encoding='utf-8')



# 예측값 정확도 높이기
train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')
train_data.info()
test_data.info()

# 1. 의미 없는 변수 제거 ID 칼럼
numeric_train = train_data.select_dtypes(include=['int64','float64'])
numeric_train = numeric_train.drop('Id',axis=1)

numeric_test = test_data.select_dtypes(include=['int64','float64'])
numeric_test = numeric_test.drop('Id',axis=1)

# 2. 결측치 대체
# LotFrontage /  MasVnrArea / GarageYrBlt 
numeric_train.info() 
numeric_train.shape
# 결측치 제거
numeric_train.dropna(subset=['MasVnrArea'],inplace=True)
# 평균값
numeric_train.fillna(numeric_train.mean(),inplace=True)

# 원본 훼손 방지를 위해 복사본 생성
df = numeric_train.copy()
train_y = df['SalePrice']
len(train_y)
df.shape
df.drop('SalePrice',axis=1,inplace=True)
model.fit(df, train_y)
model.intercept_
# test도 똑같이 열 추가
numeric_test.info()
numeric_test.fillna(numeric_test.mean(),inplace=True)
submit_data['SalePrice'] = model.predict(numeric_test)
submit_data.to_csv('tree2.cs' \
'v',index=False,
                 encoding='utf-8')

###########################################################################

# 사이킷런 활용해서 데이터 누수 없이 성능 올리기
train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')
train_data.info()
test_data.info()

# 결측치 제거
train_data.dropna(subset=['MasVnrArea'],inplace=True)

# 데이터 분리
train_X = train_data.drop(['SalePrice'], axis = 1)
train_y = train_data['SalePrice']

# train 데이터 train/validation 으로 분할
from sklearn.model_selection import train_test_split
train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 1)

# 수치형 변수만 선택
numeric_train = train_X_sub.select_dtypes(include=['int64','float64'])
numeric_val = valid_X.select_dtypes(include=['int64','float64'])
numeric_test = test_data.select_dtypes(include=['int64','float64'])

# id 칼럼 제거
numeric_train = numeric_train.drop('Id',axis=1)
numeric_val = numeric_val.drop('Id',axis=1)
numeric_test = numeric_test.drop('Id',axis=1)

# 평균값 대체
numeric_train.info()
from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy = 'most_frequent').set_output(transform='pandas')
# LotFrontage
numeric_train = imputer_mean.fit_transform(numeric_train)
numeric_train.isna().sum()
numeric_val = imputer_mean.transform(numeric_val)
numeric_val.isna().sum()
numeric_test = imputer_mean.transform(numeric_test)
numeric_train.isna().sum()

# 모델 학습
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(numeric_train, train_y_sub)

# 모델 평가
from sklearn.metrics import root_mean_squared_error
pred_val = lr.predict(numeric_val)
print('valid RMSE:' , root_mean_squared_error(valid_y, pred_val))

# mean = 31181.317716614016
# median = 31184.179231714752
# most_frequent = 31103.01163287705
# knn = 31190.37533084124

# knn
# 사이킷런 활용해서 데이터 누수 없이 성능 올리기
train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')

# 결측치 제거
train_data.dropna(subset=['MasVnrArea'],inplace=True)

# 데이터 분리
train_X = train_data.drop(['SalePrice'], axis = 1)
train_y = train_data['SalePrice']

# train 데이터 train/validation 으로 분할
from sklearn.model_selection import train_test_split
train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 1)

# 수치형 변수만 선택
numeric_train = train_X_sub.select_dtypes(include=['int64','float64'])
numeric_val = valid_X.select_dtypes(include=['int64','float64'])
numeric_test = test_data.select_dtypes(include=['int64','float64'])

# id 칼럼 제거
numeric_train = numeric_train.drop('Id',axis=1)
numeric_val = numeric_val.drop('Id',axis=1)
numeric_test = numeric_test.drop('Id',axis=1)

# knn 대체
from sklearn.impute import KNNImputer
knnimputer = KNNImputer(n_neighbors = 5)

train_X5_num_imputed = knnimputer.fit_transform(numeric_train)
val_X5_num_imputed = knnimputer.transform(numeric_val)
test_X5_num_imputed = knnimputer.transform(numeric_test)
                       
train_X5_num_imputed = pd.DataFrame(train_X5_num_imputed, 
                                    columns=numeric_train.columns, 
                                    index = numeric_train.index)
val_X5_num_imputed = pd.DataFrame(val_X5_num_imputed, 
                                   columns=numeric_val.columns, 
                                   index = numeric_val.index)
test_X5 = pd.DataFrame(test_X5_num_imputed, 
                       columns=numeric_test.columns,
                       index=numeric_test.index)

train_X5 = train_X5_num_imputed
val_X5 = val_X5_num_imputed
test_X5
print('학습 데이터 변수 결측치 확인 :', train_X5.isna().sum())

# 모델 학습
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X5, train_y_sub)

# 모델 평가
from sklearn.metrics import root_mean_squared_error
pred_val = lr.predict(val_X5)
print('valid RMSE:' , root_mean_squared_error(valid_y, pred_val))
# 최종 test 데이터 예측
pred_test = lr.predict(test_X5)
submit_data['SalePrice'] = pred_test
submit_data.to_csv('submission1631.csv', index=False)

print("최종 예측 완료! submission.csv 생성됨")


##############3
# 최종
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# 데이터 불러오기
train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')

# 결측치 제거 (필요한 경우)
train_data.dropna(subset=['MasVnrArea'], inplace=True)

# X, y 분리
train_X = train_data.drop(['SalePrice'], axis=1)
train_y = train_data['SalePrice']

# 수치형 변수만 선택
numeric_train = train_X.select_dtypes(include=['int64','float64']).drop('Id', axis=1)
numeric_test = test_data.select_dtypes(include=['int64','float64']).drop('Id', axis=1)

# 최빈값 대체 (train 전체 데이터에 fit)
imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
numeric_train = imputer.fit_transform(numeric_train)
numeric_test = imputer.transform(numeric_test)

# 모델 학습 (전체 train 데이터 사용)
lr = LinearRegression()
lr.fit(numeric_train, train_y)

# 최종 test 데이터 예측
pred_test = lr.predict(numeric_test)

# 제출파일 생성
submit_data['SalePrice'] = pred_test
submit_data.to_csv('submission1630.csv', index=False)

print("최종 예측 완료! submission.csv 생성됨")

# 




# 데이터 불러오기
train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')
submit_data = pd.read_csv('./data/house_prediction/sample_submission.csv')

# 결측치 제거 (필요한 경우)
train_data.dropna(subset=['MasVnrArea'], inplace=True)

# X, y 분리
train_X = train_data.drop(['SalePrice'], axis=1)
train_y = train_data['SalePrice']

# 수치형 변수만 선택
numeric_train = train_X.select_dtypes(include=['int64','float64']).drop('Id', axis=1)
numeric_test = test_data.select_dtypes(include=['int64','float64']).drop('Id', axis=1)

# 최빈값 대체 (train 전체 데이터에 fit)
imputer = SimpleImputer(strategy='most_frequent').set_output(transform='pandas')
numeric_train = imputer.fit_transform(numeric_train)
numeric_test = imputer.transform(numeric_test)

# 표준화
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
stdscaler = StandardScaler()
mc_transformer = make_column_transformer(
    (stdscaler, numeric_train.columns),
    remainder='passthrough'
    ).set_output(transform = 'pandas')
    
train_X_transformed = mc_transformer.fit_transform(numeric_train)
test_X_transformed = mc_transformer.transform(numeric_test)

# 모델 학습 (전체 train 데이터 사용)
lr = LinearRegression()
lr.fit(train_X_transformed, train_y)

# 최종 test 데이터 예측
pred_test = lr.predict(test_X_transformed)

# 제출파일 생성
submit_data['SalePrice'] = pred_test
submit_data.to_csv('./data/house_prediction/submission1731.csv', index=False)

print("최종 예측 완료! submission.csv 생성됨")



#####
# 데이터 불러오기
train_df = pd.read_csv('./data/house_prediction/train.csv')
test_df = pd.read_csv('./data/house_prediction/test.csv')
submit_df = pd.read_csv('./data/house_prediction/sample_submission.csv')

train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 2025, shuffle=True)
train_data