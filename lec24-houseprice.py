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

