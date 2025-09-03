import pandas as pd

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/house_prediction/train.csv')
test_df = pd.read_csv('./data/house_prediction/test.csv')

train_df = train_df.select_dtypes(include=['number'])

# 결측치 제거 (간단히 처리)
train_df = train_df.dropna()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 모델 학습
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df = test_df.select_dtypes(include=['number'])
test_df = test_df.fillna(train_df.mean())

# 예측
y_pred = lr.predict(test_df)
submit = pd.read_csv('./data/house_prediction/sample_submission.csv')
submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('./data/house_prediction/baseline.csv', index=False)


# 데이터 분할
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X_train,  
                   y_train, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

# random_state=2025
# shuffle = True
# test size = 0.3
# 정보를 사용해서 
# train_df 를 train_X, test_X, train_y, test_y로
# 나눠보는 코드 작성. 넘파이 판다스 사용.

# 1) tr_df 총 행수 확인
# 2) 벡터가 주어졌을때 랜덤으로 순서를 섞는 함수 알아낼것
#    1:n 벡터를 섞어주세요!
# 3) 앞의 80% 인덱스는 train으로 뒤 20% 인덱스는 test로 설정
import numpy as np
# train_df.iloc[np.array([2, 3]), :]
n=train_df.shape[0]
idx_vec=np.arange(n)
np.random.shuffle(idx_vec)
k1=int(n*0.33)
k2=int(n*0.66)

df1_Xy=train_df.iloc[idx_vec[:k1], :]
df2_Xy=train_df.iloc[idx_vec[k1:k2], :]
df3_Xy=train_df.iloc[idx_vec[k2:], :]


# 모의고사 세트(validation set) 3개를 활용해서 모델
# 성능 점수를 계산

# df1_xy
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
train_1 = pd.concat([df2_Xy,df3_Xy],axis=0)
train_X1 = train_1.drop('SalePrice',axis=1)
train_y1 = train_1['SalePrice']
lr.fit(train_X1,train_y1)

test_X1 = df1_Xy.drop('SalePrice',axis=1)
test_y1 = df1_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred = lr.predict(test_X1)
print('valid RMSE:' , root_mean_squared_error(test_y1, y_pred)) # 39224.77906301603

# df2_xy
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
train_2 = pd.concat([df1_Xy,df3_Xy],axis=0)
train_X2 = train_2.drop('SalePrice',axis=1)
train_y2 = train_2['SalePrice']
lr.fit(train_X2,train_y2)

test_X2 = df2_Xy.drop('SalePrice',axis=1)
test_y2 = df2_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred2 = lr.predict(test_X2)
print('valid RMSE:' , root_mean_squared_error(test_y2, y_pred2)) # 47974.03577073524

# df3_xy
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
train_3 = pd.concat([df1_Xy,df2_Xy],axis=0)
train_X3 = train_3.drop('SalePrice',axis=1)
train_y3 = train_3['SalePrice']
lr.fit(train_X3,train_y3)

test_X3 = df3_Xy.drop('SalePrice',axis=1)
test_y3 = df3_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred3 = lr.predict(test_X3)
print('valid RMSE:' , root_mean_squared_error(test_y3, y_pred3)) # 41265.0955439471


# knn


# df1_xy
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
train_1 = pd.concat([df2_Xy,df3_Xy],axis=0)
train_X1 = train_1.drop('SalePrice',axis=1)
train_y1 = train_1['SalePrice']
knn.fit(train_X1,train_y1)

test_X1 = df1_Xy.drop('SalePrice',axis=1)
test_y1 = df1_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred = knn.predict(test_X1)
print('valid RMSE:' , root_mean_squared_error(test_y1, y_pred)) # 51415.62317797282

# df2_xy
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
train_2 = pd.concat([df1_Xy,df3_Xy],axis=0)
train_X2 = train_2.drop('SalePrice',axis=1)
train_y2 = train_2['SalePrice']
knn.fit(train_X2,train_y2)

test_X2 = df2_Xy.drop('SalePrice',axis=1)
test_y2 = df2_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred2 = knn.predict(test_X2)
print('valid RMSE:' , root_mean_squared_error(test_y2, y_pred2)) #  49703.56966539102

# df3_xy
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
train_3 = pd.concat([df1_Xy,df2_Xy],axis=0)
train_X3 = train_3.drop('SalePrice',axis=1)
train_y3 = train_3['SalePrice']
knn.fit(train_X3,train_y3)

test_X3 = df3_Xy.drop('SalePrice',axis=1)
test_y3 = df3_Xy['SalePrice']

# 모델 평가
from sklearn.metrics import root_mean_squared_error
y_pred3 = knn.predict(test_X3)
print('valid RMSE:' , root_mean_squared_error(test_y3, y_pred3)) #  47731.61385887418





