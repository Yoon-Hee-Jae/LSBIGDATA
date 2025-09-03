import pandas as pd

# 집가격 데이터 불러오세요!
train_df = pd.read_csv('./data/houseprice/train.csv')
test_df = pd.read_csv('./data/houseprice/test.csv')

train_df = train_df.select_dtypes(include=['number'])

# 결측치 제거 (간단히 처리)
train_df = train_df.dropna()

# 독립변수(X)와 종속변수(y) 분리
X_train = train_df.drop(columns='SalePrice')
y_train = train_df['SalePrice']

from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha= 0.1,
                     l1_ratio= 0.6)
elastic.fit(X_train, y_train)

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
test_df = test_df.select_dtypes(include=['number'])
test_df = test_df.fillna(train_df.mean())

# 예측
y_pred = elastic.predict(test_df)
submit = pd.read_csv('./data/houseprice/sample_submission.csv')
submit["SalePrice"]=y_pred

# CSV로 저장
submit.to_csv('./data/houseprice/elasticnet.csv', index=False)
