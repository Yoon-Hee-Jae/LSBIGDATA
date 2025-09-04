import pandas as pd
import numpy as np

df = pd.read_csv('fluentcom-6c2c277f9eb0f9e3c73aea101c02360e-fluentcom-problem1.csv')

df.info()

# 데이터 전처리

# BOX PLOT
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=df)
plt.title("Boxplot with Seaborn")
plt.show()

# 히스토그렘

for col in df.columns:
    sns.histplot(df[col], bins=5, kde=True)  # kde=True 하면 밀도 곡선도 함께 표시
    plt.title(f"{col} Histogram")
    plt.show()

# age" 칼럼명 수정
df = df.rename(columns={'Age"': "Age"})

# Tri 이상치
df['Tri'].hist(bins=100)
df['Tri'].describe()
Q1 = df['Tri'].quantile(0.25)
Q3 = df['Tri'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 기준
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 탐지
outliers = df['Tri'][(df['Tri'] < lower_bound) | (df['Tri'] > upper_bound)]
print("이상치:\n", outliers)
len(outliers) # 215
215/3403*100
df['Tri'] = df['Tri'][(df['Tri'] >= lower_bound) & (df['Tri'] <= upper_bound)]
df.info()

# ALT 이상치
df['ALT'].hist(bins=100)
df['ALT'].describe()
Q1 = df['ALT'].quantile(0.25)
Q3 = df['ALT'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 기준
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 탐지
outliers = df['ALT'][(df['Tri'] < lower_bound) | (df['ALT'] > upper_bound)]
print("이상치:\n", outliers)
len(outliers) # 300
215/3403*100
df['ALT'] = df['ALT'][(df['ALT'] >= lower_bound) & (df['ALT'] <= upper_bound)]
df.info()

# 결측치 제거
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

# 다중공선성 확인
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 상수항 추가
X = add_constant(df)

# VIF 계산
vif = pd.DataFrame()
vif["feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# 다중공선성 없음

# 데이터 분리
train_X = df.drop(['DBP'], axis = 1)
train_y = df['DBP']

# 데이터 분리
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 1)
train_X.shape
test_X.shape
1150/2681*100 # 4:6 정도?

# 독립변수의 차원 축소는 필요없다
# 변수의 수가 많지도 않고, 다중공선성 의심되는 것도 없기 때문이다.

# 회귀분석의 기본 가정

# 1. 독립변수와 종속변수간의 상관성
# 약한 상관관계를 지니고 있음
for col in df.columns:
    plt.scatter(df[col], df['DBP'], color='blue', label='data points')
    plt.title('Scatter Plot of X vs Y')
    plt.xlabel(col)
    plt.ylabel('Y')
    plt.legend()
    plt.show()
# 피어슨 상관rPtn
from scipy.stats import pearsonr
for col in df.columns:
    r, p = pearsonr(df[col], df['DBP'])
    print(f"Pearson {col} 상관계수: {r:.3f}")
    print(f"p-value: {p:.3f}\n")

# 종속변수를 로그 취할 경우
np.log1p(df['DBP']).hist()
for col in df.columns:
    plt.scatter(df[col], np.log1p(df['DBP']), color='blue', label='data points')
    plt.title(f'Scatter Plot of {col} vs Y')
    plt.xlabel(col)
    plt.ylabel('Y')
    plt.legend()
    plt.show()
# 피어슨 상관rPtn
from scipy.stats import pearsonr
for col in df.columns:
    r, p = pearsonr(df[col], np.log1p(df['DBP']))
    print(f"Pearson {col} 상관계수: {r:.3f}")
    print(f"p-value: {p:.3f}\n")

# 모델 3개 선택
# 1. 랜덤포레스트
# 선형관계가 약하기 때문에 비선형 모델을 선택함
# 표준화
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler().set_output(transform='pandas')

train_X = std_scaler.fit_transform(train_X)
test_X = std_scaler.transform(test_X)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error

dct = DecisionTreeRegressor()
DecisionTreeRegressor().get_params()
# 고려할 파라미터 경우의 수
dct_params = {'max_depth' : np.arange(1, 10, 1)}

from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
dct_search = GridSearchCV(estimator=dct, 
                              param_grid=dct_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

dct_search.fit(train_X, train_y)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(dct_search.cv_results_))

# best prameter
print(dct_search.best_params_)

# 교차검증 best score 
print(dct_search.best_score_)

# 테스트 데이터도 숫자형만 선택하고, 결측치는 평균으로 채움
# test_df = test_df.select_dtypes(include=['number'])
# test_df = test_df.fillna(train_df.mean())
# 예측
y_pred_dct = dct_search.predict(test_X)

print('valid RMSE:' , root_mean_squared_error(test_y, y_pred_dct))


#  2. 엘라스틱 
from sklearn.linear_model import ElasticNet
elastic = ElasticNet()

# 파라미터 확인 
ElasticNet().get_params()

# 고려할 파라미터 경우의 수
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

# 교차검증
from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 그리드서치
elastic_search = GridSearchCV(estimator=elastic, 
                              param_grid=elastic_params, 
                              cv = cv, 
                              scoring='neg_mean_squared_error')

elastic_search.fit(train_X, train_y)

# 그리드서치 파라미터 성능 확인
print(pd.DataFrame(elastic_search.cv_results_))

# best prameter
print(elastic_search.best_params_)

# 교차검증 best score 
print(-elastic_search.best_score_)

# 예측
y_pred_E = elastic_search.predict(test_X)

print('valid RMSE:' , root_mean_squared_error(test_y, y_pred_E))


# 3. 다중선형회귀






