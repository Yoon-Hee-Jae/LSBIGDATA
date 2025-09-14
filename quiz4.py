import pandas as pd
import numpy as np
import joblib

# 저장된 모델 불러오기
lasso_model = joblib.load("lasso_model.pkl")

pd.read_csv('problem2.csv')
pd.read_csv('problem4_33.csv')
pd.read_csv('problem15.csv')
pd.read_csv('problem19.csv')
pd.read_csv('problem19_test.csv')

# 문제 1번
df1 = pd.read_csv('datasetSalaries.csv')
df1.info()
df1.head()
from scipy.stats import ttest_ind
male = df1[df1['sex'] == 'Male']
female = df1[df1['sex'] == 'Female']
t_statistic, p_value = ttest_ind(male['salary'], female['salary'],
equal_var=False, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)

# 문제 2번
from scipy.stats import f_oneway
# 각 그룹의 데이터를 추출
Professor  = df1[df1['rank'] == 'Professor']['salary']
Assistant_Professor = df1[df1['rank'] == 'Assistant Professor']['salary']
Associate_Professor = df1[df1['rank'] == 'Associate Professor']['salary']
# 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(Professor, Assistant_Professor,Associate_Professor)

print(f'F-statistic: {f_statistic}, p-value: {p_value}')
p_value<0.05

# 문제 3번
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('salary ~ C(rank)',
data=df1).fit()

import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')

# 문제 4번
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
            endog=df1['salary'],
            groups=df1['rank'],
            alpha=0.05)
print(tukey)

# 문제 5번
from scipy.stats import t

sample = [1.95, 1.80, 2.10, 1.82, 1.75, 2.01, 1.83, 1.90]
# 표본 평균
mean = np.mean(sample)
# 표본 크기
n = len(sample)
# 표준 오차
se = np.std(sample, ddof=1) / np.sqrt(n)

ci = t.interval(0.90, loc=mean, scale=se, df=n-1)
ci

# 문제 6번

# 문제 7번
sample = [96, 95, 103.2, 101.0, 100.7, 99.9, 98.6, 100.1, 97.3, 98.4, 99.5, 100.2, 101.4, 100.9, 102.0, 96.8, 99.1]
a=0
for i in range(len(sample)):
    if sample[i] < 97:
        a+=1
        print(a)
b=0
for i in range(len(sample)):
    if (sample[i] < 99) & (sample[i]>=97):
        b+=1
        print(b)
c=0
for i in range(len(sample)):
    if (sample[i] < 101) & (sample[i]>=99):
        c+=1
        print(c)
d=0
for i in range(len(sample)):
    if sample[i] >=101:
        d+=1
        print(d)

# 문제 8번
from scipy.stats import chisquare
from scipy.stats import norm
observed = np.array([3,3,7,4])
expected = np.array([norm.cdf(97, loc=100, scale=5) * len(sample),
                    (norm.cdf(99, loc=100, scale=5)-norm.cdf(97, loc=100, scale=5)) * len(sample),
                    (norm.cdf(101, loc=100, scale=5)-norm.cdf(99, loc=100, scale=5)) * len(sample),
                    (1-norm.cdf(101, loc=100, scale=5)) * len(sample)
                    ])

statistic, p_value = chisquare(observed, f_exp=expected)
print("Test statistic: ", statistic.round(3))

# 문제 9번
old_env = [72, 68, 74, 70, 65, 69, 71, 73, 67, 66]
new_env = [78, 70, 76, 74, 69, 72, 75, 77, 70, 72]
np.mean(old_env)
np.mean(new_env)
from scipy.stats import ttest_rel
# 단측 검정 (큰 쪽)
t_statistic, p_value = ttest_rel(new_env, old_env, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)

# 문제 10번

# 문제 11번
df11 = pd.read_csv('problem4_33.csv')
df11.head()
df11.info()
# 전처리 1
df11['delay_under_10'] = df11['delay_0_5'] +  df11['delay_5_10']
df11['delay_10_20'] = df11['delay_10_15'] +  df11['delay_15_20']
df11['delay_over_20'] = df11['delay_20_25'] +  df11['delay_25_30']
# 전처리 2
df11['delay_category'] = None
df11.info()
for i in range(len(df11)):
    if df11['delay_under_10'][i] > 0:
        df11['delay_category'][i] = 'Under10'
    elif df11['delay_10_20'][i] > 0:
        df11['delay_category'][i] = '10to20'
    elif df11['delay_over_20'][i] > 0:
        df11['delay_category'][i] = 'Over20'
    else:
        df11['delay_category'][i] = 'NoDelay'
df11[['delay_category','Line']].groupby('delay_category').value_counts()
Under10  = df11[df11['rank'] == 'Professor']['salary']
Assistant_Professor = df1[df1['rank'] == 'Assistant Professor']['salary']
Associate_Professor = df1[df1['rank'] == 'Associate Professor']['salary']
# 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(Professor, Assistant_Professor,Associate_Professor)

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('delay_category ~ C(Line)',
            data=df11[['delay_category','Line']]).fit()
anova_results = sm.stats.anova_lm(model, typ=2)

# 문제 12번
df12 = pd.read_csv('problem2.csv')
df12.info()
df12['건강검진일'] = df12['건강검진일'].str.replace('_','')
df12['건강검진일'] = pd.to_datetime(df12['건강검진일'],format='%Y%m%d')
df12['생년월일'] = pd.to_datetime(df12['생년월일'],format='%Y%m%d')

df12['만나이'] = (df12['건강검진일']-df12['생년월일'])
df12['만나이'] = round(df12['만나이'].dt.days /365,0)

df12['BMI'] = round(df12['weight'] / ((df12['키']/100)**2),1)

df12['분류']=0
for i in range(len(df12)):
    if (df12['성별'][i]=='남성') & ((df12['BMI'][i]>=17.3)&(df12['BMI'][i]<=18.4)) & (df12['만나이'][i]==16):
        df12['분류'][i] = '적정'
    elif (df12['성별'][i]=='남성') & ((df12['BMI'][i]<17.3)&(df12['BMI'][i]>18.4)) & (df12['만나이'][i]==16):
        df12['분류'][i] = '불량'
    elif (df12['성별'][i]=='남성') & ((df12['BMI'][i]>=17.8)&(df12['BMI'][i]<=20.4)) & (df12['만나이'][i]==17):
        df12['분류'][i] = '적정'
    elif (df12['성별'][i]=='남성') & ((df12['BMI'][i]<17.8)&(df12['BMI'][i]>20.4)) & (df12['만나이'][i]==17):
        df12['분류'][i] = '불량'
    elif (df12['성별'][i]=='여성') & ((df12['BMI'][i]>=16.7)&(df12['BMI'][i]<=18.7)) & (df12['만나이'][i]==16):
        df12['분류'][i] = '적정'
    elif (df12['성별'][i]=='여성') & ((df12['BMI'][i]<16.7)&(df12['BMI'][i]>18.7)) & (df12['만나이'][i]==16):
        df12['분류'][i] = '불량'
    elif (df12['성별'][i]=='여성') & ((df12['BMI'][i]>=16.7)&(df12['BMI'][i]<=18.7)) & (df12['만나이'][i]==17):
        df12['분류'][i] = '적정'
    elif (df12['성별'][i]=='여성') & ((df12['BMI'][i]<16.7)&(df12['BMI'][i]>18.7)) & (df12['만나이'][i]==17):
        df12['분류'][i] = '불량'       
df12[['만나이','성별','분류']].groupby('만나이').value_counts()


# 14번
data = fetch_openml(name="energy_efficiency", version=1, as_frame=True)

# 16번
df16 = pd.read_csv('problem15.csv')
df16['absences'].hist()
np.sum(df16['absences']>100)
np.sum(df16['age']>=24)
df16.isnull().sum()
df16.dropna()











# 문제 17





# 문제 19
df19 = pd.read_csv('problem19.csv')
lasso_model = joblib.load("lasso_model.pkl")
alphas = np.arange(0.1, 0.9, 0.1)  # 200개 구간 분할
# 고려할 파라미터 경우의 수
lasso_model().get_params()

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







# 문제 23
df23 = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_3_2.csv')
df23[['churn','Phone_Service']].groupby('Phone_Service').sum()
(171/500) / (1-171/500)


# 문제 24번



# 문제 25번
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
model = ols(
    "bill_length_mm ~ bill_depth_mm + species + bill_depth_mm:species", data=train_data
).fit()
print(model.summary())






