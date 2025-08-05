import pandas as pd
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
filename = "problem5_27.csv"
dat = pd.read_csv(filename)
dat

# 문제 1
# 귀무가설 : 상류와 하류의 생물 다양성 점수에 차이가 없다
# 대립가설 : 차이가 있다
from scipy.stats import ttest_rel
upper = dat.iloc[:,1]
lower = dat.iloc[:,2]
t_statistic, p_value = ttest_rel(upper, lower, alternative='two-sided')
print('t_statisitc =', t_statistic, "p_value = ", p_value)
# p-value가 0.09 이므로 귀무가설을 기각하고 즉 차이가 있다.

# 문제 2
filename = "problem5_32.csv"
dat = pd.read_csv(filename)
# 귀무가설 : 차이가 없다
# 대립가설 : 차이가 있다
dat.info()

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.boxplot(x='Gender', y='Salary', data=dat)
plt.title('성별에 따른 급여 분포')
plt.xlabel('성별')
plt.ylabel('급여')
plt.show()

# 그룹 간 분산의 차이가 있는지 판단
male = dat[dat['Gender']=='Male'].reset_index(drop=True)
female = dat[dat['Gender']=='Female'].reset_index(drop=True)
male_var = male['Salary'].var()
female_var = female['Salary'].var()
female_var*1.5 > male_var
# 분산의 차이가 존재

# 가설에 대한 검정 통계량과 유의 확률
from scipy.stats import ttest_ind
t_statistic, p_value = ttest_ind(male['Salary'], female['Salary'],
                                 equal_var=False, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)
# p-value가 굉장히 낮아서 귀무가설을 기각하고 차이는 있음

# 이렇게 하면 안되는 이유는?
from scipy.stats import ttest_1samp
a = male['Salary'] - female['Salary']
t_statistic, p_value = ttest_1samp(a, popmean=0,alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)
# p-value 가 굉장히 낮으므로 귀무가설을 기각하고 차이가 있음

# t 통계량 공식상 n 값이 올라가면 자동으로 t검정 결과값이 커진다
# 우선 시각화부터 보는 것이 맞음

# 3번
filename = "heart_disease.csv"
dat = pd.read_csv(filename)
dat.groupby('target')['chol'].mean()
# 귀무가설 : 차이가 없다
# 대립 : 차이가 있다
# 결측치 제거
dat_1 = dat[['target','chol']]
dat_1.info()
dat_1.dropna(inplace=True)
dat_1.info()

# 시각화
plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='chol', data=dat_1)
plt.title('target별 콜레스테롤(chol) 분포 비교')
plt.xlabel('target')
plt.ylabel('cholesterol (chol)')
plt.show()

# 그룹 간 분산 차이
dat_yes = dat_1[dat_1['target']=='yes']
dat_no = dat_1[dat_1['target']=='no']
same = dat_yes['chol'].var() < dat_no['chol'].var()*1.5
same
# 분산 차이가 없다

# 검정 통계량 유의확률
t_statistic, p_value = ttest_ind(dat_yes['chol'], dat_no['chol'],
                                 equal_var=True, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)
# 차이가 있다는 귀무가설을 기각해서 차이가 없다

# 4번
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()
dat.info()
# 귀무가설 : 당뇨병이 있는 사람과 없는 사람의 평균 bmi 차이가 없다
# 대립가설 : 있다

# 시각화
dat_4 = dat[['BMI','Outcome']]
dat_4
plt.figure(figsize=(8,6))
sns.boxplot(x='Outcome', y='BMI', data=dat_4)
plt.title('분포 비교')
plt.xlabel('Outcome')
plt.ylabel('BMI')
plt.show()

healthy = dat_4[dat_4['Outcome']==0]
weak = dat_4[dat_4['Outcome']==1]

same = max(healthy['BMI'].var(),weak['BMI'].var()) < min(healthy['BMI'].var(),weak['BMI'].var())*1.5
same
# 차이가 없다

# 검통 유의확률
t_statistic, p_value = ttest_ind(healthy['BMI'], weak['BMI'],
                                 equal_var=True, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)

# p-value 가 작고 기각 불가능 즉 차이가 없다





