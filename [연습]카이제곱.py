import pandas as pd
from scipy.stats import chi2_contingency
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 불러오기
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 임신 유무 파생변수 생성
dat['Pregnancy_status'] = (dat['Pregnancies'] > 0).astype(int)

dat.info()

# 귀무가설 : 임신과 당뇨는 독립이다
# 대립가설 : 독립이 아니다

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 (0: 없음, 1: 있음)
# 교차표
ct = pd.crosstab(dat['Pregnancy_status'], dat['Outcome'])

# 히트맵
plt.figure(figsize=(6,4))
sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.title('임신 여부 vs 당뇨 여부 교차표')
plt.ylabel('임신')
plt.xlabel('당뇨')
plt.show()

# 그룹 막대그래프
plt.figure(figsize=(6,4))
sns.countplot(x='Pregnancy_status', hue='Outcome', data=dat)
plt.title('임신 여부와 당뇨 여부 관계')
plt.show()

# 교차표
dat.shape
# 임신 
y_p = dat[dat['Pregnancy_status']==1].shape[0] / 768
# 임신 ㄴ
n_p =dat[dat['Pregnancy_status']==0].shape[0] / 768

# 당뇨
y_d = dat[dat['Outcome']==1].shape[0] / 768
n_d = dat[dat['Outcome']==0].shape[0] / 768

# 실제값
dat[(dat['Pregnancy_status']==1)&(dat['Outcome']==1)].shape[0] # 230
dat[(dat['Pregnancy_status']==1)&(dat['Outcome']==0)].shape[0] # 427
dat[(dat['Pregnancy_status']==0)&(dat['Outcome']==1)].shape[0] # 38
dat[(dat['Pregnancy_status']==0)&(dat['Outcome']==0)].shape[0] # 73

# 기대빈도
768 * y_p*y_d # 229.265625 임신 당뇨
768 * y_p*n_d # 427.734375 임신 노당뇨
768 * n_p * y_d # 38.734375 노임신 당뇨
768 * n_p * n_d # 72.265625 노임신 노당뇨

# 카이제곱 검정
from scipy.stats import chi2_contingency
# 교차표 생성
ct = pd.crosstab(dat['Pregnancy_status'], dat['Outcome'])

# 카이제곱 독립성 검정
chi2, p, df, expected = chi2_contingency(ct, correction=False)

print("Chi-square 통계량:", round(chi2, 3))
print("자유도:", df)
print("p-value:", round(p, 3))
print("\n기대빈도표:\n", expected)


# 문제 2번
dat.info()
dat['Age_group'] = (dat['Age'] >= 40).astype(int)

# 귀무가설 : 젊은이와 노인의 당뇨발생확률은 동일하다
# 대립가설 : 아니다

# 관계 시각화
# 그룹 막대그래프
plt.figure(figsize=(6,4))
sns.countplot(x='Age_group', hue='Outcome', data=dat)
plt.title('나이와 당뇨 여부 관계')
plt.show()

# 교차표 실제값
dat[(dat['Age_group']==1)&(dat['Outcome']==1)].shape[0] # 108
dat[(dat['Age_group']==1)&(dat['Outcome']==0)].shape[0] # 99
dat[(dat['Age_group']==0)&(dat['Outcome']==1)].shape[0] # 160
dat[(dat['Age_group']==0)&(dat['Outcome']==0)].shape[0] # 401

# 기대빈도




