import pandas as pd
import numpy as np
odors = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10, 11]
minutes_rosemary = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_peppermint = [18, 17, 18, 16, 17, 19, 18, 17, 18, 19]
anova_data = pd.DataFrame({
'Odor': np.repeat(odors, 10),
'Minutes': minutes_lavender + minutes_rosemary + minutes_peppermint})

anova_data
# ANOVA 검정
# 귀무가설: mu_l = mu_r = mu_p
anova_data.groupby('Odor').describe() # 3표준편차를 +-하면 커버가능한 범위가 생성됨
from scipy.stats import f_oneway
# 각 그룹의 데이터를 추출
lavender = anova_data[anova_data['Odor'] == 'Lavender']['Minutes']
rosemary = anova_data[anova_data['Odor'] == 'Rosemary']['Minutes']
peppermint = anova_data[anova_data['Odor'] == 'Peppermint']['Minutes']
# 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')
# 귀무가설이 기각이 되고 평균이 다른게 존재한다.

# 사후검정의 필요성
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(
endog=anova_data['Minutes'],
groups=anova_data['Odor'],
alpha=0.05)
print(tukey) # p-adj 사후검정을 통해 알아서 p-value를 조정함


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Minutes ~ C(Odor)',
data=anova_data).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
anova_results
model.resid

import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues, model.resid)
plt.show() 
# t 검정 1표본으로 잔차들의 평균이 0 인지 확인
# 0을 기준으로 잔차가 고르게 펴져있는지
# 3그룹의 퍼짐 정도가 비슷한지 즉 등분산인지

# 분포가 정규분포를 따르는지 확인 방법
# qq 플랏을 그려봄
# 샤키로이 검정

import scipy.stats as sp
W, p = sp.shapiro(model.resid)
print(f'검정통계량: {W:.3f}, 유의확률: {p:.3f}')

# bartlett 검정
from scipy.stats import bartlett
groups = ['Lavender', 'Rosemary', 'Peppermint']
grouped_residuals = [model.resid[anova_data['Odor'] == group] for group in groups]
test_statistic, p_value = bartlett(*grouped_residuals)
print(f"검정통계량: {test_statistic}, p-value: {p_value}")


