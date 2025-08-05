import numpy as np
import pandas as pd

sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
# 귀무가설: 뮤가 10이다
from scipy.stats import ttest_1samp
from scipy.stats import norm
from scipy.stats import t

t_statistic, p_value = ttest_1samp(sample, popmean=10, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)
# 유의수준 5% 하에서는 p-value가 0.06이기에 10이라는 귀무가설을 기각하지 못한다
sam_mean = np.mean(sample)
t_statistic = (sam_mean-10) / (np.std(sample)/np.sqrt(len(sample)))
(1-t(df=len(sample)).cdf(t_statistic))*2

# 귀무가설: 뮤가 10 이하이다
t_statistic, p_value = ttest_1samp(sample, popmean=10, alternative='less')
print("t-statistic:", t_statistic, "p-value:", p_value)

# 독립 2표본 t 검정
sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["Female"]*7 + ["Male"]*5

my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
my_tab2.head(3)

from scipy.stats import ttest_ind
male = my_tab2[my_tab2['gender'] == 'Male']
female = my_tab2[my_tab2['gender'] == 'Female']
t_statistic, p_value = ttest_ind(male['score'], female['score'],
                                 equal_var=True, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)

# 대응 표본 t 검정
before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])
from scipy.stats import ttest_rel
# 단측 검정 (큰 쪽)
t_statistic, p_value = ttest_rel(after, before, alternative='greater')
print("t-statistic:", t_statistic, "p-value:", p_value)
# 1표본 으로 바꾸기
a = after-before
t_statistic, p_value = ttest_1samp(a, popmean=0, alternative='two-sided')
print("t-statistic:", t_statistic, "p-value:", p_value)





