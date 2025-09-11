import pandas as pd
import numpy as np
admission_data = pd.read_csv("./data/admission.csv")
print(admission_data.head())
print(admission_data.shape)

p_hat = admission_data['admit'].mean()
print(np.round(p_hat/(1-p_hat),3))

unique_ranks = sorted(admission_data['rank'].unique())
print(unique_ranks)

grouped_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)


import numpy as np
import matplotlib.pyplot as plt
p = np.arange(0, 1.01, 0.01)
log_odds = np.log(p / (1 - p))
plt.plot(p, log_odds)
plt.xlabel('p')
plt.ylabel('log_odds')
plt.title('Plot of log odds')
plt.show()

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

#import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())
odds_data