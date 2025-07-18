import seaborn as sns
import pandas as pd
df = sns.load_dataset('titanic')
df = df.dropna()
df.head()
# 1번
ans = df.groupby('sex',as_index=False)['age'].sum()
ans
abs(ans.iloc[0,1] - ans.iloc[1,1])

#2번

df.head()
df40 = df[ (df['age']>=40) & (df['age']<50) ]
mean_num = df40.groupby('sex',as_index=False)['fare'].mean().iloc[1,1]

# 3번
import numpy as np
# a = np.array([[1,2],[3,4]])
x = np.array([[2,4],
              [1,7],
              [7,8]])
x
y = np.array([[10],
              [5],
              [15]])
y
# np.linalg.inv()
H = x @ np.linalg.inv(x.T @ x) @ x.T
H @ y

# 4번
np.random.seed(2025)
array_2d = np.random.randint(1, 13, 200).reshape((50, 4))
array_2d[:4,:] # 50행 4열
>>> array([[ 3,  9,  4,  4],
>>>        [ 1,  7,  9,  6],
>>>        [11,  2,  9,  6],
>>>        [ 8,  6,  5,  1]], dtype=int32)
# 3 + 9 + 4 + 4
23/4
# 4번
max(np.mean(array_2d,axis=1))
# 5번
np.sum(array_2d.max(axis=1) - array_2d.min(axis=1))


