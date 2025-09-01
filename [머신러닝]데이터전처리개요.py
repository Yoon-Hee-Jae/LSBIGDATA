import pandas as pd 
import numpy as np

dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')
dat.info()
dat.head()
y = dat.grade
X = dat.drop(['grade'], axis = 1)
X['school'].value_counts()
# 데이터 분할
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,
                   random_state = 0,
                   shuffle = True,
                   stratify = None
                   )

print('trainX shape : ',  train_X.shape)
print('trainy shape : ',  train_y.shape)
print('testX shape : ',  test_X.shape)
print('testy shape : ',  test_y.shape)

# train test y 히스토그렘
import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout()
plt.show()

# train test y의 분포가 같은지 확인하는 검정 = two sample ks 검정
from scipy.stats import ks_2samp

# train_y와 test_y 분포 비교
stat, p_value = ks_2samp(train_y, test_y)

print(f"KS statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value > 0.05:
    print("Train과 Test의 분포가 비슷합니다. ✅")
else:
    print("Train과 Test의 분포가 다릅니다. ⚠️")

# 층화 추출을 통해 train/test 분할
train_X, test_X, train_y, test_y = train_test_split(
                    X, 
                    y, 
                    test_size = 0.2,                                                        stratify = X['school'], 
                    random_state = 0)
# 분포 비교
import matplotlib.pyplot as plt
fig, axs = plt.subplots(nrows=1, ncols=2)
train_y.hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('histogram of train y')
test_y.hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout(); 
plt.show();

# school 분포 비교
# train/test 'school' 값의 빈도 계산
train_counts = train_X['school'].value_counts().sort_index()
test_counts = test_X['school'].value_counts().sort_index()

# figure 생성
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

# train bar plot
axs[0].bar(train_counts.index, train_counts.values, color='blue', alpha=0.7)
axs[0].set_title('Train school distribution')
axs[0].set_xlabel('School')
axs[0].set_ylabel('Count')

# test bar plot
axs[1].bar(test_counts.index, test_counts.values, color='red', alpha=0.7)
axs[1].set_title('Test school distribution')
axs[1].set_xlabel('School')
axs[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# 결측치 처리 - 평균대치법
dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')
print(dat.isna().sum(axis = 0))

y = dat.grade
X = dat.drop(['grade'], axis = 1)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                   X,  
                   y, 
                   test_size = 0.2,                     
                   random_state = 0, 
                   shuffle = True, 
                   stratify = None 
                   )

from sklearn.impute import SimpleImputer
train_X1 = train_X.copy()
test_X1 = test_X.copy()
imputer_mean = SimpleImputer(strategy = 'mean')
train_X1['goout'] = imputer_mean.fit_transform(train_X1[['goout']])
test_X1['goout'] = imputer_mean.transform(test_X1[['goout']])
print('학습 데이터 goout 변수 결측치 확인 :', train_X1['goout'].isna().sum())

# 결츠기 처리 - knn
from sklearn.impute import KNNImputer
train_X5 = train_X.copy()
test_X5 = test_X.copy()
train_X5_num = train_X5.select_dtypes('number')
test_X5_num = test_X5.select_dtypes('number')
train_X5_cat = train_X5.select_dtypes('object')
test_X5_cat = test_X5.select_dtypes('object')

knnimputer = KNNImputer(n_neighbors = 5)
train_X5_num_imputed = knnimputer.fit_transform(train_X5_num)
test_X5_num_imputed = knnimputer.transform(test_X5_num)
                       
train_X5_num_imputed = pd.DataFrame(train_X5_num_imputed, 
                                    columns=train_X5_num.columns, 
                                    index = train_X5.index)
test_X5_num_imputed = pd.DataFrame(test_X5_num_imputed, 
                                   columns=test_X5_num.columns, 
                                   index = test_X5.index)
train_X5 = pd.concat([train_X5_cat, train_X5_num_imputed], axis = 1)
test_X5 = pd.concat([test_X5_cat, test_X5_num_imputed], axis = 1)
print('학습 데이터 goout 변수 결측치 확인 :', train_X5['goout'].isna().sum())

# 출력결과를 numpy가 아닌 pandas로 바로 출력되도록해줌 .set_output
knnimputer2 = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')
train_X5_num_imputed2 = knnimputer2.fit_transform(train_X5_num)
test_X5_num_imputed2 = knnimputer2.transform(test_X5_num)
# 판다스 데이터프레임 출력 
print(train_X5_num_imputed2.head())

# 변수 변환 방법
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
import warnings
np.warnings = warnings

bike_data = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv")

import matplotlib.pyplot as plt
bike_data['count'].hist();
plt.show();

box_tr = PowerTransformer(method = 'box-cox')
bike_data['count_boxcox'] = box_tr.fit_transform(
    bike_data[['count']])
print('lambda : ', box_tr.lambdas_)

bike_data['count_boxcox'].hist();
plt.show();
