import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head(2))

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 4, random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)

df['cluster_label'] = labels
print(df.head(2))

# 펭귄 분류

from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna() # dataframe의 함수인 dropna를 사용
penguins
penguins.shape 
df_0 = penguins
df = penguins[['bill_length_mm','bill_depth_mm']]

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 3, random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)

df['cluster_label'] = labels
print(df.head(2))
df['cluster_label'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

# bill_length_mm, bill_depth_mm, cluster 컬럼이 있다고 가정
# cluster는 KMeans 같은 군집화 결과 (0, 1, 2 ...)
# k-means 군집화 결과 시각화
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, 
    x="bill_length_mm", 
    y="bill_depth_mm", 
    hue="cluster_label",   # 군집별 색깔 구분
    palette="Set2",  # 색상 팔레트
    s=80,            # 점 크기
    edgecolor="k"    # 점 테두리
)

plt.title("Bill Length vs Bill Depth (Clustered)")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.legend(title="Cluster")
plt.show()

# 실제 종 구분 시각화
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df_0,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="species",   # 종별 색상
    palette="Set1",
    s=80,
    edgecolor="k"
)
plt.title("Bill Length vs Bill Depth (Species)")
plt.show()