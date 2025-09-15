import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 데이터 불러오기 및 전처리
penguins = load_penguins()
penguins = penguins.dropna()
df = penguins[["bill_length_mm", "bill_depth_mm"]]

# 표준화
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), 
                        columns=numeric_data.columns)

# 2. 실루엣 계수 계산
scores = []
K_range = range(2, 6)   # silhouette_score는 k=1에서 계산 불가능 (클러스터가 1개면 분리도 정의 불가)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=1)
    labels = kmeans.fit_predict(df_trans)
    score = silhouette_score(df_trans, labels)
    scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

# 3. 시각화
plt.plot(list(K_range), scores, marker='o')
plt.title("실루엣 계수에 따른 최적 군집 수 탐색")
plt.xlabel("군집 수 (k)")
plt.ylabel("평균 실루엣 계수")
plt.xticks(list(K_range))
plt.grid(True)
plt.show()