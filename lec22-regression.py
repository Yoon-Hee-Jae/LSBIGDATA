import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. Iris 데이터 로드
df_iris = load_iris()
# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시
# 3. 타겟(클래스) 추가
iris["species"] = df_iris.target
# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["species"] = iris["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})


import statsmodels.api as sm
import statsmodels.formula.api as smf
# ols = 회귀분석
model = smf.ols("Petal_Length ~ Petal_Width", data = iris).fit()
print(model.summary())

# 회귀 계수 가져오기
intercept, slope = model.params
print(f'절편: {intercept:.4f}, 기울기: {slope:.4f}')

# -----------------------------
# 시각화 (산점도 + 회귀직선)
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="Petal_Width", y="Petal_Length", hue="species", data=iris)

# 회귀 직선 그리기
x_vals = np.linspace(iris["Petal_Width"].min(), iris["Petal_Width"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="회귀직선")

plt.xlabel("Petal_Width")
plt.ylabel("Petal_Length")
plt.title("단순회귀: Petal_Length ~ Petal_Width")
plt.legend()
plt.show()


##################################################################
# 변수가 2개

# 1. Iris 데이터 로드
df_iris = load_iris()
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
iris["species"] = df_iris.target
iris["species"] = iris["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# -----------------------------
# 다중 회귀 (독립변수 2개)
# -----------------------------
model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length", data=iris).fit()
print(model.summary())
model.tvalues
model.conf_int() # 신뢰구간
# 회귀 계수 가져오기
intercept = model.params["Intercept"]
coef_pw = model.params["Petal_Width"]
coef_sl = model.params["Sepal_Length"]

print(f'절편: {intercept:.4f}, Petal_Width 계수: {coef_pw:.4f}, Sepal_Length 계수: {coef_sl:.4f}')

# -----------------------------
# 3D 시각화
# -----------------------------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

# 실제 데이터 산점도
ax.scatter(iris["Petal_Width"], iris["Sepal_Length"], iris["Petal_Length"],
           c=iris["species"].map({"setosa":"blue", "versicolor":"green", "virginica":"red"}),
           alpha=0.6, label="데이터")

# 회귀평면 생성
x_surf, y_surf = np.meshgrid(
    np.linspace(iris["Petal_Width"].min(), iris["Petal_Width"].max(), 20),
    np.linspace(iris["Sepal_Length"].min(), iris["Sepal_Length"].max(), 20)
)
z_surf = intercept + coef_pw * x_surf + coef_sl * y_surf

ax.plot_surface(x_surf, y_surf, z_surf, color="orange", alpha=0.3)

# 축 라벨
ax.set_xlabel("Petal_Width")
ax.set_ylabel("Sepal_Length")
ax.set_zlabel("Petal_Length")
ax.set_title("다중회귀: Petal_Length ~ Petal_Width + Sepal_Length")

plt.show()


# 범주형 변수가 포함될 경우
model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Length + C(species)", data=iris).fit()
print(model.summary())
