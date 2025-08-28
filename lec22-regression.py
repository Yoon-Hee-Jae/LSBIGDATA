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

# F-검정
import statsmodels.api as sm
from statsmodels.formula.api import ols
model1 = ols('Petal_Length ~ Petal_Width', data=iris).fit() #mod1
model2 = ols('Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width',

data=iris).fit() #mod2
table = sm.stats.anova_lm(model1, model2) #anova
print(table)

# 잔차 등분산성 검정
import scipy.stats as stats
residuals = model2.resid
fitted_values = model2.fittedvalues
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)

plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt); # QQPLOT
plt.show()

model = smf.ols("Petal_Length ~ Petal_Width + C(species)", data=iris).fit()
print(model.summary())

# 예측용 데이터프레임
new_data = pd.DataFrame({
    "Petal_Width":[0.5],
    "species":["virginica"]

})

predictions = model.predict(new_data)

# 연습 문제
import pandas as pd
import numpy as np
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())
np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)

# 1. train_index 를 사용하여 펭귄 데이터에서 인덱스에 대응하는 표본들을 뽑아서 
# train_data를 만드세요. (단, 결측치가 있는 경우 제거)
train_data = penguins.loc[train_index,:]
train_data.info()
train_data.dropna(inplace=True)

#2. train_data의 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)를 
# 사용하여 산점도를 그려보세요.
sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", hue="species", data=train_data)

# 3. 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)의 상관계수를 구하고, 
# 두 변수 사이에 유의미한 상관성이 존재하는지 검정해보세요.
x = train_data['bill_depth_mm'].values
y = train_data['bill_length_mm'].values
x_mean = np.mean(x)
y_mean = np.mean(y)
upper = np.sum((x-x_mean)*(y-y_mean))
lower_left = np.sqrt(np.sum((x-x_mean)**2))
lower_right = np.sqrt(np.sum((y-y_mean)**2))
correlation_coef = upper / (lower_left*lower_right) #-0.24938519717051552
# 상관 계수 검정
corr_coeff, p_value = stats.pearsonr(x, y)
print(f"피어슨 상관계수 (r): {corr_coeff:.4f}")
print(f"p-value: {p_value:.4f}") # 0.05보다 작으므로 귀무가설을 기각하고 상관관계 있음

# 4. 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)를 사용하여 설명하는
#  회귀 모델을 적합시킨 후 2번의 산점도에 회귀 직선을 나타내 보세요. (모델 1)
model1 = smf.ols("bill_length_mm ~ bill_depth_mm", data=train_data).fit()
model1.params[0]
intercept = model1.params[0]
slope = model1.params[1]
plt.figure(figsize=(8,6))
sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", hue="species", data=train_data)
# 회귀직선
x_vals = np.linspace(train_data["bill_depth_mm"].min(), train_data["bill_depth_mm"].max(), 100)
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="회귀직선")

# 5. 적합된 회귀 모델이 통계적으로 유의한지 판단해보세요.
print(model.summary())
# anova 검정
sm.stats.anova_lm(model1)
# p-value 값이 0.05 보다 작으므로 모든 회귀계수가 0이라는 귀무가설을 기각
# 통계적으로 유의함

# 6. 𝑅^2 값을 구한 후 의미를 해석해 보세요.
model1.rsquared # 0.062
# 부리길이의 변동성을 약 6.2% 정도 설명한다

# 7. 적합된 회귀 모델의 계수를 해석해 보세요.
model1.params
# 부리깊이가 1단위 증가할 때 부리길이는 -0.7만큼 감소한다

# 8. 1번에서 적합한 회귀 모델에 새로운 변수 (종 - species) 변수를 추가하려고 합니다. 
# 성별 변수 정보를 사용하여 점 색깔을 다르게 시각화 한 후 적합된 모델의 회귀 직선을 시각화 해보세요. (모델2)
model2 = smf.ols("bill_length_mm ~ bill_depth_mm + C(species)", data=train_data).fit()
model2.summary()
# 산점도 시각화 (종에 따라 색깔 구분)
plt.figure(figsize=(8,6))
sns.scatterplot(x="bill_depth_mm", y="bill_length_mm", hue="species", data=train_data)
species_list = train_data["species"].unique()
# 회귀 직선 시각화
x_vals = np.linspace(train_data["bill_depth_mm"].min(), train_data["bill_depth_mm"].max(), 100)

for sp in species_list:
    intercept = model2.params['Intercept'] + model2.params.get(f"C(species)[T.{sp}]", 0)
    slope = model2.params['bill_depth_mm']
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, label=f"{sp} 회귀선")
    
plt.legend()
plt.xlabel("Bill Depth (mm)")
plt.ylabel("Bill Length (mm)")
plt.title("회귀 모델 2: Bill Length ~ Bill Depth + Species")
plt.show()


# 9. 종 변수가 새로 추가된 모델 2가 모델 1 보다 더 좋은 모델이라는 근거를 제시하세요
sm.stats.anova_lm(model1, model2)

# 10.모델 2의 계수에 대한 검정과 그 의미를 해석해 보세요.
sm.stats.anova_lm(model2)

# 모델 2 에 잔차 그래프를 그리고, 회귀모델 가정을 만족하는지 검증을 수행해주세요.
residuals = model2.resid
fitted_values = model2.fittedvalues

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals);

plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt);
plt.show()
from statsmodels.stats.diagnostic import het_breuschpagan
bptest = het_breuschpagan(model2.resid, model2.model.exog)
print('BP-test statistics: ', bptest[0])
print('p-value: ', bptest[1])
