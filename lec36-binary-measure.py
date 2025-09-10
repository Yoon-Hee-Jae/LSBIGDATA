import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
print(df.head(2))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), 
                                                    df['target'], 
                                                    test_size=0.3, 
                                                    random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000, random_state = 0)
model.fit(X_train, y_train)

y_prob_org = model.predict_proba(X_test)
print(pd.DataFrame(y_prob_org[:4].round(3)))

y_prob_org.shape
y_test.shape

y_pred = model.predict(X_test)
print(pd.DataFrame(y_pred, columns = ['pred']).head(4))

# np.set_printoptions(precision=6, suppress=True)

y_pred_ths = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print('값이 같은지 확인:', np.array_equal(y_pred_ths, y_pred))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred);
isp = ConfusionMatrixDisplay(confusion_matrix=cm);
isp.plot(cmap=plt.cm.Blues);
plt.show()
type(y_test)
# 과제
a = pd.Series([1,0,0,1,1,0,0,0,1])

# 예측 확률
b = pd.Series([0.87,0.3,0.006,0.996,0.7,0.3,0.4,0.2,0.8])
# 0.25일땐
# 1, 1, 0, 1, 1, 1, 1, 0 , 1
# recall = 4 / 4
# precision = 4 / 4 + 3
# f1 = 2 * 4/7 / 11/7 = 8 / 11
# 확률 → 0/1 라벨 변환 (threshold=0.5)
b_pred = (b >= 0.5).astype(int)

# 혼동행렬 계산
cm = confusion_matrix(a, b_pred)

# 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
# recall = 











