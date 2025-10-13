import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
# 1. 불균형 데이터 생성 (정상: 0, 이상: 1)
X, y = make_classification(n_samples=1000, n_features=5, 
                           n_informative=2, n_redundant=0, 
                           weights=[0.95, 0.05], # 95% : 5%
                           random_state=42)
print("클래스 분포:", pd.Series(y).value_counts())
# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 3. 로지스틱 회귀 모형 적합
model = LogisticRegression()
model.fit(X_train, y_train)
# 4. 예측
y_pred = model.predict(X_test)
# 5. 평가
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("혼동 행렬:\n", cm)
print(f"정확도(Accuracy): {acc:.3f}")
print(f"민감도(Recall): {recall:.3f}")



# pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
ros = RandomOverSampler(random_state=42)
X_res_over, y_res_over = ros.fit_resample(X_train, y_train)
model_over = LogisticRegression()
model_over.fit(X_res_over, y_res_over)
y_pred_over = model_over.predict(X_test)
print("\n[RandomOverSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_over))
print("정확도:", accuracy_score(y_test, y_pred_over))
print("민감도:", recall_score(y_test, y_pred_over))
# ----------------------------
# [언더샘플링] RandomUnderSampler 적용
# ----------------------------
under = RandomUnderSampler(random_state=42)
X_res_under, y_res_under = under.fit_resample(X_train, y_train)
model_under = LogisticRegression()
model_under.fit(X_res_under, y_res_under)
y_pred_under = model_under.predict(X_test)
print("\n[RandomUnderSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_under))
print("정확도:", accuracy_score(y_test, y_pred_under))
print("민감도:", recall_score(y_test, y_pred_under))


#파이프라인 활용해서 데이터 누수 방지하기
# 검정 데이터셋 , 테스트 데이터셋에는 오버/언더 샘플링 하면 안되기 때문

# imblearn pipeline을 불러와야함
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pipe_over_rf = Pipeline(
    [
        ("over", RandomOverSampler(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
)
param_over_rf = {"clf__n_estimators": [200, 400], "clf__min_samples_split": [2, 5]}
gs_over_rf = GridSearchCV(
    estimator=pipe_over_rf,
    param_grid=param_over_rf,
    scoring="f1",
    cv=kf,
    verbose=0,
)
gs_over_rf.fit(X_train, y_train)
print("[GridSearch - RF OverSampling] best params:", gs_over_rf.best_params_)
print("best CV F1:", gs_over_rf.best_score_)
y_pred_over_rf = gs_over_rf.best_estimator_.predict(X_test)
print("\n[RandomUnderSampler 결과]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_over_rf))
print("정확도:", accuracy_score(y_test, y_pred_over_rf))
print("민감도:", recall_score(y_test, y_pred_over_rf))


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, TunedThresholdClassifierCV
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
# -------------------------------
# 0) 기준선: 기본 0.5 임계값 로지스틱
# -------------------------------
logit_base = LogisticRegression(max_iter=1000)
logit_base.fit(X_train, y_train)
y_pred_base = logit_base.predict(X_test)
print("\n[Baseline Logistic (threshold=0.5)]")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_base))
print(f"Accuracy : {accuracy_score(y_test, y_pred_base):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred_base):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_base, zero_division=0):.3f}")
print(f"F1       : {f1_score(y_test, y_pred_base):.3f}")
# -------------------------------
# 1) 임계값 튜닝 (목표 지표: Recall)
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer_recall_pos1 = make_scorer(recall_score, pos_label=1)
logit_tt = TunedThresholdClassifierCV(
    estimator=LogisticRegression(max_iter=1000),
    scoring=scorer_recall_pos1,  # 재현율(양성=1) 극대화
    cv=kf,
)
logit_tt.fit(X_train, y_train)
y_pred_tt = ldogit_tt.predict(X_test)
best_thr = getattr(logit_tt, "best_threshold_", None)
print(f"\n[TunedThreshold] 최적 threshold (by Recall): {best_thr}")
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_tt))
print(f"Accuracy : {accuracy_score(y_test, y_pred_tt):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred_tt):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_tt, zero_division=0):.3f}")
print(f"F1       : {f1_score(y_test, y_pred_tt):.3f}")