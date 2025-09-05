import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv') 
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')


train.info()
train["grade"]

train_X = train.drop(['grade'], axis = 1)
train_y = train['grade']
test_X = test.drop(['grade'], axis = 1)
test_y = test['grade']


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline

num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()
cat_preprocess = make_pipeline(
    #SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown="ignore",
                  sparse_output=False)
)
num_preprocess = make_pipeline(
    SimpleImputer(strategy="mean"), 
    StandardScaler()
)
preprocess = ColumnTransformer(
    [("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)]
)


from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
full_pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("regressor", KNeighborsRegressor())
    ]
)


KNeighborsRegressor().get_params()

import numpy as np
knn_param = {'regressor__n_neighbors': np.arange(5, 10, 1)}

from sklearn.model_selection import GridSearchCV
knn_search = GridSearchCV(full_pipe,
                          param_grid = knn_param,
                          cv = 3, 
                          scoring = 'neg_mean_squared_error')
knn_search.fit(train_X, train_y)
knn_search.best_params_

np.floor(knn_search.predict(test_X))