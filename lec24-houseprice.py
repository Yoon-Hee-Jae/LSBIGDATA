import pandas as pd
import numpy as np

train_data = pd.read_csv('./data/house_prediction/train.csv')
test_data = pd.read_csv('./data/house_prediction/test.csv')

train_data.info()
test_data.info()

train_data['SalePrice']
