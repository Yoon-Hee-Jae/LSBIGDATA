import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head(2))

from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))