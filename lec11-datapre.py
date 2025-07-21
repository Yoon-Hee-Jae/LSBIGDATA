import pandas as pd
import numpy as np

data = {
    'date': ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08', '2021-04-01 14:15:16'],
    'value': [100, 201, 302, 404]
}
df = pd.DataFrame(data)

print(df.info())
df['date'] = pd.to_datetime(df['date'])
df.info()

pd.to_datetime('02-02-2024')
pd.to_datetime('02-2024-01',format = '%m-%Y-%d')
# Y: 4자리 년도
# y: 2자리 년도
# m: 월 정보
# M: 분 정보

df['date'].dt.month
df['date'].dt.year
df['date'].dt.day

current_date = pd.to_datetime('2025-07-21')
(current_date - df['date']).dt.days

date_range= pd.date_range(start='2021-01-01',end='2021-12-10',freq='D')
date_range

df['date2'] = pd.to_datetime(
    dict(
        year=df['date'].dt.year,
        month=df['date'].dt.month,
        day=df['date'].dt.day
    )
)