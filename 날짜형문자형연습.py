import pandas as pd
df = pd.read_csv('./data/bike_data.csv')
df.head()
df.shape
df = df.astype({'datetime' : 'datetime64[ns]', 'weather' : 'int64', 
                'season' : 'object', 'workingday' : 'object', 
                'holiday' : 'object'})

# 1번
df1 = df[df['season']==1]
df1
df1['hour'] = df1['datetime'].dt.hour
df1
max(df1.groupby('hour')['count'].sum()) # 13시 1417개

# 2번
df.groupby('season')['count'].mean()

# 3번 특정 달(month) 동안의 총 대여량(count)을 구하시오.
df['month'] = df['datetime'].dt.month
df_jan = df[df['month'] == 1]
df_jan['count'].sum()
df['datetime']
# 4번 가장 대여량이 많은 날짜를 구하시오.
df['date'] = df['datetime'].dt.date
a = df.groupby('date',as_index=False)[['count']].sum()
a

import numpy as np
a
max_idx = np.argmax(a['count'])
a.iloc[max_idx,:]
a.iloc[198,:]

# 5번 시간대(hour)별 평균 대여량(count)을 구하시오.
df['hour'] = df['datetime'].dt.hour
df.groupby('hour',as_index=False)[['count']].mean()

# 6번 특정 요일(weekday) 동안의 총 대여량(count)을 구하시오.
df['weekday'] = df['datetime'].dt.weekday
df.groupby('weekday',as_index=False)['count'].sum()

# 7번
df
df_melt = pd.melt(df,id_vars=['datetime','season',],
        value_vars=['casual','registered'],
        var_name='user_type',
        value_name='user_count')

# 8번
df_melt.groupby(['season','user_type'])[['user_count']].mean().reset_index()

# 9번 로그 칼럼에서 숫자 정보만 추출하시오.
pd.set_option('display.max_columns', None) # 전체 칼럼 정보 프린트 옵션
df = pd.read_csv('./data/logdata.csv')
print(df.head(2))
df['숫자'] = df['로그'].str.findall(r'\d+')
# 10번 로그 칼럼에서 모든 시간 정보를 추출하시오.
df['시간'] = df['로그'].str.extract(r'([\d]{2}:[\d]{2}:[\d]{2})')
df
# 11번 로그 칼럼에서 한글 정보만 추출하시오.
df['User'] = df['로그'].str.extract(r'([가-힣]+)')
df['User']
# 12번 로그 칼럼에서 특수 문자를 제거하시오.
df['제거'] = df['로그'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)

# 13번 로그 칼럼에서 유저, Amount 값을 추출한 후 각 유저별 Amount의 평균값을 계산하시오.
df
df['숫자']
def get_last_number(lst):
    if lst:  # 리스트가 비어있지 않으면
        return lst[-1]
    else:
        return None

df['amount'] = df['로그'].str.findall(r'\d+').apply(get_last_number)
df['amount']
type(df['amount'][0])
df['amount'] = df['amount'].astype(int)
type(df['amount'][0])

ans = df.groupby('User',as_index=False)[['amount']].mean()
ans.sort_values('User')



