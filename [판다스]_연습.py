import pandas as pd
# 1
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_1.csv')
df
df['합계'] = df['금액1'] + df['금액2']
df_group = df.groupby(['year','gender','지역코드'])['합계'].sum().reset_index()
type(df_group)
df_pivot = pd.pivot_table(
    df_group,
      index=['year','지역코드'],
      columns='gender',
      values='합계',
      fill_value=0
      )
df_pivot['절대값차이'] = abs(df_pivot[0]-df_pivot[1])
df_pivot
df_pivot['절대값차이'].sort_values(ascending=False)
maxidx = df_pivot['절대값차이'].idxmax()

# 2-1
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv')
df
df_crime = df.iloc[::2,:].set_index("연도").drop(columns="구분")
df_caught = df.iloc[1::2,:].set_index("연도").drop(columns="구분")

df_ratio = df_caught/df_crime
df_ratio
col_names = list(df_ratio.columns.str.extract(r'(범죄유형\d+)')[0])

df_melt = pd.melt(df_ratio.reset_index(),
        id_vars='연도',
        value_vars = col_names)
df_melt
df_melt = df_melt.loc[df_melt.value==1,]

caught_melt = pd.melt(
    df_caught.reset_index(),
    id_vars='연도',
    value_vars=col_names
    )

merged = pd.merge(caught_melt,df_melt,how='right',on=['연도','variable'])
merged['value_x'].sum()

# 2-2
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv')
df
df_crime = df.iloc[::2,:].set_index("연도").drop(columns="구분")
df_caught = df.iloc[1::2,:].set_index("연도").drop(columns="구분")
df_ratio = df_caught/df_crime
max_idx = df_ratio.max(axis=1)

def is_max_col(col):
    return col == max_idx

mask = df_ratio.apply(is_max_col,axis=0)
df_1 = df_caught[mask].fillna(0)
df_1.sum().sum()

# 3
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_3.csv')
df.isnull().sum()
good_mean = df['평균만족도'].mean()
df['평균만족도'].fillna(good_mean,inplace=True)
df_group = df.groupby(['부서','등급'])['근속연수'].mean().reset_index()
df_group['근속연수'] = np.floor(df_group['근속연수'])
df_group

df_merge = pd.merge(df,df_group,how='left',on=['부서','등급'])
df_merge.rename(columns={'근속연수_y':'평균근속연수','근속연수_x':'근속연수'},inplace=True)
df_merge
df_merge['근속연수'] = np.where(df_merge['근속연수'].isna(),df_merge['평균근속연수'],df_merge['근속연수'])
df_merge[(df['부서']=='HR')&(df_merge['등급']=='A')]['근속연수'].mean()

# 4
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_1.csv')
df_group = df.groupby('대륙')['맥주소비량'].mean().reset_index()
sa = df_group['대륙'][df_group['맥주소비량'].idxmax()]

df_sa = df[df['대륙']==sa]
sa_group = df_sa.groupby('국가')['맥주소비량'].sum().reset_index()
sa_group.sort_values('맥주소비량',ascending=False,inplace=True)
country = sa_group['국가'][4]
nums = df[df['국가']==country]['맥주소비량'].mean()
nums = round(nums,0)
nums.astype(int)

# 5
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_2.csv')
df_group = df.groupby('국가').sum().reset_index()
df_group['총합계'] = df_group.iloc[:,1:].sum(axis=1)
df_group['관광객비율'] = df_group['관광'] / df_group['총합계']
df_group = df_group.sort_values('관광객비율',ascending=False)
df_group.iloc[1,:]

# 7
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_1.csv')
df.info()
df.iloc[:,3:7] = df.iloc[:,3:7].astype(str)
df['신고시간'] = df['신고일자']+df['신고시각']
df['처리시간'] = df['처리일자']+df['처리시각']
df['신고시간'] = pd.to_datetime(df['신고시간'],format='%Y%m%d%H%M%S')
df['처리시간'] = pd.to_datetime(df['처리시간'],format='%Y%m%d%H%M%S')
df['시간차이'] = ((df['처리시간']-df['신고시간']).dt.total_seconds())/3600
df_group = df.groupby('공장명')['시간차이'].mean().reset_index()
df_group.sort_values('시간차이')

# 8
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_2.csv')
df['STATION_ADDR1'].str.extract(r'([가-힣]+구)')

# 9
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_3.csv')
df[['년도','월']] = df['기간'].str.extract(r'(\d+)년_(\d+)월')
df['년월'] = df['년도'] + df['월']
df['분기'] = pd.to_datetime(df['년월'],format='%Y%m').astype('period[Q]')
df.columns.str.startswith('제품')
df['총판매량'] = df.loc[:,df.columns.str.startswith('제품')].sum(axis=1)
df_group = df.groupby('분기')['총판매량'].mean().reset_index()
df_group.sort_values('총판매량',ascending=False,inplace=True)