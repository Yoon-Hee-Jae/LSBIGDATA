import numpy as np
import pandas as pd

# 1번
df = pd.read_csv('./data/problem1.csv')
df.head()
#1
df.shape
#2
df['퇴거여부'].value_counts()
# 3
df_group = df.groupby(['아파트 이름','성별'])['보증금(원)'].mean().reset_index()
df_pivot = df_group.pivot_table(index="아파트 이름",
                     columns='성별',
                     values='보증금(원)').reset_index()
df_pivot['평균보증금차이'] = abs(df_pivot['남']-df_pivot['여'])
df_pivot.sort_values('평균보증금차이',ascending=False)
# 4
df.info()
df.groupby('아파트 이름')['월세(원)'].max().reset_index().sort_values('월세(원)',ascending=False)
# 5
df.head()
df.groupby('층')['거주자 수'].mean().reset_index().sort_values('거주자 수',ascending=False)
# 6
valid_df = df[df['계약구분']=='유효'].reset_index(drop=True)
valid_df = valid_df[df['재계약횟수']>=5].reset_index(drop=True)
valid_df.groupby('평형대')['나이'].mean().reset_index()
# 7
df.info()
df.groupby('계약자고유번호')['거주연도'].max().reset_index().shape
# 8 
df.shape
df.isnull().sum()
df_1 = df.copy()
df_1.shape
df_1 = df_1[(~df_1['계약구분'].isna())&(~df_1['아파트 평점'].isna())] # true는 값이 있는 곳
df_1.shape
# 9 
df_2 = df[df['퇴거여부']=='미퇴거']
(~df_2['퇴거연도'].isna()).sum()
# 10
df.info()
df_10 =df.copy()
median_num = df_10['재계약횟수'].median()
df_10['중앙값구분'] = np.where(df_10['재계약횟수']>=median_num,"높음","낮음")
df_10_group = df_10.groupby('중앙값구분')['거주개월'].mean().reset_index()
round(df_10_group['거주개월'],2)
# 11
df_10.groupby('중앙값구분')['나이'].median()
# 12 나중에 풀기
high = df_10[df_10['중앙값구분']=='높음']
high['성별'].value_counts()[0] / len(high) # 높음 그룹 여자 비율
high['성별'].value_counts()[1] / len(high) # 높음 그룹 남자 비율

low = df_10[df_10['중앙값구분']=='낮음']
low['성별'].value_counts()[0] / len(low) # 높음 그룹 여자 비율
low['성별'].value_counts()[1] / len(low) # 높음 그룹 남자 비율

# 13 
df_13 = pd.read_csv('./data/problem2.csv')
df_13.isnull().sum()
df_14 = df_13.copy() # df_14 결측치 제거
df_14.dropna(inplace=True)
df_14.isnull().sum()
df_14.head()
df_14.info()
df_14.shape
df_14.info() #a2_1 a4_1 a7_1
df_14['a2_1'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
df_14['a4_1'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
df_14['a7_1'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
df_14.iloc[~[1],:] # 188 - 8

# 14
df_15 = pd.read_csv('./data/problem2_1.csv')
score_cols = df_15.columns.str.startswith('b')
score_cols[0] = True
df_15_score = df_15.loc[:,score_cols]
df_15_score.iloc[:,1:].mean().max()

# 15
act_cols = df_15.columns.str.startswith('a')
act_cols[0] = True
act_cols
df_16 = df_15.loc[:,act_cols] # df_16 행동 데이터프레임
first_cols = df_16.columns.str.endswith('1')
first_cols[0] = True
second_cols = df_16.columns.str.endswith('2')
second_cols[0] = True
first_player = df_16.loc[:,first_cols]
second_player = df_16.loc[:,second_cols]
round(abs(first_player.iloc[:,1:].mean().mean() - second_player.iloc[:,1:].mean().mean()),2)

# 16 나중에 풀기
first_player['1_평균행동'] = first_player.iloc[:,1:].mean(axis=1)
second_player['2_평균행동'] = second_player.iloc[:,1:].mean(axis=1)
df_15_score['평균득점'] = df_15_score.iloc[:,1:].mean(axis=1)
merge_1 = pd.merge(first_player.iloc[:,[0,10]], second_player.iloc[:,[0,10]], how='inner',on='game_id')
merge_1 = pd.merge(merge_1,df_15_score.iloc[:,[0,10]], how='inner',on='game_id')
merge_1.iloc[:,1:] = round(merge_1.iloc[:,1:],2)
merge_1['diff'] = merge_1['1_평균행동']+merge_1['2_평균행동']-merge_1['평균득점']
merge_1['diff'].max()
merge_1[merge_1['diff']==11.89]

# 17
df_17 = pd.read_csv('./data/problem2_2.csv')
df_17
one_good = df_17.loc[(df_17['ining1_move'].isin([1,2,3,6,8])),:]
one_good = one_good.loc[~(one_good['ining2_move']==4)]
one_good['ining2_move'].unique()
one_good.shape

# 18
df_18 = pd.read_csv('./data/problem2_3.csv')
df_18
df_18['score_index'] = np.where(df_18['score']>0,1,0)
df18_group = df_18.groupby('score_index')['ining2_move'].mean().reset_index()
round(df18_group['ining2_move'],2)

# 19
max_score = df_18['score'].max()
new_df = df_18[df_18['score']==max_score]
new_df.groupby(['ining1_move','ining2_move'])['score'].max().reset_index()

# 20
df_20 = pd.read_csv('./data/problem2_3.csv')
group_20 = df_20.groupby(['ining1_move','ining2_move'])['score'].mean().reset_index()
group_20.sort_values('score',ascending=False,inplace=True)
group_20
# 3 8
a1 = df_20[(df_20['ining1_move']==3)&(df_20['ining2_move']==8)].shape[0] 
# 6 8
a2 = df_20[(df_20['ining1_move']==6)&(df_20['ining2_move']==8)].shape[0] 
# 3 6
a3 = df_20[(df_20['ining1_move']==3)&(df_20['ining2_move']==6)].shape[0] 
# 3 1
a4 = df_20[(df_20['ining1_move']==3)&(df_20['ining2_move']==1)].shape[0] 
# 2 2
a5 = df_20[(df_20['ining1_move']==2)&(df_20['ining2_move']==2)].shape[0] 

a1 + a2 + a3 + a4 + a5
