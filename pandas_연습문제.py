import pandas as pd
import numpy as np

df = pd.read_csv('./data/grade.csv')
df.head()

# 1번 데이터 프레임의 정복 출력 후 데이터 타입 확인
df.info()

# 2번 중간 점수 85점 이상인 학생 필터링
df[df['midterm']>=85]

# 3번 기말점수 기준 내리차순 정렬 후 첫 5행 출력
df.sort_values("final",ascending=False).head()

# 4번 gener 기준 그룹화 후 그룹별 중간 기말 점수 평균 계산
df.groupby('gender')[['midterm','final']].mean()

# 5번

# 6번assignment의 최대, 최소값을 가지는 행을 출력하세요
df.head(10)
max_ass = df.iloc[np.where(df['assignment']==max(df['assignment']))[0],:]
min_ass = df.iloc[np.where(df['assignment']==min(df['assignment']))[0],:]
print(max_ass,'/n',min_ass)

# 10번 중간 기말 과제의 평균을 구하고 average열을 생성하세요
# 성별, 성적유형 별 평균 점수를 구하세요
df['average'] = df[["midterm","final","assignment"]].mean(axis=1)
answer_10 = df.groupby('gender')[['assignment','average','final','midterm']].mean()
type(answer_10)
answer_10
# 실제 문제 풀이 답안
# reset_index 사용하면 그룹별 첫 행만이 아닌 모든 행에 그룹 이름이 출력됨
pd.melt(df,
        id_vars=['gender'],
        var_name='variable',
        value_vars=["midterm",'final','assignment','average'],
        value_name='score').groupby(['gender','variable']).mean().reset_index()

# 강사님 풀이
df_1 = df.groupby('gender')[["midterm",'final','assignment','average']].mean()
df_1 # 이때 gender는 1번째 열이 아닌 index로 설정되어 있기 때문에 조정이 필요하다
# group by 인수로 as_index=False로 설정 // index가 0,1로 바뀐 것을 확인 가능
result = df.groupby('gender',as_index=False)[["midterm",'final','assignment','average']].mean()
result
result = pd.melt(result,id_vars='gender',var_name='variable',value_name='score')
result = result.sort_values(['gender','variable'])
result.reset_index()

# 11번 중간 기말 과제의 평균을 구하고 average열을 생성하세요
# 최대 평균 성적을 가진 학생의 이름과 평균 성적을 출력하세요
df
max_mean_idx = np.argmax(df['average'])
max_mean_idx

df.loc[max_mean_idx,['name','average']] # 답안

