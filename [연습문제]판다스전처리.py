import pandas as pd
import numpy as np


#1. 각 연도, 성별, 지역코드별 총 대출액 합계의 절대값 차이를 구하시오. 해당 데이터를 활용하여 성별의 절대값 차이가 가장 큰 지역코드를 구하시오.
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_1.csv")
df.info()
df['합계액'] = df[['금액1','금액2']].sum(axis=1)
df_group = df.groupby(['gender','지역코드'],as_index=False)['합계액'].sum()
df_group[df_group['gender']==0]
df_group[df_group['gender']==1]

df_pivot = df_group.pivot_table(index='지역코드',
                                columns='gender',
                                values='합계액')
df_pivot.fillna(0,inplace=True)
df_pivot['정답'] = (df_pivot[0]-df_pivot[1]).abs()
df_pivot.sort_values('정답',ascending=False)

#2. 각 연도별 최대 검거율을 가진 범죄유형을 찾아서 해당 연도 및 유형의 검거건수들의 총합을 구하시오.
# (검거율 = 검거건수 / 발생건수)
df =pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_2.csv")
df.info()
df.head()

df_crime = df[df['구분']=="발생건수"]
df_catch = df[df['구분']=="검거건수"]

df_crime = pd.melt(df_crime, 
        id_vars="연도",
        value_vars=['범죄유형1','범죄유형2','범죄유형3','범죄유형4','범죄유형5','범죄유형6','범죄유형7','범죄유형8','범죄유형9','범죄유형10'],
        var_name="범죄유형",
        value_name="건수")

df_crime.sort_values("연도",inplace=True)

df_catch = pd.melt(df_catch, 
        id_vars="연도",
        value_vars=['범죄유형1','범죄유형2','범죄유형3','범죄유형4','범죄유형5','범죄유형6','범죄유형7','범죄유형8','범죄유형9','범죄유형10'],
        var_name="범죄유형",
        value_name="건수")

df_catch.sort_values("연도",inplace=True)
df_catch

df_catch['검거율'] = df_catch['건수']/df_crime['건수']
df_catch

df_catch.groupby('연도',as_index=False)[['검거율']].max()

a = df_catch[df_catch['검거율']==1].groupby("연도",as_index=False)['건수'].sum()

a['건수'].sum()

# 3번
#  결측치 처리 
# ① 평균만족도 : 결측치는 평균만족도 컬럼의 전체 평균으로 채우시오.
# ② 근속연수 : 결측치는 각 부서와 등급별 평균 근속연수로 채우시오. (평균값의 소수점은 버림 처리)

#  조건에 따른 평균 계산 
# ③ A : 부서가 ’HR’이고 등급이 ’A’인 사람들의 평균 근속연수를 계산하시오.
# ④ B : 부서가 ’Sales’이고 등급이 ’B’인 사람들의 평균 교육참가횟수를 계산하시오.
# ⑤ A와 B를 더한 값을 구하시오.
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/9_1_3.csv")
df
good_mean = df['평균만족도'].mean()
df['평균만족도'].fillna(good_mean,inplace=True)
#
df_group = df.groupby(["부서","등급"],as_index=False)[['근속연수']].mean()
df_group['근속연수'] = df_group['근속연수'].astype(int)
df_group
for i in range(len(df)):
    if pd.isnull(df['근속연수'].iloc[i]) == True:
        df['근속연수'][i] = df_group['근속연수'][(df_group['부서']==df['부서'][i])&(df_group['등급']==df['등급'][i])]
df

A = df[(df['부서']=="HR")&(df['등급']=='A')]['근속연수'].mean()
B = df[(df['부서']=="Sales")&(df['등급']=='B')]['교육참가횟수'].mean()
A+B

# 4번 다음의 데이터는 대륙별 국가의 맥주소비량을 조사한 것이다
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_1.csv")
df
df.groupby('대륙',as_index=False)['맥주소비량'].mean().max()

# 4-2 이전 문제에서 구한 대륙에서 5번째로 맥주소비량이 많은 나라를 구하시오.
df_sa = df[df['대륙']=='SA']
df_sa.groupby('국가',as_index=False)['맥주소비량'].sum().sort_values('맥주소비량',ascending=False)

# 4-3 이전 문제에서 구한 나라의 평균 맥주소비량을 구하시오. (단, 소수점 첫째 자리에서 반올림하고, 정수형으로 표기)
ven_mean = df_sa[df_sa['국가']=='Venezuela']['맥주소비량'].mean()
ven_mean = round(ven_mean,1)
ven_mean = ven_mean.astype(int)
print(ven_mean)

# 5 다음의 데이터는 국가별로 방문객 유형을 조사한 것이다.
# 관광객비율 = 관광/합계(소수점 넷째 자리에서 반올림)
# 합계 = 관광 + 사무 + 공무 + 유학 + 기타
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_2.csv')
df_group = df.groupby('국가',as_index=False).sum()
df_group
df_group['합계'] = df_group.iloc[:,1:].sum(axis=1)
df_group
df_group['관광객비율'] = df_group['관광']/df_group['합계']
df_group
df_group = df_group.sort_values('관광객비율',ascending=False)
df_group

# 5-2 관광 수가 두번째로 높은 나라의 ‘공무’ 수의 평균을 구하시오. (단, 소수점 첫째 자리에서 반올림하고, 정수형으로 표기)
df_group2 = df.groupby('국가',as_index=False).agg({'관광':'sum','공무':'mean'})
df_group2['공무'] = df_group2['공무'].round(0).astype(int)
df_group2.sort_values('관광',ascending=False)

# 6CO(GT), NMHC(GT) 칼럼에 대해서 Min-Max 스케일러를 실행하고, 스케일링된 CO(GT), NMHC(GT) 칼럼의 표준편차를 구하시오. (소수점 셋째 자리에서 반올림)
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/8_1_3.csv")
df.info()

from sklearn.preprocessing import MinMaxScaler

# 스케일러 객체 생성
scaler = MinMaxScaler()

# 스케일링 대상 열만 선택
cols_to_scale = ['CO(GT)', 'NMHC(GT)']

# 스케일링 수행
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
round(df['CO(GT)'].std(), 2)
round(df['NMHC(GT)'].std(), 2)

# 7 각 제품보고서별 처리 시간(처리시각과 신고시각의 차이) 칼럼(초단위)을 생성 후 공장별 처리 시간의 평균을 산출하시오. 산출된 결과를 바탕으로 평균 처리 시간이 3번째로 적은 공장명을 구하시오.
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_1.csv')
df.info()
df['신고일시'] = pd.to_datetime(df['신고일시'])
df['처리일시'] = pd.to_datetime(df['처리일시'])
df.info()
df['처리시간'] = df['처리일시'] - df['신고일시']
df_group = df.groupby('공장명',as_index=False)['처리시간'].mean()
df_group.sort_values('처리시간')

# 8 STATION_ADDR1 변수에서 구 정보만 추출한 후, 마포구, 성동구의 평균 이동 거리를 구하시오. (단, 소수점 셋 째자리에서 반올림하여 표기)
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_2.csv")
df['구'] = df['STATION_ADDR1'].str.extract(r'([가-힣]+구)')
df_mapo = df[df['구']=='마포구']
df_songdong = df[df['구']=='성동구']
df_mapo['dist'].mean().round(2)
df_songdong['dist'].mean().round(2)

# 9 분기별 총 판매량(제품A~E 합계)의 월평균을 구하고, 월평균이 최대인 연도와 분기를 구하시오.
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/exam/7_1_3.csv')
df['분기'] = np.concatenate([np.repeat(['2018-1분기','2018-2분기','2018-3분기','2018-4분기'],3),np.repeat(['2019-1분기','2019-2분기','2019-3분기','2019-4분기'],3)])
df
df['총판매량'] = df[['제품A','제품B','제품C','제품D','제품E']].sum(axis=1)
df_group = df.groupby('분기',as_index=False)['총판매량'].sum().sort_values('총판매량',ascending=False)
round(df_group['총판매량']/3,6)

