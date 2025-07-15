import pandas as pd
# 데이터 프레임 생성
df = pd.DataFrame({
    'col1': ['one', 'two', 'three', 'four', 'five'],
    'col2': [6, 7, 8, 9, 10]
})
print(df)
df.shape
df['col1'][0:2]

# 시리즈
data = [10, 20, 30]
df_s = pd.Series(data, index=['one', 'two', 'three'], 
                 name = 'count')
print(df_s)

# 데이터 프레임 생성
my_df = pd.DataFrame({
    'name': ['issac', 'bomi'],
    'birthmonth': [5, 4]
})
print(my_df)
my_df.info()

url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)
print(mydata.head())
mydata.info()
mydata.shape
mydata.head()
mydata[["gender"]].head()
mydata['gender']
mydata[["midterm","final"]].head()
mydata[mydata["midterm"]>15].head()
# iloc의 i = indexing
mydata.iloc[:,1]
mydata.iloc[1:5,2]
mydata.iloc[1:4,:3]
# loc은 인덱싱할 때 마지막 숫자가 포함됨!!!!!!!!!!!!!!!!!
mydata.loc[1:4,"midterm"]

mydata.iloc[:,[1]].squeeze() # squeeze: series로 바꿔주는 함수
mydata.loc[1:4,["midterm","final"]] # loc: 문자로 불러오기
mydata.loc[mydata["midterm"]<=15, ["gender","student_id"]]

mydata['midterm'].isin([28,38,52]).sum()
#중간고사 점수 28,38,52인 애들의 기말고사 점수와 성별 정보 가져오세요
mydata.info()
mydata.loc[mydata['midterm'].isin([28,38,52]),["final","gender"]]
# iloc을 사용할 경우
import numpy as np
np.where(mydata['midterm'].isin([28,38,52]))[0]
mydata.iloc[np.where(mydata['midterm'].isin([28,38,52]))[0],[1,3]]

# 일부 데이터를 NA로 설정
mydata.iloc[0, 1] = np.nan
mydata.iloc[4, 0] = np.nan

print(mydata)
mydata["gender"].isna().sum()
mydata.dropna().shape
#1번
mydata["student_id"].isna()
#2번
vec_2 = ~mydata["student_id"].isna()
# ~: 모든 불린 값을 반대로 바꿔준다
#3번
vec_3 = ~mydata["gender"].isna()

mydata[vec_2 & vec_3].shape # dropna와 같음
# true false 논리연산자 and로 묶어주는 방법mydata["student_id"].isna()

# 구성원소 추가/삭제/변경
mydata['total'] = mydata['midterm'] + mydata['final']
print(mydata.iloc[0:3, [3, 4]])
mydata["average"] = (mydata['total'] / 2).rename("average")
mydata.head()

mydata["average^2"] = mydata["average"]**2
mydata["average^2"].squeeze()

# 다른 방식으로 구성원소 추가
mydata = pd.concat([mydata, (mydata['total'] / 3).rename('average_3')], axis=1)
mydata
# 삭제
del mydata["average_3"]
mydata
# 이름 변경
mydata.rename(columns={'student_id': 'std-id'}, inplace=True)
print(mydata.head())

df1 = pd.DataFrame( {
    'A': ['A0','A1','A2'],
    'B': ['B0','B1','B2']
})
df2 = pd.DataFrame( {
    'C': ['C0','C1','C2'],
    'D': ['D0','D1','D2']
})

pd.concat([df1,df2],axis=1)

df4 = pd.DataFrame({
'A': ['A2', 'A3', 'A4'],
'B': ['B2', 'B3', 'B4'],
'C': ['C2', 'C3', 'C4']
})
result1 = pd.concat([df1, df4], join='inner')
result2 = pd.concat([df1, df4], join='outer')
result1
result2

# Q
df = pd.read_csv('./data/penguins.csv')
df.head()
df.shape

# Q1. bill_length_mm,bill_depth_mm,flipper_lenght_mm,body_mass_g
# 결측치가 하나라도 있는 행은 몇개인가요?
df.info()
bill_len = ~df['bill_length_mm'].isna() # false가 결측치
bill_dep = ~df['bill_depth_mm'].isna()
flipper_len = ~df['flipper_length_mm'].isna()
body_mass = ~df['body_mass_g'].isna()
answer = bill_len & bill_dep & flipper_len & body_mass
# false가 하나라도 있으면 결측치가 있는 행 = true가 값이 있는 행
np.sum(~answer)
#####################################################
# 보다 간단한 풀이
bill_len2 = df['bill_length_mm'].isna() 
bill_dep2 = df['bill_depth_mm'].isna()
flipper_len2 = df['flipper_length_mm'].isna()
body_mass2 = df['body_mass_g'].isna()
answer2 = bill_len2 | bill_dep2 | flipper_len2 | body_mass2
answer2.sum()

# 더더 간단한 풀이
df_new = df.iloc[:,2:6]
df_new[np.sum(df_new.isna(), axis =1) >= 1] # axis 0 세로합 1 가로합

# 더더더 간단한 풀이
df.shape[0] - df.iloc[:,2:6].dropna().shape[0]

# Q2. 몸무게가 4000g 이상 5000g 이하인 펭귄은 몇마리인가요?

df
df1 = df[df['body_mass_g']>=4000]
df1
df2 = df1[df1['body_mass_g'] <= 5000]
df2
df2.shape
############################
df4500 = df[(df['body_mass_g']>=4000) & (df['body_mass_g']<=5000)]
len(df4500) # 116마리

# Q3. 펭귄 종 별로 평균 부리 길이는 어떻게 되나요?
df.info()
df3 = df[["species","bill_length_mm"]]
df3["species"].unique()
adelie = df3[df3["species"]==df3["species"].unique()[0]]
chinstrap = df3[df3["species"]==df3["species"].unique()[1]]
gentoo = df3[df3["species"]==df3["species"].unique()[2]]
chinstrap
round(adelie["bill_length_mm"].mean(),2)
round(chinstrap["bill_length_mm"].mean(),2)
round(gentoo["bill_length_mm"].mean(),2)

# Q4. 성별이 결측치가 아닌 데이터 중, 성별 비율은 각각 몇 퍼센트인가요?
# count는 결측치를 제외
# len은 전부 세줌
df.info()
df['sex'].isna()
df2 = df[~df['sex'].isna()]
male_ratio = np.sum(df2['sex']=="Male") / len(df2) * 100
male_ratio
female_ratio = 100 - male_ratio
female_ratio
f'성별이 남자인 펭귄의 비율은 {np.round(male_ratio,2)}%이고, 여성 펭귄의 비율은 {np.round(female_ratio,2)}% 입니다.'

# Q5. 섬 별로 평균 날개 길이가 가장 긴 섬은 어디인가요?

df['island'].unique()
tor = df[df["island"]==df["island"].unique()[0]]
bis = df[df["island"]==df["island"].unique()[1]]
dream = df[df["island"]==df["island"].unique()[2]]

mean_data = {
"Torgersen" : tor['flipper_length_mm'].mean(),
"Biscoe" : bis['flipper_length_mm'].mean(), 
"Dream" : dream['flipper_length_mm'].mean()
}

max(mean_data, key=mean_data.get)
# groub by 풀이법
df_ans = df.groupby('island')['flipper_length_mm'].mean()
df_ans
df_ans.argmax()
df_ans.index[df_ans.argmax()]


# groub by 사용하면 간단함
df_group = df[['island',"flipper_length_mm"]]
df_group = df_group.groupby('island').mean()
df_group
df_group.iloc[np.argmax(df_group),0]
a = df.groupby('species').mean(numeric_only=True) # numeric_only 숫자형 데이터만 포함
type(a)
a
a.iloc[0:1,:]


df.describe()
df.sort_values(by=['body_mass_g','flipper_length_mm'],ignore_index=True,ascending=[False,True])

