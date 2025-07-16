import pandas as pd
df = pd.read_csv('./data/penguins.csv')
df.info()

df.describe()
df.sort_values('bill_length_mm')
# 출력값은 series지만 인덱스가 이름으로 되어 있음
df.groupby('species')['bill_length_mm'].mean()
df['bill_length_mm']

result = df.groupby('species')['bill_length_mm'].mean()
result.index
result.values.argmax()
re_idx = result.argmax()
result.index[re_idx]
# 위 내용이 idxmax() 이다
result.idxmax() # idxmax는 result라는 pandas series로부터 점을 찍고 사용해서 pandas 함수
result.values # numpy 벡터 values.argmax()는 numpy 벡터로부터 점을 찍고 사용해서 numpy 함수

# 예제 데이터 프레임 생성
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
# 두 데이터 프레임 병합
merged_df = pd.merge(df1, df2, on='key', how='inner')
print(merged_df)
merged_df2 = pd.merge(df1, df2, on='key', how='outer')
merged_df2

# 사용 예시
mid_df1 = pd.DataFrame({'std_id': ['1', '2', '4','5'], 'score': [10,20,40,30]})
fin_df2 = pd.DataFrame({'std_id': ['1', '2', '3','6'], 'score': [4,5,7,2]})
mid_df1
fin_df2
# 중간 & 기말 둘다 본 학생들의 데이터를 만들어보세요
midfin_df = pd.merge(mid_df1,
                     fin_df2,
                     on='std_id',
                     how='inner')
# 학생들 중간 & 기말 전체 데이터 만들기
total_df = pd.merge(mid_df1,fin_df2,on='std_id',how='outer')
# merge의 how에는 left,right라는 옵션도 있음
# on에 사용할 열(cols)의 이름이 다를 경우
# left_on='열 이름', right_on='열 이름' 으로 작성해주면 된다
# pd.merge(mid_df1,fin_df2,left_on='std_id',right_on='student_id,how='left')
# del df['std_id']
# 왼 오 on에 사용된 열이 두개 생성되므로 하나를 지워주는 작업이 필요함
left_df = pd.merge(mid_df1,fin_df2,on='std_id',how='left')
left_df
# dataframe 열 이름 변경 함수 rename
left_df.rename(columns={'score_x':'first'})


df1 = pd.DataFrame({
    'emp_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

df2 = pd.DataFrame({
    'employee_id': [1, 2, 4],
    'salary': [50000, 60000, 70000]
})

# 열 이름이 다를 때 merge
merged = pd.merge(df1, df2, left_on='emp_id', right_on='employee_id',how='inner')

print(merged)
del merged['employee_id'] # 중복 열 제거
merged


# pandas melt -- 자격증에 자주 출제되는 문제
wide_df = pd.DataFrame({
    '학생' : ['철수','영희','민수'],
    '수학' : [90,80,70],
    '영어' : [86,95,75]
})
wide_df

long_df = pd.melt(wide_df, id_vars="학생",var_name="과목",value_name="점수")
long_df

# 연습 1
w_df = pd.DataFrame({
    '학년' : [1,1,2],
    "반" : ['A','B','C'],
    "1월" : [20,18,22],
    "2월" : [19,20,21],
    "3월" : [21,17,23]
})
w_df

l_df = pd.melt(w_df, id_vars=['학년',"반"], var_name="월",value_name="출석일수")
l_df

# 연습2
df3 = pd.DataFrame({
    "학생" : ['철수','영희','민수'],
    "국어" : [90,80,85],
    "수학" : [70,90,75],
    "영어" : [88,92,79],
    "학급" : ['1반','2반','3반'],
})
df3
pd.melt(df3,id_vars=["학급",'학생'],
        var_name="언어과목",
        value_vars=["국어","영어"],
        value_name="성적")
