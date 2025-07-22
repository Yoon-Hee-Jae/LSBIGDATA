import pandas as pd

df = pd.read_csv('data/regex_practice_data.csv')
df
([\w\.]+@[\w\.]+)
# 문제 1번
df['전체_문자열'].str.extract(r'([a-z]+[._][a-z]+[@][a-z]+[.][a-z]+)')

# 문제 2번
df
df['전체_문자열'].str.extract(r'(010-[0-9]+-[0-9]+)')
df['전체_문자열'].str.extract(r'(010-[0-9\-]+)')

# 문제 3번
df
phone_num = df['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
~phone_num.iloc[:,0].str.startswith("01")
phone_num.loc[~phone_num.iloc[:,0].str.startswith("01")]

# 문제 4번
df['전체_문자열'].str.extract(r'([가-힣]+구)')
df['전체_문자열'].str.extract(r'(\b\w+구)\b')
df
# 문제 5번
df['전체_문자열'].str.extract(r'(\d{4}-\d{2}-\d{2})')
# 문제 6번
df['전체_문자열'].str.extract(r'(\d{4}\W\d{2}\W\d{2})')
₩ ㅁㅇㄴ