import pandas as pd
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins

# 문제 1: 펭귄 종별 평균 부리 길이 구하기
# 펭귄 데이터에서 각 종(species)별로 평균 부리 길이(bill_length_mm)를 구하는 pivot_table()을 작성하세요.
species_mean = pd.pivot_table(penguins,
                              columns='species',
                              values='bill_length_mm',
                              aggfunc='mean')

# 강사님 풀이
penguins.pivot_table(
    index='species',
    values='bill_length_mm',
    aggfunc='mean'
).reset_index()

# 문제 2: 섬별 몸무게 중앙값 구하기
# 펭귄 데이터에서 각 섬(island)별로 몸무게(body_mass_g)의 중앙값(median)을 구하는 pivot_table()을 작성하세요.
island_median = pd.pivot_table(penguins,
                               columns='island',
                               values='body_mass_g',
                               aggfunc="median")

# 문제 3: 성별에 따른 부리 길이와 몸무게 평균 구하기
# 펭귄 데이터에서 성별(sex)과 종(species)별로 부리 길이(bill_length_mm)와 몸무게(body_mass_g)의 평균을 구하는 pivot_table()을 작성하세요.
pd.pivot_table(penguins,
               columns=['sex','species'],
               values=['bill_length_mm','body_mass_g'],
               aggfunc='mean')

# 문제 4: 종과 섬에 따른 평균 지느러미 길이 구하기
# 펭귄 데이터에서 각 종(species)과 섬(island)별로 지느러미 길이(flipper_length_mm)의 평균을 구하는 pivot_table()을 작성하세요.
# pd.pivot_table()의 인수인 dropna= 는 값이 없어도 nan으로 표시해줘서 나타내주는 인수
# fill_value='' nan 값인 부분을 채워주는 인수
pd.pivot_table(penguins,
               columns=['species','island'],
               values='flipper_length_mm',
               aggfunc="mean",
               dropna=False)
#강사님 풀이
penguins.pivot_table(
    index=['species','island'],
    values='flipper_length_mm',
    aggfunc='mean',
    dropna=False
).reset_index()

penguins.pivot_table(
    index='species',
    columns='island',
    values='flipper_length_mm',
    aggfunc='mean',
    dropna=False,
    fill_value="없음"
).reset_index()

# 문제 5: 종과 성별에 따른 부리 깊이 합계 구하기
# 펭귄 데이터에서 종(species)과 성별(sex)별로 부리 깊이(bill_depth_mm)의 총합(sum)을 구하는 pivot_table()을 작성하세요.
pd.pivot_table(penguins,
               columns=['species','sex'],
               values='bill_depth_mm',
               aggfunc='sum')

# 문제 6: 종별 몸무게의 변동 범위(Range) 구하기
# 펭귄 데이터에서 각 종(species)별로 몸무게(body_mass_g)의 변동 범위 (최댓값 – 최솟값) 를 구하는 pivot_table()을 작성하세요.
def max_min(x):
    return ( max(x)-min(x) )

max(penguins['body_mass_g'])
max_min(penguins['body_mass_g'])

df = pd.pivot_table(penguins,
               columns='species',
               values='body_mass_g',
               aggfunc=max_min)
df
pd.melt(df,var_name='species',value_name='body_mass')
