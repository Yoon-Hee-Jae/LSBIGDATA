import pandas as pd
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)

df_melted = pd.melt(df, 
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='측정요소', 
                    value_name='측정값')
df_melted

# 원래 형식으로 변환
# Date, Temperature, Humidity

df_pivot = pd.pivot_table(df_melted,
               index='Date',
               columns="측정요소",
               values="측정값").reset_index()
df_pivot 
df_pivot.columns.name = None # 맨 앞 측정요소 지우기
# 원래 데이터프레임과 다르게 7월 3일이 2개였는데 1개로 줄여짐
# index를 date로 지정이 되었기 때문에 unique한 값만 가능하기 때문에 평균을 내서 하나로 만들어준 것

# aggfunc 인수를 통해서 index가 중첩될 경우 합쳐지는 방식을 설정할 수 있다. (sum,mean 등등)
df_pivot2 = pd.pivot_table(df_melted,
               index='Date',
               columns="측정요소",
               values="측정값",
               aggfunc="sum").reset_index()
df_pivot2

# 학생성적 데이터
df = pd.read_csv('./data/dat.csv')
df.head()
df.columns
# 데이터 열 이름 변경
df = df.rename( columns= {'Dalc':'dalc', 'Walc':'walc'} )
# 데이터 타입 변경
df.loc[:, ['famrel', 'dalc']].astype({'famrel' : 'object', 'dalc' : 'float64'}).update()
df.info()
# 함수를 통해서 새로운 값으로 대체해주기
def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    elif famrel <= 4:
        return 'Medium'
    else:
        return 'High'
# assign()이라는 값을 대체하는데 사용하는 매서드 사용 + apply()를 통해서 classify_famel 함수 적용
df = df.assign(famrel=df['famrel'].apply(classify_famrel))
df
# select_dtypes() 특정 데이터 타입만 선택 
df.select_dtypes('object')
df.select_dtypes('number')

import numpy as np
# np.nanmean()과 np.mean()의 차이점
# nanmean()의 경우 nan을 제외하고 평균을 계산하지만 np.mean()의 경우 nan을 포함하기에 하나라도 nan이 있으면 결과값도 nan이다
# np.std() 표준편차
def standardize(x):
    return ( ( x - np.nanmean(x) )/np.std(x) ) 
# 해당 함수는 평균을 0으로 맞추어주고 표준편차를 1로 맞추는 작업이다
vec_a = np.arange(5)

standardize(vec_a)
# 숫자형 데이터 타입에 전부 standarize 함수 적용
df_std = df.select_dtypes('number').apply(standardize)
df_std.mean(axis=0) # 2.050576e-16 << -16은 10의 -16승 즉 0.000...2 거의 0이다
# f로 시작하는 열만 가져오기
df.columns
df.columns.str.startswith('f')
index_f = df.columns.str.startswith('f')
df.loc[:,index_f]

# csv로 저장하기
df.to_csv("test_dat.csv", index=False)