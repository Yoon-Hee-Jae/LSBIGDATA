import pandas as pd
data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}

df = pd.DataFrame(data)
df['제품명_길이'] = df['가전제품'].str.len()
df['브랜드_길이'] = df['브랜드'].str.len()
df

df['브랜드'] = df['브랜드'].str.lower()
df

df['브랜드'].str.contains('i')
df['브랜드'].str.startswith('i')
df['브랜드'].str.endswith('i')

df['가전제품'].str.replace('에어컨','선풍기')

df['가전제품'].str.replace('기','')

df['브랜드'].str.split('a') # 기준이 되는 애는 사라짐
df['브랜드'].str.split('a',expand=True) # expand=True 데이터 프레임 형식으로 보여줌
df['제품_브랜드'] = df['가전제품'].str.cat(df['브랜드'], sep=', ')
df

# 간단 예제
data = {
    '주소': ['서울특별시 강남구 테헤란로 123', '부산광역시 해운대구 센텀중앙로 45', '대구광역시 수성구 동대구로 77-9@@##', '인천광역시 남동구 예술로 501&amp;&amp;, 아트센터', '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)
print(df.head(2))

df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
print(df.head(2))


df = pd.DataFrame({
    'text': [
        'apple',        # [aeiou], (a..e), ^a
        'banana',       # [aeiou], (ana), ^b
        'Hello world',  # ^Hello, world$
        'abc',          # (abc), a.c
        'a1c',          # a.c
        'xyz!',         # [^aeiou], [^0-9]
        '123',          # [^a-z], [0-9]
        'the end',      # d$, e.
        'space bar',    # [aeiou], . (space)
        'hi!'           # [^0-9], [aeiou]
    ]
})
print(df)
df['text'].str.extract(r'([aeiou])')
df['text'].str.extractall(r'([aeiou])')
df['text'].str.extractall(r'([^0-9])')
df['text'].str.extractall(r'(a.c)')