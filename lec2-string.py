# 작은 따옴표 / 큰 따옴표 둘 다 사용 
"Issac's name"
'Issac"s name'

# \를 앞에 붙여주면 따옴표를 특수 기호 형식으로 넣을 수 있다
"Issac"s name"
"Issac\"s name"


# 줄 바꿈  \n
"Issac\ns name"
print("Issac\ns name")

# 멀티라인 << 코드 돌릴 때는 전체를 드래그한 다음 돌려야 오류가 뜨지 않음
b='''
안녕하세요
파이썬
입니다
'''
b
print(b)


"안녕" + "Python"
"안녕" * 3

len("sss")
len("안녕" + " Python") # len은 빈칸도 같이 세준다

# 인덱싱, 슬라이싱
str_a = "안녕" + " Python"
str_a
str_a[1] # 0부터 시작
str_a[1:4] # a:b a부터 b-1 까지
str_a[3:]
str_a[:5]
str_a[-3]
str_a[1:-3]
str_a[:len(str_a)]

# f 접두사를 붙이고 작은 괄호를 써서 나타낼 수 있음
name = "이삭"
age = 30
f'나의 이름은 {name}, 나이는 {age}입니다.'
f'나의 이름은 {name}, 나이는 {age+10}입니다.' # 사칙연산도 가능

# 변수의 데이터 타입 
str_a
type(str_a)
num_a = 10
type(num_a)

# str 데이터 타입인 변수에 .를 붙이고 뒤에 나오는 f는 function
str_a.count('n') #count: str_a안에 있는 "n"의 개수를 세어줌
str_a.find('녕') # find: '녕'의 인덱스 번호
str_a[2:].strip() # strip: 양옆 공백을 지워줌 lstrip/rstrip 왼/오
str_a.replace("안녕","반가워") # replace: 괄호 왼쪽=바꾸고싶은 단어, 오른쪽=바꿀 단어
# 업데이트를 하고 싶을 때
str_a = str_a.replace("안녕","반가워")
str_a
str_a.split() # split: 공백을 기준으로 나눠주지만 설정하면 커스텀 가능