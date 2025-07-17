def add(a,b):
    return a+b
add(3,4)
# 매개변수 : parameter
# 인수 : arguments

def say():
    return "hi"

say()

#121pg
money= 1000
card = True

if (money>4500 or card == True):
    print("take a taxi")
else:
    print("just walk")

# 시험점수가 60점 이상이면 합격, 그렇지 않으면 불합격을 출력하는 조건문 생성
score = 50
if score >=60:
    print("합격")
else:
    print("불합격")

# x의 수가 홀수이면서 7의 배수이면
# "7의 배수이면서 홀수"를 출력,
# 그렇지 않으면, "조건 불만족 출력"
x=14
if x%7==0 & x%2==1:
    print("7의 배수이면서 홀수")
else:
    print("조건 불만족")

gender = "M"

if gender == "M":
    print("남자")
elif gender == "F":
    print("여자")
else:
    print("비어있음")

# 다음은 공원 입장료 정보입니다
# 유아 7세이하 : 무료
# 어린이: 3000원
# 성인 : 7000원
# 노인 60세이상 : 5000원

age = 30
if age >= 60:
    print("5000원")
elif age >= 20:
    print("7000원")
elif age > 7:
    print("3000원")
else:
    print("무료")

# cal_price 함수 만들기
def cal_price(age):
    if age >= 60:
        price = 5000
    elif age >= 20:
        price = 7000
    elif age > 7:
        price = 3000
    else:
        price = 0
    return price
cal_price(age)

treehit = 0
while treehit < 10:
    treehit = treehit + 1
    print("나무를 %d번 찍었습니다" % treehit)
    if treehit == 10:
        print("나무 넘어집니다")

a=0
while a < 10:
    a += 1
    if a % 2 == 0:
        continue
    print(a)

# for 루프
for a in [5,6,7,8]:
    print(a)

a = [(1,2),(3,4),(5,6)]
for (first,last) in a:
    print(first + last)

# Q. 1에서 100까지 넘파이 벡터 만들기
import numpy as np
a = np.arange(1,101)
for i in a:
    if (i%7==0):
        continue
    print(i)
# append함수를 이용해 for문을 돌리는 것보다 리스트 컴프리헨션을 사용하는 것이
# 성능이 더 빠르다
a = [1,2,4,3,5]
# a의 각원소에 3을 곱한 값을 다시 리스트로 만들기
[3*a for a in a]

# 1에서부터 10까지의 정수 중 각 수의 제곱값을 요소로 가지는 리스트를 만드세요
[x**2 for x in range(1,11)]
# 1~20까지 짝수만 골라서 리스트 만들기
[x for x in range(2,21,2)]
[x for x in range(1,21) if (x % 2 == 0)]
a = []
for i in range(1,21):
    if i % 2 == 0:
        a.append(i) 
print(a)
# 음수는 0으로 바꾸는 리스트컴픠헨션
num = [-3,5,-1,0,8]

for x in num:
    if x < 0:
        x = 0
    else:
        x

[x if x >= 0 else 0 for x in num]

words = ['apple','banana','cherry','avocado']
for i in words:
    if i.startswith('a'):
        print(i)
# else를 써야할 경우 if를 앞으로 빼고 반대의 경우 if를 뒤로 보냄
[x for x in words if x.startswith('a')]

def add_many(*nums):
    result = 0
    for i in nums:
        result = result + i
    return result

add_many(3,4,2)

def cal_how(method,*nums):
    if method == 'add':
        result = 0
        for i in nums:
            result += i
    elif method == 'mul':
        result = 1
        for i in nums:
            result *= i
    else:
        print("go")
        result = None
    return result

cal_how('add',3,2,5)
cal_how('mul',3,2,5)
cal_how('minus',3,2,5)

def add_and_mul(a,b):
    return a+b, a*b

def add_and_mul(a=3,b=5): # 기본값 설정
    return a+b, a*b
add_and_mul(3,4)
add_and_mul(3)