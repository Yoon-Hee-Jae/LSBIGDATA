# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)

type(fruits[0])
fruits[1:]

numbers * 3
numbers + numbers

mixed_list[2][1]

len(mixed_list)
len(mixed_list[2])

a=[1,2,3]
a[2]
a[2] + 'hi' # 숫자형과 문자형은 연산 불가
str(a[2]) + 'hi'

numbers[2] = 10
numbers

del(numbers[2])
numbers

type(numbers)
numbers.append(200)
numbers.append([1,4])
numbers

del(numbers[-1])
numbers
# 리스트 원소는 대소 비교가 불가능
numbers.sort()
numbers.append([1,4])
numbers.sort()


# 튜플 생성 예제
a = (10, 20, 30) # a = 10, 20, 30 과 동일
b = (42,)
print("좌표:", a)

a[1]
a[1:]
b = (2,4)
a+b
a*3
d=(2) # 원소 하나로는 튜플이 만들어지지 않음
d = (2,) # 이런식으로 만들어줘야함
d
d[0]
a[2] = 3 # 튜플은 기존 원소에 대한 수정은 불가능

tup_e = (1,3,"a",[1,2]) # 튜플도 여러 데이터 타입을 넣어줄 수 있음

# 리스트는 부가 기능이 많아서 운영적인 측면에서 무겁고 부담스러울 수 있음
# 튜플은 비교적 가벼움 + 코딩 중간 변하지 않을 데이터 설정시 사용


# 딕셔너리 생성 예제
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}
print("Person:", person)
# 딕셔너리 순서가 없다 no index > 인덱싱 슬라이싱 불가능
# 딕셔너리의 원소는 대괄호로 표시되어 있어서 리스트처럼 보이지만 아니다.
person[0]

person.get('name') # get: key를 통해 value를 찾는 매서드
person.keys() # keys: key들을 불러오는 매서드

a = {1:'hi'}
a[2] = 'b' 
a['name'] = 'issac'
del a[1]
a[2]
list(a.keys())[1] #딕셔너리의 원소를 리스트로 바꾸어서 빼내는 방법
list(a.values())[1]
'name' in a # in: 딕셔너리안에 있는 key가 있는지 확인하는 방법
'issac' in a

# 집합
s1 = set([1,2,3]) 
s1[0] # 집합 순서가 없다. + 중복 허용 X
s2= set([4,5,6,1,2,7])
s1
s2
s1 & s2 # &: s1과 s2의 교집합
s1 | s2 # &: s1과 s2의 합집합 s1.union(s2)도 가능
s1 - s2 # &: s1과 s2의 차집합


