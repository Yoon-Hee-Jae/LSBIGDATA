# 변수 개념

# 1번
city_name = "용인"
age = 28
is_student = False

#2번
정답 = a,c,e

#3번
x = 12
y = 4

(x-y) * y/x # 2.6666666666666665
x ** y // x # 1728
(x%y)+2 # 2

#4번
age >= 18 
is_raining = False
is_warm = True

#5번
price = 12000
quantity = 3
total_price = price * quantity
result = 0
if total_price >= 10000 and total_price <= 50000:
    result = True
else:
    result = False
result

# 데이터타입 이해하기

#1번
num = 100
pi = 3.14
name = "111"
fruits = ["apple", "banana", "cherry"]
data = (10, 20, 30)
person = {"name": "Tom", "age": 25}
flags = {True, False}
# 타입 검증 및 True 개수 계산 
results = [
    type(num) == int,
    type(pi) == float,
    type(name) == int,
    type(fruits) == list,
    type(data) == tuple,
    type(person) == dict,
    type(flags) == set
]
print("검증 결과:", results)
print("True 개수:", sum(results))


#2번
nums = [5,10,10,15]
nums = tuple(nums)
nums = set(nums)
len(nums)


# 3번
profile = {
    "name": "Jane",
    "age": 27,
    "city": "Busan"
}
profile["age"] = 28
del profile['city']
print(profile)

# 4번
set_x = {1, 2, 3, 4}
set_y = {3, 4, 5, 6}
# 합집합
set_x.union(set_y)
set_x | set_y

# 교집합
set_x & set_y
# 차집합
set_x - set_y

# 5번
sentence = "Python Is FUN"
sentence = sentence.replace("Python",sentence[:sentence.find("n")+1].upper())
sentence
sentence = sentence.replace(" Is FUN",sentence[sentence.find("N")+1:].lower())
sentence

# 5번 다른 풀이
sentence = "Python Is FUN"
sen_a = sentence[:sentence.find("n")+1].upper()
sen_b = sentence[sentence.find("n")+1:].lower()
sentence = sen_a + sen_b
sentence

# 5번 또 다른 풀이
sentence = "Python Is FUN"
result = sentence.lower().replace("python", "PYTHON")
print(result)



