import numpy as np
import pandas as pd

# 연습문제1
def add_numbers(a=1,b=2):
    return a+b
add_numbers() # 기본값
add_numbers(5,7)

# 연습문제2
def check_sign(x):
    if x > 0:
        print("양수")
    elif x < 0:
        print("음수")
    else:
        print("0")

check_sign(10)
check_sign(-5)
check_sign(0)

# 연습문제3
def print_numbers():
    for i in range(1,11):
        print(i)
print_numbers()

# 연습문제4
def outer_function(x):
    def inner_function(y):
        return y +2
    return inner_function(x)
outer_function(5)

# 연습문제5
def find_even(start):
    while True:
        if start % 2 == 1:
            start+=1
        else:
            return start
            break
find_even(3)


# 추가 연습문제
# 1 번
ans = [x**2 for x in range(1,11) if x%2==0]

# 2번
import copy
customers = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 34},
    {"name": "Charlie", "age": 29},
    {"name": "David", "age": 32},
    {"name": "Eve", "age": 22}
]

# 2-1번
customers2 = copy.deepcopy(customers) 
# copy.deepcopy() 리스트안에 리스트/딕셔너리 같은 복잡한 객체가 있을 떄 독립해서 복사하는 방법
for i in range(len(customers)):
    customers2[i]['age'] = customers[i]['age']+1
customers
customers2

# 2-1번 강사님 풀이 -- 수정필요
updated_ages = []
# 루프 돌아가는 방식 확인
for c in customers:
    print(c)
# 풀이시작
customers[0]["name"]
for c in customers:
    up_customer = {"name":c["name"],"age":c["age"]+1}
    updated_ages.append(up_customer)
updated_ages

# 2-2번
ans = []
for i in range(len(customers)):    
    if customers[i]['age']>=30:
        ans.append(customers[i])
ans

# 2-3번
sum_age = 0
for i in range(len(customers)):
    sum_age += customers[i]['age']
sum_age
# 강사님 풀이 
# 딕셔너리이니까 인덱스 접근법보다는 key와 value를 통해 접근하는 방식으로 최대한 하자
for c in customers:
    sum_age += c['age']
print(sum_age)

# 2-4번
name = []
for i in range(len(customers)):
    if customers[i]['age'] < 30 and customers[i]['name'][0] == "A":
        name.append(customers[i]['name'])
name

# 마찬가지 다른 풀이
under30_a = []
for c in customers:
    if c["age"] < 30 and c["name"][0] == "A":
        under30_a.append(c["name"])
under30_a


# 3번
sales_data = {
    "January": 90,"February": 110,
    "March": 95,"April": 120,
    "May": 80,"June": 105,
    "July": 135,"August": 70,
    "September": 85,"October": 150,
    "November": 125,"December": 95
}

# 3-1번
item_list = list(sales_data.items())
item_list[0]
ans_3 = []
for i in range(len(item_list)):
    if item_list[i][1] >= 100:
        ans_3.append(item_list[i][0])
print(ans_3)

# 3-2번
# 연간총판매량
sum_sales = 0
for i in range(len(item_list)):
    sum_sales += item_list[i][1]
print(sum_sales)
# 월 평균 판매량
sum_avg = sum_sales / len(item_list)
print(sum_avg)

# 3-3번
list_values = list(sales_data.values())
max_idx = int(np.argmax(list_values))
print(item_list[max_idx])
    
# 3번 강사님 풀이
# 3-1
high_sales = []
for month, sales in sales_data.items():
    if sales >= 100:
        high_sales.append(month)
high_sales

# 3-2
sum_sale = 0
for month, sales in sales_data.items():
    sum_sale += sales
mean_sale = sum_sale / len(sales_data)
mean_sale

# 3-3
for month, sales in sales_data.items():
    if sales == max(sales_data.values()):
        print(month,sales)
#
highest_month = list(sales_data.keys())[0]
highest_sales = sales_data[highest_month]

for month, sales in sales_data.items():
    if sales > highest_sales:
        highest_month = month
        highest_sales = sales
print(highest_month,highest_sales)




