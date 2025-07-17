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
    inner_function = x +2
    return inner_function
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