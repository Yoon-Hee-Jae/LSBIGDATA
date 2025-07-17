import numpy as np
# 연습 문제 1
a = np.array([1, 2, 3, 4, 5])
plus_a = a + 5
print([plus_a])

# 연습 문제 2
a = np.array([12, 21, 35, 48, 5])
a[::2]

# 연습 문제 3
a = np.array([1, 22, 93, 64, 54])
np.max(a)
max(a)

# 연습 문제 4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
a
a.dtype
list(np.unique(a))[0]

##################################################

set_a = set(a)
list_a = list(set_a)
np.array(set_a)

# 연습 문제 5
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(a.size+b.size,dtype=a.dtype)
c[::2] = a
c[1::2] = b
c

# 연습 문제 6
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
c = a[:4] + b
c

# 연습 문제 7 
a = np.array([1, 3, 3, 2, 1, 3, 4, 2, 2, 2, 5, 6, 6, 6, 6])
unique, counts = np.unique(a,return_counts=True)
most_frequent = unique[counts == np.max(counts)] # argmax는 최대값의 위치를 반환
most_frequent

# 연습 문제 8
a = np.array([12, 5, 18, 21, 7, 9, 30, 25, 3, 6])
multiples_3 = a[a % 3 == 0]
multiples_3

# 연습 문제 9
a = np.array([10, 20, 5, 7, 15, 30, 25, 8])
np.median(a)
lower_a = a[a<np.median(a)]
upper_a = a[a>np.median(a)]
print(lower_a)
print(upper_a)

# 연습 문제 10
a = np.array([12, 45, 8, 20, 33, 50, 19])
median_num = np.median(a)
closest_num = a[np.argmin(np.abs(a - median_num))]
closest_num


# 연습 문제 11
np.array([[3,5,7],
         [2,3,6]])

# 연습 문제 12
np.random.seed(2023)
# 행렬 B 생성
B = np.random.choice(range(1, 11), 20, replace=True).reshape(5, 4)
print("행렬 B:\n", B)
new_B = B[[1,3,4],:]
new_B

# 연습 문제 13
new_B[:,2][new_B[:,2] > 3]

# 연습 문제 14
B
ANS = B[np.sum(B,axis=1)>=20,:]
ANS

# 연습 문제 15
B
ANS_IDX = np.where(np.mean(B,axis=0)>=5)[0]
ANS_IDX

# 연습 문제 16
B
big7 = np.sum(B>7,axis=1)>=1
B[big7,:] # 행 보여주기
np.where(np.sum(B>7,axis=1)>=1)[0] # 인덱스가 필요할 경우

# 연습 문제 17
import numpy as np
x = np.array([1, 2, 3, 4, 5])  
y = np.array([2, 4, 5, 4, 5])  # 종속 변수
print("x 벡터:", x)
print("y 벡터:", y)

up_num = sum( (x-np.mean(x)) * (y-np.mean(y)) ) # 분자
down_num = sum( (x-np.mean(x))**2 ) # 분모

beta_one = up_num / down_num
beta_one

# 연습 문제 18
X = np.array([[2, 4, 6],
              [1, 7, 2],
              [7, 8, 12]])
y = np.array([[10],
              [5],
              [15]])

Xt = X.T
Xt

Vector = np.linalg.inv(Xt @ X) @ Xt @ y
Vector


