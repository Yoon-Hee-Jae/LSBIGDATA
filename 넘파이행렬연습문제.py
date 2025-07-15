import numpy as np
# 기본적인 행렬 곱셈
# 1번 
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
A @ B 
B @ A

# 2번
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8],[9,10],[11,12]])
A.shape
B.shape # 곱셈 가능
np.matmul(A,B)

# 3번 행렬의 경우 AB와 BA는 다르다
A = np.array([[2,3],[4,5]])
I = np.array([[1,0],[0,1]])
A @ I 
I @ A

# 4번
A = np.array([[1,2],[3,4]])
Z = np.array([[0,0],[0,0]])
A @ Z

# 5번
A = np.array([[4,5],[6,7]])
D = np.array([[2,0],[0,3]])
np.matmul(A,D)

# 6번  가중 평균 / 0.5일경우 평균
A = np.array([[1,2],[3,4],[5,6]])
v = np.array([[0.4],[0.6]])
A.shape
v.shape
A @ v

# 7번
T = np.array([[[1,2],[3,4]],
             [[5,6],[7,8]]])
C = np.array([[9,10],[11,12]])
T @ C
# np.stack 활용

# 8번 본인끼리 곱했을 때 본인이 나옴 = 멱등행렬(idempotent)
S = np.array([[3,-6],[1,-2]])
S @ S
np.linalg.inv(S)

# 9번
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.array([[9,10],[11,12]])
(A@B)@C
A@(B@C)
(B@C)@A
(C@A)@B # 행렬의 순서가 변하면 값은 변하지만 대각선의 합은 항상 같다. 
# 행렬의 TRACE 성질

#10번
A = np.array([[3,2,-1],
              [2,-2,4],
              [-1,0.5,-1]])
b = np.array([[1],
              [-2],
              [0]])
A.shape
b.shape
# x는 (3,1)의 벡터 (a,b,c)
3a+2b-c = 1
2a-2b+4c = -2
-a+0.5b-c = 0 
# Ax=b에서 A를 넘기면 역함수A가 된다.
np.linalg.inv(A) @ b
np.linalg.solve(A,b)





