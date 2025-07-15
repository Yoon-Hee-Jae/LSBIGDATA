import numpy as np # ctrl + shift p > python 인터프리터 > 가상환경 설정
import pandas as pd

# 벡터: 1차원 행렬: 2차원 배열: n차원
# 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5])  # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])  # 문자형 벡터 생성
c = np.array([True, False, True, True])  # 논리형 벡터 생성
print("Numeric Vector:", a)

type(a) # numpy.ndarray > numpy를 통한 배열 
a[2:4]
d = np.array(["q",2])
d # numpy의 배열은 데이터 타입을 통일시켜줘야함 // 2를 텍스트 2로 자동 저장
a
b   
c    # dtype u6라면 최대 6의 크기를 갖는 배열이다. 변경은 가능
d
a + 3
a * 20
b = np.array([6,7,8,9,10])
a+b
a**2
a.cumsum() # cumsum: 원소의 누적합을 구해주는 메서드


np.arange(10) # numpy로부터 .을 찍은 것이기 때문에 메서드보다는 함수라고 봐야함
# np.arange([start, ]stop, [step, ]dtype=None)
a = np.arange(4,10,0.5)
len(a)

# 1000이하의 7의 배수를 발생시켜보세요!
vec_a = np.arange(7,1000,7)
sum(vec_a)
vec_a.sum()
vec_a.cumsum()
np.cumsum(vec_a)



# pip install palmerpenguins
# 데이터로드
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()
penguins = penguins.dropna() # dataframe의 함수인 dropna를 사용
penguins
penguins.shape 
# ()를 붙이면 에러가 나는 이유: 
# shape은 함수가 아님 '속성'이라고 함
penguins.columns # columns도 마찬가지
penguins["body_mass_g"] # numpy의 배열과는 다름 / pandas.series라는 것
vec_m = np.array(penguins["body_mass_g"]) # numpy 배열로 변환
vec_m.shape

np.max(vec_m) # float은 numpy에서 표현하는 수치형 데이터 타입
np.min(vec_m)

vec_m.argmax() # argmax: 가장 큰 값의 위치를 찾는 함수 
vec_m.argmin()

np.mean(vec_m)
vec_m.mean()

# 평균 4.2kg라고 하면 평균보다 작은 펭귄들은 몇마리인가?
sum(vec_m < 4200)

# 3kg이상인 펭귄들은 몇마리 인가요?
sum(vec_m >= 3000)
# 3kg 이상이면서 4.2kg 미만인 펭귄들은 몇마리인가요?
sum((vec_m < 4200) & (vec_m >= 3000))

np.arange(1,10,0.5)
np.linspace(1,10,20,endpoint=False)

np.repeat(3,10)
np.tile([1,3,5],4) # 전체 반복
np.repeat([1,3,5],4) # 리스트 원소마다 반복

arr = np.array([[1,2],[3,4]])
arr.shape

np.repeat(arr,3,axis=1).shape
np.repeat(arr,3,axis=0).shape

np.repeat([1,2,4],[1,2,3])

# 벡터의 길이, 모양, 사이즈
vec_a = np.array([2,1,4])
len(vec_a)
vec_a.shape
vec_a.size
# 행렬의 길이, 모양, 사이즈
len(arr)
arr.shape
arr.size

# 브로드캐스팅
a = np.array([1,2,3,4])
b = np.array([1,2])

a + b # 배열의 길이가 달라서 오류
a + np.tile(b,2)
a + np.repeat(b,2)

# 브로드 캐스팅: 배열의 크기가 다를 경우 자동으로 크기를 맞추어주는 기능
# 2차원 배열 생성
matrix = np.array([[ 0.0,  0.0,  0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
print(matrix.shape, vector.shape)

matrix + vector

vector = np.array([1.0, 2.0, 3.0, 4.0])

matrix + vector # 오류 발생함 > vector를 세로로 바꿔줘야함
vector = vector.reshape(4,1)

matrix + vector

matrix.reshape(3,4)
matrix.reshape(3,-1)

np.linspace(0,100,num=20).reshape(4,5)

vec_a = np.arange(20)
vec_a[:7:2]

a = np.array([1,5,7,8,10])
a<7
np.where(a<7)[0][1] # where: 괄호 안 조건을 만족하는 인덱스를 반환

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
                          np.arange(12, 16)))
print("행렬:\n", matrix)

y = np.zeros((2, 2))
y

# order='C': 행 우선 순서 (row-major order).  행을 먼저 채웁니다.
# order='F': 열 우선 순서 (column-major order).  열을 먼저 채웁니다.
np.arange(1,5).reshape((2,2),order='C') # 기본 값
np.arange(1,5).reshape((2,2),order='F')
np.arange(1,5).reshape((2,2))


x = np.arange(1,11).reshape((5,2)) * 2
x.shape
x[2] # 인데스 2번의 행을 갖고 옴 0, 1, 2
x[2][0]
x[2,0] # 행, 열 구분은 콤마 ","!!!
x[2:4,0].shape
x[1:4]
x[[1,3,4],0]
x[[1,2,3],[0,1,1]]

vec_a[[1,3,4]]

x[2:4,0] # 기본적인 방법
x[2:4,[0]] # 리스트 형태로 인덱스를 넣어주면 행렬 형태를 유지한 상태로 추출
x
x > 10
x[x>10]
x[x[:,0] > 10]


# Q 문제 난수 생성
np.random.seed(2025)
vec_a = np.random.choice(np.arange(1,101),size=3000,replace=True)

# Q1. 두번째 열(기말고사점수) 점수가 10점 이하인 학생들 데이터를 필터링하면?

x[x[:,1]<10,:]
x

# 중간 기말 평균
c = mat_a.mean(axis=0)
midterm = c[0]
midterm # 50.52066666666666
finalterm = c[1]
finalterm # 50.839333333333336

# Q2. 중간고사 성적이 50점 이상인 학생의 수는?

vec_a[:5]
mat_a = vec_a.reshape((-1,2))
mat_a
mat_a.shape

answer_mat = mat_a[mat_a[:,0]>=50,:]
answer_mat.shape
answer = answer_mat.shape[0]
print( "중간고사 점수가 50점 이상인 학생의 수는",answer,"명 입니다")

# Q3. 그 학생들의 기말고사 성적 평균은?
mean_score = answer_mat[:,1].mean()
mean_score # 51.0158940397351
print( "기말고사 점수의 평균은",mean_score,"점 입니다")

# Q4. 중간고사 최고점을 맞은 학생의 기말고사 성적은?

sum(mat_a[:,0]==mat_a[:,0].max()) # 중간고사 최고점 17명
mat_a[mat_a[:,0]==mat_a[:,0].max(),1]

#################
mid_score = mat_a[:,0]
final_score = mat_a[:1]

max(mid_score)

# Q5. 중간고사 성적이 평균보다 높은 학생들의 기말고사 성적 평균은?
mid_mean = mat_a[:,0].mean() # 중간고사 성적 평균
mat_a[mat_a[:,0]>mid_mean,1].mean() # 51.06891891891892

# Q6. 중간고사 대비 기말고사 성적이 향상된 학생들은 몇명인가요?
sum(mat_a[:,0] < mat_a[:,1])

# Q7. 반대로 성적이 떨어진 학생들은 어디에 위치해 있나요? 인덱스 정보
mat_a[:,0] > mat_a[:,1]
ans_idx = np.where(mat_a[:,0] > mat_a[:,1])[0] #where은 튜플로 출력하기 때문에 [0]을 해줘야함
mat_a[ans_idx].shape
mat_a[ans_idx,:].shape