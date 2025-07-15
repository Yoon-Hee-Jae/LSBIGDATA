import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
img1

# 행렬을 이미지로 표시
plt.figure(figsize=(10, 5))  # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

# 행렬 다운로드
img_mat = np.loadtxt('./data/img_mat.csv', delimiter=',', skiprows=1)
img_mat

print("행렬의 크기:", img_mat.shape)
img_mat.max() # 최댓값 255
img_mat.min() # 최솟값 0

# 행렬 값을 0과 1 사이로 변환
img_mat = img_mat / 255.0 # 시각화를 위해서 0~1사이로 바꿔줘야함
# 행렬을 이미지로 변환하여 출력
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();


# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)

# 행렬을 전치
transposed_x = x.transpose() # 행 > 열 전환 # 1열을 1행으로 보냄
print("전치된 행렬 x:\n", transposed_x)

# 밝기를 높여주기 > 숫자를 키워주면 됨
1단계 : 전체에 0.2를 더함
2단계 : 1이상 값을 가지는 애들 -> 1 변환

# 내 풀이
bright_img = (img_mat + 0.2)/1.2

plt.imshow(bright_img, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

# 정답
# 필터링: 1보다 큰 값은 1로 설정
img_mat_2 = img_mat + 0.2
img_mat_2[img_mat>1] = 1.0
plt.imshow(img_mat_2, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();


# 2행 3열의 행렬 y 생성
x
y = np.arange(1, 7).reshape((2, 3))
print("행렬 x:\n", x)
print("행렬 y:\n", y)
print("행렬 x, y의 크기:", x.shape, y.shape)

# 행렬곱 계산
dot_product = x.dot(y)
np.matmul(x,y) # 행렬곱 함수
print("행렬곱 x * y:\n", dot_product)

mat_a = np.array([[1,2],[3,4]])
mat_b = np.array([[2,1],[3,1]])
print("행렬 mat_a:\n", mat_a)
print("행렬 mat_b:\n", mat_b)

# 행렬곱 3가지 표현 방법
dot_ab = mat_a.dot(mat_b)
np.matmul(mat_a,mat_b)
mat_a @ mat_b
# * 는 행렬곱이 아닌 똑같은 위치의 원소끼리 곱셈을 해주는 기호
mat_a * mat_b

# 역행렬 
# 행렬의 세게: 1 == 단위행렬
# 행렬의 세게: 역수 == 역행렬
mat_A = np.array([[1, 2], [4, 3]])
np.linalg.inv(mat_A)
mat_A @ np.linalg.inv(mat_A) # 행렬 * 역행렬 = 단위행렬
# 단위행렬: 대각선이 모두 1이고 나머지는 모두 0
np.eye(4) # eye: 단위행렬을 만드는 함수
mat_A @ np.eye(2) # 행렬 * 단위행렬은 언제나 행렬이 그대로 나옴

# 역행렬이 존재하는 행렬도 있고 없는 행렬도 있다.
mat_C = np.array([[3,1],
                  [6,2]])
inv_C = np.linalg.inv(mat_C) 
# Singular matrix라는 에러코드가 뜸 = mat_C는 역행렬이 존재하지 않음

# 역행렬이 존재하는 행렬 VS. 존재하지 않는 행렬
# non-singular vs. singular
# mat_C의 경우 1번째 칼럼(열)에 3을 곱하면 0번째 칼럼을 만들 수 있음
# 이럴 경우 역행렬이 존재하지 않음
# 데이터 분석 측면에서 이해하자면 각 열이 독립적이지 않고 연관성이 깊다는 의미
# 칼럼이 선형 독립인 경우 -> 역행렬 존재
# 칼럼이 선형 종속인 경우 -> 역행렬 존재 X


# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(3, 2)
mat2 = np.arange(7, 13).reshape(3, 2)
print("mat1 \n",mat1,"\n","mat2 \n",mat2)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array
my_array.shape  #(차원수,행,열)
my_array[:,1,:]
my_array.reshape(2,3,2)
my_array.reshape(-1,3,2)

import imageio
# 이미지 읽기
cat = imageio.imread("cat.png")
print("이미지 클래스:", type(cat))
print("이미지 차원:", cat.shape) # 마지막이 4인 이유 rgb + 알파(투명도)

plt.imshow(cat);
plt.axis('off');
plt.show();
# 흑백으로 전환
bw_cat = np.mean(cat[:, :, :3], axis=2)
plt.imshow(bw_cat, cmap='gray');
plt.axis('off');
plt.show();


####
data = np.array([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]]
                 ,
                 [[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]]])

# 2 x 3 x 4 행렬 (2채널 3행 4열)

print(data.shape, end="\n\n")   #(2, 3, 4)

print(np.mean(data, axis=0),end="\n\n") # 각 채널 같은 위치 원소끼리 평균 (3x4)
print(np.mean(data, axis=1),end="\n\n") # 각 채널의 열끼리의 평균 (2x4)
print(np.mean(data, axis=2),end="\n\n") # 각 채널의 행끼리의 평균 (2x3)

print(np.mean(data, axis=(1,2))) # 각 채널의 평균 (1x2)