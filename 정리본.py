1. idmax pandas 전용 함수 - dataframe / series

2. np.where 사용법
max_ass = df.iloc[np.where(df['assignment']==max(df['assignment']))[0],:]

3. sort_values() 
기본값 ascendnig=True 오름차순
df.sort_values("final",ascending=False).head()


4. groupby()
as_index=False 그룹화 기준 열 행 삭제
df.groupby('gender',as_index=False)[['midterm','final']].mean()
여러개를 그룹화하고 sort_values를 할 경우 
l_df.sort_values(['학년','반'],ascending=[True,False])
이런식으로 해주면 된다

5. .mean()
axis=0 열 평균 // axis=1 행 평균

6. np.unique()
# 인수의 종류 뿐만 아니라 각 갯수까지 파악 가능 return_counts
a = np.array([1, 3, 3, 2, 1, 3, 4, 2, 2, 2, 5, 6, 6, 6, 6])
unique, counts = np.unique(a,return_counts=True)

7.random 숫자 추출
np.random.seed(2023)
# 행렬 B 생성
B = np.random.choice(range(1, 11), 20, replace=True)

8. 역행렬
np.linalg.inv()
mat_A @ np.linalg.inv(mat_A) # 행렬 * 역행렬 = 단위행렬
# 행렬의 세게: 1 == 단위행렬 : 대각선이 모두 1이고 나머지는 모두 0

9. .T
전치행렬 구하기

10. 데이터프레임은 기본적으로 딕셔너리로 구성한다.
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})

11. merge(
    total_df = pd.merge(mid_df1,fin_df2,on='std_id',how='outer')
# merge의 how에는 left,right라는 옵션도 있음
# on에 사용할 열(cols)의 이름이 다를 경우
# left_on='열 이름', right_on='열 이름' 으로 작성해주면 된다
# pd.merge(mid_df1,fin_df2,left_on='std_id',right_on='student_id,how='left')
# del df['std_id']
# 왼 오 on에 사용된 열이 두개 생성되므로 하나를 지워주는 작업이 필요함
left_df = pd.merge(mid_df1,fin_df2,on='std_id',how='left')
left_df
# dataframe 열 이름 변경 함수 rename
left_df.rename(columns={'score_x':'first'})
)

12.series 생성법
data = [10, 20, 30]
df_s = pd.Series(data, index=['one', 'two', 'three'], 
                 name = 'count')
print(df_s)

13. 행렬곱 표현 방식 3가지
dot_ab = mat_a.dot(mat_b)
np.matmul(mat_a,mat_b)
mat_a @ mat_b

14. numpy의 배열은 데이터 타입을 통일시켜줘야함
d = np.array(["q",2]) # 2를 텍스트 2로 자동 저장
a.cumsum() # 원소의 누적합을 구해줌
np.tile([1,3,5],4) # 전체 반복
np.repeat([1,3,5],4) # 리스트 원소마다 반복

15. 튜플 생성 주의 사항
b = (42,) # 원소를 하나만 입력할 땐 ,를 꼭 해줘야함
ㅁ = 34, 43 
ㅁ # 자동으로 튜플 생성

16. 데이터타입 헷갈리는 것들
tuple은 기존 원소에 대한 수정이 불가능하다. 추가 및 제거는 가능
변경 불가능한 데이터타입 = 문자열, 튜플
# 유일한 변경 튜플
tup = tuple(['foo',[1,2],True])
tup[1].append(3)
# 튜플내에 저장된 객체는 그 위치에서 바로 변경 가능
set은 순서가 없다 (dict는 원래는 순서가 없지만 현재 파이썬 버전은 부분적으로 인정)
dictionary의 key는 변경이 불가능한 값만 와야하기 때문에 튜플은 가능하지만
리스트는 key로서 설정이 불가능하다

17. \를 앞에 붙여주면 따옴표를 특수 기호 형식으로 넣을 수 있다
"Issac\"s name"

18. str 데이터 함수
find() - 찾고 싶은 문자의 인덱스를 얻어내는 함수
pin = '881120-1068234'
num = pin[pin.find('-')+1:]
replace() - 특정 문자 교체
a="a:b:c:d"
b=a.replace(":","#")
join() - 리스트 속 문자열을 이어주는 문자를 지정
a=["Life","is","too","short"]    
result = " ".join(a)
strip() - 공백제거 왼오도 가능
count() - 문자 개수 세기
a = 'hobby'
a.count('b')
index() - 해당 문자열의 첫번째 인덱스 추출
a.index('b')
replace() - 문자 교체
a.replace("h","b")
split() - 문자열 나누기
a = "life is too short"
a.split() # (':')안에 특정 문자 지정 가능

19. list 함수
a = [1,2,3]
insert() - 지정 위치 추가
a.insert(0,4)
remove() - 첫번째로 나오는 x를 삭제
a.remove(2)
pop() - 맨 마지막 요소를 리턴하고 삭제
a.pop()
a.pop(1) # 지정된 인덱스의 요소를 리턴하고 삭제

19. 행렬곱을 이용한 실사용 사례 가중평균
# 가중 평균 / 0.5일경우 평균
A = np.array([[1,2],[3,4],[5,6]])
v = np.array([[0.4],[0.6]])
A.shape
v.shape
A @ v
 
20. 본인끼리 곱했을 때 본인이 나옴 = 멱등행렬(idempotent)

21. Trace 성질
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = np.array([[9,10],[11,12]])
(A@B)@C
A@(B@C)
(B@C)@A
(C@A)@B # 행렬의 순서가 변하면 값은 변하지만 대각선의 합은 항상 같다. 
# 행렬의 TRACE 성질

22. 집합 연산
# 합집합
set_x.union(set_y)
set_x | set_y
# 교집합
set_x & set_y
# 차집합
set_x - set_y

23. 튜플에서 여러개의 값을 반환해야할 때
values = 1,2,3,4,5
values
a,b, *rest = values
rest

24. 기존에 리스트가 있을 경우 extend()
x = [4,None,'foo']
x.extend([7,8,(2,3)])
x
