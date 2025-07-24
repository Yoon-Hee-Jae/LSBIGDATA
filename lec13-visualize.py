import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.plot([4,1,3,2])
plt.ylabel('Some Number')
plt.show()
# marker = 'o' 점표시 // linestyle = '' 선 스타일
plt.plot([4,1,3,2], marker='o', linestyle='None')
plt.ylabel('Some Number')
plt.show()
# x-y plot ( 산점도 그래프 )
plt.plot([1,2,3,4],[1,4,9,16], marker='o',linestyle='None')
plt.show()

plt.plot((np.arange(10),
          np.arange(10)),
            marker='o', linestyle='None')
plt.ylabel('Some Number')
plt.show()

# 펭귄 데이터 불러오기
df = pd.read_csv('./data/penguins.csv')
df.info()

# 부리 길이 x 축 부리 깊이 y축
plt.plot(df['bill_length_mm'],
         df['bill_depth_mm'],
           marker='o', linestyle='None',color='red')
plt.ylabel('bill_depth')
plt.xlabel('bill_length')
plt.show()

# 산점도 쉽게 그리는 방법
plt.scatter(df['bill_length_mm'],
            df['bill_depth_mm'],
            c='red')
# 점별로 다른 색깔 
x = np.repeat('red',100)
y = np.repeat('blue',244)
my_color = np.concatenate([x,y])

plt.scatter(df['bill_length_mm'],
            df['bill_depth_mm'],
            c= my_color)
# 펭귄 종류별로 다른 색깔
df['species'].unique()
df.isnull().sum()

df['color'] = None
# for 문 사용법
for i in range(len(df)):
    if df['species'][i] == 'Adelie':
        df['color'][i] = 'red'
    elif df['species'][i] == 'Chinstrap':
        df['color'][i] = 'blue'
    else:
        df['color'][i] = 'green'

plt.scatter(df['bill_length_mm'],
            df['bill_depth_mm'],
            c= df['color'])

# dict 사용법
color_map = {
    "Adelie":"red",
    "Chinstrap":"blue",
    "Gentoo":"green"
}
color_vec = df['species'].map(color_map)
color_vec

plt.scatter(df['bill_length_mm'],
            df['bill_depth_mm'],
            c= color_vec)


##
plt.scatter(df['bill_length_mm'],
            df['bill_depth_mm'],
            c= np.repeat(1,344)) # 색깔은 숫자로도 표현 가능하다

df['species'] = df['species'].astype('category')
df['species'].cat.codes

# subplot >> (a,b,c) a세로 * b가로 * c인덱스

plt.plot([1, 2, 3, 4], [10, 20, 30, 40])
plt.text(2, 25, # 특정 위치에 텍스트 추가
        'Important Point', 
        fontsize=12, 
        color='red')
plt.show()

####################################################################


# 펭귄 데이터 불러오기
df = pd.read_csv('./data/penguins.csv')
df.info()

# 펭귄데이터 불러오자
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/penguins.csv')
df.info()

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 종별 평균 부리 길이 계산 및 내림차순 정렬
mean_bill_length = df.groupby('species')['bill_length_mm'].mean().sort_values(ascending=False)

# 색상 지정: Chinstrap은 빨간색, 나머지는 회색
colors = ['red' if species == 'Chinstrap' else 'gray' for species in mean_bill_length.index]

# 막대그래프 그리기
plt.bar(mean_bill_length.index, mean_bill_length.values, color=colors)
plt.xlabel('펭귄 종류')
plt.ylabel('평균 부리 길이 (mm)')
plt.title('펭귄 종류별 평균 부리 길이 (내림차순)')

# 막대 위에 값 표시
for idx, value in enumerate(mean_bill_length.values):
    plt.text(idx, value + 0.1, f'{value:.1f} mm', ha='center', va='bottom', fontsize=10)

plt.show()

# 섬별 몸무게 평균 막대그래프

df.info()
# 섬별 평균 몸무게 계산 (결측치 자동 제외됨)
mean_weight = df.groupby('island')['body_mass_g'].mean().sort_values()

# 색상 지정: 가장 몸무게가 낮은 섬은 파랑, 나머지는 회색
colors = ['blue' if island == mean_weight.idxmin() else 'gray' for island in mean_weight.index]

# 막대그래프 그리기
plt.bar(mean_weight.index, mean_weight.values, color=colors)
plt.xlabel('섬')
plt.ylabel('평균 몸무게 (g)')
plt.title('펭귄 섬별 평균 몸무게')

# 막대 위에 숫자 표시
for idx, value in enumerate(mean_weight.values):
    plt.text(idx, value + 20, f'{value:.1f}g', ha='center', va='bottom', fontsize=10)

plt.show()

# 펭귄 종별 부리길이(x) vs 깊이(y) 산점도
# 한글 제목, x축, y축 제목 설정
# 아델리 - 빨간색
# 찬스트랩 - 회색
# 겐투 - 파란색
# 범례 - 오른쪽 하단 위치
# 아델리 평균 중심점 표시
# 점찍고 텍스트로 아래와 같이 출력
# 평균 부리길이: xx.xx mm, 평균 부리깊이: xx.xx mm


# 부리 길이 및 깊이에 결측치 제거
df_plot = df[['species', 'bill_length_mm', 'bill_depth_mm']].dropna()

# 종별 색상 지정
color_map = {
    'Adelie': 'red',
    'Chinstrap': 'gray',
    'Gentoo': 'blue'
}

# 산점도 그리기
plt.figure(figsize=(8, 6))
for species, color in color_map.items():
    subset = df_plot[df_plot['species'] == species]
    plt.scatter(subset['bill_length_mm'], subset['bill_depth_mm'], 
                label=species, color=color)

# 아델리 평균 중심점 계산 및 표시
adelie = df_plot[df_plot['species'] == 'Adelie']
mean_x = adelie['bill_length_mm'].mean()
mean_y = adelie['bill_depth_mm'].mean()
plt.scatter(mean_x, mean_y, color='black', s=100, marker='x', label='Adelie 평균')

# 평균점 텍스트 표시 (오른쪽 위에 배치하고 화살표로 연결)
text_str = f'평균 부리길이: {mean_x:.2f} mm\n평균 부리깊이: {mean_y:.2f} mm'
plt.annotate(
    text_str,
    xy=(mean_x, mean_y),              # 화살표 출발점
    xytext=(mean_x + 2, mean_y + 2),  # 텍스트 위치 (오른쪽 위)
    textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray')
)

# 제목 및 축
plt.title('펭귄 종별 부리 길이 vs 깊이')
plt.xlabel('부리 길이 (mm)')
plt.ylabel('부리 깊이 (mm)')

# 범례: 오른쪽 하단
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
##################################

# 필요한 열에서 결측치 제거
df_plot = df[['species', 'bill_length_mm', 'bill_depth_mm']].dropna()

# Adelie vs 기타종으로 분리
adelie = df_plot[df_plot['species'] == 'Adelie']
others = df_plot[df_plot['species'] != 'Adelie']

# 산점도 그리기
plt.figure(figsize=(8, 6))
plt.scatter(others['bill_length_mm'], others['bill_depth_mm'],
            color='gray', label='기타종', alpha=0.7)
plt.scatter(adelie['bill_length_mm'], adelie['bill_depth_mm'],
            color='red', label='Adelie', alpha=0.7)

# 아델리 평균 중심점 계산 및 표시
mean_x = adelie['bill_length_mm'].mean()
mean_y = adelie['bill_depth_mm'].mean()
plt.scatter(mean_x, mean_y, color='black', s=100, marker='x', label='Adelie 평균')

# 평균 텍스트: 왼쪽 하단 배치
text_str = f'평균 부리길이: {mean_x:.2f} mm\n평균 부리깊이: {mean_y:.2f} mm'
plt.annotate(
    text_str,
    xy=(mean_x, mean_y),
    xytext=(mean_x - 3, mean_y - 1),
    textcoords='data',
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray')
)

# 제목 및 축
plt.title('펭귄 종별 부리 길이 vs 깊이')
plt.xlabel('부리 길이 (mm)')
plt.ylabel('부리 깊이 (mm)')

# 범례: 오른쪽 하단
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


######################################################
# 섬 이름 한글 매핑
island_kor_map = {
    'Biscoe': '비스코 섬',
    'Dream': '드림 섬',
    'Torgersen': '토거센 섬'
}

# 종별 색상 매핑
species_color_map = {
    'Adelie': 'red',
    'Chinstrap': 'green',
    'Gentoo': 'blue'
}

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('각 성별 펭귄 종 서식 현황', fontsize=24)

species_labels = ['Adelie', 'Chinstrap', 'Gentoo']
handles = []

for ax, island in zip(axes, df['island'].dropna().unique()):
    data = df[df['island'] == island]['species'].value_counts()
    labels = data.index
    sizes = data.values
    colors = [species_color_map.get(label, 'gray') for label in labels]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        textprops={'fontsize': 20},
        colors=colors
    )
    ax.set_title(island_kor_map[island], fontsize=18)

    if not handles:  # 첫 번째 반복일 때만 범례용 핸들 생성
        handles = wedges

# 범례 추가 (전체 하나만)
fig.legend(handles, species_labels, loc='lower right', fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()