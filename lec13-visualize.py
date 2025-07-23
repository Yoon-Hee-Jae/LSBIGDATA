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