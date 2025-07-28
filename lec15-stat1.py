import numpy as np

# 1~10까지 랜덤 숫자 5개 뽑기
samples = np.random.choice(np.arange(1, 11), size=5, replace=False)
# repalce=False 복원추출 여부
samples
